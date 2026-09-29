"""
Punto de entrada de alto nivel del pipeline completo: preprocesamiento +
OCR + parser + validaciones + reintentos + máquina de estados +
enriquecimiento con timbre electrónico, todo en una sola llamada. Esto
es lo que la Etapa 8 conecta al endpoint de la API.
"""
import logging

import numpy as np

from app.config import Settings
from app.models.enums import InputSource
from app.models.schemas import DatosExtraidos, ScanResponse
from app.ocr.ocr_engine import MotorOCRError
from app.pipeline.retry_orchestrator import ejecutar_pipeline
from app.pipeline.state_machine import determinar_estado
from app.pipeline.timbre_enrichment import enriquecer_con_timbre
from app.responses.error_responses import respuesta_imagen_borrosa, respuesta_timeout

logger = logging.getLogger(__name__)

# Rótulos genéricos para mostrar cuando el dato no se pudo identificar —
# a pedido explícito de Camilo. "Sin especificar" es literalmente el
# mismo texto que ya usa el selector de método de pago en el formulario
# de "Ingresar manualmente" de Blynn (ese campo ya es opcional ahí).
# Fecha y total quedan afuera a propósito: son los dos campos donde un
# rótulo de relleno podría llegar a confundirse con un dato real (una
# fecha o un monto con pinta de correcto) — eso sí chocaría con el
# contrato de `DatosExtraidos` ("nunca se inventa información
# faltante"). "Sin título" y "Sin especificar" no: son evidentemente
# genéricos, no plausiblemente reales.
_SIN_TITULO = "Sin título"
_SIN_ESPECIFICAR = "Sin especificar"


def aplicar_valores_por_defecto(datos: DatosExtraidos) -> DatosExtraidos:
    """Sustituye, solo para mostrar, comercio y método de pago ausentes
    por un rótulo genérico. Se aplica al final, después de que
    confianza/advertencias/estado ya se calcularon con el dato real
    (`None`) — esta sustitución nunca cambia el puntaje ni el estado,
    solo lo que termina viendo el usuario en la respuesta.
    """
    return datos.model_copy(
        update={
            "comercio": datos.comercio or _SIN_TITULO,
            "metodo_pago": datos.metodo_pago or _SIN_ESPECIFICAR,
        }
    )


def procesar_boleta(imagen_bgr: np.ndarray, source: InputSource, settings: Settings) -> ScanResponse:
    """Ejecuta el pipeline completo sobre una imagen ya validada (Etapa 2)
    y devuelve la respuesta final lista para el frontend.

    Lanza MotorOCRError si se intentó al menos una estrategia y TODAS
    fallaron con un error del motor OCR (situación de infraestructura,
    no de calidad de la boleta); la Etapa 8 la traduce a una respuesta
    controlada (HTTP 503) en el endpoint.

    Si en cambio no se llegó a intentar ninguna estrategia, hay dos
    motivos posibles: la imagen resultó demasiado borrosa/sin contenido
    (se corta antes de la Etapa 4, ver Settings.UMBRAL_NITIDEZ_MINIMA) o
    el límite de tiempo se agotó incluso antes de la primera estrategia
    (caso extremo). En ambos casos es sobre la calidad/tamaño de la
    imagen, no sobre el motor OCR, así que se devuelve una respuesta
    normal en vez de lanzar una excepción.
    """
    resultado_pipeline = ejecutar_pipeline(imagen_bgr, settings)
    mejor = resultado_pipeline.mejor_intento

    if mejor is None:
        if resultado_pipeline.limite_alcanzado == "calidad":
            return respuesta_imagen_borrosa(source)
        if resultado_pipeline.intentos_realizados == 0:
            return respuesta_timeout(source)
        raise MotorOCRError(
            "No fue posible completar el reconocimiento con ninguna estrategia."
        )

    datos_finales = mejor.datos
    evaluacion_final = mejor.evaluacion
    timbre_decodificado = False

    # El enriquecimiento con timbre es una mejora de "mejor esfuerzo": se
    # omite si el orquestador ya agotó el presupuesto de tiempo (evita
    # que esta mejora opcional empuje una solicitud lenta hacia el
    # timeout duro del endpoint). Se usa el documento ya recortado, sin
    # las mejoras de texto (ver ResultadoPipeline.imagen_para_timbre) en
    # vez de la foto original entera.
    if resultado_pipeline.limite_alcanzado != "tiempo":
        imagen_para_timbre = (
            resultado_pipeline.imagen_para_timbre
            if resultado_pipeline.imagen_para_timbre is not None
            else imagen_bgr
        )
        enriquecimiento = enriquecer_con_timbre(
            imagen_para_timbre, mejor.datos, mejor.evaluacion, mejor.confianza_ocr
        )
        datos_finales = enriquecimiento.datos
        evaluacion_final = enriquecimiento.evaluacion
        timbre_decodificado = enriquecimiento.timbre_decodificado

    estado, mensaje = determinar_estado(
        datos_finales, evaluacion_final, mejor.confianza_ocr, source, settings
    )

    # El folio se extrae para "facilitar el reconocimiento del documento"
    # (spec, sección "Información a extraer"): se usa aquí para
    # identificar la boleta en los logs del servidor, sin exponerlo
    # nunca en la respuesta al frontend.
    logger.info(
        "Boleta procesada: folio=%s estado=%s estrategia=%s intentos=%d confianza=%.1f timbre=%s",
        mejor.folio or "desconocido",
        estado.value,
        mejor.estrategia,
        resultado_pipeline.intentos_realizados,
        evaluacion_final.confianza_general,
        timbre_decodificado,
    )

    return ScanResponse(
        estado=estado,
        mensaje=mensaje,
        confianza_general=evaluacion_final.confianza_general,
        confianza_por_campo=evaluacion_final.confianza_por_campo,
        advertencias=evaluacion_final.validacion.advertencias,
        datos=aplicar_valores_por_defecto(datos_finales),
    )

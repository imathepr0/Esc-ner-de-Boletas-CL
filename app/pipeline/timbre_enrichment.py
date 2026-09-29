"""
Enriquecimiento del resultado del pipeline con datos del timbre
electrónico (PDF417), cuando se logra decodificar (ver `app/timbre/`).

Es una mejora de "mejor esfuerzo" sobre el resultado ya obtenido por OCR
(Etapas 4-7): nunca bloquea ni retrasa el flujo principal si el timbre
no decodifica —el caso más común en una foto de la boleta completa—, y
nunca degrada un resultado ya bueno del OCR. Solo reemplaza fecha/total
cuando el timbre efectivamente trae ese dato, porque un código de
barras leído es más preciso que texto reconocido por OCR sobre una foto.
"""
from dataclasses import dataclass

import numpy as np

from app.models.schemas import DatosExtraidos
from app.receipt_validation.consistency_checks import validar_datos
from app.receipt_validation.document_confidence import calcular_confianza_general
from app.receipt_validation.receipt_validator import ResultadoEvaluacion
from app.timbre.barcode_decoder import decodificar_timbre
from app.timbre.ted_parser import DatosTED, parsear_ted

# Confianza asignada a un campo reemplazado por el dato del timbre: un
# código de barras decodificado correctamente es una lectura exacta, no
# una estimación — se le asigna la confianza máxima.
_CONFIANZA_DATO_DE_TIMBRE = 100.0


@dataclass
class ResultadoEnriquecimiento:
    """Resultado de intentar enriquecer con el timbre: si se decodificó
    algo útil o no, y los datos/evaluación resultantes (idénticos a los
    originales, sin copiar de más, si no se decodificó nada útil)."""

    timbre_decodificado: bool
    datos: DatosExtraidos
    evaluacion: ResultadoEvaluacion


def _tiene_datos_utiles(ted: DatosTED) -> bool:
    return ted.fecha_emision is not None or ted.monto_total is not None


def enriquecer_con_timbre(
    imagen_bgr: np.ndarray,
    datos: DatosExtraidos,
    evaluacion: ResultadoEvaluacion,
    confianza_ocr_general: float,
) -> ResultadoEnriquecimiento:
    """Intenta decodificar el timbre electrónico de la imagen y, si trae
    fecha y/o total, los usa para reemplazar (con confianza 100) los
    valores que había extraído el OCR — recalculando validaciones y
    confianza general para que sigan siendo consistentes con los datos
    finales (p. ej. el cuadre suma-productos-vs-total debe compararse
    contra el total ya actualizado, no el original).

    Si no se detecta/decodifica ningún timbre, o decodifica pero no trae
    ningún dato utilizable, devuelve `datos`/`evaluacion` sin modificar
    (mismos objetos, no copias) — el resultado es idéntico a como si
    esta función no se hubiera llamado.
    """
    texto_ted = decodificar_timbre(imagen_bgr)
    if texto_ted is None:
        return ResultadoEnriquecimiento(False, datos, evaluacion)

    ted = parsear_ted(texto_ted)
    if not _tiene_datos_utiles(ted):
        return ResultadoEnriquecimiento(False, datos, evaluacion)

    nuevos_datos = datos.model_copy()
    nueva_confianza_campos = evaluacion.confianza_por_campo.model_copy()

    if ted.fecha_emision is not None:
        nuevos_datos.fecha = ted.fecha_emision
        nueva_confianza_campos.fecha = _CONFIANZA_DATO_DE_TIMBRE

    if ted.monto_total is not None:
        nuevos_datos.total = ted.monto_total
        nueva_confianza_campos.total = _CONFIANZA_DATO_DE_TIMBRE

    nueva_validacion = validar_datos(nuevos_datos, nueva_confianza_campos)
    nueva_confianza_general = calcular_confianza_general(
        nueva_confianza_campos, confianza_ocr_general, nueva_validacion.penalizacion
    )

    nueva_evaluacion = ResultadoEvaluacion(
        confianza_por_campo=nueva_confianza_campos,
        confianza_general=nueva_confianza_general,
        validacion=nueva_validacion,
    )

    return ResultadoEnriquecimiento(True, nuevos_datos, nueva_evaluacion)

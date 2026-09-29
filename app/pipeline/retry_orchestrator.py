"""
Orquestador del reintento automático: prueba distintas estrategias
(Etapa 7) hasta encontrar una lectura suficientemente buena, respetando
límites de cantidad de intentos y de tiempo total de procesamiento (ver
"Reintentos automáticos" en las especificaciones).

El bucle recorre una lista FINITA de estrategias (nunca un while True),
así que nunca puede entrar en un ciclo infinito.
"""
import time
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.image_processing.orientation import estimar_rotacion_via_osd
from app.image_processing.preprocessor import preprocesar_imagen, rotar_resultado_preprocesamiento
from app.models.schemas import DatosExtraidos
from app.ocr.ocr_engine import MotorOCRError, PalabraOCR, ResultadoOCR, reconocer_texto
from app.parsing.receipt_parser import parsear_boleta
from app.pipeline.strategies import ESTRATEGIAS, obtener_imagen_para_estrategia
from app.receipt_validation.receipt_validator import ResultadoEvaluacion, evaluar_lectura


@dataclass
class IntentoLectura:
    """El resultado completo de un único intento (una estrategia)."""

    estrategia: str
    datos: DatosExtraidos
    palabras: list[PalabraOCR]
    confianza_ocr: float
    evaluacion: ResultadoEvaluacion
    folio: str | None = None


@dataclass
class ResultadoPipeline:
    """Resultado final del pipeline completo, tras probar las
    estrategias necesarias."""

    mejor_intento: IntentoLectura | None
    intentos_realizados: int
    tiempo_total_segundos: float
    documento_detectado: bool
    limite_alcanzado: str | None  # "intentos", "tiempo", "calidad", o None
    nitidez: float = 0.0
    factor_escala_aplicado: float = 1.0
    # Documento ya recortado (si se detectó), a color y SIN las mejoras
    # de contraste/ruido/nitidez pensadas para texto — para reusar en el
    # enriquecimiento con timbre (ver pipeline_service.py): decodificar
    # el PDF417 sobre esto en vez de la foto original entera le da mejor
    # proporción código/imagen (sin fondo/mesa alrededor de la boleta),
    # y evitar las mejoras de texto importa tanto como el recorte en sí
    # — CLAHE/filtro bilateral/enfoque degradan el patrón del código lo
    # suficiente como para que deje de decodificar (confirmado en esta
    # sesión).
    imagen_para_timbre: np.ndarray | None = None


def _puntaje_intento(intento: IntentoLectura) -> float:
    """Puntaje para comparar intentos entre sí: la confianza general
    calculada en la Etapa 6."""
    return intento.evaluacion.confianza_general


def _es_resultado_suficiente(evaluacion: ResultadoEvaluacion, settings: Settings) -> bool:
    """Un resultado ya es lo bastante bueno como para no seguir
    intentando otras estrategias (evita reprocesar innecesariamente)."""
    return (
        evaluacion.confianza_general >= settings.MIN_CONFIDENCE_ACCEPTABLE
        and not evaluacion.validacion.campos_faltantes
    )


def _puntaje_ocr_suficiente(resultado_ocr: ResultadoOCR, settings: Settings) -> bool:
    """Indica si la lectura cruda de OCR de la variante de imagen actual
    ya es lo bastante buena como para no seguir probando otras variantes
    (gris rotada, binaria, binaria rotada) — Opción 6 de
    VISION_Y_ROADMAP.md.

    A diferencia de `_es_resultado_suficiente`, no depende del parsing
    downstream (Etapas 5/6): usa directamente el puntaje crudo de OCR
    (Etapa 4), que es la señal que de verdad separa una variante bien
    orientada de una rotada. Se necesita como corte aparte porque, contra
    fotos reales, la confianza general de campos nunca llegó a
    MIN_CONFIDENCE_ACCEPTABLE en la primera variante bien orientada —
    ese corte existente en la práctica nunca se activaba, y las 3
    variantes configuradas (MAX_RETRY_ATTEMPTS) se probaban siempre
    completas, con sus 4 PSM cada una.
    """
    return resultado_ocr.puntaje >= settings.UMBRAL_PUNTAJE_OCR_TEMPRANO


def _eje_probablemente_incorrecto(resultado_ocr: ResultadoOCR, settings: Settings) -> bool:
    """Señal barata de que el PRIMER intento de OCR podría estar leyendo
    el eje equivocado — ver README, "Quinta ronda: una foto real expone
    un bug en la corrección de rotación".

    Un puntaje bajo en el primer intento es ambiguo por sí solo: puede
    significar "está boca abajo dentro del eje correcto" (lo resuelve la
    siguiente estrategia, gris_rotada_180, sin gastar en OSD) o "la
    Etapa 3 eligió el eje equivocado" (eso ninguna estrategia existente
    puede corregir sola, porque todas rotan sobre el mismo eje que
    decidió la Etapa 3). Reusa el umbral de la Opción 6 a propósito: es
    exactamente el mismo criterio de "esto no se ve como texto real
    todavía" — si no lo cruza, vale la pena pagar el OSD (más confiable,
    pero 1-3s más caro que la heurística de la Etapa 3) antes de seguir
    gastando estrategias sobre un eje que podría estar mal desde el
    principio.
    """
    return resultado_ocr.puntaje < settings.UMBRAL_PUNTAJE_OCR_TEMPRANO


def ejecutar_pipeline(imagen_bgr: np.ndarray, settings: Settings) -> ResultadoPipeline:
    """Ejecuta el pipeline completo de reconocimiento sobre una imagen ya
    validada (Etapa 2): preprocesamiento (una sola vez) + OCR/parser/
    validación (repetido por estrategia) hasta encontrar una lectura
    aceptable o agotar los límites de intentos/tiempo/calidad.
    """
    inicio = time.monotonic()
    preprocesado = preprocesar_imagen(
        imagen_bgr, settings.MAX_IMAGE_DIMENSION, settings.ANCHO_OBJETIVO_ESCALADO
    )

    # Si la imagen está demasiado borrosa/sin contenido (ver
    # Settings.UMBRAL_NITIDEZ_MINIMA), se corta ACÁ, antes de intentar
    # cualquier estrategia de OCR: ahorra el tiempo que se habría
    # gastado en una imagen condenada a fallar.
    if preprocesado.nitidez < settings.UMBRAL_NITIDEZ_MINIMA:
        return ResultadoPipeline(
            mejor_intento=None,
            intentos_realizados=0,
            tiempo_total_segundos=time.monotonic() - inicio,
            documento_detectado=preprocesado.documento_detectado,
            limite_alcanzado="calidad",
            nitidez=preprocesado.nitidez,
            factor_escala_aplicado=preprocesado.factor_escala_aplicado,
            imagen_para_timbre=preprocesado.imagen_recortada,
        )

    mejor_intento: IntentoLectura | None = None
    intentos_realizados = 0
    limite_alcanzado: str | None = None

    estrategias_a_probar = ESTRATEGIAS[: settings.MAX_RETRY_ATTEMPTS]

    for idx, estrategia in enumerate(estrategias_a_probar):
        if time.monotonic() - inicio >= settings.MAX_PROCESSING_TIME_SECONDS:
            limite_alcanzado = "tiempo"
            break

        intentos_realizados += 1
        imagen_intento = obtener_imagen_para_estrategia(preprocesado, estrategia)

        try:
            resultado_ocr = reconocer_texto(imagen_intento, settings)
        except MotorOCRError:
            continue

        if idx == 0 and _eje_probablemente_incorrecto(resultado_ocr, settings):
            rotacion_osd = estimar_rotacion_via_osd(imagen_intento)
            if rotacion_osd in (90, 270):
                # La Etapa 3 probablemente eligió el eje equivocado (ver
                # README, "Quinta ronda"): corregir y repetir esta misma
                # estrategia sobre las imágenes ya rotadas. De acá en
                # adelante el resto del loop usa el `preprocesado`
                # corregido — no cuenta como un intento nuevo, es
                # arreglar el primero antes de seguir.
                preprocesado = rotar_resultado_preprocesamiento(preprocesado, rotacion_osd)
                imagen_intento = obtener_imagen_para_estrategia(preprocesado, estrategia)
                try:
                    resultado_ocr = reconocer_texto(imagen_intento, settings)
                except MotorOCRError:
                    continue

        resultado_parser = parsear_boleta(resultado_ocr.texto_completo)
        evaluacion = evaluar_lectura(
            resultado_parser.datos, resultado_ocr.palabras, resultado_ocr.confianza_promedio
        )

        intento = IntentoLectura(
            estrategia=estrategia.nombre,
            datos=resultado_parser.datos,
            palabras=resultado_ocr.palabras,
            confianza_ocr=resultado_ocr.confianza_promedio,
            evaluacion=evaluacion,
            folio=resultado_parser.folio,
        )

        if mejor_intento is None or _puntaje_intento(intento) > _puntaje_intento(mejor_intento):
            mejor_intento = intento

        if _es_resultado_suficiente(evaluacion, settings):
            break

        if _puntaje_ocr_suficiente(resultado_ocr, settings):
            # La lectura de OCR de esta variante ya es lo bastante buena
            # (ver `_puntaje_ocr_suficiente`) como para no seguir
            # probando otras variantes de imagen, aunque el parsing
            # downstream no haya alcanzado el umbral de confianza
            # general de campos que exige `_es_resultado_suficiente`.
            break

    if limite_alcanzado is None and intentos_realizados >= len(estrategias_a_probar):
        limite_alcanzado = "intentos"

    return ResultadoPipeline(
        mejor_intento=mejor_intento,
        intentos_realizados=intentos_realizados,
        tiempo_total_segundos=time.monotonic() - inicio,
        documento_detectado=preprocesado.documento_detectado,
        limite_alcanzado=limite_alcanzado,
        nitidez=preprocesado.nitidez,
        factor_escala_aplicado=preprocesado.factor_escala_aplicado,
        imagen_para_timbre=preprocesado.imagen_recortada,
    )

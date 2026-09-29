"""
Determinación del estado final del procesamiento (máquina de estados) y
construcción del mensaje descriptivo correspondiente, adaptado según el
origen de la imagen (cámara o archivo).

Ver "Estados posibles" y "Diferenciar origen" en las especificaciones.
Nota: UNSUPPORTED_FILE y el IMAGE_UNREADABLE por archivo corrupto/vacío
ya se resuelven antes, en la validación de entrada (Etapa 2); esta
máquina de estados decide entre los estados restantes, que dependen del
contenido semántico reconocido (o no) en la imagen.
"""
from app.config import Settings
from app.models.enums import InputSource, ProcessingState
from app.models.schemas import DatosExtraidos
from app.receipt_validation.receipt_validator import ResultadoEvaluacion
from app.responses.error_responses import sugerencia_por_origen


def _sugerencia_incompleta(source: InputSource) -> str:
    """Sugerencia específica para boletas que parecen cortadas."""
    if source == InputSource.CAMERA:
        return "Intenta capturar la boleta completa, incluyendo la parte inferior con el total."
    return "Verifica que la imagen incluya la boleta completa, incluyendo la parte inferior con el total."


def determinar_estado(
    datos: DatosExtraidos,
    evaluacion: ResultadoEvaluacion,
    mejor_confianza_ocr: float,
    source: InputSource,
    settings: Settings,
) -> tuple[ProcessingState, str]:
    """Decide el estado final según los datos extraídos y su confianza,
    y construye un mensaje descriptivo adaptado al origen de la imagen.

    Orden de evaluación (del caso más severo al más favorable):
    1. Prácticamente no se reconoció texto -> IMAGE_UNREADABLE
    2. Se reconoció texto, pero nada con forma de boleta -> NO_RECEIPT_DETECTED
    3. Falta el total o no hay productos (posible corte) -> INCOMPLETE_RECEIPT
    4. Confianza general baja -> LOW_CONFIDENCE
    5. Faltan campos secundarios o no cuadra la suma -> PARTIAL_SUCCESS
    6. Todo presente y con buena confianza -> SUCCESS
    """
    algun_campo_presente = bool(
        datos.comercio or datos.fecha or datos.total is not None or datos.productos
    )

    if mejor_confianza_ocr < settings.MIN_CONFIDENCE_LOW:
        return (
            ProcessingState.IMAGE_UNREADABLE,
            f"No fue posible leer el contenido de la imagen. {sugerencia_por_origen(source)}",
        )

    if not algun_campo_presente:
        return (
            ProcessingState.NO_RECEIPT_DETECTED,
            f"No se detectó una boleta en la imagen. {sugerencia_por_origen(source)}",
        )

    if datos.total is None or not datos.productos:
        campo_faltante = "el total" if datos.total is None else "ningún producto"
        return (
            ProcessingState.INCOMPLETE_RECEIPT,
            f"La boleta parece estar incompleta: no se pudo identificar "
            f"{campo_faltante}. {_sugerencia_incompleta(source)}",
        )

    if evaluacion.confianza_general < settings.MIN_CONFIDENCE_ACCEPTABLE:
        return (
            ProcessingState.LOW_CONFIDENCE,
            "La boleta se procesó, pero con confianza baja. Se recomienda "
            "revisar los datos manualmente.",
        )

    campos_secundarios_faltantes = not (datos.comercio and datos.fecha and datos.metodo_pago)
    cuadre_malo = evaluacion.validacion.suma_coincide_con_total is False

    if campos_secundarios_faltantes or cuadre_malo:
        return (
            ProcessingState.PARTIAL_SUCCESS,
            "Se reconocieron los datos principales de la boleta, pero "
            "algunos detalles no pudieron confirmarse con certeza.",
        )

    return ProcessingState.SUCCESS, "Boleta procesada correctamente."

"""
Construcción de respuestas para los casos detectados en la capa de
validación de entrada (formato no soportado, tamaño, imagen corrupta).

Los mensajes se adaptan según el origen de la imagen (cámara o
archivo), tal como pide la sección "Diferenciar origen" del spec.
"""
from app.models.enums import InputSource, ProcessingState
from app.models.schemas import ScanResponse
from app.validation.file_validation import RazonInvalidez, ResultadoValidacion

# Razones que implican que el archivo es del tipo/tamaño correcto, pero
# no se pudo leer contenido de imagen válido a partir de él.
_RAZONES_IMAGEN_NO_LEGIBLE = {
    RazonInvalidez.ARCHIVO_VACIO,
    RazonInvalidez.IMAGEN_CORRUPTA,
}

# Razones que implican que el archivo, en sí mismo, no cumple los
# requisitos de entrada (nunca se llegó a interpretar su contenido).
_RAZONES_FORMATO_NO_SOPORTADO = {
    RazonInvalidez.FORMATO_NO_SOPORTADO,
    RazonInvalidez.TIPO_CONTENIDO_NO_SOPORTADO,
    RazonInvalidez.ARCHIVO_DEMASIADO_GRANDE,
}


def sugerencia_por_origen(source: InputSource) -> str:
    """Sugerencia adaptada según si la imagen vino de la cámara o un
    archivo. Pública porque también la reutiliza la máquina de estados
    de la Etapa 7 (app/pipeline/state_machine.py)."""
    if source == InputSource.CAMERA:
        return (
            "Intenta mejorar la iluminación, acercar la cámara y encuadrar "
            "la boleta completa dentro del recuadro."
        )
    return (
        "Intenta subir una imagen de mayor resolución o verifica que el "
        "archivo corresponda efectivamente a una boleta."
    )


def respuesta_desde_validacion(
    resultado: ResultadoValidacion, source: InputSource
) -> ScanResponse:
    """Traduce un resultado de validación fallido en una respuesta pública."""
    if resultado.razon in _RAZONES_IMAGEN_NO_LEGIBLE:
        return ScanResponse(
            estado=ProcessingState.IMAGE_UNREADABLE,
            mensaje=f"{resultado.detalle} {sugerencia_por_origen(source)}",
            confianza_general=0.0,
            advertencias=[],
            datos=None,
        )

    if resultado.razon in _RAZONES_FORMATO_NO_SOPORTADO:
        return ScanResponse(
            estado=ProcessingState.UNSUPPORTED_FILE,
            mensaje=resultado.detalle or "El archivo no cumple los requisitos de entrada.",
            confianza_general=None,
            advertencias=[],
            datos=None,
        )

    # Resguardo defensivo: no debería alcanzarse (todas las razones de
    # RazonInvalidez quedan cubiertas por los dos conjuntos anteriores).
    return ScanResponse(
        estado=ProcessingState.UNSUPPORTED_FILE,
        mensaje=resultado.detalle or "No fue posible validar el archivo recibido.",
        confianza_general=None,
        advertencias=[],
        datos=None,
    )


def respuesta_timeout(source: InputSource) -> ScanResponse:
    """Respuesta cuando el procesamiento excede el tiempo máximo permitido
    (Etapa 8: "timeout del procesamiento" en la sección "Manejo de
    errores"). Se trata como IMAGE_UNREADABLE: no se logró obtener una
    lectura utilizable dentro del tiempo disponible.
    """
    return ScanResponse(
        estado=ProcessingState.IMAGE_UNREADABLE,
        mensaje=(
            "El procesamiento de la imagen demoró más de lo esperado y se "
            f"detuvo antes de completarse. {sugerencia_por_origen(source)}"
        ),
        confianza_general=None,
        advertencias=["El procesamiento se detuvo por exceder el tiempo máximo permitido."],
        datos=None,
    )


def respuesta_imagen_borrosa(source: InputSource) -> ScanResponse:
    """Respuesta cuando la imagen está demasiado borrosa (o sin
    contenido reconocible) como para siquiera intentar el OCR — ver
    `Settings.UMBRAL_NITIDEZ_MINIMA` y `retry_orchestrator.py`. Se corta
    antes de gastar el tiempo de varios intentos de OCR en una imagen
    condenada a fallar.

    Se trata como IMAGE_UNREADABLE (mismo estado público que otras
    imágenes ilegibles), pero con un mensaje específico sobre nitidez en
    vez del genérico — más accionable para quien vuelve a intentar.
    """
    return ScanResponse(
        estado=ProcessingState.IMAGE_UNREADABLE,
        mensaje=(
            "La imagen está demasiado borrosa o no tiene suficiente "
            f"definición para ser leída. {sugerencia_por_origen(source)}"
        ),
        confianza_general=0.0,
        advertencias=["El procesamiento se detuvo por baja nitidez de la imagen."],
        datos=None,
    )

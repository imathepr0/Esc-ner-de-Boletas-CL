"""
Enumeraciones centrales utilizadas por el resto del backend.

Mantener estos valores en un solo lugar evita "strings mágicos" repartidos
entre el parser, las validaciones, las respuestas y la API.
"""
from enum import Enum


class ProcessingState(str, Enum):
    """Estados posibles del resultado de un intento de escaneo."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    IMAGE_UNREADABLE = "IMAGE_UNREADABLE"
    INCOMPLETE_RECEIPT = "INCOMPLETE_RECEIPT"
    UNSUPPORTED_FILE = "UNSUPPORTED_FILE"
    NO_RECEIPT_DETECTED = "NO_RECEIPT_DETECTED"


class InputSource(str, Enum):
    """Origen de la imagen recibida, usado para adaptar los mensajes."""

    CAMERA = "camera"
    FILE = "file"


class ImageFormat(str, Enum):
    """Formatos de imagen soportados por el sistema."""

    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"

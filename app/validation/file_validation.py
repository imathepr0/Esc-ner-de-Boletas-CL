"""
Validación de entrada: formato, tamaño e integridad del archivo recibido.

Esta capa se ejecuta ANTES de cualquier preprocesamiento o intento de
OCR. Su único trabajo es determinar si el archivo recibido es un
candidato válido para el pipeline de reconocimiento; nunca interpreta
el contenido de la boleta.
"""
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from app.config import Settings


class RazonInvalidez(str, Enum):
    """Motivo interno por el cual un archivo no pasó la validación."""

    ARCHIVO_VACIO = "archivo_vacio"
    FORMATO_NO_SOPORTADO = "formato_no_soportado"
    TIPO_CONTENIDO_NO_SOPORTADO = "tipo_contenido_no_soportado"
    ARCHIVO_DEMASIADO_GRANDE = "archivo_demasiado_grande"
    IMAGEN_CORRUPTA = "imagen_corrupta"


@dataclass
class ResultadoValidacion:
    """Resultado de validar un archivo de entrada.

    Si `es_valido` es True, `imagen` contiene la matriz decodificada
    (BGR, formato OpenCV) lista para el pipeline de preprocesamiento.
    Si es False, `razon` y `detalle` describen qué falló.
    """

    es_valido: bool
    imagen: np.ndarray | None = None
    razon: RazonInvalidez | None = None
    detalle: str | None = None


def _extension_valida(nombre_archivo: str | None, settings: Settings) -> bool:
    """Verifica que la extensión del archivo esté entre las permitidas."""
    if not nombre_archivo or "." not in nombre_archivo:
        return False
    extension = nombre_archivo.rsplit(".", 1)[-1].lower()
    return extension in settings.ALLOWED_EXTENSIONS


def _mime_type_valido(content_type: str | None, settings: Settings) -> bool:
    """Verifica que el content-type declarado esté entre los permitidos."""
    return bool(content_type) and content_type.lower() in settings.ALLOWED_MIME_TYPES


def _decodificar_imagen(contenido: bytes) -> np.ndarray | None:
    """Intenta decodificar los bytes recibidos como una imagen válida.

    Devuelve la matriz de la imagen (formato OpenCV, BGR) o None si el
    archivo está corrupto o no es una imagen real. Nunca lanza excepciones:
    cualquier problema de decodificación se traduce en un resultado None.
    """
    try:
        buffer = np.frombuffer(contenido, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


def validar_archivo(
    contenido: bytes,
    nombre_archivo: str | None,
    content_type: str | None,
    settings: Settings,
) -> ResultadoValidacion:
    """Ejecuta todas las validaciones de entrada sobre el archivo recibido.

    Orden de verificación: archivo vacío -> extensión -> tipo de
    contenido -> tamaño -> decodificación. Se detiene en el primer
    problema encontrado (fail-fast).
    """
    if len(contenido) == 0:
        return ResultadoValidacion(
            es_valido=False,
            razon=RazonInvalidez.ARCHIVO_VACIO,
            detalle="El archivo recibido está vacío.",
        )

    if not _extension_valida(nombre_archivo, settings):
        return ResultadoValidacion(
            es_valido=False,
            razon=RazonInvalidez.FORMATO_NO_SOPORTADO,
            detalle=(
                "Formato de archivo no soportado. Formatos permitidos: "
                f"{', '.join(sorted(settings.ALLOWED_EXTENSIONS))}."
            ),
        )

    if not _mime_type_valido(content_type, settings):
        return ResultadoValidacion(
            es_valido=False,
            razon=RazonInvalidez.TIPO_CONTENIDO_NO_SOPORTADO,
            detalle=(
                f"Tipo de contenido no soportado ({content_type}). "
                f"Tipos permitidos: {', '.join(sorted(settings.ALLOWED_MIME_TYPES))}."
            ),
        )

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contenido) > max_bytes:
        return ResultadoValidacion(
            es_valido=False,
            razon=RazonInvalidez.ARCHIVO_DEMASIADO_GRANDE,
            detalle=(
                "El archivo excede el tamaño máximo permitido "
                f"({settings.MAX_FILE_SIZE_MB} MB)."
            ),
        )

    imagen = _decodificar_imagen(contenido)
    if imagen is None or imagen.size == 0:
        return ResultadoValidacion(
            es_valido=False,
            razon=RazonInvalidez.IMAGEN_CORRUPTA,
            detalle="El archivo está corrupto o no corresponde a una imagen válida.",
        )

    return ResultadoValidacion(es_valido=True, imagen=imagen)

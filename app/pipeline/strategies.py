"""
Definición de las estrategias de reintento: distintas combinaciones de
variante de imagen (escala de grises / binarizada) y orientación
(normal / rotada 180°) que se prueban en orden si la primera lectura no
es satisfactoria (ver "Reintentos automáticos" en las especificaciones).

Se reutilizan las imágenes ya calculadas por el preprocesamiento de la
Etapa 3 (no se vuelve a ejecutar la detección de documento ni la
corrección geométrica en cada intento): cada estrategia solo decide qué
variante de esas imágenes usar para el OCR.

El orden prioriza la variante en escala de grises (empíricamente más
confiable que la binarizada sobre fotos con ruido, ver Etapa 4) en sus
dos orientaciones posibles antes de recurrir a la binarizada: así se
resuelve primero la ambigüedad de orientación 0°/180° que quedó
pendiente desde la Etapa 3, con la imagen que suele dar mejores
resultados.
"""
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from app.image_processing.preprocessor import ResultadoPreprocesamiento


class VarianteImagen(str, Enum):
    GRIS = "gris"
    BINARIA = "binaria"


@dataclass
class Estrategia:
    """Una combinación concreta de variante de imagen + orientación a
    probar en un intento de OCR."""

    nombre: str
    variante: VarianteImagen
    rotar_180: bool


ESTRATEGIAS: list[Estrategia] = [
    Estrategia("gris", VarianteImagen.GRIS, rotar_180=False),
    Estrategia("gris_rotada_180", VarianteImagen.GRIS, rotar_180=True),
    Estrategia("binaria", VarianteImagen.BINARIA, rotar_180=False),
    Estrategia("binaria_rotada_180", VarianteImagen.BINARIA, rotar_180=True),
]


def obtener_imagen_para_estrategia(
    preprocesado: ResultadoPreprocesamiento, estrategia: Estrategia
) -> np.ndarray:
    """Devuelve la imagen concreta (ya calculada por la Etapa 3) que
    corresponde a una estrategia."""
    base = (
        preprocesado.imagen_gris
        if estrategia.variante == VarianteImagen.GRIS
        else preprocesado.imagen_binaria
    )
    return cv2.rotate(base, cv2.ROTATE_180) if estrategia.rotar_180 else base

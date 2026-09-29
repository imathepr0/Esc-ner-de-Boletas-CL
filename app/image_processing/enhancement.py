"""
Mejora de la imagen antes del OCR: escala de grises, reducción de
sombras, mejora de contraste, eliminación de ruido, nitidez y
binarización.

Cada función es independiente y puede reutilizarse por separado (por
ejemplo, para probar distintas combinaciones como estrategias de
reintento en la Etapa 7).
"""
import cv2
import numpy as np


def convertir_a_grises(imagen: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises si la imagen viene en color."""
    if imagen.ndim == 2:
        return imagen
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def reducir_sombras(imagen_gris: np.ndarray) -> np.ndarray:
    """Normaliza la iluminación para atenuar sombras y fondos desparejos.

    Estima el "fondo" mediante dilatación + desenfoque de mediana, y
    resta ese fondo estimado de la imagen original.
    """
    dilatada = cv2.dilate(imagen_gris, np.ones((7, 7), np.uint8))
    fondo = cv2.medianBlur(dilatada, 21)
    diferencia = 255 - cv2.absdiff(imagen_gris, fondo)
    normalizada = cv2.normalize(diferencia, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalizada


def mejorar_contraste(imagen_gris: np.ndarray) -> np.ndarray:
    """Mejora el contraste local con CLAHE (mejor que ecualización global
    para documentos con iluminación despareja)."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(imagen_gris)


def eliminar_ruido(imagen_gris: np.ndarray) -> np.ndarray:
    """Suaviza ruido preservando bordes de texto (filtro bilateral)."""
    return cv2.bilateralFilter(imagen_gris, d=9, sigmaColor=50, sigmaSpace=50)


def aumentar_nitidez(imagen_gris: np.ndarray) -> np.ndarray:
    """Aumenta la nitidez con máscara de desenfoque (unsharp masking)."""
    desenfocada = cv2.GaussianBlur(imagen_gris, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(imagen_gris, 1.5, desenfocada, -0.5, 0)


def binarizar(imagen_gris: np.ndarray) -> np.ndarray:
    """Binariza usando umbral adaptativo (mejor que un umbral global
    cuando la iluminación de la foto es despareja)."""
    return cv2.adaptiveThreshold(
        imagen_gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )


def binarizar_otsu(imagen_gris: np.ndarray) -> np.ndarray:
    """Binarización alternativa con el método de Otsu (más simple; útil
    como estrategia distinta en los reintentos de la Etapa 7)."""
    _, binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria

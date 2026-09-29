"""
Métrica de nitidez de la imagen, usada para decidir si está demasiado
borrosa como para intentar el OCR (ver `retry_orchestrator.py`).

Heurística clásica de visión por computador — varianza del Laplaciano —
sin ningún componente de IA/aprendizaje, tal como exige el spec del
proyecto. Es la misma técnica que usan herramientas como OpenCV para
detección de desenfoque en general: una imagen nítida tiene bordes
marcados (segunda derivada con varianza alta); una imagen borrosa tiene
transiciones suaves entre píxeles vecinos (varianza baja).
"""
import cv2
import numpy as np


def calcular_nitidez(imagen: np.ndarray) -> float:
    """Calcula un puntaje de nitidez de la imagen: la varianza del
    Laplaciano (operador de segunda derivada).

    No es una probabilidad ni un porcentaje: es una magnitud relativa,
    sensible al contenido y al tamaño de la imagen, pensada para
    compararse contra un umbral calibrado (`Settings.UMBRAL_NITIDEZ_MINIMA`),
    no como valor absoluto con significado propio. Una imagen vacía o
    completamente lisa (sin bordes) da 0.0, no un error.
    """
    if imagen is None or imagen.size == 0:
        return 0.0

    gris = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())

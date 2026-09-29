"""
Localización y decodificación del código PDF417 (timbre electrónico)
dentro de la imagen de la boleta.

Se prueban dos librerías, en orden de costo creciente, como estrategias
de mejor esfuerzo (mismo espíritu que las estrategias de OCR de la
Etapa 4 y las de reintento de la Etapa 7): `pyzbar` (envuelve ZBar, muy
rápida) primero; `pdf417decoder` (pura Python, más lenta pero más
confiable con los códigos generados sintéticamente que se probaron en
este proyecto) como respaldo si la primera no encuentra nada.

Nunca lanza excepciones: cualquier problema (código ausente, ilegible,
demasiado pequeño, imagen corrupta, librería no disponible) se traduce
en `None`. Esto es intencional — decodificar el timbre es una mejora
opcional sobre el flujo principal basado en OCR, nunca un requisito: si
falla, el resto del pipeline sigue funcionando exactamente igual que si
esta función no existiera.
"""
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Tope de tamaño antes de intentar decodificar: acota el peor caso de
# tiempo (una imagen grande sin ningún código de barras puede tardar
# varios segundos en descartarse, sobre todo con pdf417decoder). No
# afecta la resolución del código en sí mientras el documento recortado
# no sea desproporcionadamente más grande que la boleta real — medido
# empíricamente en esta sesión: ~2.8s peor caso a 2000x1500px sin
# reescalar, un costo acotado y aceptable para un intento único.
_DIMENSION_MAXIMA_DECODIFICACION = 2000

# Segundo intento, sobre un recorte de la franja inferior de la imagen,
# reescalado agresivamente. Se agregó tras confirmar (corriendo 10 fotos
# reales entre dos rondas de validación) que el timbre nunca decodificaba
# en la imagen completa — la causa raíz ya identificada es que el código
# ocupa muy pocos píxeles por módulo en una foto de la boleta entera.
# Recortar antes de reescalar ataca eso directamente: con muchos menos
# píxeles totales que mover, se puede agrandar mucho más agresivo sin
# pagar el costo de tiempo de reescalar la imagen completa a ese mismo
# factor, y el timbre chileno siempre aparece impreso en la franja
# inferior del documento (confirmado en las 10 fotos reales probadas).
_FRACCION_INFERIOR_TIMBRE = 0.32
_ANCHO_OBJETIVO_RECORTE_TIMBRE = 1500
_FACTOR_ESCALA_MAXIMO_RECORTE_TIMBRE = 4.0


# Margen que se recorta antes de intentar decodificar: la imagen que
# llega acá ya pasó por la detección/recorte de documento (Etapa 3), que
# a veces deja un borde delgado y oscuro cuando el recorte no calza
# perfecto con el límite real de la boleta. Confirmado en esta sesión:
# ese borde le basta a ambas librerías para no lograr localizar el
# código, aunque el código en sí esté perfectamente nítido — quitar un
# margen chico antes de intentar lo resuelve.
_MARGEN_RECORTE_FRACCION = 0.02


def _recortar_margen(imagen: np.ndarray) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    margen_y = int(alto * _MARGEN_RECORTE_FRACCION)
    margen_x = int(ancho * _MARGEN_RECORTE_FRACCION)
    if alto - 2 * margen_y <= 0 or ancho - 2 * margen_x <= 0:
        return imagen
    return imagen[margen_y : alto - margen_y, margen_x : ancho - margen_x]


def _preparar_imagen(imagen: np.ndarray) -> tuple[np.ndarray, "Image.Image"] | None:
    """Recorta un margen chico, convierte a escala de grises y acota el
    tamaño. Devuelve None si la imagen viene vacía o no se puede
    interpretar."""
    if imagen is None or imagen.size == 0:
        return None

    imagen = _recortar_margen(imagen)
    if imagen.size == 0:
        return None

    try:
        gris = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return None

    alto, ancho = gris.shape[:2]
    lado_mayor = max(alto, ancho)
    if lado_mayor > _DIMENSION_MAXIMA_DECODIFICACION:
        factor = _DIMENSION_MAXIMA_DECODIFICACION / lado_mayor
        nuevo_tamano = (max(int(ancho * factor), 1), max(int(alto * factor), 1))
        gris = cv2.resize(gris, nuevo_tamano, interpolation=cv2.INTER_AREA)

    try:
        imagen_pil = Image.fromarray(gris, mode="L")
    except (ValueError, TypeError):
        return None

    return gris, imagen_pil


def _recortar_y_escalar_franja_inferior(imagen: np.ndarray) -> np.ndarray | None:
    """Recorta el `_FRACCION_INFERIOR_TIMBRE` inferior de la imagen (donde
    va impreso el timbre en una boleta electrónica chilena) y lo agranda
    con interpolación Lanczos hacia `_ANCHO_OBJETIVO_RECORTE_TIMBRE`, con
    tope `_FACTOR_ESCALA_MAXIMO_RECORTE_TIMBRE`. Devuelve None ante
    cualquier problema (imagen vacía o demasiado angosta para recortar).
    """
    if imagen is None or imagen.size == 0:
        return None

    alto, ancho = imagen.shape[:2]
    if alto <= 0 or ancho <= 0:
        return None

    inicio = int(alto * (1 - _FRACCION_INFERIOR_TIMBRE))
    recorte = imagen[inicio:, :]
    if recorte.size == 0:
        return None

    alto_recorte, ancho_recorte = recorte.shape[:2]
    lado_mayor = max(alto_recorte, ancho_recorte)
    if lado_mayor <= 0 or lado_mayor >= _ANCHO_OBJETIVO_RECORTE_TIMBRE:
        return recorte

    factor = min(_ANCHO_OBJETIVO_RECORTE_TIMBRE / lado_mayor, _FACTOR_ESCALA_MAXIMO_RECORTE_TIMBRE)
    nuevo_tamano = (max(int(ancho_recorte * factor), 1), max(int(alto_recorte * factor), 1))
    return cv2.resize(recorte, nuevo_tamano, interpolation=cv2.INTER_LANCZOS4)


def _intentar_pyzbar(gris: np.ndarray) -> str | None:
    """Primera estrategia: ZBar vía pyzbar. Rápida (decenas de ms), pero
    en las pruebas de este proyecto no siempre reconoce PDF417 —
    se mantiene como primer intento porque cuando sí funciona es casi
    gratis, no porque sea la más confiable de las dos."""
    try:
        from pyzbar import pyzbar
    except Exception:  # librería no disponible / falta libzbar del sistema
        logger.debug("pyzbar no disponible, se omite esa estrategia.")
        return None

    try:
        resultados = pyzbar.decode(gris, symbols=[pyzbar.ZBarSymbol.PDF417])
    except Exception:
        return None

    for resultado in resultados:
        try:
            texto = resultado.data.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if texto:
            return texto

    return None


def _intentar_pdf417decoder(imagen_pil: "Image.Image") -> str | None:
    """Segunda estrategia: pdf417decoder (pura Python). Más lenta que
    pyzbar pero más confiable en las pruebas de este proyecto."""
    try:
        from pdf417decoder import PDF417Decoder
    except Exception:  # librería no disponible
        logger.debug("pdf417decoder no disponible, se omite esa estrategia.")
        return None

    try:
        decoder = PDF417Decoder(imagen_pil)
        cantidad_encontrados = decoder.decode()
        if cantidad_encontrados <= 0:
            return None
        texto = decoder.barcode_data_index_to_string(0)
    except Exception:
        return None

    texto = (texto or "").strip()
    return texto or None


def _intentar_en_imagen(imagen: np.ndarray) -> str | None:
    """Prepara una imagen (gris + tope de tamaño) e intenta decodificar
    con las dos librerías, en orden. Función compartida por los dos
    intentos de `decodificar_timbre` (imagen completa y recorte
    inferior reescalado) para no duplicar la lógica de preparación."""
    preparada = _preparar_imagen(imagen)
    if preparada is None:
        return None
    gris, imagen_pil = preparada

    texto = _intentar_pyzbar(gris)
    if texto:
        return texto

    return _intentar_pdf417decoder(imagen_pil)


def decodificar_timbre(imagen_bgr: np.ndarray) -> str | None:
    """Busca y decodifica un código PDF417 (timbre electrónico SII) en
    la imagen dada.

    Mejor esfuerzo, en dos pasadas: primero sobre la imagen completa;
    si no encuentra nada, sobre un recorte de la franja inferior
    (`_FRACCION_INFERIOR_TIMBRE`) reescalado agresivamente — ahí es
    donde va impreso el timbre en una boleta chilena, y reescalar solo
    esa franja permite agrandarla mucho más de lo que sería razonable
    para la imagen completa (ver notas junto a `_FRACCION_INFERIOR_TIMBRE`).
    Cada pasada prueba las dos librerías de decodificación disponibles.

    Devuelve el texto crudo decodificado (formato XML-like del TED, ver
    `ted_parser.py`), o `None` si ninguna pasada encontró/pudo
    decodificar ningún código. Nunca lanza excepciones ni escribe la
    imagen a disco — todo el procesamiento ocurre en memoria (misma
    política de privacidad que el resto del pipeline).
    """
    texto = _intentar_en_imagen(imagen_bgr)
    if texto:
        return texto

    recorte = _recortar_y_escalar_franja_inferior(imagen_bgr)
    if recorte is None:
        return None

    return _intentar_en_imagen(recorte)

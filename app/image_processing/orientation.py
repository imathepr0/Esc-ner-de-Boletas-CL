"""
Corrección de orientación de la imagen: inclinación fina (deskew) y
alineación del eje de lectura (0°/90°/180°/270°).

Nota de diseño: distinguir 0° de 180° (texto realmente "boca abajo")
dentro del eje ya corregido no es confiable usando solo visión clásica
sin leer el texto. Ese caso ambiguo se deja para el mecanismo de
reintentos (Etapa 7), que puede probar una variante rotada 180° como
una de sus estrategias si la primera lectura del OCR falla.
"""
import cv2
import numpy as np
import pytesseract


def _binarizar_para_analisis(imagen: np.ndarray) -> np.ndarray:
    """Binarización rápida usada solo para estimar orientación (no es el
    resultado final del pipeline de preprocesamiento)."""
    gris = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria


def estimar_angulo_inclinacion(imagen: np.ndarray) -> float:
    """Estima el ángulo de inclinación fina (grados) de la boleta.

    Usa el rectángulo de área mínima que envuelve los píxeles de primer
    plano (texto/tinta). Devuelve un ángulo en grados en el rango
    (-45, 45], donde 0 significa "sin inclinación detectada".
    """
    binaria = _binarizar_para_analisis(imagen)
    coordenadas = cv2.findNonZero(binaria)
    if coordenadas is None or len(coordenadas) < 20:
        return 0.0

    rectangulo = cv2.minAreaRect(coordenadas)
    angulo = rectangulo[-1]

    # Distintas versiones de OpenCV devuelven el ángulo en rangos
    # ligeramente distintos; lo normalizamos siempre a (-45, 45].
    if angulo < -45:
        angulo = 90 + angulo
    elif angulo > 45:
        angulo = angulo - 90

    return float(angulo)


def corregir_inclinacion(
    imagen: np.ndarray, angulo: float | None = None
) -> tuple[np.ndarray, float]:
    """Corrige la inclinación fina de la imagen.

    Si `angulo` es None, se estima automáticamente con
    `estimar_angulo_inclinacion`. Devuelve (imagen_corregida, angulo_aplicado).
    """
    if angulo is None:
        angulo = estimar_angulo_inclinacion(imagen)

    if abs(angulo) < 0.5:
        return imagen, 0.0

    alto, ancho = imagen.shape[:2]
    centro = (ancho // 2, alto // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    corregida = cv2.warpAffine(
        imagen,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corregida, angulo


def _puntaje_lineas_horizontales(imagen: np.ndarray) -> float:
    """Calcula qué tan "periódico" es el perfil de proyección horizontal
    de la imagen: valores altos indican líneas de texto horizontales
    bien definidas (la orientación de lectura correcta).

    El perfil se normaliza por el ancho de fila (densidad de píxeles de
    primer plano por fila, no la suma cruda) y por la densidad media
    (coeficiente de variación). Es fundamental para que las 4 rotaciones
    candidatas sean comparables entre sí: al rotar 90°, el ancho y alto
    de la imagen se intercambian, y sin esta normalización esa sola
    diferencia de dimensiones ya alteraba el puntaje, sin relación con
    si el texto quedó realmente alineado en horizontal.
    """
    binaria = _binarizar_para_analisis(imagen)
    ancho_fila = binaria.shape[1]
    perfil = binaria.sum(axis=1).astype(np.float64) / ancho_fila

    media = perfil.mean()
    if media < 1e-6:
        return 0.0

    return float(perfil.std() / media)


def estimar_rotacion_cardinal(imagen: np.ndarray) -> int:
    """Estima cuál de las rotaciones {0, 90, 180, 270} alinea mejor las
    líneas de texto en horizontal, comparando la "periodicidad" del
    perfil de proyección horizontal en cada caso.

    0 y 180 producen perfiles casi idénticos (el texto queda horizontal
    en ambos casos), por lo que esta función resuelve el EJE correcto
    (horizontal vs. vertical) y no si el resultado quedó boca abajo.
    """
    candidatos = {
        0: imagen,
        90: cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(imagen, cv2.ROTATE_180),
        270: cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    mejor_rotacion = 0
    mejor_puntaje = -1.0
    for rotacion, candidata in candidatos.items():
        puntaje = _puntaje_lineas_horizontales(candidata)
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_rotacion = rotacion

    return mejor_rotacion


def corregir_rotacion_cardinal(
    imagen: np.ndarray, rotacion: int | None = None
) -> tuple[np.ndarray, int]:
    """Corrige la rotación de eje (0/90/180/270).

    Devuelve (imagen_corregida, rotacion_aplicada).
    """
    if rotacion is None:
        rotacion = estimar_rotacion_cardinal(imagen)

    if rotacion == 90:
        return cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE), 90
    if rotacion == 180:
        return cv2.rotate(imagen, cv2.ROTATE_180), 180
    if rotacion == 270:
        return cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE), 270
    return imagen, 0


def estimar_rotacion_via_osd(imagen: np.ndarray) -> int | None:
    """Respaldo más confiable (pero más caro) que `estimar_rotacion_cardinal`
    para el mismo problema: usa el OSD (Orientation and Script Detection)
    de Tesseract, que mira la forma real de los caracteres, en vez de
    periodicidad de píxeles. `estimar_rotacion_cardinal` puede fallar con
    confianza en boletas con bloques grandes (recuadros, códigos de
    barras) mezclados con líneas cortas tipo "etiqueta: valor" — ver
    README, "Quinta ronda: una foto real expone un bug en la corrección
    de rotación" — porque ese caso ni siquiera es "ambiguo" para la
    heurística visual: el puntaje que le da al eje incorrecto puede ser
    varias veces más alto que el del eje correcto.

    Por su costo (1-3s medido contra fotos reales, contra prácticamente
    gratis la heurística de arriba), no reemplaza a
    `estimar_rotacion_cardinal` como primer paso — Etapa 7
    (`retry_orchestrator._eje_probablemente_incorrecto`) la usa solo
    como respaldo, cuando el primer intento de OCR da un puntaje bajo.

    Devuelve 0/90/180/270 (la rotación que Tesseract sugiere aplicar), o
    None si no pudo determinarlo (imagen sin texto suficiente, script no
    reconocible, etc.) — nunca lanza.
    """
    try:
        resultado = pytesseract.image_to_osd(imagen, output_type=pytesseract.Output.DICT)
        return int(resultado["rotate"]) % 360
    except Exception:
        return None

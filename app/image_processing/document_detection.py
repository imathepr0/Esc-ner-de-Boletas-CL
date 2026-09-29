"""
Detección automática del documento dentro de la foto, recorte automático
y corrección de perspectiva.

Implementa el enfoque clásico de "escáner de documentos": se buscan los
bordes de la boleta dentro de la imagen completa (que puede incluir
mesa, mano, fondo, etc.) y, si se encuentra un contorno rectangular
plausible, se aplica una transformación de perspectiva para obtener una
vista plana de frente.

Si no se logra detectar un contorno confiable (boleta ocupa toda la
imagen, bordes poco definidos, fondo muy similar al papel, etc.) se
devuelve la imagen original sin recortar: no se asume que todas las
imágenes estarán perfectamente tomadas.
"""
import cv2
import numpy as np

# Fracción del área de la imagen (ya con el margen de _PADDING_PX
# agregado) que debe cubrir el contorno candidato para considerarlo un
# documento válido. El límite inferior evita recortes sobre ruido o
# elementos pequeños del fondo. El límite superior evita seleccionar el
# borde de la FOTO COMPLETA como si fuera el documento: cuando el fondo
# de la foto (mesa, mano) llega hasta los bordes del encuadre, ese borde
# también pasa el chequeo de "4 puntos" igual de bien que el documento
# real — encontrado corriendo fotos reales (ver README).
_AREA_MINIMA_RELATIVA = 0.20
_AREA_MAXIMA_RELATIVA = 0.85

# Margen blanco agregado antes de detectar bordes (ver
# detectar_y_recortar_documento). Cuando el borde del documento en la
# foto original coincide o casi coincide con el borde del encuadre —
# frecuente en fotos de boletas angostas tomadas sin mucho margen — no
# hay nada "afuera" con qué contrastar, así que Canny no genera ningún
# borde ahí y el contorno del documento queda abierto por ese lado
# (encontrado corriendo fotos reales, ver README). Agregar un margen
# blanco artificial le da a ese lado algo contra qué contrastar.
_PADDING_PX = 15


def _ordenar_puntos(puntos: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como [superior-izq, superior-der, inferior-der, inferior-izq]."""
    puntos = puntos.reshape(4, 2)
    ordenados = np.zeros((4, 2), dtype=np.float32)

    suma = puntos.sum(axis=1)
    ordenados[0] = puntos[np.argmin(suma)]  # superior-izquierda: x+y mínimo
    ordenados[2] = puntos[np.argmax(suma)]  # inferior-derecha: x+y máximo

    diferencia = np.diff(puntos, axis=1)
    ordenados[1] = puntos[np.argmin(diferencia)]  # superior-derecha: x-y mínimo
    ordenados[3] = puntos[np.argmax(diferencia)]  # inferior-izquierda: x-y máximo

    return ordenados


def _buscar_contorno_documento(imagen: np.ndarray) -> np.ndarray | None:
    """Busca un contorno cuadrilátero que probablemente sea la boleta.

    Devuelve las 4 esquinas (sin ordenar) o None si no se encontró un
    candidato suficientemente grande, rectangular y con proporciones de
    boleta (más alto que ancho — ver _PADDING_PX y el chequeo de
    proporción más abajo).
    """
    gris = imagen if imagen.ndim == 2 else cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    difuminada = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(difuminada, 50, 150)
    bordes = cv2.dilate(bordes, np.ones((5, 5), np.uint8), iterations=1)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    area_imagen = imagen.shape[0] * imagen.shape[1]
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:8]

    for contorno in contornos:
        fraccion_area = cv2.contourArea(contorno) / area_imagen
        if fraccion_area < _AREA_MINIMA_RELATIVA or fraccion_area > _AREA_MAXIMA_RELATIVA:
            continue

        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

        if len(aproximado) != 4:
            continue

        _, _, ancho_caja, alto_caja = cv2.boundingRect(aproximado)
        # Una boleta chilena (rollo térmico) siempre es más alta que
        # ancha; un candidato "acostado" casi siempre resulta ser un
        # elemento impreso DENTRO del documento (una tabla, un recuadro,
        # el timbre) y no el documento mismo — encontrado corriendo
        # fotos reales (ver README).
        if alto_caja < ancho_caja:
            continue

        return aproximado.reshape(4, 2).astype(np.float32)

    return None


def _recortar_con_perspectiva(imagen: np.ndarray, puntos: np.ndarray) -> np.ndarray:
    """Aplica la transformación de perspectiva dados los 4 puntos del documento."""
    tl, tr, br, bl = puntos

    ancho_superior = np.linalg.norm(tr - tl)
    ancho_inferior = np.linalg.norm(br - bl)
    ancho_destino = max(int(max(ancho_superior, ancho_inferior)), 1)

    alto_izquierdo = np.linalg.norm(bl - tl)
    alto_derecho = np.linalg.norm(br - tr)
    alto_destino = max(int(max(alto_izquierdo, alto_derecho)), 1)

    destino = np.array(
        [
            [0, 0],
            [ancho_destino - 1, 0],
            [ancho_destino - 1, alto_destino - 1],
            [0, alto_destino - 1],
        ],
        dtype=np.float32,
    )

    matriz = cv2.getPerspectiveTransform(puntos, destino)
    return cv2.warpPerspective(imagen, matriz, (ancho_destino, alto_destino))


def detectar_y_recortar_documento(imagen: np.ndarray) -> tuple[np.ndarray, bool]:
    """Intenta detectar la boleta dentro de la foto y recortarla con
    corrección de perspectiva.

    Devuelve (imagen_resultante, documento_detectado). Si no se detecta
    un contorno confiable, imagen_resultante es la imagen original y
    documento_detectado es False.

    La búsqueda y el recorte ocurren sobre la imagen con un margen
    blanco agregado (ver _PADDING_PX), no sobre la original: así, si el
    documento detectado sí llegaba hasta el borde real de la foto, el
    recorte no queda "cortado" por accidente. El margen de sobra que
    pueda quedar en el resultado es blanco y no afecta al OCR.
    """
    con_margen = cv2.copyMakeBorder(
        imagen,
        _PADDING_PX,
        _PADDING_PX,
        _PADDING_PX,
        _PADDING_PX,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    contorno = _buscar_contorno_documento(con_margen)
    if contorno is None:
        return imagen, False

    puntos_ordenados = _ordenar_puntos(contorno)
    try:
        recortada = _recortar_con_perspectiva(con_margen, puntos_ordenados)
    except cv2.error:
        return imagen, False

    if recortada.size == 0:
        return imagen, False

    return recortada, True

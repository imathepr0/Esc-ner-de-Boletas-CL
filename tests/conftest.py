"""
Fixtures compartidas por toda la suite de pruebas.

Las boletas de prueba son **sintéticas** (generadas con OpenCV, texto
renderizado con cv2.putText), no fotografías reales de boletas
chilenas. Simulan condiciones realistas (perspectiva, sombra, ruido),
pero no reemplazan probar con fotos reales antes de un uso en
producción — ver la nota en el README principal.
"""
import cv2
import numpy as np
import pytest

from app.config import Settings

LINEAS_BOLETA_SUPERMERCADO = [
    "SUPERMERCADO EJEMPLO LTDA",
    "RUT: 76.123.456-7",
    "BOLETA ELECTRONICA N 001234",
    "FECHA: 14/07/2026   HORA: 18:32",
    "--------------------------------",
    "PAN HALLULLA          1     1200",
    "LECHE DESCREMADA 1L   2     1990",
    "ARROZ GRADO 1 1KG      1     1450",
    "COCA COLA 1.5L          1     1690",
    "--------------------------------",
    "SUBTOTAL                     8320",
    "TOTAL               CLP      8320",
    "MEDIO DE PAGO: DEBITO",
    "GRACIAS POR SU COMPRA",
]

LINEAS_BOLETA_RESTAURANTE = [
    "CAFETERIA DON JOSE SPA",
    "Av. Providencia 1234, Santiago",
    "--------------------------------",
    "14-07-2026 13:15",
    "Mesa: 5   Garzon: Maria",
    "--------------------------------",
    "2 Cafe Americano         3400",
    "1 Sandwich Barros Luco   4990",
    "1 Jugo Natural Naranja   2200",
    "--------------------------------",
    "Subtotal              13990",
    "Propina sugerida (10%)  1399",
    "TOTAL A PAGAR          15389",
    "Forma de pago: EFECTIVO",
    "Vuelvan pronto!",
]


def _crear_boleta_con_texto(lineas, ancho=520, alto=620):
    img = np.full((alto, ancho, 3), 255, dtype=np.uint8)
    y = 40
    for linea in lineas:
        cv2.putText(img, linea, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 38
    return img


def _crear_foto_con_perspectiva(pts_destino, lineas, seed=7):
    """Simula una foto de la boleta sobre una mesa: perspectiva, sombra
    y ruido gaussiano, tal como saldría de la cámara de un teléfono."""
    np.random.seed(seed)
    boleta = _crear_boleta_con_texto(lineas)
    ba, bh = boleta.shape[1], boleta.shape[0]
    pts_origen = np.float32([[0, 0], [ba, 0], [ba, bh], [0, bh]])
    ancho_fondo, alto_fondo = 950, 1050
    fondo = np.random.randint(90, 130, (alto_fondo, ancho_fondo, 3), dtype=np.uint8)
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    warp = cv2.warpPerspective(boleta, matriz, (ancho_fondo, alto_fondo), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full((bh, ba), 255, dtype=np.uint8), matriz, (ancho_fondo, alto_fondo)
    )
    foto = fondo.copy()
    foto[mask > 0] = warp[mask > 0]

    sombra = np.zeros((alto_fondo, ancho_fondo), dtype=np.uint8)
    cv2.circle(sombra, (750, 200), 420, 55, -1)
    sombra = cv2.GaussianBlur(sombra, (151, 151), 0)
    for c in range(3):
        foto[:, :, c] = cv2.subtract(foto[:, :, c], sombra)

    ruido = np.random.normal(0, 5, foto.shape).astype(np.int16)
    return np.clip(foto.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def boleta_limpia() -> np.ndarray:
    """Boleta de supermercado, con texto real, sin ruido ni perspectiva."""
    return _crear_boleta_con_texto(LINEAS_BOLETA_SUPERMERCADO)


@pytest.fixture
def boleta_restaurante() -> np.ndarray:
    """Boleta de restaurante: formato distinto (cantidad al inicio,
    guiones, 'TOTAL A PAGAR'), sin ruido ni perspectiva."""
    return _crear_boleta_con_texto(LINEAS_BOLETA_RESTAURANTE)


@pytest.fixture
def foto_normal() -> np.ndarray:
    """Foto sintética con perspectiva moderada, sombra y ruido — no
    dispara la ambigüedad de orientación 0°/180° (Etapa 3/7)."""
    return _crear_foto_con_perspectiva(
        np.float32([[60, 40], [890, 30], [900, 1010], [50, 1020]]),
        LINEAS_BOLETA_SUPERMERCADO,
    )


@pytest.fixture
def foto_volteada() -> np.ndarray:
    """Foto sintética cuya perspectiva más pronunciada hace que el
    recorte de documento salga boca abajo: caso real para probar el
    reintento con rotación 180° de la Etapa 7."""
    return _crear_foto_con_perspectiva(
        np.float32([[160, 100], [720, 80], [760, 900], [140, 930]]),
        LINEAS_BOLETA_SUPERMERCADO,
    )


def bytes_jpg(imagen: np.ndarray) -> bytes:
    """Codifica una imagen (array de OpenCV) como bytes JPEG, tal como
    llegaría un archivo subido de verdad por HTTP."""
    ok, buffer = cv2.imencode(".jpg", imagen)
    assert ok
    return buffer.tobytes()

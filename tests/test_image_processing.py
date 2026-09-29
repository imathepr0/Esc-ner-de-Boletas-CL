"""Pruebas del preprocesamiento de imagen con OpenCV (Etapa 3)."""
import cv2
import numpy as np

from app.image_processing.document_detection import detectar_y_recortar_documento
from app.image_processing.orientation import (
    corregir_inclinacion,
    corregir_rotacion_cardinal,
    estimar_rotacion_cardinal,
    estimar_rotacion_via_osd,
)
from app.image_processing.preprocessor import preprocesar_imagen, rotar_resultado_preprocesamiento


def test_deteccion_recorta_y_endereza_documento(foto_normal):
    recortada, detectado = detectar_y_recortar_documento(foto_normal)
    assert detectado is True
    # El recorte debe ser sustancialmente más chico que la foto completa
    # (se quitó el fondo de la mesa).
    assert recortada.shape[0] * recortada.shape[1] < foto_normal.shape[0] * foto_normal.shape[1]


def test_deteccion_sin_documento_devuelve_original():
    fondo_sin_boleta = np.random.randint(60, 200, (800, 600, 3), dtype=np.uint8)
    resultado, detectado = detectar_y_recortar_documento(fondo_sin_boleta)
    assert detectado is False
    assert resultado.shape == fondo_sin_boleta.shape


def test_rotacion_cardinal_corrige_eje_90_grados(boleta_limpia):
    rotada = cv2.rotate(boleta_limpia, cv2.ROTATE_90_CLOCKWISE)
    estimacion = estimar_rotacion_cardinal(rotada)
    # 90 o 270 corrigen el EJE (horizontal vs vertical). Cuál de las dos
    # queda "boca arriba" es ambiguo sin leer texto (ver docstring del
    # módulo); eso lo resuelve el reintento de la Etapa 7, no esta función.
    assert estimacion in (90, 270)
    corregida, _ = corregir_rotacion_cardinal(rotada)
    assert corregida.shape == boleta_limpia.shape


def test_deskew_reduce_inclinacion_artificial(boleta_limpia):
    alto, ancho = boleta_limpia.shape[:2]
    matriz = cv2.getRotationMatrix2D((ancho // 2, alto // 2), 8, 1.0)
    inclinada = cv2.warpAffine(boleta_limpia, matriz, (ancho, alto), borderValue=(255, 255, 255))
    _, angulo_aplicado = corregir_inclinacion(inclinada)
    assert angulo_aplicado != 0.0


def test_osd_estima_0_grados_en_imagen_derecha(boleta_limpia):
    """Respaldo de `estimar_rotacion_cardinal` (Etapa 7, ver README,
    "Quinta ronda"): más confiable porque lee forma real de caracteres,
    no periodicidad de píxeles."""
    assert estimar_rotacion_via_osd(boleta_limpia) == 0


def test_osd_estima_180_grados_en_imagen_al_reves(boleta_limpia):
    al_reves = cv2.rotate(boleta_limpia, cv2.ROTATE_180)
    assert estimar_rotacion_via_osd(al_reves) == 180


def test_osd_no_lanza_sin_texto_legible():
    """Ante una imagen sin texto reconocible, Tesseract lanza un error
    interno — se captura y se devuelve None en vez de propagarlo."""
    en_blanco = np.full((800, 600, 3), 255, dtype=np.uint8)
    assert estimar_rotacion_via_osd(en_blanco) is None


def test_rotar_resultado_preprocesamiento_intercambia_dimensiones(foto_normal):
    preprocesado = preprocesar_imagen(foto_normal)
    alto_orig, ancho_orig = preprocesado.imagen_gris.shape[:2]

    rotado = rotar_resultado_preprocesamiento(preprocesado, 90)

    assert rotado.imagen_gris.shape[:2] == (ancho_orig, alto_orig)
    assert rotado.imagen_binaria.shape[:2] == (ancho_orig, alto_orig)
    assert rotado.rotacion_cardinal_aplicada == (preprocesado.rotacion_cardinal_aplicada + 90) % 360
    # El resto de lo ya calculado en la Etapa 3 no depende de la
    # orientación — no debería recalcularse ni cambiar.
    assert rotado.nitidez == preprocesado.nitidez
    assert rotado.documento_detectado == preprocesado.documento_detectado


def test_rotar_resultado_preprocesamiento_respeta_imagen_recortada_none():
    from app.image_processing.preprocessor import ResultadoPreprocesamiento

    gris = np.zeros((100, 80), dtype=np.uint8)
    binaria = np.zeros((100, 80), dtype=np.uint8)
    preprocesado = ResultadoPreprocesamiento(
        imagen_binaria=binaria,
        imagen_gris=gris,
        documento_detectado=False,
        rotacion_cardinal_aplicada=0,
        angulo_inclinacion_aplicado=0.0,
        imagen_recortada=None,
    )
    rotado = rotar_resultado_preprocesamiento(preprocesado, 270)
    assert rotado.imagen_recortada is None


def test_rotar_resultado_preprocesamiento_rechaza_angulos_invalidos(foto_normal):
    preprocesado = preprocesar_imagen(foto_normal)
    for angulo_invalido in (0, 180, 45):
        try:
            rotar_resultado_preprocesamiento(preprocesado, angulo_invalido)
            assert False, f"debería haber lanzado ValueError para {angulo_invalido}"
        except ValueError:
            pass


def test_pipeline_completo_produce_imagenes_validas(foto_normal):
    resultado = preprocesar_imagen(foto_normal)
    assert resultado.documento_detectado is True
    assert resultado.imagen_binaria.ndim == 2
    assert resultado.imagen_gris.ndim == 2
    assert resultado.imagen_binaria.shape == resultado.imagen_gris.shape


def test_pipeline_no_crashea_con_imagen_minuscula():
    diminuta = np.full((10, 10, 3), 255, dtype=np.uint8)
    resultado = preprocesar_imagen(diminuta)
    assert resultado.imagen_binaria.shape[0] > 0


def test_pipeline_redimensiona_imagenes_grandes(foto_normal):
    grande = cv2.resize(foto_normal, (3600, 4400))
    resultado = preprocesar_imagen(grande, dimension_maxima=2000)
    assert max(resultado.imagen_gris.shape) <= max(foto_normal.shape) * 3

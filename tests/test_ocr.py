"""Pruebas del motor OCR basado en Tesseract (Etapa 4)."""
import sys

import cv2
import numpy as np
import pytest

from app.image_processing.preprocessor import preprocesar_imagen
from app.ocr.ocr_engine import (
    MotorOCRError,
    PalabraOCR,
    _agrupar_en_filas,
    _reconstruir_linea,
    reconocer_texto,
)
from app.parsing.product_parser import extraer_productos


def test_reconoce_texto_limpio(settings, boleta_limpia):
    resultado = reconocer_texto(boleta_limpia, settings)
    texto = resultado.texto_completo.upper()
    assert "SUPERMERCADO" in texto or "EJEMPLO" in texto
    assert "TOTAL" in texto
    assert resultado.cantidad_palabras > 15
    assert resultado.confianza_promedio > 60
    # Puntaje crudo (Opción 6 de VISION_Y_ROADMAP.md): con texto limpio y
    # confianza alta, debe superar cómodamente el umbral de salida
    # anticipada por defecto (ver Settings.UMBRAL_PUNTAJE_OCR_TEMPRANO).
    assert resultado.puntaje >= settings.UMBRAL_PUNTAJE_OCR_TEMPRANO


def test_integracion_preprocesamiento_y_ocr(settings, foto_normal):
    """Etapa 3 + Etapa 4: sobre una foto con perspectiva/sombra/ruido,
    al menos una de las variantes preprocesadas debe dar un total
    plausible de texto reconocido."""
    prep = preprocesar_imagen(foto_normal)
    resultado = reconocer_texto(prep.imagen_gris, settings)
    assert resultado.cantidad_palabras > 5


def test_imagen_en_blanco_no_lanza_excepcion(settings):
    blanco = np.full((300, 300, 3), 255, dtype=np.uint8)
    resultado = reconocer_texto(blanco, settings)
    assert resultado.cantidad_palabras == 0
    assert resultado.confianza_promedio == 0.0
    assert resultado.puntaje == 0.0


def test_tesseract_no_disponible_lanza_motor_ocr_error():
    """Se corre en un proceso aparte para no ensuciar el estado global
    de pytesseract (tesseract_cmd) del resto de la suite."""
    codigo = """
import sys
sys.path.insert(0, {path!r})
import numpy as np
from app.config import Settings
from app.ocr.tesseract_config import configurar_tesseract
from app.ocr.ocr_engine import reconocer_texto, MotorOCRError

settings = Settings(TESSERACT_CMD="/ruta/que/no/existe/tesseract")
configurar_tesseract(settings)
img = np.full((100, 100, 3), 255, dtype=np.uint8)
try:
    reconocer_texto(img, settings)
    sys.exit(1)
except MotorOCRError:
    sys.exit(0)
""".format(path=str(__import__("pathlib").Path(__file__).resolve().parents[1]))

    import subprocess

    proceso = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert proceso.returncode == 0, proceso.stderr


def _palabra_ocr(texto, x, y, ancho=None, alto=20, confianza=90.0, linea=0, bloque=0):
    if ancho is None:
        ancho = len(texto) * 12  # ancho "de mentira" consistente: 12px por carácter
    return PalabraOCR(
        texto=texto, confianza=confianza, x=x, y=y, ancho=ancho, alto=alto, linea=linea, bloque=bloque
    )


def test_agrupar_en_filas_tolera_pequeno_desalineamiento_vertical():
    """Nombre y precio rara vez caen en el mismo píxel exacto de altura
    (distinta línea base de fuente) — deben seguir agrupándose en la
    misma fila si la diferencia es chica, pero sin fusionarse con una
    fila realmente distinta."""
    fila_1 = [
        _palabra_ocr("PRODUCTO", x=20, y=100, alto=20),
        _palabra_ocr("990", x=300, y=104, alto=18),  # 4px más abajo, alto levemente distinto
    ]
    fila_2 = [_palabra_ocr("OTRO", x=20, y=150, alto=20)]  # fila distinta, bien separada

    filas = _agrupar_en_filas(fila_1 + fila_2)

    assert len(filas) == 2
    assert {p.texto for p in filas[0]} == {"PRODUCTO", "990"}
    assert [p.texto for p in filas[1]] == ["OTRO"]


def test_agrupar_en_filas_reconstruye_columnas_que_tesseract_separo_en_bloques():
    """Reproduce el modo de falla documentado en el README ("Cuarta
    ronda de boletas reales", boleta Líder: 0 de 13 productos
    reconocidos): alguna de las estrategias de PSM probadas detecta el
    nombre de producto y el precio como bloques de texto SEPARADOS (la
    columna vacía entre ambos es ancha) y los entrega, en su propio
    array de palabras, primero TODOS los nombres y recién al final
    TODOS los precios — sin relación posicional entre ellos.

    Se arma esa misma forma a mano (cada nombre en un `bloque`/`linea`
    de Tesseract distinto al de su precio, pero con la posición Y real
    correcta por fila) para verificar dos cosas: que el agrupamiento
    anterior (por bloque/línea de Tesseract) efectivamente no podía
    juntarlos, y que agrupar por posición Y real sí — al punto de que
    el parser de productos (Etapa 5, sin cambios) los reconoce bien.
    """
    productos = [
        (["PAN", "HALLULLA"], "1200"),
        (["LECHE", "ENTERA"], "1990"),
        (["ARROZ", "1KG"], "1450"),
        (["FIDEOS", "SPAGUETTI"], "990"),
    ]
    alto_fila = 40
    palabras: list[PalabraOCR] = []
    for i, (palabras_nombre, precio) in enumerate(productos):
        y = 100 + i * alto_fila
        x = 20
        for palabra_texto in palabras_nombre:
            p = _palabra_ocr(palabra_texto, x=x, y=y, bloque=1, linea=i)
            palabras.append(p)
            x += p.ancho + 12  # espacio normal entre palabras de un mismo nombre
        # el precio queda en un bloque de Tesseract DISTINTO (columna
        # separada por un espacio ancho, como en la boleta real de
        # Líder), pero a la MISMA altura Y que su producto.
        palabras.append(_palabra_ocr(precio, x=520, y=y, bloque=2, linea=i))

    # --- comportamiento ANTERIOR: agrupar por (bloque, línea) de Tesseract ---
    grupos_tesseract: dict[tuple[int, int], list[PalabraOCR]] = {}
    for p in palabras:
        grupos_tesseract.setdefault((p.bloque, p.linea), []).append(p)
    lineas_anteriores = [
        _reconstruir_linea(sorted(g, key=lambda p: p.x)) for g in grupos_tesseract.values()
    ]
    for palabras_nombre, precio in productos:
        nombre_completo = " ".join(palabras_nombre)
        assert not any(nombre_completo in linea and precio in linea for linea in lineas_anteriores)

    # --- comportamiento NUEVO: agrupar por posición Y real ---
    filas = _agrupar_en_filas(palabras)
    assert len(filas) == 4  # una fila por producto, no "bloque de nombres + bloque de precios"

    texto_completo = "\n".join(_reconstruir_linea(fila) for fila in filas)
    extraidos = extraer_productos(texto_completo)

    assert len(extraidos) == 4
    for extraido, (palabras_nombre, precio) in zip(extraidos, productos):
        assert extraido.nombre == " ".join(palabras_nombre)
        assert extraido.precio_unitario == float(precio)


def test_reconocer_texto_extremo_a_extremo_con_layout_de_dos_columnas(settings):
    """Igual que el test anterior, pero de punta a punta contra el
    binario real de Tesseract (no con PalabraOCR construidas a mano):
    una imagen sintética con nombre de producto a la izquierda y precio
    en una columna bien separada a la derecha.

    No depende de que Tesseract efectivamente colapse los bloques en
    esta corrida particular (su salida no es 100% determinista, ver
    CONTEXTO_PARA_IA.md) — solo de que, los haya colapsado o no, nombre
    y precio terminen en la misma línea del texto reconstruido, que es
    lo que este fix garantiza y el agrupamiento anterior no podía.
    """
    img = np.full((260, 700, 3), 255, dtype=np.uint8)
    filas_boleta = [
        ("PAN HALLULLA", "1200"),
        ("LECHE ENTERA", "1990"),
        ("ARROZ GRADO", "1450"),
    ]
    y = 60
    for nombre, precio in filas_boleta:
        cv2.putText(img, nombre, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, precio, (560, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        y += 65

    resultado = reconocer_texto(img, settings)
    lineas = resultado.texto_completo.upper().splitlines()

    for nombre, precio in filas_boleta:
        assert any(nombre in linea and precio in linea for linea in lineas), (
            f"'{nombre}' y '{precio}' no quedaron en la misma línea reconstruida:\n"
            f"{resultado.texto_completo}"
        )

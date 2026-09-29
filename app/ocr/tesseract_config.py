"""
Configuración del motor Tesseract: idioma, modos de segmentación de
página (PSM) a probar, y localización del binario cuando no está en PATH.
"""
import pytesseract

from app.config import Settings

# Modos de segmentación de página (PSM) que se prueban, en orden, para
# maximizar la precisión (sección "OCR" del spec: "debe intentar
# maximizar la precisión antes de aceptar una lectura").
#   6  = bloque uniforme de texto (mejor por defecto para una boleta)
#   4  = columna de texto de tamaños variables (útil si hay encabezados
#        con letra más grande que el detalle de productos)
#   3  = segmentación automática completa (más general)
#   11 = texto disperso, sin orden particular (respaldo para layouts difíciles)
PSM_ESTRATEGIAS: list[int] = [6, 4, 3, 11]


def configurar_tesseract(settings: Settings) -> None:
    """Aplica la configuración global de pytesseract (ruta del binario).

    Se llama una única vez al iniciar la aplicación (ver app/main.py).
    """
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def construir_config(psm: int, settings: Settings) -> str:
    """Construye el string de configuración de Tesseract para un PSM dado."""
    return f"--oem {settings.TESSERACT_OEM} --psm {psm}"

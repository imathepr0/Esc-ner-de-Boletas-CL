"""
Motor de OCR basado en Tesseract (pytesseract), configurado para español.

Implementado en la Etapa 4 (ver tesseract_config.py y ocr_engine.py).
"""
from app.ocr.ocr_engine import MotorOCRError, PalabraOCR, ResultadoOCR, reconocer_texto
from app.ocr.tesseract_config import configurar_tesseract

__all__ = [
    "MotorOCRError",
    "PalabraOCR",
    "ResultadoOCR",
    "reconocer_texto",
    "configurar_tesseract",
]

"""
Parser de boletas: transforma el texto crudo del OCR en datos
estructurados (comercio, fecha, método de pago, productos, total), con
reglas flexibles para distintos formatos de boletas chilenas.

Implementado en la Etapa 5 (ver patterns.py, monetary.py,
field_extractors.py, product_parser.py y receipt_parser.py).
"""
from app.parsing.receipt_parser import ResultadoParser, parsear_boleta

__all__ = ["ResultadoParser", "parsear_boleta"]

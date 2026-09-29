"""
Validaciones internas post-OCR (consistencia entre productos y total,
existencia de campos clave) y cálculo del puntaje de confianza general
y por campo.

Implementado en la Etapa 6 (ver field_confidence.py,
consistency_checks.py, document_confidence.py y receipt_validator.py).
"""
from app.receipt_validation.consistency_checks import ResultadoValidacionInterna
from app.receipt_validation.receipt_validator import ResultadoEvaluacion, evaluar_lectura

__all__ = ["ResultadoValidacionInterna", "ResultadoEvaluacion", "evaluar_lectura"]

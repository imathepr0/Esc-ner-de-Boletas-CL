"""
Cálculo de la confianza general del documento, combinando la confianza
por campo (ponderada) con el resultado de las validaciones internas.
"""
from app.models.schemas import ConfianzaPorCampo

# Peso de cada campo en el promedio ponderado que forma la base de
# confianza_general. Repesado en sept. 2026 a pedido explícito de
# Camilo, mirando el formulario de "Ingresar manualmente" de Blynn: los
# primeros tres campos que pide (comercio, monto, fecha) más la
# descripción (nombre y precio de los productos) son los datos que de
# verdad importan; el monto total es el más importante de todos; método
# de pago es el único campo opcional del formulario, así que pesa menos
# que el resto. Un campo ausente (None) simplemente no entra al
# promedio — no hace falta un peso "cero" aparte.
_PESO_TOTAL = 3.0  # el más importante — es el dato que se guarda como gasto
_PESO_CAMPO_PRINCIPAL = 1.0  # comercio, fecha, productos — misma importancia entre sí
_PESO_METODO_PAGO = 0.3  # único campo opcional del formulario manual


def calcular_confianza_general(
    confianza_por_campo: ConfianzaPorCampo,
    confianza_ocr_general: float,
    penalizacion: float,
) -> float:
    """Combina la confianza por campo (ponderada, cuando hay datos
    suficientes) con la confianza general del OCR, y resta la
    penalización ya calculada por `consistency_checks.validar_datos`
    (que pesa cada advertencia según qué tan grave es para un tracker de
    gastos, no un monto plano por advertencia).

    Si ningún campo tiene confianza calculable (ej. no se identificó
    nada), se usa directamente la confianza general del OCR como base.
    """
    campos_ponderados = [
        (confianza_por_campo.comercio, _PESO_CAMPO_PRINCIPAL),
        (confianza_por_campo.fecha, _PESO_CAMPO_PRINCIPAL),
        (confianza_por_campo.metodo_pago, _PESO_METODO_PAGO),
        (confianza_por_campo.productos, _PESO_CAMPO_PRINCIPAL),
        (confianza_por_campo.total, _PESO_TOTAL),
    ]
    presentes = [(c, peso) for c, peso in campos_ponderados if c is not None]

    if presentes:
        suma_pesos = sum(peso for _, peso in presentes)
        base = sum(c * peso for c, peso in presentes) / suma_pesos
    else:
        base = confianza_ocr_general

    return round(max(0.0, min(100.0, base - penalizacion)), 1)

"""
Utilidad para interpretar montos en formato chileno: punto como
separador de miles (también coma, por la confusión frecuente del OCR
entre ambos — ver PATRON_MONTO) y coma opcional para decimales (poco
común en CLP, pero se soporta por robustez).
"""
from app.parsing.patterns import PATRON_MONTO

# Techo de plausibilidad: ningún monto de una boleta de consumo chilena
# (compra, cuenta de servicios) llega ni cerca de esto. Sin este techo,
# ruido de OCR sobre una secuencia larga de dígitos (ej. un código de
# barras o número de folio mal recortado como si fuera un monto) puede
# arrojar un número astronómicamente grande en vez de fallar limpio —
# encontrado corriendo fotos reales: una línea de puntos de fidelización
# con un código de barras de 30 dígitos se interpretó como un precio de
# producto de ese mismo largo (ver README).
_MONTO_MAXIMO_PLAUSIBLE = 100_000_000.0


def parsear_monto_clp(texto: str) -> float | None:
    """Convierte un monto en formato chileno a un número.

    Ejemplos: "9.020" -> 9020.0, "$1.234.567" -> 1234567.0,
    "1990" -> 1990.0, "9.020,50" -> 9020.5, "19,00%" -> 19.0.

    Un sufijo de exactamente 2 dígitos sobre un entero SIN separador de
    miles propio (ej. "4,25", "6,50") se trata por defecto como un
    grupo de miles truncado por el OCR, no como decimales genuinos: en
    boletas CLP casi no existen centavos reales, y corriendo fotos
    reales se vio más de una vez que el OCR pierde el último dígito de
    un monto de 3 cifras (ej. "$6.500" leído como "$6,50d", perdiendo
    la "0" final) — interpretarlo como decimal daba totales
    absurdamente chicos (6.5 en vez de 6500). Dos excepciones: si el
    sufijo viene seguido de "%" (ej. "19,00%" de IVA) sí son decimales
    genuinos, y si el entero YA trae su propio separador de miles (ej.
    "9.020,50") no hay ambigüedad de truncamiento, así que también se
    trata como decimal.

    Devuelve None si no se encuentra un monto válido, o si el número
    encontrado supera el techo de plausibilidad (ver
    _MONTO_MAXIMO_PLAUSIBLE) — nunca inventa ni acepta un número
    absurdo.
    """
    coincidencia = PATRON_MONTO.search(texto)
    if not coincidencia:
        return None

    entero, decimales = coincidencia.groups()
    entero_limpio = entero.replace(".", "").replace(",", "")
    try:
        valor = float(entero_limpio)
    except ValueError:
        return None

    if decimales:
        resto = texto[coincidencia.end() :].lstrip()
        entero_ya_agrupado = "." in entero or "," in entero
        if resto.startswith("%") or entero_ya_agrupado:
            valor += float(decimales) / 100
        else:
            valor = valor * 1000 + float(decimales) * 10

    if valor > _MONTO_MAXIMO_PLAUSIBLE:
        return None

    return valor

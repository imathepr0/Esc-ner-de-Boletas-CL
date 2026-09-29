"""
Cálculo de confianza individual para cada dato extraído, a partir de la
confianza que Tesseract asignó a las palabras del OCR que dieron origen
a ese dato (ver "Puntaje de confianza" en las especificaciones).

Cada función busca, entre las palabras reconocidas por el OCR
(app.ocr.PalabraOCR), aquellas que probablemente originaron el valor ya
extraído por el parser (Etapa 5), y promedia su confianza. El parser no
guarda esa trazabilidad directamente (trabaja sobre texto plano), así
que se reconstruye aquí buscando coincidencias de texto.
"""
import unicodedata

from app.models.schemas import DatosExtraidos
from app.ocr.ocr_engine import PalabraOCR
from app.parsing.monetary import parsear_monto_clp
from app.parsing.patterns import METODOS_PAGO


def _sin_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def _normalizar_token(token: str) -> str:
    return _sin_acentos(token).strip(".,:;-$()").upper()


def _confianza_por_tokens(valor: str | None, palabras: list[PalabraOCR]) -> float | None:
    """Promedia la confianza de las palabras del OCR cuyo texto coincide
    (normalizado) con algún token del valor extraído. Sirve para campos
    de texto que el parser no transforma (comercio, nombre de producto)."""
    if not valor:
        return None

    tokens_valor = {_normalizar_token(t) for t in valor.split() if t}
    tokens_valor.discard("")
    if not tokens_valor:
        return None

    coincidencias = [p.confianza for p in palabras if _normalizar_token(p.texto) in tokens_valor]
    if not coincidencias:
        return None

    return round(sum(coincidencias) / len(coincidencias), 1)


def confianza_comercio(comercio: str | None, palabras: list[PalabraOCR]) -> float | None:
    """Confianza del comercio: promedio de confianza de las palabras del
    OCR que coinciden con su texto (no requiere transformación de
    formato, a diferencia de fecha/monto)."""
    return _confianza_por_tokens(comercio, palabras)


def confianza_fecha(fecha_iso: str | None, palabras: list[PalabraOCR]) -> float | None:
    """La fecha se guarda en formato ISO, que no coincide textualmente
    con lo escrito en la boleta (ej. "14/07/2026"); se buscan palabras
    del OCR cuyos dígitos contengan el día, mes y año extraídos.
    """
    if not fecha_iso:
        return None

    anio, mes, dia = fecha_iso.split("-")
    anio_corto = anio[-2:]

    coincidencias = []
    for p in palabras:
        digitos = "".join(c for c in p.texto if c.isdigit())
        if not digitos:
            continue
        if dia in digitos and mes in digitos and (anio in digitos or anio_corto in digitos):
            coincidencias.append(p.confianza)

    if not coincidencias:
        return None

    return round(sum(coincidencias) / len(coincidencias), 1)


def confianza_metodo_pago(metodo_pago: str | None, palabras: list[PalabraOCR]) -> float | None:
    """El valor guardado está normalizado (ej. "Débito"); se buscan las
    palabras clave originales que mapean a ese valor (ej. "DEBITO",
    "REDCOMPRA") entre las palabras del OCR."""
    if not metodo_pago:
        return None

    claves_originales = {
        _normalizar_token(clave) for clave, valor in METODOS_PAGO.items() if valor == metodo_pago
    }

    coincidencias = [
        p.confianza for p in palabras if _normalizar_token(p.texto) in claves_originales
    ]
    if not coincidencias:
        return None

    return round(sum(coincidencias) / len(coincidencias), 1)


def confianza_total(total: float | None, palabras: list[PalabraOCR]) -> float | None:
    """Busca la palabra del OCR cuyo valor monetario coincide exactamente
    con el total extraído."""
    if total is None:
        return None

    coincidencias = [p.confianza for p in palabras if parsear_monto_clp(p.texto) == total]
    if not coincidencias:
        return None

    return round(sum(coincidencias) / len(coincidencias), 1)


def confianza_productos(
    datos: DatosExtraidos, palabras: list[PalabraOCR]
) -> tuple[list[float | None], float | None]:
    """Calcula la confianza de cada producto (por nombre) y el promedio
    general de la lista de productos.

    Efecto secundario intencional: completa `producto.confianza` en cada
    elemento de `datos.productos` (viene en None desde el parser de la
    Etapa 5, que no calcula confianza).
    """
    confianzas_individuales: list[float | None] = []
    for producto in datos.productos:
        c = _confianza_por_tokens(producto.nombre, palabras)
        confianzas_individuales.append(c)
        producto.confianza = c

    validas = [c for c in confianzas_individuales if c is not None]
    promedio = round(sum(validas) / len(validas), 1) if validas else None
    return confianzas_individuales, promedio

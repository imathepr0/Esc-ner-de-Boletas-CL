"""
Interpretación del texto XML-like del Timbre Electrónico (TED) del SII,
una vez decodificado el código PDF417 (ver `barcode_decoder.py`).

Formato real (simplificado): `<TED><DD><RE>76543210-5</RE><TD>39</TD>
<F>1234</F><FE>2026-07-14</FE>...<MNT>8320</MNT>...</DD><FRMT>...
</FRMT></TED>`.

Se usan expresiones regulares independientes por campo — no un parser
XML estricto — a propósito: una decodificación de PDF417 sobre una foto
real puede venir parcial o con caracteres corruptos (ver README, sección
"Timbre electrónico"), y un parser XML fallaría por completo ante el
primer tag mal formado. Buscar cada campo por su cuenta significa que
uno corrupto no impide extraer los demás.
"""
import re
from dataclasses import dataclass
from datetime import date

_PATRON_FECHA_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


@dataclass
class DatosTED:
    """Campos relevantes extraídos del TED. Todos opcionales: una
    decodificación parcial puede traer solo algunos, y nunca se
    inventa un valor para el que falta."""

    rut_emisor: str | None = None
    tipo_documento: str | None = None
    folio: str | None = None
    fecha_emision: str | None = None  # Normalizada a ISO (YYYY-MM-DD)
    monto_total: float | None = None


def _extraer_tag(texto: str, tag: str) -> str | None:
    """Busca `<TAG>valor</TAG>` en el texto. Tolerante a que falte el
    cierre exacto `</TAG>` (a veces se pierde en decodificaciones
    parciales): en ese caso corta en la siguiente apertura de tag `<`,
    o en el final del texto si el tag venía cortado justo al final.
    """
    coincidencia = re.search(
        rf"<{re.escape(tag)}>(.*?)(?:</{re.escape(tag)}>|<|$)",
        texto,
        re.IGNORECASE | re.DOTALL,
    )
    if not coincidencia:
        return None
    valor = coincidencia.group(1).strip()
    return valor or None


def _normalizar_fecha(valor: str | None) -> str | None:
    """El TED trae la fecha ya en formato ISO (YYYY-MM-DD); esto solo
    valida que sea una fecha calendario real, sin inventar una si no lo
    es (misma política que `extraer_fecha` del parser de OCR)."""
    if not valor:
        return None
    coincidencia = _PATRON_FECHA_ISO.match(valor)
    if not coincidencia:
        return None
    anio, mes, dia = (int(g) for g in coincidencia.groups())
    try:
        return date(anio, mes, dia).isoformat()
    except ValueError:
        return None


def _normalizar_monto(valor: str | None) -> float | None:
    """El TED trae el monto como entero plano (sin puntos de miles), a
    diferencia del formato chileno con puntos que maneja
    `parsear_monto_clp` para el texto de OCR."""
    if not valor:
        return None
    limpio = re.sub(r"[^\d]", "", valor)
    if not limpio:
        return None
    try:
        return float(limpio)
    except ValueError:
        return None


def parsear_ted(texto_ted: str) -> DatosTED:
    """Extrae los campos relevantes del texto crudo ya decodificado del
    timbre. Nunca lanza excepciones ni inventa datos: cada campo queda
    en `None` si no se pudo reconocer con confianza a partir del texto
    disponible.
    """
    if not texto_ted:
        return DatosTED()

    return DatosTED(
        rut_emisor=_extraer_tag(texto_ted, "RE"),
        tipo_documento=_extraer_tag(texto_ted, "TD"),
        folio=_extraer_tag(texto_ted, "F"),
        fecha_emision=_normalizar_fecha(_extraer_tag(texto_ted, "FE")),
        monto_total=_normalizar_monto(_extraer_tag(texto_ted, "MNT")),
    )

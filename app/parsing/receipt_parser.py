"""
Orquestación del parser: combina los extractores de campos individuales
para transformar el texto reconocido por el OCR en datos estructurados
(DatosExtraidos), tal como pide la sección "Parser" de las
especificaciones.
"""
from dataclasses import dataclass

from app.models.schemas import DatosExtraidos
from app.parsing.field_extractors import (
    extraer_comercio,
    extraer_fecha,
    extraer_folio,
    extraer_metodo_pago,
    extraer_total,
)
from app.parsing.product_parser import extraer_productos


@dataclass
class ResultadoParser:
    """Datos listos para el frontend, más el folio (uso interno)."""

    datos: DatosExtraidos
    folio: str | None


def parsear_boleta(texto: str) -> ResultadoParser:
    """Transforma el texto crudo del OCR en datos estructurados.

    Nunca inventa información: cada campo queda en None (o lista vacía
    para productos) si no se pudo identificar con las reglas disponibles.
    """
    datos = DatosExtraidos(
        comercio=extraer_comercio(texto),
        fecha=extraer_fecha(texto),
        metodo_pago=extraer_metodo_pago(texto),
        productos=extraer_productos(texto),
        total=extraer_total(texto),
    )
    folio = extraer_folio(texto)

    return ResultadoParser(datos=datos, folio=folio)

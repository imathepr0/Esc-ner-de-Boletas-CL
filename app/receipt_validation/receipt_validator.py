"""
Orquestación de la Etapa 6: combina el cálculo de confianza por campo,
la confianza general del documento y las validaciones internas.
"""
from dataclasses import dataclass

from app.models.schemas import ConfianzaPorCampo, DatosExtraidos, ProductoExtraido
from app.ocr.ocr_engine import PalabraOCR
from app.receipt_validation.consistency_checks import (
    UMBRAL_CAMPO_IMPRECISO,
    ResultadoValidacionInterna,
    validar_datos,
)
from app.receipt_validation.document_confidence import calcular_confianza_general
from app.receipt_validation.field_confidence import (
    confianza_comercio,
    confianza_fecha,
    confianza_metodo_pago,
    confianza_productos,
    confianza_total,
)


@dataclass
class ResultadoEvaluacion:
    """Resultado completo de evaluar la calidad de una lectura: la
    confianza por campo, la confianza general y las validaciones
    internas (advertencias, campos faltantes, cuadre de totales)."""

    confianza_por_campo: ConfianzaPorCampo
    confianza_general: float
    validacion: ResultadoValidacionInterna


def _renombrar_productos_imprecisos(productos: list[ProductoExtraido], umbral: float) -> None:
    """Reemplaza, en el lugar, el nombre de los productos con confianza
    de OCR baja por un rótulo genérico ("Artículo 1", "Artículo 2", ...)
    en vez de mostrar el texto que garabateó el OCR — a pedido explícito
    de Camilo: si hay 3 líneas de producto pero no se leen bien, que se
    vea "Artículo 1/2/3" en vez de basura, sin necesidad de que el
    precio de cada una también se haya leído bien.

    Se aplica ANTES de `validar_datos` a propósito: así la advertencia
    de "producto impreciso" que genera esa función ya habla del mismo
    nombre que ve el usuario ("El producto 'Artículo 2' podría no..."),
    no del texto garabateado original.
    """
    for indice, producto in enumerate(productos, start=1):
        if producto.confianza is not None and producto.confianza < umbral:
            producto.nombre = f"Artículo {indice}"


def evaluar_lectura(
    datos: DatosExtraidos,
    palabras: list[PalabraOCR],
    confianza_ocr_general: float,
) -> ResultadoEvaluacion:
    """Evalúa la calidad de los datos extraídos por el parser (Etapa 5).

    Calcula la confianza de cada campo (comparando contra la confianza
    que Tesseract asignó a las palabras que lo originaron), ejecuta las
    validaciones internas (existencia de campos, cuadre de totales,
    inconsistencias) y combina todo en una confianza general.

    Nota: esta función completa el campo `confianza` de cada producto en
    `datos.productos` como efecto secundario (viene en None desde el
    parser de la Etapa 5), y también renombra los productos imprecisos
    (ver `_renombrar_productos_imprecisos`).
    """
    _confianzas_por_producto, promedio_productos = confianza_productos(datos, palabras)
    _renombrar_productos_imprecisos(datos.productos, UMBRAL_CAMPO_IMPRECISO)

    confianza_campos = ConfianzaPorCampo(
        comercio=confianza_comercio(datos.comercio, palabras),
        fecha=confianza_fecha(datos.fecha, palabras),
        metodo_pago=confianza_metodo_pago(datos.metodo_pago, palabras),
        productos=promedio_productos,
        total=confianza_total(datos.total, palabras),
    )

    validacion = validar_datos(datos, confianza_campos)

    confianza_general = calcular_confianza_general(
        confianza_campos, confianza_ocr_general, validacion.penalizacion
    )

    return ResultadoEvaluacion(
        confianza_por_campo=confianza_campos,
        confianza_general=confianza_general,
        validacion=validacion,
    )

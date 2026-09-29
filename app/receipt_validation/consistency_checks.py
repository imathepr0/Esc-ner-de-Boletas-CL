"""
Validaciones internas posteriores al OCR y al parseo: existencia de
campos clave, consistencia entre la suma de productos y el total,
detección de información inconsistente, y precisión de los campos que
sí se identificaron (ver "Validaciones internas" en las
especificaciones).

Cada chequeo que agrega una advertencia también aporta su propia
penalización a `confianza_general` (ver `ResultadoValidacionInterna.
penalizacion`), en vez de que todas las advertencias resten lo mismo.
El criterio de peso (repesado en sept. 2026, a pedido explícito de
Camilo, mirando el formulario de "Ingresar manualmente" de Blynn): el
monto total es el dato más importante de todos — es lo que se guarda
como gasto. Comercio, fecha y productos (nombre y precio) pesan igual
entre sí, un escalón abajo del total — son los otros datos que pide el
formulario manual. Método de pago, el único campo opcional del
formulario, pesa menos que todo el resto. Que la suma de productos no
coincida con el total sigue siendo la señal más grave de todas, porque
pone en duda directamente el dato más importante.
"""
from dataclasses import dataclass, field

from app.models.schemas import ConfianzaPorCampo, DatosExtraidos

# Tolerancia absoluta (CLP) al comparar la suma de productos con el
# total: se permite un pequeño margen por redondeos o cargos no
# desglosados como línea de producto (ej. envío, propina).
_TOLERANCIA_CUADRE_CLP = 1.0

# Confianza de campo (0-100), aun estando presente, por debajo de la
# cual se considera que el dato podría no ser exacto y se agrega una
# advertencia de precisión (distinta de "campo faltante").
UMBRAL_CAMPO_IMPRECISO = 50.0

# --- Pesos de penalización sobre confianza_general ---
# Total: el dato más importante — sin esto no hay gasto que registrar.
_PENALIZACION_TOTAL_FALTANTE = 30.0
_PENALIZACION_TOTAL_IMPRECISO = 25.0
# Comercio, fecha y productos: mismo peso entre sí, un escalón abajo del
# total — son los otros datos que pide el formulario de Blynn. En 10.0
# (sept. 2026) pantufla-180° —total y productos perfectos, plata
# cuadra, pero comercio mal leído y fecha ausente— caía de 80.1 a 59.6:
# a Camilo le pareció excesivo ("solo por la fecha y el comercio bajar
# tanto"), esperaba algo más cerca de 70. Bajado a 5.0: la misma foto
# queda en 69.6.
_PENALIZACION_CAMPO_PRINCIPAL = 5.0  # comercio/fecha faltantes o imprecisos; productos faltantes
_PENALIZACION_PRODUCTO_IMPRECISO = 5.0  # por cada producto impreciso — puede repetirse varias veces
# Método de pago: único campo opcional del formulario — pesa menos que
# el resto (y, como antes, no se penaliza si falta, solo si es impreciso).
_PENALIZACION_METODO_PAGO_IMPRECISO = 3.0
# La suma no coincide con el total: la señal más grave de todas — pone
# en duda directamente el dato más importante, aunque todo se haya
# leído con letra clara.
_PENALIZACION_DESCUADRE = 40.0
_PENALIZACION_INCONSISTENCIA = 10.0
# Tope de penalización total: más alto que antes (era 70) porque ahora
# los campos de nivel medio (comercio/fecha/productos) pesan más y se
# acumulan más rápido en un caso realmente malo.
_PENALIZACION_MAXIMA = 80.0


@dataclass
class ResultadoValidacionInterna:
    """Resultado de las validaciones posteriores al OCR/parser."""

    advertencias: list[str] = field(default_factory=list)
    campos_faltantes: list[str] = field(default_factory=list)
    suma_coincide_con_total: bool | None = None
    diferencia_suma_total: float | None = None
    penalizacion: float = 0.0


def _formatear_clp(monto: float) -> str:
    """Formatea un monto como CLP, con punto como separador de miles."""
    return f"${monto:,.0f}".replace(",", ".")


def _verificar_existencia(datos: DatosExtraidos) -> tuple[list[str], list[str], float]:
    advertencias: list[str] = []
    faltantes: list[str] = []
    penalizacion = 0.0

    if not datos.comercio:
        advertencias.append("No se pudo identificar el comercio.")
        faltantes.append("comercio")
        penalizacion += _PENALIZACION_CAMPO_PRINCIPAL
    if not datos.fecha:
        advertencias.append("No se pudo identificar la fecha.")
        faltantes.append("fecha")
        penalizacion += _PENALIZACION_CAMPO_PRINCIPAL
    if datos.total is None:
        advertencias.append("No se pudo identificar el total.")
        faltantes.append("total")
        penalizacion += _PENALIZACION_TOTAL_FALTANTE
    if not datos.productos:
        advertencias.append("No se identificó ningún producto.")
        faltantes.append("productos")
        penalizacion += _PENALIZACION_CAMPO_PRINCIPAL

    return advertencias, faltantes, penalizacion


def _verificar_cuadre_total(
    datos: DatosExtraidos,
) -> tuple[list[str], bool | None, float | None, float]:
    """Compara la suma de productos con el total cuando sea posible."""
    if datos.total is None or not datos.productos:
        return [], None, None, 0.0

    if any(p.cantidad is None or p.precio_unitario is None for p in datos.productos):
        return [], None, None, 0.0

    suma = sum(p.cantidad * p.precio_unitario for p in datos.productos)
    diferencia = round(suma - datos.total, 2)

    if abs(diferencia) <= _TOLERANCIA_CUADRE_CLP:
        return [], True, diferencia, 0.0

    advertencia = (
        f"La suma de los productos ({_formatear_clp(suma)}) no coincide "
        f"con el total declarado ({_formatear_clp(datos.total)})."
    )
    return [advertencia], False, diferencia, _PENALIZACION_DESCUADRE


def _verificar_inconsistencias(datos: DatosExtraidos) -> tuple[list[str], float]:
    """Detecta valores que, aunque se extrajeron, no tienen sentido."""
    advertencias: list[str] = []
    penalizacion = 0.0

    if datos.total is not None and datos.total <= 0:
        advertencias.append("El total identificado no es un monto válido (debe ser mayor a cero).")
        penalizacion += _PENALIZACION_INCONSISTENCIA

    for producto in datos.productos:
        if producto.precio_unitario is not None and producto.precio_unitario <= 0:
            advertencias.append(f"El producto '{producto.nombre}' tiene un precio inválido.")
            penalizacion += _PENALIZACION_INCONSISTENCIA
        if producto.cantidad is not None and producto.cantidad <= 0:
            advertencias.append(f"El producto '{producto.nombre}' tiene una cantidad inválida.")
            penalizacion += _PENALIZACION_INCONSISTENCIA

    return advertencias, penalizacion


def _verificar_precision(
    datos: DatosExtraidos, confianza_por_campo: ConfianzaPorCampo, umbral: float
) -> tuple[list[str], float]:
    """Señala campos que sí se identificaron, pero con confianza de OCR
    baja — complementa `_verificar_existencia`, que solo cubre campos
    ausentes. Cada campo que aplica genera su propio mensaje, para que
    el frontend pueda mostrar cuál dato conviene confirmar a mano (ver
    Opción 5 de VISION_Y_ROADMAP.md)."""
    advertencias: list[str] = []
    penalizacion = 0.0

    if (
        datos.comercio
        and confianza_por_campo.comercio is not None
        and confianza_por_campo.comercio < umbral
    ):
        advertencias.append(f"El comercio identificado ('{datos.comercio}') podría no ser exacto.")
        penalizacion += _PENALIZACION_CAMPO_PRINCIPAL

    if (
        datos.fecha
        and confianza_por_campo.fecha is not None
        and confianza_por_campo.fecha < umbral
    ):
        advertencias.append(f"La fecha identificada ({datos.fecha}) podría no ser exacta.")
        penalizacion += _PENALIZACION_CAMPO_PRINCIPAL

    if (
        datos.metodo_pago
        and confianza_por_campo.metodo_pago is not None
        and confianza_por_campo.metodo_pago < umbral
    ):
        advertencias.append(
            f"El método de pago identificado ('{datos.metodo_pago}') podría no ser exacto."
        )
        penalizacion += _PENALIZACION_METODO_PAGO_IMPRECISO

    if (
        datos.total is not None
        and confianza_por_campo.total is not None
        and confianza_por_campo.total < umbral
    ):
        advertencias.append(
            f"El total identificado ({_formatear_clp(datos.total)}) podría no ser exacto."
        )
        penalizacion += _PENALIZACION_TOTAL_IMPRECISO

    for producto in datos.productos:
        if producto.confianza is not None and producto.confianza < umbral:
            advertencias.append(f"El producto '{producto.nombre}' podría no estar bien identificado.")
            penalizacion += _PENALIZACION_PRODUCTO_IMPRECISO

    return advertencias, penalizacion


def validar_datos(
    datos: DatosExtraidos, confianza_por_campo: ConfianzaPorCampo
) -> ResultadoValidacionInterna:
    """Ejecuta todas las validaciones internas sobre los datos ya
    extraídos y estructurados por el parser (Etapa 5). Necesita la
    confianza por campo (Etapa 6, calculada antes de esta llamada) para
    poder distinguir "campo ausente" de "campo presente pero impreciso"."""
    advertencias_existencia, faltantes, penalizacion_existencia = _verificar_existencia(datos)
    advertencias_cuadre, coincide, diferencia, penalizacion_cuadre = _verificar_cuadre_total(datos)
    advertencias_inconsistencia, penalizacion_inconsistencia = _verificar_inconsistencias(datos)
    advertencias_precision, penalizacion_precision = _verificar_precision(
        datos, confianza_por_campo, UMBRAL_CAMPO_IMPRECISO
    )

    penalizacion_total = min(
        penalizacion_existencia + penalizacion_cuadre + penalizacion_inconsistencia + penalizacion_precision,
        _PENALIZACION_MAXIMA,
    )

    return ResultadoValidacionInterna(
        advertencias=advertencias_existencia
        + advertencias_cuadre
        + advertencias_inconsistencia
        + advertencias_precision,
        campos_faltantes=faltantes,
        suma_coincide_con_total=coincide,
        diferencia_suma_total=diferencia,
        penalizacion=penalizacion_total,
    )

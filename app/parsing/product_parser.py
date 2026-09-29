"""
Parser de la lista de productos: identifica, línea por línea, cuáles
corresponden a un producto (nombre + cantidad + precio) y cuáles son
encabezado, pie o totales.

No asume una posición fija ni un único formato: reconoce tanto boletas
donde la cantidad y el precio van al final de la línea ("PAN HALLULLA
1 1200") como boletas donde la cantidad va al inicio ("2 Cafe
Americano 3400"), típico de restaurantes.
"""
import re

from app.models.schemas import ProductoExtraido
from app.parsing.monetary import parsear_monto_clp
from app.parsing.patterns import (
    PALABRAS_CLAVE_NO_PRODUCTO,
    PATRON_CANTIDAD_INICIAL,
    PATRON_NUMERO_EN_LINEA,
    PATRON_SOLO_NUMEROS,
    PATRON_SOLO_SEPARADORES,
)

_CANTIDAD_MAXIMA_PLAUSIBLE = 999
_ESPACIOS_MINIMOS_COLUMNA = 3

# --- Formato con cantidad/precio/descuento/total en una línea y
# código/descripción en la línea siguiente (ej. "1  $2.990  $0  $2.990"
# seguida de "2646146500226 PANTUFLA HPG D1 IN26-44/45-B") — encontrado
# en boletas reales de una cadena de retail de vestuario/calzado (mismo
# encabezado "CANTIDAD PRECIO UNITARIO DESCUENTO TOTAL" / "CÓDIGO
# DESCRIPCIÓN" en dos fotos distintas de esa cadena). Es el orden
# inverso del que ya maneja _fusionar_lineas_divididas (ahí el nombre
# va primero, los montos después) — necesita su propia detección.
_PATRON_LINEA_SOLO_CANTIDAD_Y_MONTOS = re.compile(r"^[\d\s.,$]+$")
# 4+ dígitos: los códigos de producto reales vistos son de 10-13
# dígitos, y "cantidad al inicio" (_parsear_con_cantidad_inicial) ya
# cubre 1-3 dígitos (_CANTIDAD_MAXIMA_PLAUSIBLE) — el corte en 4 evita
# que ambos formatos se solapen.
_PATRON_CODIGO_Y_DESCRIPCION = re.compile(
    r"^\d{4,}\s+(?P<descripcion>[A-Za-zÁÉÍÓÚÑáéíóúñ].*)$"
)


def _es_linea_excluida(linea_mayus: str) -> bool:
    if not linea_mayus.strip():
        return True
    if PATRON_SOLO_SEPARADORES.match(linea_mayus):
        return True
    if PATRON_SOLO_NUMEROS.match(linea_mayus):
        return True
    return any(palabra in linea_mayus for palabra in PALABRAS_CLAVE_NO_PRODUCTO)


def _es_numero_de_columna(linea: str, inicio: int) -> bool:
    """Un número se considera columna (cantidad/precio) si está separado
    del texto anterior por 2+ espacios: así se evita confundir números
    que son parte del nombre del producto (p.ej. "ACEITE 1L", "ARROZ
    GRADO 1") con las columnas alineadas a la derecha de una boleta.

    Excepción: un número pegado directamente a "$" (sin espaciado
    amplio, ej. "FRUG KREM $2.490") se reconoce como columna sin
    importar el espaciado — el símbolo de moneda ya es una señal
    suficientemente fuerte de que es un precio, y boletas reales suelen
    imprimirlo así, pegado al monto.
    """
    texto_previo = linea[:inicio]
    if texto_previo.rstrip().endswith("$"):
        return True
    espacios_previos = len(texto_previo) - len(texto_previo.rstrip())
    return espacios_previos >= _ESPACIOS_MINIMOS_COLUMNA


def _parsear_con_cantidad_inicial(linea: str) -> ProductoExtraido | None:
    """Formato "CANTIDAD NOMBRE ... PRECIO" (ej. "2 Cafe Americano 3400")."""
    coincidencia_cantidad = PATRON_CANTIDAD_INICIAL.match(linea)
    if not coincidencia_cantidad:
        return None

    cantidad = float(coincidencia_cantidad.group(1))
    if not (0 < cantidad <= _CANTIDAD_MAXIMA_PLAUSIBLE):
        return None

    inicio_nombre = coincidencia_cantidad.end()
    numeros_restantes = list(PATRON_NUMERO_EN_LINEA.finditer(linea, inicio_nombre))
    if not numeros_restantes:
        return None

    ultimo = numeros_restantes[-1]
    precio_unitario = parsear_monto_clp(ultimo.group())
    if precio_unitario is None or precio_unitario <= 0:
        return None

    nombre = linea[inicio_nombre : ultimo.start()].strip(" .-:$")
    if len(nombre) < 2:
        return None

    return ProductoExtraido(nombre=nombre, cantidad=cantidad, precio_unitario=precio_unitario)


def _parsear_con_columnas_finales(linea: str) -> ProductoExtraido | None:
    """Formato "NOMBRE ... [CANTIDAD] PRECIO" (columnas al final,
    alineadas a la derecha, típico de supermercados/tiendas)."""
    coincidencias = list(PATRON_NUMERO_EN_LINEA.finditer(linea))
    columnas = [c for c in coincidencias if _es_numero_de_columna(linea, c.start())]
    if not columnas:
        return None

    valores = [(c, parsear_monto_clp(c.group())) for c in columnas]
    valores = [(c, v) for c, v in valores if v is not None and v > 0]
    if not valores:
        return None

    nombre = linea[: valores[0][0].start()].strip(" .-:$")
    if len(nombre) < 2:
        return None

    if len(valores) == 1:
        cantidad, precio_unitario = 1.0, valores[0][1]
    else:
        primer_valor = valores[0][1]
        if primer_valor.is_integer() and 0 < primer_valor <= _CANTIDAD_MAXIMA_PLAUSIBLE:
            cantidad, precio_unitario = primer_valor, valores[1][1]
        else:
            cantidad, precio_unitario = 1.0, valores[-1][1]

    return ProductoExtraido(nombre=nombre, cantidad=cantidad, precio_unitario=precio_unitario)


def _parsear_linea_producto(linea: str) -> ProductoExtraido | None:
    return _parsear_con_cantidad_inicial(linea) or _parsear_con_columnas_finales(linea)


def _es_linea_solo_monto(linea: str) -> bool:
    """Verifica si una línea es *solo* un monto o cantidad (dígitos y
    puntos de miles, nada más).

    Es más estricto que PATRON_SOLO_NUMEROS a propósito: ese patrón
    también acepta guiones y dos puntos (útil para excluir líneas
    separadoras "----" u horas "14-07-2026 13:15" de la lista de
    productos), pero eso mismo lo hace poco confiable para decidir si
    una línea es una columna numérica "huérfana" que hay que fusionar
    con el nombre de producto anterior.
    """
    limpio = linea.strip()
    return bool(limpio) and limpio.replace(".", "").replace(",", "").isdigit()


def _fusionar_lineas_divididas(lineas: list[str]) -> list[str]:
    """Recompone filas de producto que Tesseract detectó como varias
    líneas separadas (nombre en una línea, columnas de cantidad/precio
    en la o las siguientes). Esto puede ocurrir con layouts en columnas
    bajo ciertos modos de segmentación de página.

    Se une una línea con letras (que no sea un encabezado/pie conocido)
    con las líneas que sean *solo* un monto y que la siguen de
    inmediato, insertando espacio amplio entre ellas para que se
    reconozcan claramente como columnas al parsear la línea fusionada.
    """
    resultado: list[str] = []
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        letras = sum(1 for c in linea if c.isalpha())
        hay_siguiente_numerica = i + 1 < len(lineas) and _es_linea_solo_monto(lineas[i + 1])

        if letras >= 3 and hay_siguiente_numerica and not _es_linea_excluida(linea.upper()):
            fusionada = linea
            j = i + 1
            agregados = 0
            while j < len(lineas) and _es_linea_solo_monto(lineas[j]) and agregados < 3:
                fusionada += "    " + lineas[j].strip()
                agregados += 1
                j += 1
            resultado.append(fusionada)
            i = j
            continue

        resultado.append(lineas[i])
        i += 1

    return resultado


def _es_linea_solo_cantidad_y_montos(linea: str) -> bool:
    """Una línea de totales de ítem sin nombre: solo cantidad y montos
    en formato "$X.XXX" (cantidad, precio unitario, [descuento], total),
    sin ninguna letra. Se exige al menos 2 números (cantidad + precio
    unitario como mínimo) para no confundirla con una línea de un solo
    monto suelto — esa la maneja _es_linea_solo_monto, para el patrón
    inverso."""
    limpio = linea.strip()
    if not limpio or not _PATRON_LINEA_SOLO_CANTIDAD_Y_MONTOS.match(limpio):
        return False
    return len(PATRON_NUMERO_EN_LINEA.findall(limpio)) >= 2


def _fusionar_codigo_despues_de_montos(lineas: list[str]) -> list[str]:
    """Recompone el formato "cantidad/precio/descuento/total en una
    línea, código/descripción en la línea siguiente" en una única línea
    con la forma "NOMBRE ... CANTIDAD PRECIO", que
    _parsear_con_columnas_finales ya sabe interpretar sin cambios.

    Toma los dos primeros números de la línea de montos como cantidad y
    precio unitario respectivamente — el orden que indica el encabezado
    de esta boleta ("CANTIDAD PRECIO UNITARIO DESCUENTO TOTAL") — e
    ignora descuento/total si están presentes: tomar el precio unitario
    real (no el total de la línea) importa cuando la cantidad es mayor
    a 1, donde ambos difieren.
    """
    resultado: list[str] = []
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        siguiente = lineas[i + 1].strip() if i + 1 < len(lineas) else ""
        coincidencia_codigo = _PATRON_CODIGO_Y_DESCRIPCION.match(siguiente)

        if _es_linea_solo_cantidad_y_montos(linea) and coincidencia_codigo:
            cantidad_texto, precio_texto = PATRON_NUMERO_EN_LINEA.findall(linea)[:2]
            descripcion = coincidencia_codigo.group("descripcion").strip()
            resultado.append(f"{descripcion}    {cantidad_texto}    {precio_texto}")
            i += 2
            continue

        resultado.append(lineas[i])
        i += 1

    return resultado


def extraer_productos(texto: str) -> list[ProductoExtraido]:
    """Recorre el texto línea por línea y extrae los productos plausibles."""
    lineas = _fusionar_codigo_despues_de_montos(texto.splitlines())
    lineas = _fusionar_lineas_divididas(lineas)

    productos: list[ProductoExtraido] = []
    for linea in lineas:
        linea = linea.strip()
        if _es_linea_excluida(linea.upper()):
            continue

        producto = _parsear_linea_producto(linea)
        if producto is not None:
            productos.append(producto)

    return productos

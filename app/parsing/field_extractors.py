"""
Extractores de campos individuales (comercio, fecha, método de pago,
total, folio) a partir del texto reconocido por el OCR. Cada función es
independiente y usa reglas generales, no posiciones fijas.
"""
import re
from datetime import date

from app.parsing.monetary import parsear_monto_clp
from app.parsing.patterns import (
    ETIQUETAS_TOTAL_ALTERNATIVAS,
    METODOS_PAGO,
    PALABRAS_CLAVE_NO_PRODUCTO,
    PATRON_ETIQUETA_RUT,
    PATRON_FECHA,
    PATRON_SOLO_NUMEROS,
    PATRON_SOLO_SEPARADORES,
    SUFIJOS_RAZON_SOCIAL,
)

_PATRON_FOLIO = re.compile(
    r"(?:FOLIO|BOLETA(?:\s+ELECTRONICA)?|N[°ºo]?)\s*[:\-]?\s*(\d{4,})", re.IGNORECASE
)

# Líneas de desglose de impuestos que contienen "TOTAL" pero no son el
# total final de la boleta (ej. "TOTAL AFECTO", "TOTAL EXENTO", "TOTAL
# IVA"): sin esta exclusión le ganaban a la línea del TOTAL real porque
# aparecen antes en el documento (ver ronda de validación con boletas
# reales en el README).
_FRASES_TOTAL_EXCLUIDAS = ("TOTAL AFECTO", "TOTAL EXENTO", "TOTAL IVA")

# Encabezados de columna de una tabla de productos (ej. "Cód Prod
# P.Unit Cant Total") a veces incluyen la palabra "TOTAL" como nombre de
# columna — el total de ESA fila, no el de la boleta. Se reconocen por
# traer 2+ de estas palabras típicas de encabezado junto con "TOTAL", y
# se excluyen por completo (encontrado corriendo fotos reales: sin esto,
# el algoritmo tomaba el encabezado como la línea del total y de ahí
# terminaba sacando un número de la primera fila de productos).
_PALABRAS_ENCABEZADO_TABLA = (
    "CANT", "CÓD", "COD", "PROD", "DESCRIPCION", "DESCRIPCIÓN",
    "P.UNIT", "PUNIT", "PRECIO", "VALOR",
)

# Etiquetas de OTROS campos monetarios: si una línea con "TOTAL" no trae
# monto consigo y se mira 1-2 líneas más abajo buscándolo (Tesseract a
# veces separa la etiqueta del monto), una línea con alguna de estas
# etiquetas pertenece a OTRO campo distinto y no debe tomarse como si
# fuera el monto del total (ej. una línea "TOTALES" seguida de "NETO $"
# y "EXENTO $": ninguna de esas dos es el total).
_ETIQUETAS_OTRO_CAMPO = ("NETO", "EXENTO", "IVA", "SUBTOTAL", "DESCUENTO", "AFECTO")

# Frases que, cuando aparecen, casi siempre marcan el monto FINAL real a
# pagar — tienen prioridad sobre un "TOTAL" genérico que aparezca antes
# en el documento. Encontrado corriendo fotos reales: un "TOTAL" (o
# "Total Mes") genérico y más temprano en el documento a veces es un
# subtotal o un monto parcial, y el monto verdaderamente final trae uno
# de estos calificativos más específicos (ver README).
_FRASES_TOTAL_PRIORITARIAS = ("TOTAL FINAL", "A PAGAR")

_ETIQUETAS_TOTAL_PRINCIPALES = ("TOTAL",) + ETIQUETAS_TOTAL_ALTERNATIVAS


def extraer_fecha(texto: str) -> str | None:
    """Busca una fecha en el texto y la normaliza a ISO (YYYY-MM-DD).

    Prueba cada coincidencia hasta encontrar una que sea una fecha
    calendario válida; nunca inventa una fecha si no encuentra ninguna.

    Ignora dos señales de la referencia fija a la Resolución SII N°80
    (que autoriza el timbre electrónico) que trae en el pie toda boleta
    electrónica chilena: una línea que mencione "RESOLUC" (de
    "Resolución"/"Resolucion"), y la fecha exacta 22-08-2014 en
    cualquier línea (la resolución se cita a veces abreviada, "Res. 80
    del 22-08-2014", sin la palabra completa "Resolución" — encontrado
    corriendo fotos reales, ver README). Esa fecha no tiene relación con
    la fecha real de la boleta, pero si la fecha real sale ilegible del
    OCR, esta era la siguiente candidata y terminaba ganando por error.
    """
    fecha_resolucion_sii = date(2014, 8, 22)
    for linea in texto.splitlines():
        if "RESOLUC" in linea.upper():
            continue
        for coincidencia in PATRON_FECHA.finditer(linea):
            dia_str, mes_str, anio_str = coincidencia.groups()
            try:
                dia, mes, anio = int(dia_str), int(mes_str), int(anio_str)
                if anio < 100:
                    anio += 2000
                fecha_normalizada = date(anio, mes, dia)
            except ValueError:
                continue
            if fecha_normalizada == fecha_resolucion_sii:
                continue
            return fecha_normalizada.isoformat()
    return None


def _parece_encabezado_de_tabla(linea_mayus: str) -> bool:
    return sum(1 for palabra in _PALABRAS_ENCABEZADO_TABLA if palabra in linea_mayus) >= 2


def _monto_de_linea_total(linea: str) -> float | None:
    """Parsea el monto de una línea candidata a total, con una
    salvedad: si la línea tiene ":", solo se considera el texto
    DESPUÉS del último ":" (si eso no da un monto válido, cae al
    comportamiento normal sobre la línea completa).

    Encontrado corriendo fotos reales: cuando el OCR confunde el
    símbolo "$" con un dígito suelto justo después de la etiqueta (ej.
    "TOTAL 9: 5970", donde "9" es en realidad un "$" mal leído), tomar
    el PRIMER número de la línea agarraba ese dígito espurio en vez del
    monto real que sigue al ":" — que es casi siempre donde va el monto
    real en boletas chilenas ("ETIQUETA $: MONTO", "ETIQUETA: MONTO").
    Restringir la búsqueda a después del ":" evita ese problema sin
    cambiar el comportamiento de parsear_monto_clp en general (lo usan
    otras partes del parser donde tomar el primer número SÍ es lo
    correcto).
    """
    if ":" in linea:
        monto = parsear_monto_clp(linea.rsplit(":", 1)[-1])
        if monto is not None:
            return monto
    return parsear_monto_clp(linea)


def extraer_total(texto: str) -> float | None:
    """Busca el total de la boleta.

    Primero busca una línea con una frase que casi siempre marca el
    monto final real ("TOTAL FINAL", "... A PAGAR") — tiene prioridad
    sobre un "TOTAL" genérico que aparezca antes en el documento, porque
    ese "TOTAL" (o "Total Mes") temprano a veces es solo un subtotal o
    un monto parcial, no el monto final (encontrado corriendo fotos
    reales, ver README).

    Si no encuentra ninguna, prioriza la primera línea que contenga
    "TOTAL" (o una etiqueta alternativa equivalente que algunos sistemas
    de punto de venta usan en su lugar, ver ETIQUETAS_TOTAL_ALTERNATIVAS
    en patterns.py) y no sea "SUBTOTAL", un encabezado de columna de
    tabla (ej. "Cant Total"), ni una línea de desglose de impuestos
    ("TOTAL AFECTO", "TOTAL EXENTO", "TOTAL IVA") — esas suelen aparecer
    antes que el total real en boletas chilenas y no deben confundirse
    con él. Ejemplos válidos: "TOTAL", "TOTAL A PAGAR", "TOTAL CLP",
    "MONTO BOLETA ELECTRONICA". Si esa línea no trae un monto consigo,
    revisa las 2 líneas siguientes (Tesseract a veces separa la etiqueta
    del monto en líneas de detección distintas), saltándose cualquiera
    que a su vez traiga la etiqueta de OTRO campo (NETO, EXENTO, etc.)
    — esa no es el monto del total, es de otro campo. Si no encuentra
    ninguna, usa SUBTOTAL como respaldo.
    """
    lineas = texto.splitlines()

    for i, linea in enumerate(lineas):
        linea_mayus = linea.upper()
        frase = next((f for f in _FRASES_TOTAL_PRIORITARIAS if f in linea_mayus), None)
        if frase is None:
            continue

        # Buscar el monto SOLO después de la frase encontrada, no en
        # toda la línea: a veces el OCR pega texto de otro campo antes
        # de la frase en la misma línea (ej. una fecha de "próxima
        # lectura" quedó pegada justo antes de "TOTAL A PAGAR" en una
        # boleta real — sin esto, el primer número de esa fecha ganaba
        # por error en vez del monto real que sigue a la frase; ver
        # README).
        texto_desde_frase = linea[linea_mayus.index(frase) + len(frase) :]
        monto = _monto_de_linea_total(texto_desde_frase) if texto_desde_frase.strip() else None
        if monto is None:
            for siguiente in lineas[i + 1 : i + 3]:
                if any(etiqueta in siguiente.upper() for etiqueta in _ETIQUETAS_OTRO_CAMPO):
                    continue
                monto = _monto_de_linea_total(siguiente)
                if monto is not None:
                    break
        if monto is not None:
            return monto

    candidato_subtotal = None

    for i, linea in enumerate(lineas):
        linea_mayus = linea.upper()
        etiqueta = next((e for e in _ETIQUETAS_TOTAL_PRINCIPALES if e in linea_mayus), None)
        if etiqueta is None:
            continue
        if any(frase in linea_mayus for frase in _FRASES_TOTAL_EXCLUIDAS):
            continue
        if _parece_encabezado_de_tabla(linea_mayus):
            continue

        # Buscar el monto SOLO después de la etiqueta encontrada, no en
        # toda la línea (mismo motivo que en la pasada de frases
        # prioritarias arriba): un dígito suelto de OCR antes de la
        # etiqueta en la misma línea —por ejemplo la "S" de "SUB TOTAL"
        # leída como "5"— no debe ganarle al monto real que sigue a la
        # etiqueta (encontrado corriendo fotos reales, ver README).
        texto_desde_etiqueta = linea[linea_mayus.index(etiqueta) + len(etiqueta) :]
        monto = _monto_de_linea_total(texto_desde_etiqueta) if texto_desde_etiqueta.strip() else None
        if monto is None:
            for siguiente in lineas[i + 1 : i + 3]:
                if any(etiqueta_otro in siguiente.upper() for etiqueta_otro in _ETIQUETAS_OTRO_CAMPO):
                    continue
                monto = _monto_de_linea_total(siguiente)
                if monto is not None:
                    break
        if monto is None:
            continue

        if "SUBTOTAL" in linea_mayus:
            if candidato_subtotal is None:
                candidato_subtotal = monto
            continue

        return monto

    return candidato_subtotal


def extraer_metodo_pago(texto: str) -> str | None:
    """Busca palabras clave de método de pago en el texto completo."""
    texto_mayus = texto.upper()
    for palabra_clave, valor_normalizado in METODOS_PAGO.items():
        if palabra_clave in texto_mayus:
            return valor_normalizado
    return None


def extraer_folio(texto: str) -> str | None:
    """Busca un número de folio/boleta.

    Uso interno para identificar el documento; no se entrega al
    frontend (ver sección "Información a extraer" del spec).
    """
    coincidencia = _PATRON_FOLIO.search(texto)
    return coincidencia.group(1) if coincidencia else None


def _es_encabezado_o_pie(linea_mayus: str) -> bool:
    if PATRON_ETIQUETA_RUT.search(linea_mayus):
        return True
    return any(palabra in linea_mayus for palabra in PALABRAS_CLAVE_NO_PRODUCTO)


def _tiene_sufijo_razon_social(linea_mayus: str) -> bool:
    # .replace(".", "") además de .strip(",") -- no alcanza con strip():
    # "S.A.".strip(".,") da "S.A" (el punto interno, entre S y A, no
    # está en un extremo del token, así que strip() no lo toca). Sin
    # esto, "S.A." y "E.I.R.L." (con puntos, la forma más común de
    # escribirlos) nunca calzaban contra el set de sufijos sin puntos —
    # encontrado corriendo fotos reales, ver README.
    tokens = {token.strip(",").replace(".", "") for token in linea_mayus.split()}
    return bool(tokens & SUFIJOS_RAZON_SOCIAL)


def extraer_comercio(texto: str) -> str | None:
    """Identifica el nombre del comercio.

    Prioriza, entre las primeras 10 líneas CON CONTENIDO del documento
    (al menos 3 caracteres — se filtran antes de contar la ventana de
    10: fotos reales a veces traen ruido de OCR de 1-2 caracteres al
    principio, que agotaba la ventana de 10 líneas antes de llegar a
    contenido real; ver README), la primera que tenga un sufijo de razón
    social chileno (S.A., LTDA, SPA, etc.); si no encuentra ninguna, usa
    la primera línea "plausible" (con letras, que no sea un encabezado
    conocido como RUT/fecha/folio).
    """
    lineas = [linea.strip() for linea in texto.splitlines() if len(linea.strip()) >= 3][:10]

    for linea in lineas:
        if _tiene_sufijo_razon_social(linea.upper()):
            return linea

    for linea in lineas:
        linea_mayus = linea.upper()
        if PATRON_SOLO_NUMEROS.match(linea) or PATRON_SOLO_SEPARADORES.match(linea):
            continue
        if _es_encabezado_o_pie(linea_mayus):
            continue
        if len(linea) < 3:
            continue
        return linea

    return None

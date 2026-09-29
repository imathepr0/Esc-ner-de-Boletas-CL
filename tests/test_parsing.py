"""Pruebas del parser de boletas chilenas (Etapa 5)."""
import pytest

from app.parsing.field_extractors import (
    extraer_comercio,
    extraer_fecha,
    extraer_folio,
    extraer_metodo_pago,
    extraer_total,
)
from app.parsing.monetary import parsear_monto_clp
from app.parsing.product_parser import extraer_productos
from app.parsing.receipt_parser import parsear_boleta


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("9.020", 9020.0),
        ("$1.234.567", 1234567.0),
        ("1990", 1990.0),
        ("9.020,50", 9020.5),
        ("TOTAL CLP 9020", 9020.0),
        ("sin numero aqui", None),
        # Real: OCR confunde "." con "," (ver README, "Tercera ronda").
        ("TOTAL 4,250", 4250.0),
        # Real: sufijo de 2 dígitos sobre un entero SIN separador
        # propio se trata por defecto como miles truncados por el OCR,
        # no decimales genuinos (CLP casi no tiene centavos reales) —
        # ver README.
        ("4,25", 4250.0),
        # Excepción: seguido de "%" sí son decimales genuinos (IVA).
        ("19,00%", 19.0),
        # Real: 30 dígitos de un código de barras mal leídos como monto
        # (ver README) -- debe rechazarse, no devolver un número absurdo.
        ("202608223585005904502691986201", None),
    ],
)
def test_parsear_monto_clp(texto, esperado):
    assert parsear_monto_clp(texto) == esperado


def test_extraer_fecha_formatos_variados():
    assert extraer_fecha("FECHA: 14/07/2026 HORA: 18:32") == "2026-07-14"
    assert extraer_fecha("14-07-26") == "2026-07-14"
    assert extraer_fecha("no hay fecha aqui") is None
    assert extraer_fecha("fecha invalida 32/13/2026 pero real 05/03/2025") == "2025-03-05"


def test_extraer_metodo_pago_variantes():
    assert extraer_metodo_pago("MEDIO DE PAGO: DEBITO") == "Débito"
    assert extraer_metodo_pago("Forma de pago: EFECTIVO") == "Efectivo"
    assert extraer_metodo_pago("pagado con transferencia bancaria") == "Transferencia"
    assert extraer_metodo_pago("sin info de pago") is None
    # "TARJETA DE CREDITO" ya se cubre por substring de "CREDITO".
    assert extraer_metodo_pago("PAGO CON TARJETA DE CREDITO") == "Crédito"
    # Comprobantes Transbank que solo imprimen la marca de la tarjeta.
    assert extraer_metodo_pago("VISA **** **** **** 1234") == "Tarjeta"
    assert extraer_metodo_pago("MASTERCARD CONTACTLESS") == "Tarjeta"
    # Etiqueta en inglés (algunos POS la imprimen así) — encontrado
    # corriendo una foto real.
    assert extraer_metodo_pago("Debit            $22.990") == "Débito"
    assert extraer_metodo_pago("CREDIT CARD $5.000") == "Crédito"


def test_extraer_comercio_prioriza_sufijo_razon_social():
    assert extraer_comercio("SUPERMERCADO EJEMPLO LTDA\nRUT: 1-9") == "SUPERMERCADO EJEMPLO LTDA"
    assert extraer_comercio("CAFETERIA DON JOSE SPA\nAv. Siempreviva 742") == "CAFETERIA DON JOSE SPA"
    assert extraer_comercio("12345678\nMINIMARKET LOS ROBLES\nFECHA: 1/1/26") == "MINIMARKET LOS ROBLES"


def test_extraer_folio():
    assert extraer_folio("BOLETA ELECTRONICA N 001234") == "001234"
    assert extraer_folio("FOLIO: 55990") == "55990"
    assert extraer_folio("sin folio") is None


def test_extraer_productos_formato_columnas_al_final():
    texto = (
        "PAN HALLULLA          1     1200\n"
        "LECHE DESCREMADA 1L   2     1990\n"
        "ARROZ GRADO 1 1KG      1     1450\n"
        "COCA COLA 1.5L          1     1690"
    )
    productos = extraer_productos(texto)
    assert len(productos) == 4
    assert productos[0].nombre == "PAN HALLULLA"
    assert productos[0].cantidad == 1 and productos[0].precio_unitario == 1200
    assert productos[1].nombre == "LECHE DESCREMADA 1L"
    assert productos[1].cantidad == 2 and productos[1].precio_unitario == 1990
    # Los números dentro del nombre ("GRADO 1", "1L", "1.5L") no deben
    # confundirse con columnas de cantidad/precio.
    assert productos[2].nombre == "ARROZ GRADO 1 1KG"
    assert productos[2].precio_unitario == 1450
    assert productos[3].nombre == "COCA COLA 1.5L"


def test_extraer_productos_formato_cantidad_al_inicio():
    texto = (
        "2 Cafe Americano         3400\n"
        "1 Sandwich Barros Luco   4990\n"
        "1 Jugo Natural Naranja   2200"
    )
    productos = extraer_productos(texto)
    assert len(productos) == 3
    assert productos[0].nombre == "Cafe Americano"
    assert productos[0].cantidad == 2 and productos[0].precio_unitario == 3400
    assert productos[1].nombre == "Sandwich Barros Luco"


def test_parsear_boleta_supermercado_completo():
    texto = (
        "SUPERMERCADO EJEMPLO LTDA\n"
        "RUT: 76.123.456-7\n"
        "BOLETA ELECTRONICA N 001234\n"
        "FECHA: 14/07/2026   HORA: 18:32\n"
        "--------------------------------\n"
        "PAN HALLULLA          1     1200\n"
        "LECHE DESCREMADA 1L   2     1990\n"
        "ARROZ GRADO 1 1KG      1     1450\n"
        "COCA COLA 1.5L          1     1690\n"
        "--------------------------------\n"
        "SUBTOTAL                     8320\n"
        "TOTAL               CLP      8320\n"
        "MEDIO DE PAGO: DEBITO\n"
        "GRACIAS POR SU COMPRA"
    )
    resultado = parsear_boleta(texto)
    d = resultado.datos
    assert d.comercio == "SUPERMERCADO EJEMPLO LTDA"
    assert d.fecha == "2026-07-14"
    assert d.metodo_pago == "Débito"
    assert d.total == 8320.0
    assert resultado.folio == "001234"
    assert len(d.productos) == 4
    assert sum(p.cantidad * p.precio_unitario for p in d.productos) == d.total


def test_parsear_boleta_formato_restaurante_distinto():
    """Mismo parser, formato de boleta completamente distinto: guiones,
    cantidad al inicio, 'TOTAL A PAGAR' en vez de 'TOTAL'."""
    texto = (
        "CAFETERIA DON JOSE SPA\n"
        "Av. Providencia 1234, Santiago\n"
        "--------------------------------\n"
        "14-07-2026 13:15\n"
        "Mesa: 5   Garzon: Maria\n"
        "--------------------------------\n"
        "2 Cafe Americano         3400\n"
        "1 Sandwich Barros Luco   4990\n"
        "1 Jugo Natural Naranja   2200\n"
        "--------------------------------\n"
        "Subtotal              13990\n"
        "Propina sugerida (10%)  1399\n"
        "TOTAL A PAGAR          15389\n"
        "Forma de pago: EFECTIVO\n"
        "Vuelvan pronto!"
    )
    resultado = parsear_boleta(texto)
    d = resultado.datos
    assert d.comercio == "CAFETERIA DON JOSE SPA"
    assert d.fecha == "2026-07-14"
    assert d.metodo_pago == "Efectivo"
    assert d.total == 15389.0
    assert len(d.productos) == 3


def test_parsear_boleta_sin_campos_reconocibles_no_crashea():
    resultado = parsear_boleta("xqz asdkj wlekrj\nzzz 000")
    assert resultado.datos.fecha is None
    assert resultado.datos.total is None
    assert resultado.datos.productos == []


# --- Regresiones de la ronda de validación con boletas reales (ver
# README, sección "Validación con boletas reales") ---


def test_extraer_productos_no_confunde_moneda_peso_con_producto():
    """Bug real #1: 'MONEDA: PESO' se fusionaba como producto falso."""
    texto = (
        "PAN HALLULLA          1     1200\n"
        "MONEDA: PESO\n"
        "LECHE DESCREMADA 1L   2     1990"
    )
    productos = extraer_productos(texto)
    assert len(productos) == 2
    assert all("PESO" not in p.nombre.upper() for p in productos)


def test_extraer_total_ignora_desglose_de_impuestos():
    """Bug real #2: 'TOTAL AFECTO'/'TOTAL EXENTO'/'TOTAL IVA' le ganaban
    al TOTAL real porque aparecen antes en el documento."""
    texto = (
        "TOTAL AFECTO              6.990\n"
        "TOTAL EXENTO                  0\n"
        "TOTAL IVA                 1.328\n"
        "TOTAL                     8.318\n"
    )
    assert extraer_total(texto) == 8318.0


def test_extraer_total_sin_total_real_usa_subtotal():
    """Si de verdad no hay más que desglose de impuestos, sigue sin
    inventar: cae a None (no hay SUBTOTAL tampoco en este caso)."""
    texto = "TOTAL AFECTO   6.990\nTOTAL EXENTO   0\n"
    assert extraer_total(texto) is None


def test_extraer_comercio_encuentra_nombre_mas_alla_de_linea_6():
    """Bug real #3: la ventana de 6 líneas se quedaba corta en boletas
    reales con más encabezado (logo, dirección, giro) antes del nombre."""
    texto = (
        "\n"
        "76.123.456-7\n"
        "R.U.T.\n"
        "GIRO: VENTA AL POR MENOR\n"
        "AV. SIEMPRE VIVA 742\n"
        "SUCURSAL CENTRO\n"
        "MINIMARKET LOS ROBLES SPA\n"
        "FECHA: 1/1/26\n"
    )
    assert extraer_comercio(texto) == "MINIMARKET LOS ROBLES SPA"


def test_extraer_productos_precio_pegado_al_signo_peso():
    """Bug real #4 (el de mayor impacto): precios pegados a '$' sin
    espaciado amplio deben reconocerse igual como columna."""
    texto = "FRUG KREM $2.490\nYOGHUR NATURAL $990"
    productos = extraer_productos(texto)
    assert len(productos) == 2
    assert productos[0].nombre == "FRUG KREM"
    assert productos[0].precio_unitario == 2490.0
    assert productos[1].nombre == "YOGHUR NATURAL"
    assert productos[1].precio_unitario == 990.0


# --- Regresiones encontradas corriendo 6 fotos reales nuevas (ver
# README, sección "Segunda ronda de boletas reales") ---


def test_extraer_total_ignora_encabezado_de_tabla_de_productos():
    """El encabezado de una tabla de productos (ej. OCR de "Cód Prod
    P.Unit Cant Total") incluye "TOTAL" como nombre de columna — no debe
    confundirse con la línea del total real, ni usarse como punto de
    partida para buscar un monto en las líneas siguientes (que son la
    primera fila de productos, no el total)."""
    texto = (
        "Cód Prad Punt Cant Total\n"
        "CCX2 CARTON 3900. 1 3.990\n"
        "SUBTOTAL $: 5.970\n"
        "TOTAL $: 5970\n"
    )
    assert extraer_total(texto) == 5970.0


def test_extraer_total_no_confunde_otro_campo_al_buscar_monto_adelante():
    """Una línea "TOTALES" (encabezado de sección, sin monto propio)
    seguida de otros campos con SU PROPIO monto (NETO, EXENTO) no debe
    hacer que se tome el monto de esos otros campos como si fuera el
    total."""
    texto = "TOTALES\nNETO $: 5.017\nEXENTO $: 0\nIVA $: 953\nTOTAL $: 5970\n"
    assert extraer_total(texto) == 5970.0


def test_extraer_comercio_ignora_ruido_de_un_caracter_al_contar_ventana():
    """Ruido de OCR de 1-2 caracteres al inicio del documento (frecuente
    en fotos reales con textura/reflejos) no debe agotar la ventana de
    10 líneas antes de llegar al nombre real del comercio."""
    texto = (
        "s\n.\nA\n;\n,\n¿\nJ\n|\nj\n,\n"
        "COMPAÑIA GENERAL DE ELECTRICIDAD S.A.\n"
        "RUT: 76.411.321-7\n"
    )
    assert extraer_comercio(texto) == "COMPAÑIA GENERAL DE ELECTRICIDAD S.A."


def test_extraer_comercio_excluye_rut_con_puntuacion_distinta_a_la_esperada():
    """El OCR a veces lee "R.U.T." con coma en vez de punto, o sin
    espaciado consistente — debe seguir excluyéndose como encabezado,
    no tomarse como si fuera el nombre del comercio."""
    texto = "R.U,T.176,818,189-6\nBOLETA ELECTRONICA No 718927\nSOCIEDAD EJEMPLO SPA\n"
    assert extraer_comercio(texto) != "R.U,T.176,818,189-6"
    assert extraer_comercio(texto) == "SOCIEDAD EJEMPLO SPA"


def test_extraer_fecha_ignora_resolucion_sii_del_pie_de_boleta():
    """Toda boleta electrónica chilena trae en el pie una referencia fija
    a la Resolución SII N°80 del 22-08-2014 (timbre electrónico) — esa
    fecha nunca es la fecha real de la boleta. Si la fecha real sale
    ilegible del OCR pero esta referencia sale clara, no debe tomarse
    por error."""
    texto = "FECHA DE EMISION: 22-85-2026\nTIMBRE ELECTRONICO S.I.I. Res. 80 del 22-08-2014\n"
    # La fecha real (con mes "85") no es válida como fecha calendario;
    # la de la resolución sí lo sería si no se excluyera explícitamente.
    assert extraer_fecha(texto) is None


def test_extraer_fecha_real_se_reconoce_aunque_haya_resolucion_sii_despues():
    texto = "Fecha de Emisión: 22-05-2026\nRes. 80 del 22-08-2014\n"
    assert extraer_fecha(texto) == "2026-05-22"


def test_extraer_productos_no_confunde_linea_de_detalle_de_pago_con_producto():
    """Una línea de detalle de pago (ej. "TARJETA DEBITO $5.970" en la
    sección de pagos del final de la boleta) no debe confundirse con una
    línea de producto real."""
    texto = "PAN HALLULLA          1     1200\nTARJETA DEBITO $ 5970"
    productos = extraer_productos(texto)
    assert len(productos) == 1
    assert productos[0].nombre == "PAN HALLULLA"


def test_extraer_productos_no_confunde_debit_en_ingles_con_producto():
    """Mismo bug que el test de arriba, pero con la etiqueta de pago en
    inglés: "Debit $22.990" se leía como un producto con el total
    completo como precio — encontrado corriendo una foto real de una
    boleta de tienda de ropa (un solo producto: $22.990 pagado con
    "Debit", sin la palabra en español en ningún lado de la boleta)."""
    texto = "PANTALON HB RECTO BOLS          1     22990\nDebit            $22.990"
    productos = extraer_productos(texto)
    assert len(productos) == 1
    assert productos[0].nombre == "PANTALON HB RECTO BOLS"


def test_extraer_total_ignora_digito_espurio_antes_de_dos_puntos():
    """El OCR a veces confunde '$' con un dígito suelto justo después de
    la etiqueta ('TOTAL 9: 5970', donde '9' es en realidad un '$' mal
    leído) — el monto real, después del ':', no debe perderse frente a
    ese dígito espurio."""
    assert extraer_total("TOTAL 9: 5970") == 5970.0
    # Sin ':', sigue tomando el único número de la línea (sin cambios).
    assert extraer_total("TOTAL CLP 8320") == 8320.0


# --- Regresiones de la tercera ronda de boletas reales (Easy Retail,
# Popcorn, Supermercados Cugat — ver README) ---


def test_extraer_total_reconoce_etiqueta_monto_boleta_electronica():
    """Algunos sistemas de punto de venta usan "MONTO BOLETA ELECTRONICA"
    en vez de "TOTAL" para el mismo concepto (boleta real de Popcorn)."""
    texto = "CANT.ITEM SUBTOTAL\n1 Caramel Bliss Refill Ba 6.500\nMONTO BOLETA ELECTRONICA $6.500\nEl IVA de esta boleta es $1.038\n"
    assert extraer_total(texto) == 6500.0


def test_extraer_total_prefiere_total_final_sobre_total_generico():
    """Boleta real de Supermercados Cugat: "TOTAL $742" aparece antes
    que "TOTAL FINAL $739" (tras aplicar redondeo legal) — el segundo es
    el monto correcto."""
    texto = "TOTAL\n$ 742\nREDONDEO LEGAL OBLIGATORIO\n-$ 3\nTOTAL FINAL\n$ 739\n"
    assert extraer_total(texto) == 739.0


def test_extraer_total_prefiere_a_pagar_sobre_total_generico():
    """Boleta real de Aguas de Colina: "Total Mes $56.670" es un
    subtotal mensual; "TOTAL A PAGAR $459.730" (incluye saldo anterior)
    es el monto correcto."""
    texto = "Total Mes\n$ 56.670\nSaldo Anterior\n$ 403.060\nTOTAL A PAGAR\n$ 459.730\n"
    assert extraer_total(texto) == 459730.0


def test_extraer_total_a_pagar_ignora_texto_pegado_antes_en_la_misma_linea():
    """Boleta real de Aguas de Colina: el OCR pegó una fecha justo antes
    de "TOTAL A PAGAR" en la misma línea de detección — el primer
    número de esa fecha no debe ganarle al monto real que sigue a la
    frase."""
    texto = "Próxima lectura entre el 26/07/2023 y 30/07/2023         TOTAL A PAGAR     Ss    459.730"
    assert extraer_total(texto) == 459730.0


def test_extraer_comercio_reconoce_sa_con_puntos():
    """"S.A." (con puntos, la forma más común de escribirlo) debe seguir
    contando como sufijo de razón social — antes solo calzaba "SA" sin
    puntos (boleta real de Easy Retail)."""
    texto = "RUT 76568660-1\nBOLETA ELECTRONICA N 269198620\nSII Osorno\nEASY RETAIL S.A.\nAv. Cesar Ercilla 1075\n"
    assert extraer_comercio(texto) == "EASY RETAIL S.A."


def test_extraer_productos_no_confunde_texto_de_puntos_fidelizacion_con_producto():
    """Texto de programas de puntos/fidelización (ej. Puntos Cencosud)
    no debe confundirse con una línea de producto (boleta real de Easy
    Retail)."""
    texto = (
        "SOLDADURA C/PT 1MT          1     4250\n"
        "PUNTOS CENCOSUD\n"
        "PODRIAS HABER ACUMULADO : 14\n"
    )
    productos = extraer_productos(texto)
    assert len(productos) == 1
    assert productos[0].nombre == "SOLDADURA C/PT 1MT"


# --- Regresiones de correr 3 fotos reales nuevas (Cugat, Easy Retail,
# Popcorn) contra el servidor real ---


def test_extraer_total_ignora_digito_espurio_del_ocr_antes_de_la_etiqueta():
    """Boleta real de Easy Retail: el OCR leyó "SUB TOTAL" como "5UB
    TOTAL" (la "S" se leyó como el dígito "5"). Sin restringir la
    búsqueda del monto a lo que sigue a la etiqueta encontrada, ese "5"
    suelto le ganaba al monto real de la línea."""
    assert extraer_total("5UB TOTAL         4,20") == 4200.0


def test_parsear_monto_clp_trata_sufijo_de_2_digitos_como_miles_truncados():
    """Boleta real de Popcorn: "$6.500" se leyó "$6,50d" (perdió el
    último dígito) — debe interpretarse como 6500, no como 6.5."""
    assert parsear_monto_clp("MONTO BOLETA ELECTRONICA $6,50d") == 6500.0


# --- Regresiones de probar 3 fotos reales nuevas (misma cadena de
# retail de vestuario/calzado: pantufla, pantalón; y un comprobante
# Transbank) contra el servidor real ---


def test_extraer_productos_formato_codigo_y_descripcion_en_linea_siguiente():
    """Boleta real (pantufla, $2.990, efectivo): cantidad/precio unitario
    /descuento/total van en una línea, y código/descripción en la línea
    siguiente — el orden inverso del formato ya soportado
    ("_fusionar_lineas_divididas" cubre nombre-primero-números-después,
    no al revés). Texto tal cual lo devolvió Tesseract: sin este fix el
    producto se perdía por completo pese a que el OCR lo había leído
    bien."""
    texto = (
        "1          $2.990            $0            $2.990\n"
        "2646146500226 PANTUFLA HPG D1 IN26-44/45-B\n"
    )
    productos = extraer_productos(texto)
    assert len(productos) == 1
    assert productos[0].nombre == "PANTUFLA HPG D1 IN26-44/45-B"
    assert productos[0].cantidad == 1.0
    assert productos[0].precio_unitario == 2990.0


def test_extraer_productos_formato_codigo_y_descripcion_usa_precio_unitario_no_total():
    """Con cantidad > 1 y descuento > 0, precio unitario y total de
    línea difieren — debe tomarse el precio unitario (segunda columna:
    "CANTIDAD PRECIO UNITARIO DESCUENTO TOTAL"), no el total (cuarta)."""
    texto = (
        "2          $1.000            $500            $1.500\n"
        "1234567890123 PRODUCTO DE PRUEBA\n"
    )
    productos = extraer_productos(texto)
    assert len(productos) == 1
    assert productos[0].cantidad == 2.0
    assert productos[0].precio_unitario == 1000.0

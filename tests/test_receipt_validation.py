"""Pruebas de validaciones internas y puntaje de confianza (Etapa 6)."""
from app.models.schemas import ConfianzaPorCampo, DatosExtraidos, ProductoExtraido
from app.ocr.ocr_engine import PalabraOCR
from app.receipt_validation.consistency_checks import validar_datos
from app.receipt_validation.document_confidence import calcular_confianza_general
from app.receipt_validation.field_confidence import (
    confianza_comercio,
    confianza_fecha,
    confianza_metodo_pago,
    confianza_productos,
    confianza_total,
)
from app.receipt_validation.receipt_validator import evaluar_lectura


def _palabra(texto, confianza, x=0):
    return PalabraOCR(texto=texto, confianza=confianza, x=x, y=0, ancho=10, alto=10, linea=0, bloque=0)


def test_confianza_comercio_promedia_tokens_coincidentes():
    palabras = [_palabra("SUPERMERCADO", 95), _palabra("EJEMPLO", 90), _palabra("LTDA", 85)]
    assert confianza_comercio("SUPERMERCADO EJEMPLO LTDA", palabras) == 90.0
    assert confianza_comercio(None, palabras) is None
    assert confianza_comercio("ALGO QUE NO ESTA", palabras) is None


def test_confianza_fecha_usa_digitos_no_texto_literal():
    palabras = [_palabra("FECHA:", 90), _palabra("14/07/2026", 88), _palabra("HORA:", 85)]
    assert confianza_fecha("2026-07-14", palabras) == 88.0
    assert confianza_fecha("2026-07-14", [_palabra("nada relacionado", 50)]) is None


def test_confianza_metodo_pago_usa_palabra_original():
    palabras = [_palabra("MEDIO", 90), _palabra("PAGO:", 90), _palabra("DEBITO", 82)]
    assert confianza_metodo_pago("Débito", palabras) == 82.0


def test_confianza_total_por_valor_monetario():
    palabras = [_palabra("TOTAL", 95), _palabra("CLP", 95), _palabra("8320", 93)]
    assert confianza_total(8320.0, palabras) == 93.0
    assert confianza_total(9999.0, palabras) is None


def test_confianza_productos_completa_campo_confianza():
    datos = DatosExtraidos(
        productos=[
            ProductoExtraido(nombre="PAN HALLULLA", cantidad=1, precio_unitario=1200),
            ProductoExtraido(nombre="PRODUCTO INEXISTENTE", cantidad=1, precio_unitario=500),
        ]
    )
    palabras = [_palabra("PAN", 90), _palabra("HALLULLA", 88)]
    _, promedio = confianza_productos(datos, palabras)
    assert datos.productos[0].confianza == 89.0
    assert datos.productos[1].confianza is None
    assert promedio == 89.0


def test_validar_datos_caso_perfecto():
    datos = DatosExtraidos(
        comercio="SUPERMERCADO EJEMPLO LTDA",
        fecha="2026-07-14",
        metodo_pago="Débito",
        total=1200.0,
        productos=[ProductoExtraido(nombre="PAN HALLULLA", cantidad=1, precio_unitario=1200)],
    )
    confianza_alta = ConfianzaPorCampo(comercio=90, fecha=90, metodo_pago=90, productos=90, total=90)
    v = validar_datos(datos, confianza_alta)
    assert v.advertencias == []
    assert v.campos_faltantes == []
    assert v.suma_coincide_con_total is True
    assert v.penalizacion == 0.0


def test_validar_datos_campos_faltantes():
    v = validar_datos(DatosExtraidos(), ConfianzaPorCampo())
    assert set(v.campos_faltantes) == {"comercio", "fecha", "total", "productos"}
    assert len(v.advertencias) == 4
    # comercio, fecha y productos faltantes pesan igual entre sí (5 c/u);
    # el total, el dato más importante, pesa bastante más (30).
    assert v.penalizacion == 5.0 + 5.0 + 30.0 + 5.0


def test_validar_datos_detecta_descuadre():
    datos = DatosExtraidos(
        comercio="TIENDA X",
        fecha="2026-01-01",
        total=5000.0,
        productos=[ProductoExtraido(nombre="ALGO", cantidad=1, precio_unitario=1200)],
    )
    confianza_alta = ConfianzaPorCampo(comercio=90, fecha=90, productos=90, total=90)
    v = validar_datos(datos, confianza_alta)
    assert v.suma_coincide_con_total is False
    assert any("no coincide" in a for a in v.advertencias)
    # El descuadre es, a propósito, la penalización más grande de todas
    # — más que cualquier campo faltante o impreciso por separado.
    assert v.penalizacion == 40.0


def test_validar_datos_detecta_valores_invalidos():
    datos = DatosExtraidos(
        comercio="X",
        fecha="2026-01-01",
        total=-100.0,
        productos=[ProductoExtraido(nombre="RARO", cantidad=-2, precio_unitario=0)],
    )
    confianza_alta = ConfianzaPorCampo(comercio=90, fecha=90, productos=90, total=90)
    v = validar_datos(datos, confianza_alta)
    assert any("precio inválido" in a for a in v.advertencias)
    assert any("cantidad inválida" in a for a in v.advertencias)
    # 3 inconsistencias (total<=0, precio<=0, cantidad<=0) a 10 c/u, más
    # el descuadre (suma=0 vs total=-100 también difieren): 30 + 40 = 70,
    # bajo el tope máximo (80).
    assert v.penalizacion == 70.0


def test_validar_datos_marca_campo_impreciso_sin_confundirlo_con_faltante():
    """Comercio y fecha presentes pero con confianza de OCR baja generan
    una advertencia distinta a la de "campo faltante" — no deberían
    contar como campos_faltantes."""
    datos = DatosExtraidos(
        comercio="TIENDA BORROSA",
        fecha="2026-01-01",
        metodo_pago="Débito",
        total=1000.0,
        productos=[ProductoExtraido(nombre="ALGO", cantidad=1, precio_unitario=1000)],
    )
    confianza_baja_comercio_fecha = ConfianzaPorCampo(
        comercio=20.0, fecha=30.0, metodo_pago=90.0, productos=90.0, total=90.0
    )
    v = validar_datos(datos, confianza_baja_comercio_fecha)
    assert v.campos_faltantes == []
    assert any("comercio identificado" in a and "podría no ser exacto" in a for a in v.advertencias)
    assert any("fecha identificada" in a and "podría no ser exacta" in a for a in v.advertencias)
    # comercio y fecha pesan igual que productos (5 c/u) — ya no son
    # "secundarios": el único campo con menos peso es método de pago.
    assert v.penalizacion == 5.0 + 5.0


def test_validar_datos_marca_total_y_productos_imprecisos_con_mas_peso():
    datos = DatosExtraidos(
        comercio="TIENDA",
        fecha="2026-01-01",
        metodo_pago="Efectivo",
        total=1000.0,
        productos=[ProductoExtraido(nombre="ALGO RARO", cantidad=1, precio_unitario=1000)],
    )
    confianza_total_dudoso = ConfianzaPorCampo(
        comercio=90.0, fecha=90.0, metodo_pago=90.0, productos=20.0, total=10.0
    )
    # En el flujo real, `producto.confianza` lo completa confianza_productos()
    # (Etapa 6) antes de llegar acá — se simula a mano en este test unitario.
    datos.productos[0].confianza = 20.0
    v = validar_datos(datos, confianza_total_dudoso)
    assert any("total identificado" in a and "podría no ser exacto" in a for a in v.advertencias)
    assert any("ALGO RARO" in a for a in v.advertencias)
    # Total impreciso (25) + producto impreciso (5): el total pesa mucho
    # más que cualquier otro campo impreciso por separado.
    assert v.penalizacion == 25.0 + 5.0


def test_validar_datos_no_marca_impreciso_si_no_se_pudo_calcular_confianza():
    """Si confianza_por_campo trae None (no se pudo relacionar el dato
    con ninguna palabra del OCR), no se marca como "impreciso" — eso es
    un caso distinto, no una confianza baja confirmada."""
    datos = DatosExtraidos(comercio="TIENDA", total=1000.0)
    v = validar_datos(datos, ConfianzaPorCampo(comercio=None, total=None))
    assert not any("podría no ser exacto" in a for a in v.advertencias)


def test_calcular_confianza_general_el_total_pesa_mas_que_el_resto():
    """Un total en cero debería hundir la confianza general bastante más
    que si cualquier otro campo estuviera en cero, porque el total pesa
    3x en el promedio (comercio/fecha/productos pesan 1x)."""
    todo_bien_menos_total = ConfianzaPorCampo(comercio=100, fecha=100, metodo_pago=100, productos=100, total=0.0)
    todo_bien_menos_comercio = ConfianzaPorCampo(comercio=0.0, fecha=100, metodo_pago=100, productos=100, total=100)
    resultado_total_mal = calcular_confianza_general(todo_bien_menos_total, confianza_ocr_general=90, penalizacion=0.0)
    resultado_comercio_mal = calcular_confianza_general(
        todo_bien_menos_comercio, confianza_ocr_general=90, penalizacion=0.0
    )
    assert resultado_total_mal < resultado_comercio_mal


def test_calcular_confianza_general_metodo_pago_pesa_poco():
    """Método de pago en cero, con todo lo demás perfecto (incluido el
    total), apenas debería mover la confianza general — es el único
    campo opcional del formulario de Blynn."""
    campos = ConfianzaPorCampo(comercio=100, fecha=100, metodo_pago=0.0, productos=100, total=100)
    resultado = calcular_confianza_general(campos, confianza_ocr_general=90, penalizacion=0.0)
    assert resultado > 90.0


def test_calcular_confianza_general_resta_penalizacion():
    campos = ConfianzaPorCampo(comercio=95, fecha=90, metodo_pago=85, productos=92, total=97)
    base = calcular_confianza_general(campos, confianza_ocr_general=90, penalizacion=0.0)
    con_penalizacion = calcular_confianza_general(campos, confianza_ocr_general=90, penalizacion=25.0)
    assert con_penalizacion == round(base - 25.0, 1)
    # La resta nunca baja de 0, aunque la penalización sea mayor a la base.
    assert calcular_confianza_general(campos, confianza_ocr_general=90, penalizacion=999.0) == 0.0


def test_calcular_confianza_general_usa_ocr_si_no_hay_campos():
    sin_campos = ConfianzaPorCampo()
    assert calcular_confianza_general(sin_campos, confianza_ocr_general=42.0, penalizacion=0.0) == 42.0


def test_evaluar_lectura_end_to_end(boleta_limpia):
    from app.ocr.ocr_engine import reconocer_texto
    from app.parsing.receipt_parser import parsear_boleta
    from app.config import Settings

    settings = Settings()
    resultado_ocr = reconocer_texto(boleta_limpia, settings)
    resultado_parser = parsear_boleta(resultado_ocr.texto_completo)

    evaluacion = evaluar_lectura(resultado_parser.datos, resultado_ocr.palabras, resultado_ocr.confianza_promedio)

    assert evaluacion.validacion.advertencias == []
    assert evaluacion.validacion.suma_coincide_con_total is True
    assert evaluacion.confianza_general > 70
    assert all(p.confianza is not None for p in resultado_parser.datos.productos)


def test_evaluar_lectura_detecta_dato_manipulado(boleta_limpia):
    from app.ocr.ocr_engine import reconocer_texto
    from app.parsing.receipt_parser import parsear_boleta
    from app.config import Settings

    settings = Settings()
    resultado_ocr = reconocer_texto(boleta_limpia, settings)
    resultado_parser = parsear_boleta(resultado_ocr.texto_completo)

    evaluacion_original = evaluar_lectura(
        resultado_parser.datos, resultado_ocr.palabras, resultado_ocr.confianza_promedio
    )

    datos_manipulados = resultado_parser.datos.model_copy(update={"total": 99999.0})
    evaluacion_manipulada = evaluar_lectura(
        datos_manipulados, resultado_ocr.palabras, resultado_ocr.confianza_promedio
    )

    assert evaluacion_manipulada.confianza_general < evaluacion_original.confianza_general
    assert len(evaluacion_manipulada.validacion.advertencias) >= 1


def test_evaluar_lectura_renombra_productos_imprecisos():
    """A pedido explícito de Camilo: si hay productos pero se leyeron
    con confianza baja, se muestran como "Artículo N" en vez del texto
    garabateado — y la advertencia de "producto impreciso" (Etapa 6)
    tiene que hablar del mismo nombre que termina viendo el usuario."""
    datos = DatosExtraidos(
        total=1700.0,
        productos=[
            ProductoExtraido(nombre="PAN HALLULLA", cantidad=1, precio_unitario=1200),
            ProductoExtraido(nombre="GALLETA SODA", cantidad=1, precio_unitario=500),
        ],
    )
    palabras = [
        _palabra("PAN", 90),
        _palabra("HALLULLA", 88),
        _palabra("GALLETA", 20),
        _palabra("SODA", 15),
    ]

    evaluacion = evaluar_lectura(datos, palabras, confianza_ocr_general=80.0)

    assert datos.productos[0].nombre == "PAN HALLULLA"  # confianza alta -> no se toca
    assert datos.productos[1].nombre == "Artículo 2"  # confianza baja -> renombrado
    assert any(
        "'Artículo 2'" in a and "podría no estar bien identificado" in a
        for a in evaluacion.validacion.advertencias
    )
    assert not any("GALLETA SODA" in a for a in evaluacion.validacion.advertencias)

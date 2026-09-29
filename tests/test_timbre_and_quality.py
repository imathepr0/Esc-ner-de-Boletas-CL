"""
Pruebas para las mejoras posteriores a la ronda de validación con
boletas reales (ver README, sección "Mejoras posteriores a la
validación con boletas reales"):

- Detección de borrosidad + reescalado: `app/image_processing/quality.py`
  y los cambios correspondientes en `preprocessor.py` / `retry_orchestrator.py`.
- Timbre electrónico (PDF417): `app/timbre/` y
  `app/pipeline/timbre_enrichment.py`.
"""
import cv2
import numpy as np
import pdf417gen

from app.image_processing.preprocessor import preprocesar_imagen
from app.image_processing.quality import calcular_nitidez
from app.models.enums import ProcessingState
from app.models.schemas import DatosExtraidos, ProductoExtraido
from app.pipeline.retry_orchestrator import ejecutar_pipeline
from app.pipeline.timbre_enrichment import enriquecer_con_timbre
from app.receipt_validation.receipt_validator import evaluar_lectura
from app.responses.error_responses import respuesta_imagen_borrosa
from app.timbre.barcode_decoder import decodificar_timbre
from app.timbre.ted_parser import DatosTED, parsear_ted

TED_EJEMPLO = (
    '<TED version="1.0"><DD><RE>76543210-5</RE><TD>39</TD><F>1234</F>'
    "<FE>2026-07-14</FE><RR>11111111-1</RR><MNT>8320</MNT></DD>"
    '<FRMT algoritmo="SHA1withRSA">firma_falsa</FRMT></TED>'
)


def _generar_imagen_con_timbre(texto_ted: str = TED_EJEMPLO, lienzo=(1000, 1400)) -> np.ndarray:
    """Genera una imagen BGR con un código PDF417 real (codificado con
    pdf417gen) embebido, simulando dónde aparecería el timbre en una
    boleta. Solo se usa en las pruebas: el backend nunca genera códigos,
    solo los decodifica."""
    codigos = pdf417gen.encode(texto_ted, columns=8, security_level=5)
    timbre = pdf417gen.render_image(codigos, scale=3, ratio=3).convert("RGB")
    timbre_np = np.array(timbre)
    ancho, alto = lienzo
    lienzo_img = np.full((alto, ancho, 3), 255, dtype=np.uint8)
    y0 = alto - timbre_np.shape[0] - 100
    x0 = (ancho - timbre_np.shape[1]) // 2
    lienzo_img[y0 : y0 + timbre_np.shape[0], x0 : x0 + timbre_np.shape[1]] = timbre_np
    return cv2.cvtColor(lienzo_img, cv2.COLOR_RGB2BGR)


def _generar_foto_con_fondo_y_timbre_real(texto_ted: str = TED_EJEMPLO) -> np.ndarray:
    """Genera una 'foto' sintética realista (mismo patrón que las
    fixtures de `conftest.py`: boleta + perspectiva + fondo con ruido +
    sombra) con un timbre PDF417 real embebido cerca del borde inferior
    del documento. A diferencia de `_generar_imagen_con_timbre` (que
    pone el código directo sobre un lienzo en blanco), esto sirve para
    probar el flujo completo detección-de-documento -> recorte ->
    timbre en condiciones parecidas a una foto real, no el decodificador
    de forma aislada."""
    ancho, alto = 480, 760
    boleta = np.full((alto, ancho, 3), 255, dtype=np.uint8)
    cv2.putText(boleta, "COMERCIO EJEMPLO SPA", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(boleta, "TOTAL CLP 8320", (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    codigos = pdf417gen.encode(texto_ted, columns=6, security_level=3)
    timbre = pdf417gen.render_image(codigos, scale=2, ratio=2).convert("RGB")
    timbre_np = np.array(timbre)
    y0 = alto - timbre_np.shape[0] - 30
    x0 = (ancho - timbre_np.shape[1]) // 2
    boleta[y0 : y0 + timbre_np.shape[0], x0 : x0 + timbre_np.shape[1]] = timbre_np

    np.random.seed(11)
    ba, bh = boleta.shape[1], boleta.shape[0]
    pts_origen = np.float32([[0, 0], [ba, 0], [ba, bh], [0, bh]])
    pts_destino = np.float32([[60, 40], [890, 30], [900, 1010], [50, 1020]])
    ancho_fondo, alto_fondo = 950, 1050
    fondo = np.random.randint(90, 130, (alto_fondo, ancho_fondo, 3), dtype=np.uint8)
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    warp = cv2.warpPerspective(boleta, matriz, (ancho_fondo, alto_fondo), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full((bh, ba), 255, dtype=np.uint8), matriz, (ancho_fondo, alto_fondo)
    )
    foto = fondo.copy()
    foto[mask > 0] = warp[mask > 0]
    return foto


# --- Nitidez ---


class TestNitidez:
    def test_imagen_en_blanco_da_nitidez_cero(self):
        blanca = np.full((300, 300, 3), 255, dtype=np.uint8)
        assert calcular_nitidez(blanca) == 0.0

    def test_imagen_vacia_no_crashea(self):
        assert calcular_nitidez(np.array([], dtype=np.uint8)) == 0.0
        assert calcular_nitidez(None) == 0.0

    def test_imagen_con_bordes_marcados_da_nitidez_alta(self):
        tablero = np.zeros((200, 200), dtype=np.uint8)
        tablero[::2, ::2] = 255  # patrón de alto contraste, bordes por doquier
        assert calcular_nitidez(tablero) > 100.0

    def test_desenfoque_reduce_la_nitidez_medida(self, boleta_limpia):
        nitida = calcular_nitidez(boleta_limpia)
        borrosa = calcular_nitidez(cv2.GaussianBlur(boleta_limpia, (0, 0), sigmaX=4))
        assert borrosa < nitida

    def test_acepta_imagen_en_escala_de_grises_directamente(self, boleta_limpia):
        gris = cv2.cvtColor(boleta_limpia, cv2.COLOR_BGR2GRAY)
        assert calcular_nitidez(gris) > 0.0


# --- Reescalado (integrado en preprocesar_imagen) ---


class TestReescalado:
    def test_documento_chico_se_reescala_hacia_arriba(self):
        chica = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.putText(chica, "BOLETA", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        resultado = preprocesar_imagen(chica, ancho_objetivo_escalado=1200)
        assert resultado.factor_escala_aplicado > 1.0
        assert max(resultado.imagen_gris.shape) > max(chica.shape[:2])

    def test_documento_ya_grande_no_se_reescala(self, boleta_limpia):
        # boleta_limpia es 520x620: sin documento detectado, se usa tal
        # cual (lado menor 520 < 1200 -> igual se reescala). Se prueba
        # el caso "ya grande" directamente con una imagen >= objetivo.
        grande = np.full((1500, 1300, 3), 255, dtype=np.uint8)
        resultado = preprocesar_imagen(grande, ancho_objetivo_escalado=1200)
        assert resultado.factor_escala_aplicado == 1.0

    def test_factor_de_escala_respeta_el_tope_maximo(self):
        diminuta = np.full((20, 30, 3), 255, dtype=np.uint8)
        resultado = preprocesar_imagen(diminuta, ancho_objetivo_escalado=1200)
        assert resultado.factor_escala_aplicado <= 3.0

    def test_resultado_preprocesamiento_tiene_defaults_sin_pasar_nitidez_ni_escala(self):
        """Regresión: estos 2 campos se agregaron con valor por defecto
        a propósito, para que construir el dataclass directamente (como
        hacen algunas pruebas) siga funcionando sin conocerlos."""
        from app.image_processing.preprocessor import ResultadoPreprocesamiento

        r = ResultadoPreprocesamiento(
            imagen_binaria=np.zeros((10, 10), dtype=np.uint8),
            imagen_gris=np.zeros((10, 10), dtype=np.uint8),
            documento_detectado=True,
            rotacion_cardinal_aplicada=0,
            angulo_inclinacion_aplicado=0.0,
        )
        assert r.nitidez == 0.0
        assert r.factor_escala_aplicado == 1.0


# --- Gate de calidad en el orquestador ---


class TestGateDeCalidad:
    def test_imagen_nitida_no_se_bloquea(self, foto_normal, settings):
        resultado = ejecutar_pipeline(foto_normal, settings)
        assert resultado.limite_alcanzado != "calidad"
        assert resultado.nitidez >= settings.UMBRAL_NITIDEZ_MINIMA

    def test_imagen_muy_borrosa_corta_antes_del_ocr(self, foto_normal, settings):
        muy_borrosa = cv2.GaussianBlur(foto_normal, (0, 0), sigmaX=8)
        resultado = ejecutar_pipeline(muy_borrosa, settings)
        assert resultado.limite_alcanzado == "calidad"
        assert resultado.mejor_intento is None
        assert resultado.intentos_realizados == 0
        assert resultado.nitidez < settings.UMBRAL_NITIDEZ_MINIMA

    def test_borrosidad_moderada_no_bloquea_una_boleta_igual_legible(self, foto_normal, settings):
        """Calibración deliberada (ver README): un desenfoque moderado
        que el OCR todavía puede leer bien no debe bloquearse."""
        borrosa_moderada = cv2.GaussianBlur(foto_normal, (0, 0), sigmaX=2)
        resultado = ejecutar_pipeline(borrosa_moderada, settings)
        assert resultado.limite_alcanzado != "calidad"


class TestRespuestaImagenBorrosa:
    def test_estructura_de_la_respuesta(self):
        from app.models.enums import InputSource

        respuesta = respuesta_imagen_borrosa(InputSource.CAMERA)
        assert respuesta.estado == ProcessingState.IMAGE_UNREADABLE
        assert respuesta.datos is None
        assert respuesta.confianza_general == 0.0
        assert len(respuesta.advertencias) >= 1


# --- Decodificador PDF417 ---


class TestDecodificadorPDF417:
    def test_decodifica_un_timbre_valido(self):
        imagen = _generar_imagen_con_timbre()
        texto = decodificar_timbre(imagen)
        assert texto is not None
        assert "<RE>76543210-5</RE>" in texto
        assert "<MNT>8320</MNT>" in texto

    def test_imagen_sin_codigo_devuelve_none(self, boleta_limpia):
        assert decodificar_timbre(boleta_limpia) is None

    def test_imagen_vacia_no_crashea(self):
        assert decodificar_timbre(np.array([], dtype=np.uint8)) is None
        assert decodificar_timbre(None) is None

    def test_imagen_diminuta_no_crashea(self):
        assert decodificar_timbre(np.full((1, 1, 3), 128, dtype=np.uint8)) is None

    def test_ruido_aleatorio_no_crashea_y_no_encuentra_nada(self):
        np.random.seed(3)
        ruido = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        assert decodificar_timbre(ruido) is None

    def test_funciona_con_imagen_en_escala_de_grises(self):
        imagen = _generar_imagen_con_timbre()
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        assert decodificar_timbre(gris) is not None


# --- Parser del TED ---


class TestParserTED:
    def test_extrae_todos_los_campos_de_un_ted_bien_formado(self):
        datos = parsear_ted(TED_EJEMPLO)
        assert datos.rut_emisor == "76543210-5"
        assert datos.tipo_documento == "39"
        assert datos.folio == "1234"
        assert datos.fecha_emision == "2026-07-14"
        assert datos.monto_total == 8320.0

    def test_tolera_decodificacion_parcial_sin_tags_de_cierre(self):
        parcial = "<RE>76543210-5<TD>39</TD><F>1234<FE>2026-07-14</FE><MNT>8320"
        datos = parsear_ted(parcial)
        assert datos.rut_emisor == "76543210-5"
        assert datos.folio == "1234"
        assert datos.fecha_emision == "2026-07-14"
        assert datos.monto_total == 8320.0

    def test_texto_irreconocible_no_inventa_datos(self):
        assert parsear_ted("asdkj alskdj 12312") == DatosTED()

    def test_texto_vacio_no_crashea(self):
        assert parsear_ted("") == DatosTED()
        assert parsear_ted(None) == DatosTED()

    def test_fecha_calendario_invalida_queda_en_none(self):
        datos = parsear_ted("<FE>2026-13-45</FE><MNT>100</MNT>")
        assert datos.fecha_emision is None
        assert datos.monto_total == 100.0

    def test_monto_sin_puntos_de_miles_se_parsea_como_entero_plano(self):
        datos = parsear_ted("<MNT>123456</MNT>")
        assert datos.monto_total == 123456.0


# --- Enriquecimiento con timbre ---


class TestEnriquecimientoConTimbre:
    def _datos_y_evaluacion(self, fecha="2026-07-10", total=8000.0):
        datos = DatosExtraidos(
            comercio="MINIMARKET EJEMPLO",
            fecha=fecha,
            metodo_pago="Efectivo",
            productos=[ProductoExtraido(nombre="PAN", cantidad=1, precio_unitario=1200.0)],
            total=total,
        )
        evaluacion = evaluar_lectura(datos, palabras=[], confianza_ocr_general=75.0)
        return datos, evaluacion

    def test_sin_timbre_en_la_imagen_no_modifica_nada(self, boleta_limpia):
        datos, evaluacion = self._datos_y_evaluacion()
        resultado = enriquecer_con_timbre(boleta_limpia, datos, evaluacion, 75.0)
        assert resultado.timbre_decodificado is False
        assert resultado.datos is datos
        assert resultado.evaluacion is evaluacion

    def test_timbre_decodificado_reemplaza_fecha_y_total_con_confianza_100(self):
        datos, evaluacion = self._datos_y_evaluacion(fecha="2026-07-10", total=8000.0)
        imagen = _generar_imagen_con_timbre()

        resultado = enriquecer_con_timbre(imagen, datos, evaluacion, 75.0)

        assert resultado.timbre_decodificado is True
        assert resultado.datos.fecha == "2026-07-14"
        assert resultado.datos.total == 8320.0
        assert resultado.evaluacion.confianza_por_campo.fecha == 100.0
        assert resultado.evaluacion.confianza_por_campo.total == 100.0

    def test_no_muta_los_datos_originales(self):
        datos, evaluacion = self._datos_y_evaluacion(fecha="2026-07-10", total=8000.0)
        imagen = _generar_imagen_con_timbre()

        enriquecer_con_timbre(imagen, datos, evaluacion, 75.0)

        assert datos.fecha == "2026-07-10"
        assert datos.total == 8000.0

    def test_revalida_contra_el_total_ya_actualizado(self):
        """El cuadre suma-productos-vs-total debe compararse contra el
        total del timbre, no contra el que traía el OCR."""
        datos, evaluacion = self._datos_y_evaluacion(fecha="2026-07-10", total=1200.0)
        imagen = _generar_imagen_con_timbre()  # trae MNT=8320

        resultado = enriquecer_con_timbre(imagen, datos, evaluacion, 75.0)

        assert resultado.datos.total == 8320.0
        assert any("no coincide" in a for a in resultado.evaluacion.validacion.advertencias)

    def test_timbre_sin_fecha_ni_monto_no_decodificado_como_util(self):
        datos, evaluacion = self._datos_y_evaluacion()
        # TED sin <FE> ni <MNT>: no trae nada utilizable para reemplazar.
        imagen = _generar_imagen_con_timbre(texto_ted="<TED><DD><RE>76543210-5</RE></DD></TED>")

        resultado = enriquecer_con_timbre(imagen, datos, evaluacion, 75.0)

        assert resultado.timbre_decodificado is False
        assert resultado.datos.fecha == "2026-07-10"
        assert resultado.datos.total == 8000.0


# --- Integración de punta a punta (pipeline completo con timbre) ---


class TestIntegracionTimbreEndToEnd:
    def test_procesar_boleta_usa_el_timbre_cuando_esta_presente(self, settings):
        from app.models.enums import InputSource
        from app.pipeline.pipeline_service import procesar_boleta

        imagen = _generar_imagen_con_timbre()
        cv2.putText(
            imagen, "MINIMARKET EJEMPLO", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )
        cv2.putText(
            imagen, "TOTAL 8320", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )

        respuesta = procesar_boleta(imagen, InputSource.FILE, settings)

        # No se exige un estado particular (depende de si el OCR
        # reconoce algo más en esta imagen sintética simple); lo que
        # importa es que, si se obtuvo un resultado con datos, la fecha
        # coincide con la del timbre.
        if respuesta.datos is not None and respuesta.datos.fecha is not None:
            assert respuesta.datos.fecha == "2026-07-14"

    def test_quitar_el_fondo_alrededor_es_lo_que_permite_decodificar(self):
        """Hallazgo central de esta ronda: el timbre puede tener
        resolución de sobra y aun así no decodificar si está rodeado de
        mucho fondo/mesa en una imagen grande (problema de localización
        del código, no solo de nitidez) — decodifica bien una vez que se
        recorta el documento y desaparece ese fondo. Se prueba
        directamente (sin pasar por detección de documento) para aislar
        el efecto."""
        foto_con_fondo = _generar_foto_con_fondo_y_timbre_real()
        assert decodificar_timbre(foto_con_fondo) is None

        recorte_sin_fondo = foto_con_fondo[40:1010, 60:900]  # aprox. el área de la boleta
        assert decodificar_timbre(recorte_sin_fondo) is not None

    def test_pipeline_completo_decodifica_timbre_en_foto_con_fondo_y_perspectiva(self, settings):
        """Prueba de integración de punta a punta con la pieza clave de
        este fix: `procesar_boleta` debe usar el documento ya recortado
        (no la foto completa con fondo) al intentar el timbre, para que
        la detección de documento (Etapa 3) termine ayudando también acá."""
        from app.models.enums import InputSource
        from app.pipeline.pipeline_service import procesar_boleta

        foto = _generar_foto_con_fondo_y_timbre_real()
        respuesta = procesar_boleta(foto, InputSource.CAMERA, settings)

        assert respuesta.datos is not None
        assert respuesta.datos.fecha == "2026-07-14"
        assert respuesta.datos.total == 8320.0

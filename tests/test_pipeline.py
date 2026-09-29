"""Pruebas de reintentos automáticos y máquina de estados (Etapa 7)."""
import cv2
import numpy as np

from app.image_processing.preprocessor import ResultadoPreprocesamiento
from app.models.enums import InputSource, ProcessingState
from app.models.schemas import ConfianzaPorCampo, DatosExtraidos, ProductoExtraido
from app.pipeline.pipeline_service import aplicar_valores_por_defecto, procesar_boleta
from app.pipeline.retry_orchestrator import ejecutar_pipeline
from app.pipeline.state_machine import determinar_estado
from app.pipeline.strategies import ESTRATEGIAS, obtener_imagen_para_estrategia
from app.receipt_validation.consistency_checks import ResultadoValidacionInterna
from app.receipt_validation.receipt_validator import ResultadoEvaluacion


def _evaluacion(confianza_general, advertencias=None, campos_faltantes=None, suma_ok=True):
    return ResultadoEvaluacion(
        confianza_por_campo=ConfianzaPorCampo(comercio=90, fecha=90, metodo_pago=90, productos=90, total=90),
        confianza_general=confianza_general,
        validacion=ResultadoValidacionInterna(
            advertencias=advertencias or [],
            campos_faltantes=campos_faltantes or [],
            suma_coincide_con_total=suma_ok,
        ),
    )


DATOS_COMPLETOS = DatosExtraidos(
    comercio="TIENDA X",
    fecha="2026-01-01",
    metodo_pago="Efectivo",
    total=100.0,
    productos=[ProductoExtraido(nombre="ALGO", cantidad=1, precio_unitario=100)],
)


class TestEstrategias:
    def test_las_4_estrategias_devuelven_imagen_correcta(self):
        gris = np.arange(100).reshape(10, 10).astype(np.uint8)
        prep = ResultadoPreprocesamiento(
            imagen_binaria=255 - gris,
            imagen_gris=gris,
            documento_detectado=True,
            rotacion_cardinal_aplicada=0,
            angulo_inclinacion_aplicado=0.0,
        )
        assert len(ESTRATEGIAS) == 4
        assert np.array_equal(obtener_imagen_para_estrategia(prep, ESTRATEGIAS[0]), gris)
        assert np.array_equal(
            obtener_imagen_para_estrategia(prep, ESTRATEGIAS[1]), cv2.rotate(gris, cv2.ROTATE_180)
        )
        assert np.array_equal(obtener_imagen_para_estrategia(prep, ESTRATEGIAS[2]), 255 - gris)


class TestMaquinaDeEstados:
    def test_image_unreadable_confianza_ocr_muy_baja(self, settings):
        estado, msg = determinar_estado(
            DatosExtraidos(), _evaluacion(0), mejor_confianza_ocr=10.0,
            source=InputSource.FILE, settings=settings,
        )
        assert estado == ProcessingState.IMAGE_UNREADABLE

    def test_no_receipt_detected_nada_reconocido(self, settings):
        estado, msg = determinar_estado(
            DatosExtraidos(), _evaluacion(50), mejor_confianza_ocr=85.0,
            source=InputSource.CAMERA, settings=settings,
        )
        assert estado == ProcessingState.NO_RECEIPT_DETECTED
        assert "iluminación" in msg

    def test_incomplete_receipt_falta_total(self, settings):
        datos = DatosExtraidos(comercio="X", fecha="2026-01-01",
                                productos=[ProductoExtraido(nombre="A", cantidad=1, precio_unitario=1)])
        estado, msg = determinar_estado(
            datos, _evaluacion(80), mejor_confianza_ocr=85.0,
            source=InputSource.FILE, settings=settings,
        )
        assert estado == ProcessingState.INCOMPLETE_RECEIPT
        assert "total" in msg

    def test_low_confidence(self, settings):
        estado, _ = determinar_estado(
            DATOS_COMPLETOS, _evaluacion(55), mejor_confianza_ocr=60.0,
            source=InputSource.FILE, settings=settings,
        )
        assert estado == ProcessingState.LOW_CONFIDENCE

    def test_partial_success_falta_campo_secundario(self, settings):
        datos = DatosExtraidos(comercio="X", fecha="2026-01-01", total=100.0,
                                productos=[ProductoExtraido(nombre="A", cantidad=1, precio_unitario=100)])
        estado, _ = determinar_estado(
            datos, _evaluacion(85), mejor_confianza_ocr=88.0,
            source=InputSource.FILE, settings=settings,
        )
        assert estado == ProcessingState.PARTIAL_SUCCESS

    def test_partial_success_no_cuadra_suma(self, settings):
        estado, _ = determinar_estado(
            DATOS_COMPLETOS, _evaluacion(85, suma_ok=False), mejor_confianza_ocr=88.0,
            source=InputSource.FILE, settings=settings,
        )
        assert estado == ProcessingState.PARTIAL_SUCCESS

    def test_success(self, settings):
        estado, msg = determinar_estado(
            DATOS_COMPLETOS, _evaluacion(90), mejor_confianza_ocr=92.0,
            source=InputSource.FILE, settings=settings,
        )
        assert estado == ProcessingState.SUCCESS
        assert msg == "Boleta procesada correctamente."


class TestOrquestadorDeReintentos:
    def test_caso_normal_exito_al_primer_intento(self, settings, foto_normal):
        resultado = ejecutar_pipeline(foto_normal, settings)
        assert resultado.intentos_realizados == 1
        assert resultado.mejor_intento.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"
        assert resultado.mejor_intento.evaluacion.confianza_general >= settings.MIN_CONFIDENCE_ACCEPTABLE

    def test_recupera_boleta_volteada_con_reintento(self, settings, foto_volteada):
        """La ambigüedad 0/180 documentada desde la Etapa 3: la boleta
        sale boca abajo del recorte de perspectiva, y el reintento con
        rotación 180° la recupera."""
        resultado = ejecutar_pipeline(foto_volteada, settings)
        assert resultado.intentos_realizados > 1
        assert resultado.mejor_intento.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"
        assert resultado.mejor_intento.datos.total == 8320.0

    def test_respeta_limite_de_intentos(self, foto_volteada):
        from app.config import Settings

        settings_pocos_intentos = Settings(MAX_RETRY_ATTEMPTS=1)
        resultado = ejecutar_pipeline(foto_volteada, settings_pocos_intentos)
        assert resultado.intentos_realizados == 1
        assert resultado.limite_alcanzado == "intentos"

    def test_respeta_limite_de_tiempo(self, foto_normal):
        from app.config import Settings

        settings_poco_tiempo = Settings(MAX_PROCESSING_TIME_SECONDS=0.001)
        resultado = ejecutar_pipeline(foto_normal, settings_poco_tiempo)
        assert resultado.limite_alcanzado == "tiempo"
        assert resultado.intentos_realizados < len(ESTRATEGIAS)

    def test_umbral_puntaje_ocr_bajo_corta_en_la_primera_variante(self, foto_volteada):
        """Opción 6 de VISION_Y_ROADMAP.md: con un umbral de puntaje
        crudo de OCR muy bajo, el nuevo corte dispara incluso en la
        primera variante (mal orientada, boca abajo), antes de llegar a
        la rotación correcta — demuestra que el mecanismo está conectado
        al loop de variantes. El valor por defecto (13000) es
        deliberadamente más alto para no cortar así de temprano contra
        una foto real boca abajo (ver test siguiente y
        `test_recupera_boleta_volteada_con_reintento`, que sigue pasando
        con el umbral por defecto)."""
        from app.config import Settings

        settings_umbral_bajo = Settings(UMBRAL_PUNTAJE_OCR_TEMPRANO=0.0)
        resultado = ejecutar_pipeline(foto_volteada, settings_umbral_bajo)
        assert resultado.intentos_realizados == 1

    def test_umbral_puntaje_ocr_por_defecto_no_corta_variante_mal_orientada(
        self, settings, foto_volteada
    ):
        """Complemento del test anterior: con el umbral por defecto, la
        primera variante (mal orientada) NO alcanza el puntaje crudo
        necesario, así que el loop sigue probando hasta encontrar la
        rotación correcta — mismo resultado final que antes de la
        Opción 6, solo que ahora también puede cortar antes de llegar a
        `_es_resultado_suficiente` cuando el puntaje crudo ya es alto."""
        resultado = ejecutar_pipeline(foto_volteada, settings)
        assert resultado.intentos_realizados > 1
        assert resultado.mejor_intento.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"

    def test_imagen_en_blanco_se_detiene_por_baja_calidad(self, settings):
        """Antes de la detección de borrosidad, esta imagen (sin ningún
        contenido) llegaba a intentar OCR igual. Ahora se corta antes,
        con nitidez=0.0 — ahorra tiempo en una imagen condenada a
        fallar (ver Settings.UMBRAL_NITIDEZ_MINIMA)."""
        blanca = np.full((300, 300, 3), 255, dtype=np.uint8)
        resultado = ejecutar_pipeline(blanca, settings)
        assert resultado.mejor_intento is None
        assert resultado.limite_alcanzado == "calidad"
        assert resultado.intentos_realizados == 0
        assert resultado.nitidez == 0.0


class TestCorreccionDeEjeViaOSD:
    """Respaldo de la Etapa 7 para cuando la Etapa 3 (heurística visual,
    barata) elige mal el eje de lectura — ver README, "Quinta ronda: una
    foto real expone un bug en la corrección de rotación"."""

    def test_recupera_eje_incorrecto_cuando_se_dispara_el_chequeo(self, foto_normal):
        """`foto_normal` girada 90° es un caso real donde la Etapa 3 NO
        detecta que haga falta corregir nada (confirmado aparte,
        instrumentando `preprocesar_imagen` directo sobre esta misma
        imagen: devuelve rotacion_cardinal_aplicada=0). Forzando el
        chequeo de la Etapa 7 con un umbral inalcanzable — para no
        depender de que el puntaje del primer intento sea bajo, que en
        este caso puntual no lo es — el respaldo de OSD sí lo corrige."""
        from app.config import Settings

        rotada_90 = cv2.rotate(foto_normal, cv2.ROTATE_90_CLOCKWISE)
        settings_chequeo_forzado = Settings(UMBRAL_PUNTAJE_OCR_TEMPRANO=10**9)

        resultado = ejecutar_pipeline(rotada_90, settings_chequeo_forzado)

        assert resultado.mejor_intento is not None
        assert resultado.mejor_intento.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"
        assert resultado.mejor_intento.datos.total == 8320.0

    def test_no_rota_de_mas_cuando_el_eje_ya_esta_bien(self, foto_normal, settings):
        """Con el chequeo forzado mismo sobre una foto ya bien orientada,
        el OSD debe confirmar que no hace falta rotar nada — el
        resultado tiene que ser idéntico a correr sin el chequeo."""
        from app.config import Settings

        settings_chequeo_forzado = Settings(UMBRAL_PUNTAJE_OCR_TEMPRANO=10**9)
        normal = ejecutar_pipeline(foto_normal, settings)
        forzado = ejecutar_pipeline(foto_normal, settings_chequeo_forzado)

        assert forzado.mejor_intento.datos.comercio == normal.mejor_intento.datos.comercio
        assert forzado.mejor_intento.datos.total == normal.mejor_intento.datos.total


class TestProcesarBoletaEndToEnd:
    def test_flujo_completo_success(self, settings, foto_normal):
        respuesta = procesar_boleta(foto_normal, InputSource.FILE, settings)
        assert respuesta.estado == ProcessingState.SUCCESS
        assert respuesta.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"
        assert respuesta.datos.total == 8320.0

    def test_flujo_completo_recupera_boleta_volteada(self, settings, foto_volteada):
        respuesta = procesar_boleta(foto_volteada, InputSource.CAMERA, settings)
        assert respuesta.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"
        assert respuesta.datos.total == 8320.0
        assert respuesta.estado in (ProcessingState.SUCCESS, ProcessingState.PARTIAL_SUCCESS)

    def test_flujo_completo_no_pisa_comercio_real(self, settings, foto_normal):
        """`aplicar_valores_por_defecto` no debe tocar un comercio que sí
        se identificó — solo entra a tallar cuando el dato es None."""
        respuesta = procesar_boleta(foto_normal, InputSource.FILE, settings)
        assert respuesta.datos.comercio == "SUPERMERCADO EJEMPLO LTDA"
        assert respuesta.datos.comercio != "Sin título"


class TestValoresPorDefecto:
    """A pedido explícito de Camilo: mostrar un rótulo genérico en vez
    de un campo vacío cuando comercio o método de pago no se
    identificaron. Se aplica DESPUÉS de calcular confianza/advertencias
    /estado (que ya usaron el dato real, None) — nunca debería cambiar
    el puntaje, ver `TestNoAfectaPuntaje` más abajo."""

    def test_comercio_ausente_recibe_rotulo_generico(self):
        datos = DatosExtraidos(comercio=None, total=1000.0)
        resultado = aplicar_valores_por_defecto(datos)
        assert resultado.comercio == "Sin título"

    def test_metodo_pago_ausente_recibe_rotulo_generico(self):
        datos = DatosExtraidos(metodo_pago=None, total=1000.0)
        resultado = aplicar_valores_por_defecto(datos)
        assert resultado.metodo_pago == "Sin especificar"

    def test_no_pisa_comercio_ni_metodo_pago_ya_identificados(self):
        datos = DatosExtraidos(comercio="TIENDA REAL", metodo_pago="Débito", total=1000.0)
        resultado = aplicar_valores_por_defecto(datos)
        assert resultado.comercio == "TIENDA REAL"
        assert resultado.metodo_pago == "Débito"

    def test_fecha_y_total_ausentes_quedan_intactos(self):
        """Fecha y total quedan afuera a propósito — ver docstring de
        `aplicar_valores_por_defecto`: ahí un rótulo de relleno sí
        podría confundirse con un dato real."""
        datos = DatosExtraidos(comercio=None, fecha=None, total=None)
        resultado = aplicar_valores_por_defecto(datos)
        assert resultado.fecha is None
        assert resultado.total is None

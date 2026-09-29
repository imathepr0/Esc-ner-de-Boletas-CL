"""Pruebas de la capa de validación de entrada (Etapa 2)."""
from app.config import Settings
from app.validation.file_validation import RazonInvalidez, validar_archivo
from tests.conftest import bytes_jpg


def test_archivo_vacio_es_invalido(settings):
    resultado = validar_archivo(b"", "boleta.jpg", "image/jpeg", settings)
    assert not resultado.es_valido
    assert resultado.razon == RazonInvalidez.ARCHIVO_VACIO


def test_extension_no_soportada(settings):
    resultado = validar_archivo(b"contenido", "boleta.txt", "text/plain", settings)
    assert not resultado.es_valido
    assert resultado.razon == RazonInvalidez.FORMATO_NO_SOPORTADO


def test_content_type_no_soportado(settings, boleta_limpia):
    resultado = validar_archivo(bytes_jpg(boleta_limpia), "boleta.jpg", "application/pdf", settings)
    assert not resultado.es_valido
    assert resultado.razon == RazonInvalidez.TIPO_CONTENIDO_NO_SOPORTADO


def test_imagen_corrupta(settings):
    resultado = validar_archivo(b"esto no es una imagen real", "boleta.jpg", "image/jpeg", settings)
    assert not resultado.es_valido
    assert resultado.razon == RazonInvalidez.IMAGEN_CORRUPTA


def test_archivo_demasiado_grande(boleta_limpia):
    settings_pequeno = Settings(MAX_FILE_SIZE_MB=0.00001)
    resultado = validar_archivo(
        bytes_jpg(boleta_limpia), "boleta.jpg", "image/jpeg", settings_pequeno
    )
    assert not resultado.es_valido
    assert resultado.razon == RazonInvalidez.ARCHIVO_DEMASIADO_GRANDE


def test_imagen_valida_se_decodifica(settings, boleta_limpia):
    resultado = validar_archivo(bytes_jpg(boleta_limpia), "boleta.jpg", "image/jpeg", settings)
    assert resultado.es_valido
    assert resultado.imagen is not None
    assert resultado.imagen.shape[0] > 0 and resultado.imagen.shape[1] > 0

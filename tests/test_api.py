"""
Pruebas de integración HTTP de extremo a extremo (Etapa 8), usando
TestClient de FastAPI — no necesita un servidor real corriendo ni hacer
llamadas de red de verdad.
"""
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import bytes_jpg

client = TestClient(app)


def test_health_check():
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"


def test_boleta_normal_devuelve_success(foto_normal):
    respuesta = client.post(
        "/api/v1/boletas/scan",
        files={"file": ("boleta.jpg", bytes_jpg(foto_normal), "image/jpeg")},
        data={"source": "file"},
    )
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "SUCCESS"
    assert cuerpo["datos"]["comercio"] == "SUPERMERCADO EJEMPLO LTDA"
    assert cuerpo["datos"]["total"] == 8320.0
    assert len(cuerpo["datos"]["productos"]) == 4
    assert cuerpo["confianza_general"] > 70


def test_boleta_volteada_recupera_datos_via_reintento(foto_volteada):
    """Prueba HTTP real de la ambigüedad 0/180: la imagen sube
    comprimida como JPEG (con la pérdida de calidad real que eso
    implica), tal como la subiría el frontend."""
    respuesta = client.post(
        "/api/v1/boletas/scan",
        files={"file": ("boleta.jpg", bytes_jpg(foto_volteada), "image/jpeg")},
        data={"source": "camera"},
    )
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["datos"]["comercio"] == "SUPERMERCADO EJEMPLO LTDA"
    assert cuerpo["estado"] in ("SUCCESS", "PARTIAL_SUCCESS")


def test_extension_no_soportada():
    respuesta = client.post(
        "/api/v1/boletas/scan",
        files={"file": ("boleta.txt", b"no es una imagen", "text/plain")},
        data={"source": "file"},
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "UNSUPPORTED_FILE"


def test_archivo_vacio():
    respuesta = client.post(
        "/api/v1/boletas/scan",
        files={"file": ("boleta.jpg", b"", "image/jpeg")},
        data={"source": "camera"},
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "IMAGE_UNREADABLE"


def test_imagen_corrupta():
    respuesta = client.post(
        "/api/v1/boletas/scan",
        files={"file": ("boleta.jpg", b"esto no es una imagen real de verdad", "image/jpeg")},
        data={"source": "file"},
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "IMAGE_UNREADABLE"


def test_cors_headers_presentes():
    respuesta = client.options(
        "/api/v1/boletas/scan",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert respuesta.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_imagenes_limite_no_generan_error_no_controlado():
    """Imágenes inusuales pero válidas (grises, RGBA, 1x1, panorámica)
    nunca deben producir un error 500 sin controlar."""
    import cv2
    import numpy as np

    casos = {
        "gris.jpg": np.random.randint(0, 255, (200, 200), dtype=np.uint8),
        "rgba.png": np.random.randint(0, 255, (200, 200, 4), dtype=np.uint8),
        "1x1.png": np.full((1, 1, 3), 128, dtype=np.uint8),
        "panoramica.jpg": np.full((50, 5000, 3), 255, dtype=np.uint8),
    }
    for nombre, imagen in casos.items():
        ok, buffer = cv2.imencode(f".{nombre.split('.')[-1]}", imagen)
        assert ok
        respuesta = client.post(
            "/api/v1/boletas/scan",
            files={"file": (nombre, buffer.tobytes(), "image/jpeg")},
            data={"source": "file"},
        )
        assert respuesta.status_code == 200, f"{nombre} produjo status {respuesta.status_code}"
        assert respuesta.json()["estado"] in (
            "IMAGE_UNREADABLE",
            "NO_RECEIPT_DETECTED",
            "UNSUPPORTED_FILE",
        )

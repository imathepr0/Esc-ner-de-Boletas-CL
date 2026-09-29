"""
Punto de entrada de la aplicación FastAPI.

Ensambla la configuración, el middleware y los routers de la API.
La lógica de negocio (OCR, preprocesamiento, parsing, validaciones)
vive en sus propios módulos y nunca se implementa directamente aquí.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as boletas_router
from app.config import get_settings
from app.ocr.tesseract_config import configurar_tesseract
from app.utils.logging_config import configure_logging

configure_logging()
settings = get_settings()
configurar_tesseract(settings)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend de escaneo y lectura de boletas chilenas. "
        "Procesamiento 100% local con OpenCV y Tesseract OCR, "
        "sin IA ni servicios externos."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(boletas_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def manejador_errores_inesperados(request: Request, exc: Exception) -> JSONResponse:
    """Red de seguridad final (Etapa 8): cualquier excepción no
    capturada explícitamente en otro lugar se convierte en una
    respuesta controlada en vez de un error 500 sin manejar. El
    'HTTPException' de FastAPI sigue manejándose por su cuenta (es más
    específico que 'Exception'), este manejador solo atrapa lo
    verdaderamente inesperado.
    """
    logger.exception("Error no controlado procesando %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Ocurrió un error inesperado. Por favor intenta nuevamente."},
    )


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Verifica que el servicio esté activo."""
    return {"status": "ok", "service": settings.APP_NAME}

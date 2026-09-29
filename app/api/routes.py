"""
Definición de endpoints de la API (capa API).

Los endpoints solo orquestan llamadas a los módulos de validación,
procesamiento, OCR, parsing, pipeline y respuestas; nunca contienen
lógica de negocio directamente.

Etapa 8: conecta el pipeline completo (Etapa 7) al endpoint. Cubre los
casos de la sección "Manejo de errores" que dependen del límite de
tiempo o de fallas del motor OCR; los demás (archivo corrupto, formato
no soportado, imagen vacía, OCR sin resultados, documento parcial) ya
quedan resueltos por la validación de entrada (Etapa 2) y la máquina de
estados (Etapa 7). Cualquier excepción no anticipada la captura el
manejador global definido en app/main.py, para que el backend nunca
finalice abruptamente.
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.models.enums import InputSource
from app.models.schemas import ScanResponse
from app.ocr.ocr_engine import MotorOCRError
from app.pipeline.pipeline_service import procesar_boleta
from app.responses.error_responses import respuesta_desde_validacion, respuesta_timeout
from app.validation.file_validation import validar_archivo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/boletas", tags=["Boletas"])


@router.post("/scan", response_model=ScanResponse)
async def escanear_boleta(
    file: UploadFile = File(..., description="Imagen de la boleta (jpg, jpeg, png o webp)."),
    source: InputSource = Form(
        default=InputSource.FILE,
        description="Origen de la imagen: 'camera' o 'file'.",
    ),
    settings: Settings = Depends(get_settings),
) -> ScanResponse:
    """
    Recibe una imagen de boleta, la valida y ejecuta el pipeline completo
    de reconocimiento (preprocesamiento + OCR + parsing + validaciones +
    reintentos) para devolver los datos estructurados.

    Privacidad: la imagen nunca se guarda en disco. Se lee una única vez
    a memoria, se decodifica, se procesa y los bytes originales se
    liberan explícitamente apenas se obtiene la imagen decodificada; solo
    los datos extraídos (nunca la imagen) llegan a la respuesta.
    """
    contenido = await file.read()
    await file.close()

    resultado = validar_archivo(
        contenido=contenido,
        nombre_archivo=file.filename,
        content_type=file.content_type,
        settings=settings,
    )
    del contenido  # los bytes crudos ya no se necesitan una vez decodificada la imagen

    if not resultado.es_valido:
        return respuesta_desde_validacion(resultado, source)

    tiempo_maximo = settings.MAX_PROCESSING_TIME_SECONDS + settings.TIMEOUT_MARGEN_SEGUNDOS

    try:
        # El pipeline es síncrono y usa CPU (OpenCV/Tesseract): se ejecuta
        # en un thread pool para no bloquear el event loop, y con un
        # timeout duro como red de seguridad final sobre el límite
        # "suave" que ya aplica la Etapa 7 entre intentos.
        return await asyncio.wait_for(
            run_in_threadpool(procesar_boleta, resultado.imagen, source, settings),
            timeout=tiempo_maximo,
        )
    except asyncio.TimeoutError:
        logger.warning("Tiempo de procesamiento excedido (origen=%s)", source.value)
        return respuesta_timeout(source)
    except MotorOCRError as exc:
        # Falla de infraestructura (Tesseract no disponible), no de
        # calidad de la boleta: se señaliza como error de servicio, no
        # como un estado más del documento.
        logger.error("Motor OCR no disponible: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "El motor de reconocimiento (Tesseract) no está disponible "
                "en este momento. Intenta nuevamente más tarde."
            ),
        ) from exc

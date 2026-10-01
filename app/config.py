"""
Configuración central del backend.

Todos los parámetros ajustables del sistema (formatos aceptados, límites
de reintentos, umbrales de confianza, configuración de Tesseract, etc.)
se definen en un único lugar para que el resto de los módulos no dependan
de valores dispersos por el código.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de configuración del backend de escaneo de boletas."""

    # --- Información del servicio ---
    APP_NAME: str = "Boletas CL - Backend de Escaneo"
    APP_VERSION: str = "0.1.0"

    # --- Entrada del sistema: formatos de archivo aceptados ---
    ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_MIME_TYPES: set[str] = {"image/jpeg", "image/png", "image/webp"}
    MAX_FILE_SIZE_MB: float = 10.0

    # --- Preprocesamiento de imagen (se usa desde la Etapa 3) ---
    MAX_IMAGE_DIMENSION: int = 2000

    # --- OCR: configuración de Tesseract (se usa desde la Etapa 4) ---
    TESSERACT_CMD: str | None = None  # Ruta al binario; None = buscar en PATH
    TESSERACT_LANG: str = "spa"
    TESSERACT_OEM: int = 3  # 3 = motor LSTM (por defecto, el más preciso)

    # --- Reintentos automáticos (se usa desde la Etapa 7) ---
    MAX_RETRY_ATTEMPTS: int = 3
    # El OCR puede tardar bastante más en instancias pequeñas (por ejemplo,
    # Render Free con 0.1 CPU). Se permite hasta 120s de trabajo total para
    # que los reintentos terminen antes de devolver un resultado incompleto.
    MAX_PROCESSING_TIME_SECONDS: float = 120.0

    # --- Timeout duro a nivel de endpoint (se usa desde la Etapa 8) ---
    # Margen para que el pipeline salga limpiamente después de alcanzar
    # su límite "suave" entre intentos; luego actúa este timeout duro.
    TIMEOUT_MARGEN_SEGUNDOS: float = 15.0

    # --- Salida anticipada del loop de variantes de imagen (Etapa 7,
    # Opción 6 de VISION_Y_ROADMAP.md — implementada en sesión de
    # rendimiento de sept. 2026) ---
    # Puntaje crudo de OCR (ResultadoOCR.puntaje, Etapa 4 — escala de
    # miles, no confundir con confianza_promedio 0-100 ni con la
    # confianza general de campos de la Etapa 6) a partir del cual la
    # variante de imagen actual (gris/binaria, normal/rotada 180°) ya
    # se considera lo bastante buena como para no seguir probando el
    # resto. Antes de este corte, el único early-exit existente exigía
    # confianza general de campos >= MIN_CONFIDENCE_ACCEPTABLE (70), una
    # condición ligada al éxito completo del parsing downstream que
    # ninguna de las fotos reales medidas alcanzó nunca — en la práctica
    # siempre se gastaban las 3 variantes x 4 PSM = 12 llamadas a
    # Tesseract por foto.
    #
    # Calibrado instrumentando `puntaje` para las 4 estrategias x 2
    # fixtures sintéticas de este proyecto: la orientación correcta
    # separa muy claramente de la rotada 180° (~19.900-20.150 vs.
    # ~4.900-6.400, sin solape), en línea con lo medido contra fotos
    # reales en la sesión anterior (~16.000-26.000 vs. ~3.500-7.500). El
    # valor por defecto queda en el medio de esa brecha. Actúa solo
    # entre variantes de imagen, nunca dentro del loop de PSM: ahí los 4
    # modos sí pueden diferir bastante entre sí (caso Transbank de la
    # sesión anterior, PSM 6 a ~20% del mejor), así que siempre se
    # prueban los 4 antes de decidir la mejor lectura de una variante.
    # Sigue siendo un valor provisional calibrado con fotos sintéticas y
    # una muestra chica de fotos reales — no reemplaza confirmar con más
    # fotos reales, incluyendo alguna genuinamente al revés.
    UMBRAL_PUNTAJE_OCR_TEMPRANO: float = 13000.0

    # --- Puntaje de confianza (se usa desde la Etapa 6) ---
    MIN_CONFIDENCE_LOW: float = 40.0  # bajo esto => IMAGE_UNREADABLE / NO_RECEIPT_DETECTED
    MIN_CONFIDENCE_ACCEPTABLE: float = 70.0  # entre LOW y ACCEPTABLE => LOW_CONFIDENCE

    # --- Detección de borrosidad y reescalado (mejora post-validación
    # con boletas reales, sin IA: heurísticas clásicas de OpenCV) ---
    # Lado menor objetivo (px) al que se reescala el documento ya
    # recortado y enderezado, si viene más chico (interpolación Lanczos,
    # tope de 3x). Ataca directamente "texto con pocos píxeles de alto
    # por carácter", identificado como causa raíz de números mal leídos
    # en fotos reales (ver README, "Detección de borrosidad y reescalado").
    ANCHO_OBJETIVO_ESCALADO: int = 1200
    # Umbral mínimo de nitidez (varianza del Laplaciano, ver
    # app/image_processing/quality.py) bajo el cual se considera que la
    # imagen está demasiado borrosa/sin contenido como para siquiera
    # intentar el OCR — se corta antes, ahorrando el tiempo que se
    # habría gastado en una imagen condenada a fallar. Se mide ANTES de
    # reescalar, sobre el documento ya recortado.
    #
    # Calibrado en esta sesión simulando desenfoque de cámara creciente
    # (Gaussian blur) sobre la fixture `foto_normal` y comparando contra
    # el resultado real del OCR (no solo el número de nitidez): con
    # sigma=2 (nitidez≈24) el comercio se sigue reconociendo bien
    # (confianza OCR ≈90%); con sigma=3 (nitidez≈9) ya falla. 15.0 queda
    # deliberadamente en medio de ese rango, más cerca del extremo
    # permisivo: entre bloquear por error una foto aprovechable y gastar
    # unos segundos de más en una que no sirve, este proyecto prefiere
    # lo segundo (además, un resultado deficiente igual queda cubierto
    # por LOW_CONFIDENCE/PARTIAL_SUCCESS más adelante en el pipeline).
    # Sigue siendo un valor provisional: no reemplaza calibrar con fotos
    # reales (ver README, sección "Detección de borrosidad y reescalado").
    UMBRAL_NITIDEZ_MINIMA: float = 15.0

    # --- CORS: orígenes permitidos para el consumo desde el frontend ---
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de Settings (patrón singleton simple)."""
    return Settings()

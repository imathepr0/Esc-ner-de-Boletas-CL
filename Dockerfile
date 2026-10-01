# Imagen de producción para Render (y ejecución local con Docker).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSERACT_CMD=/usr/bin/tesseract

# Tesseract aporta OCR en español y libzbar permite decodificar el PDF417
# del timbre electrónico. Las bibliotecas de runtime son necesarias para
# OpenCV en la imagen slim de Debian.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libzbar0 \
        tesseract-ocr \
        tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app

# Render no requiere root para este servicio y el pipeline solo trabaja
# con imágenes en memoria, por lo que no necesita un directorio escribible.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render establece PORT en runtime (por defecto, 10000). El valor alternativo
# permite ejecutar la imagen localmente con `docker run -p 10000:10000 ...`.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

# Escáner de Boletas CL

Backend para escanear y extraer información de boletas chilenas a partir de imágenes.
Expone una API HTTP construida con FastAPI y realiza el procesamiento localmente con
OpenCV y Tesseract OCR.

El sistema puede identificar, cuando la calidad y el formato de la boleta lo permiten:

- Comercio
- Fecha
- Método de pago
- Productos, cantidades y precios unitarios
- Total
- Confianza de la lectura y advertencias de validación

No utiliza modelos de lenguaje ni servicios externos de OCR. Las imágenes se procesan
en memoria y el backend no las almacena en disco.

> Este repositorio contiene el backend. La interfaz de usuario final puede ser un
> frontend separado que consuma la API.

## Estado

El flujo principal está implementado e incluye:

- Validación de archivos de entrada
- Detección y corrección del documento dentro de la imagen
- Corrección de orientación e inclinación
- Mejora de imagen con OpenCV
- OCR con varios modos de segmentación de Tesseract
- Parsing de boletas chilenas
- Validaciones internas y confianza por campo
- Reintentos con variantes de la imagen
- Detección y enriquecimiento opcional mediante timbre PDF417
- Suite de pruebas automatizadas

La lectura de fotografías reales depende de la iluminación, resolución, enfoque,
perspectiva y formato de cada boleta. Antes de usarlo en producción se recomienda
validarlo con un conjunto representativo de documentos reales.

## Requisitos

- Python 3.11 o superior
- Tesseract OCR instalado en el sistema
- Modelo de idioma español de Tesseract (spa)
- Git, si se desea clonar el repositorio

El soporte PDF417 utiliza pdf417decoder y pyzbar. En Windows, la versión de pyzbar
incluida en las dependencias normalmente incorpora las bibliotecas necesarias. En
Linux puede ser necesario instalar también ZBar desde el sistema operativo.

## Instalación

### Windows

Instala Python desde [python.org](https://www.python.org/downloads/) y Tesseract OCR
desde una distribución compatible para Windows. Con winget, una opción es:

~~~powershell
winget install --id tesseract-ocr.tesseract --exact
~~~

Comprueba que Tesseract está disponible:

~~~powershell
tesseract --version
tesseract --list-langs
~~~

La salida de idiomas debe incluir spa. Si Tesseract no está en el PATH, configura su
ubicación mediante variables de entorno:

~~~powershell
$env:TESSERACT_CMD = "C:\Ruta\a\tesseract.exe"
$env:TESSDATA_PREFIX = "C:\Ruta\a\tessdata"
~~~

Crea el entorno virtual e instala las dependencias:

~~~powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Si PowerShell bloquea la activación del entorno, ejecuta el backend sin activarlo:

~~~powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
~~~

### Linux

En Ubuntu o Debian:

~~~bash
sudo apt update
sudo apt install python3 python3-venv tesseract-ocr tesseract-ocr-spa libzbar0
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### macOS

Con Homebrew:

~~~bash
brew install python tesseract tesseract-lang zbar
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### Render

El repositorio incluye un `Dockerfile` que instala Python 3.11, Tesseract OCR con
el idioma español (`spa`) y ZBar para leer el PDF417. Se usa Docker porque el
servicio necesita paquetes del sistema que no se instalan con `pip`. La imagen
escucha en el puerto definido por Render mediante `PORT` y procesa las imágenes
solo en memoria.

En Render crea o configura un **Web Service** conectado a este repositorio:

- **Language / Runtime:** Docker
- **Branch:** `main`
- **Root Directory:** vacío, porque `Dockerfile`, `requirements.txt` y `app/`
  están en la raíz del repositorio
- **Dockerfile Path:** `./Dockerfile` (o el valor predeterminado)
- **Docker Command:** vacío, para usar el `CMD` del Dockerfile
- **Health Check Path:** `/health`
- **Environment Variables:** no se necesitan para que Tesseract funcione; la
  imagen ya instala Tesseract y fija `TESSERACT_CMD=/usr/bin/tesseract`

No uses el runtime Python nativo, porque no incluye Tesseract ni `spa`. Tampoco
hay que ingresar `PORT`: Render lo asigna automáticamente. Para permitir llamadas
desde un frontend desplegado, agrega `CORS_ORIGINS` con el origen del frontend en
formato JSON, por ejemplo `["https://tu-frontend.example"]`.

Después del despliegue, comprueba `https://TU-SERVICIO.onrender.com/health` y
prueba `POST /api/v1/boletas/scan` con una imagen real desde `/docs`. Un `200` en
`/health` solo confirma que la API arrancó; el escaneo real permite comprobar
Tesseract. El plan gratuito dispone de recursos limitados y puede no tener RAM
suficiente para todas las imágenes; ante cierres por memoria, reduce el tamaño
de imagen de prueba o considera un plan con más memoria.

## Configuración

La configuración principal está en app/config.py y puede sobrescribirse mediante
variables de entorno o un archivo .env local. El archivo .env está excluido por
.gitignore y no debe subirse al repositorio.

| Variable | Valor predeterminado | Descripción |
|---|---:|---|
| TESSERACT_CMD | vacío | Ruta al ejecutable de Tesseract cuando no está en el PATH |
| TESSERACT_LANG | spa | Idioma usado por el OCR |
| MAX_FILE_SIZE_MB | 10 | Tamaño máximo del archivo recibido |
| MAX_PROCESSING_TIME_SECONDS | 300 | Límite de procesamiento del pipeline |
| TIMEOUT_MARGEN_SEGUNDOS | 30 | Margen del timeout HTTP |
| CORS_ORIGINS | localhost:5173 | Orígenes permitidos para el frontend |

Ejemplo de .env local en Windows:

~~~env
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
TESSDATA_PREFIX=C:\\Program Files\\Tesseract-OCR\\tessdata
~~~

Adapta las rutas al equipo donde se ejecute el backend.

## Ejecutar el servidor

Con el entorno virtual activado:

~~~bash
uvicorn app.main:app --reload --port 8000
~~~

Direcciones disponibles:

- API: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs
- Especificación OpenAPI: http://localhost:8000/openapi.json
- Health check: http://localhost:8000/health

La página /docs es una herramienta de documentación y prueba de la API; no es una
interfaz de usuario final.

## API

### GET /health

Comprueba que el proceso está activo.

~~~json
{
  "status": "ok",
  "service": "Boletas CL - Backend de Escaneo"
}
~~~

### POST /api/v1/boletas/scan

Recibe una imagen mediante multipart/form-data.

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---:|---|
| file | archivo | Sí | Imagen JPG, JPEG, PNG o WEBP |
| source | texto | No | file o camera; por defecto file |

Ejemplo con PowerShell o Windows:

~~~powershell
curl.exe -X POST "http://localhost:8000/api/v1/boletas/scan" -F "file=@boleta.jpg;type=image/jpeg" -F "source=file"
~~~

Ejemplo con Linux o macOS:

~~~bash
curl -X POST "http://localhost:8000/api/v1/boletas/scan" \
  -F "file=@boleta.jpg;type=image/jpeg" \
  -F "source=file"
~~~

Respuesta exitosa de ejemplo:

~~~json
{
  "estado": "SUCCESS",
  "mensaje": "Boleta procesada correctamente.",
  "confianza_general": 92.5,
  "confianza_por_campo": {
    "comercio": 95.0,
    "fecha": 90.0,
    "metodo_pago": 88.0,
    "productos": 91.0,
    "total": 97.0
  },
  "advertencias": [],
  "datos": {
    "comercio": "Jumbo",
    "fecha": "2026-07-14",
    "metodo_pago": "Débito",
    "productos": [
      {
        "nombre": "Pan Hallulla",
        "cantidad": 1,
        "precio_unitario": 1200,
        "confianza": 93.0
      }
    ],
    "total": 1200
  }
}
~~~

Estados posibles:

- SUCCESS: lectura completa y suficientemente confiable.
- PARTIAL_SUCCESS: se identificaron algunos datos, pero faltan otros.
- LOW_CONFIDENCE: hay datos, pero la confianza es baja.
- IMAGE_UNREADABLE: la imagen no permite una lectura útil.
- INCOMPLETE_RECEIPT: la boleta parece válida, pero faltan campos relevantes.
- UNSUPPORTED_FILE: formato, tamaño o contenido no soportado.
- NO_RECEIPT_DETECTED: no se detectó una boleta en la imagen.

Errores de infraestructura:

- 503: Tesseract no está instalado, no está configurado o no puede ejecutarse.
- 500: error inesperado no controlado.

## Pruebas

Instala las dependencias de desarrollo:

~~~bash
python -m pip install -r requirements-dev.txt
~~~

Ejecuta toda la suite:

~~~bash
pytest
pytest -v
~~~

Ejecuta un módulo específico:

~~~bash
pytest tests/test_parsing.py
~~~

Las pruebas utilizan principalmente boletas sintéticas para validar cada etapa. Esto
permite detectar regresiones de código, pero no reemplaza las pruebas con fotografías
reales en distintas condiciones de luz, enfoque y perspectiva.

## Estructura del proyecto

~~~text
boletas-backend/
├── app/
│   ├── api/                  # Endpoints FastAPI
│   ├── image_processing/     # Detección y mejora de imágenes
│   ├── models/               # Modelos Pydantic y enumeraciones
│   ├── ocr/                  # Configuración y ejecución de Tesseract
│   ├── parsing/              # Extracción de campos y productos
│   ├── pipeline/             # Flujo, reintentos y estados
│   ├── receipt_validation/   # Validaciones y confianza
│   ├── responses/            # Respuestas públicas de la API
│   ├── timbre/               # Lectura y parsing de PDF417/TED
│   ├── validation/           # Validación de archivos de entrada
│   ├── config.py             # Configuración central
│   └── main.py               # Aplicación FastAPI
├── tests/                    # Suite automatizada
├── requirements.txt          # Dependencias de ejecución
├── requirements-dev.txt      # Dependencias de pruebas
├── pytest.ini                # Configuración de pytest
├── .gitignore
└── README.md
~~~

## Privacidad y seguridad

- Las imágenes se procesan en memoria y no se guardan permanentemente.
- El backend no envía imágenes a servicios externos de OCR.
- El endpoint no incluye autenticación; debe protegerse antes de exponerlo públicamente.
- En producción se recomienda usar HTTPS, restringir CORS al dominio real del frontend
  y añadir límites de tasa.
- No subir archivos .env, credenciales ni venv/.

## Dependencias de terceros

El proyecto utiliza software y paquetes de terceros, entre ellos Python, FastAPI,
OpenCV, Tesseract OCR, Pillow y las librerías de lectura PDF417. Estas dependencias
no se convierten en código original del proyecto por el hecho de ser instaladas o
utilizadas; cada una conserva sus propios avisos y condiciones de licencia.

La licencia MIT de este repositorio se aplica al código original de Escáner de
Boletas CL. Las licencias de las dependencias deben consultarse en sus respectivos
repositorios y paquetes de distribución.

## Uso y responsabilidad

El OCR y el procesamiento de imágenes pueden producir errores de lectura debido a
la calidad de la fotografía, iluminación, enfoque, perspectiva, formato de la boleta
u otras condiciones del documento. El proyecto no garantiza exactitud total para
todas las boletas ni para todos los campos extraídos.

Los resultados son una ayuda automatizada y no reemplazan la revisión humana,
contable o tributaria, el usuario debe verificar los datos
obtenidos contra la boleta original.

El software se entrega tal cual, sin garantía de disponibilidad, exactitud,
idoneidad para un propósito particular o ausencia de errores. La licencia MIT que
acompañará al proyecto incluirá las limitaciones de responsabilidad correspondientes.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para
conocer sus condiciones completas.

## Limitaciones conocidas

- El OCR puede confundir caracteres en fotografías borrosas, inclinadas o con poca luz.
- Las boletas tienen layouts muy variados y algunas requieren reglas específicas.
- El timbre PDF417 no siempre puede decodificarse, aunque el resto sea legible.
- El procesamiento puede tardar varios segundos porque utiliza varias estrategias locales.
- Los resultados deben revisarse antes de usarse en procesos contables o fiscales.

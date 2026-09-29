"""
Modelos Pydantic que definen el contrato de datos entre el backend y el
frontend. Sus campos deben mantenerse alineados con la sección
"Respuesta JSON" de las especificaciones del proyecto.
"""
from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ProcessingState


class ProductoExtraido(BaseModel):
    """Un producto individual identificado dentro de la boleta."""

    nombre: str | None = None
    cantidad: float | None = None
    precio_unitario: float | None = None
    confianza: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Confianza (0-100) de esta línea de producto.",
    )


class DatosExtraidos(BaseModel):
    """Datos estructurados extraídos de la boleta, listos para el frontend.

    Todos los campos son opcionales: en una lectura parcial solo se llenan
    los datos efectivamente identificados. Nunca se inventa información
    faltante (ver sección "Respuesta JSON" del spec).
    """

    comercio: str | None = None
    fecha: str | None = None  # Normalizada a ISO (YYYY-MM-DD) cuando es posible
    metodo_pago: str | None = None
    productos: list[ProductoExtraido] = Field(default_factory=list)
    total: float | None = None


class ConfianzaPorCampo(BaseModel):
    """Confianza individual (0-100) de cada dato relevante del documento."""

    comercio: float | None = None
    fecha: float | None = None
    metodo_pago: float | None = None
    productos: float | None = None
    total: float | None = None


class ScanResponse(BaseModel):
    """Respuesta pública devuelta por el endpoint de escaneo de boletas."""

    estado: ProcessingState
    mensaje: str
    confianza_general: float | None = Field(default=None, ge=0, le=100)
    confianza_por_campo: ConfianzaPorCampo | None = None
    advertencias: list[str] = Field(default_factory=list)
    datos: DatosExtraidos | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "estado": "SUCCESS",
                "mensaje": "Boleta procesada correctamente.",
                "confianza_general": 92.5,
                "confianza_por_campo": {
                    "comercio": 95.0,
                    "fecha": 90.0,
                    "metodo_pago": 88.0,
                    "productos": 91.0,
                    "total": 97.0,
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
                            "confianza": 93.0,
                        }
                    ],
                    "total": 1200,
                },
            }
        }
    )

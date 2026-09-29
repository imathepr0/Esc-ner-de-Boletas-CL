"""
Orquestación del flujo completo de escaneo: reintentos automáticos con
distintas estrategias de preprocesamiento, límites de intentos y tiempo,
y determinación del estado final (máquina de estados).

Implementado en la Etapa 7 (ver strategies.py, retry_orchestrator.py,
state_machine.py y pipeline_service.py).
"""
from app.pipeline.pipeline_service import procesar_boleta
from app.pipeline.retry_orchestrator import IntentoLectura, ResultadoPipeline, ejecutar_pipeline
from app.pipeline.state_machine import determinar_estado

__all__ = [
    "procesar_boleta",
    "ejecutar_pipeline",
    "determinar_estado",
    "IntentoLectura",
    "ResultadoPipeline",
]

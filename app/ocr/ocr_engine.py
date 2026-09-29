"""
Motor de OCR: envuelve pytesseract para extraer texto y confianza por
palabra desde una imagen ya preprocesada, probando distintas estrategias
de segmentación de página (PSM) y quedándose con la de mayor confianza.
"""
import statistics
from dataclasses import dataclass, field

import numpy as np
import pytesseract
from pytesseract import Output

from app.config import Settings
from app.ocr.tesseract_config import PSM_ESTRATEGIAS, construir_config


class MotorOCRError(RuntimeError):
    """Se lanza cuando Tesseract no está disponible o todas las
    estrategias probadas fallaron.

    No se lanza cuando el OCR simplemente no reconoce texto: eso es un
    resultado válido con confianza baja (0 palabras), no un error.
    """


@dataclass
class PalabraOCR:
    """Una palabra individual reconocida, con su posición y confianza."""

    texto: str
    confianza: float
    x: int
    y: int
    ancho: int
    alto: int
    linea: int
    bloque: int


@dataclass
class ResultadoOCR:
    """Resultado estructurado de una pasada de OCR sobre una imagen."""

    texto_completo: str
    palabras: list[PalabraOCR] = field(default_factory=list)
    confianza_promedio: float = 0.0
    psm_utilizado: int = 6
    # Puntaje crudo de calidad de OCR (ver `_puntaje_palabras`), en una
    # escala de miles — no confundir con `confianza_promedio` (0-100) ni
    # con la confianza general de campos de la Etapa 6, que depende del
    # parsing downstream. Se usa desde la Etapa 7 para decidir si vale
    # la pena seguir probando otras variantes de imagen (ver
    # `retry_orchestrator._puntaje_ocr_suficiente`).
    puntaje: float = 0.0

    @property
    def cantidad_palabras(self) -> int:
        """Cantidad de palabras reconocidas con confianza válida."""
        return len(self.palabras)


def _reconstruir_linea(palabras_linea: list["PalabraOCR"]) -> str:
    """Reconstruye el texto de una línea preservando el espaciado real
    entre palabras, según su posición en píxeles.

    Es importante no unir todo con un solo espacio: el parser de la
    Etapa 5 distingue las columnas de cantidad/precio (alineadas a la
    derecha, con espacio amplio) de números que son parte del nombre
    del producto usando justamente el ancho del espacio entre palabras.

    Asume que `palabras_linea` ya viene en orden real de lectura
    izquierda-a-derecha — lo garantiza `_agrupar_en_filas`, que ordena
    por posición X real antes de llamar a esta función. Esta función
    solo calcula el espaciado entre palabras ya ordenadas, no decide el
    orden (antes sí se apoyaba en el orden que entregaba Tesseract; ver
    `_agrupar_en_filas` para por qué eso dejó de ser seguro).
    """
    if not palabras_linea:
        return ""

    anchos_por_caracter = [p.ancho / len(p.texto) for p in palabras_linea if p.texto]
    ancho_caracter = (sum(anchos_por_caracter) / len(anchos_por_caracter)) if anchos_por_caracter else 10.0
    ancho_caracter = max(ancho_caracter, 1.0)

    partes = [palabras_linea[0].texto]
    for anterior, actual in zip(palabras_linea, palabras_linea[1:]):
        espacio_px = actual.x - (anterior.x + anterior.ancho)
        num_espacios = max(1, round(espacio_px / ancho_caracter))
        partes.append(" " * min(num_espacios, 12))
        partes.append(actual.texto)

    return "".join(partes)


# Fracción de la altura mediana de palabra (en toda la imagen) que se
# tolera como diferencia vertical entre dos palabras para considerarlas
# parte de la misma fila visual. Cubre el desalineamiento normal entre
# columnas (nombre y precio rara vez quedan pixel-perfect a la misma
# altura, por diferencias de línea base de fuente) sin fusionar filas
# realmente distintas, que en una boleta impresa quedan separadas por
# bastante más que eso. _TOLERANCIA_FILA_MINIMA_PX evita un umbral
# demasiado chico en imágenes con texto muy pequeño, donde la altura
# mediana puede ser de solo unos pocos píxeles.
_TOLERANCIA_FILA_FRACCION_ALTO = 0.6
_TOLERANCIA_FILA_MINIMA_PX = 5.0


def _agrupar_en_filas(palabras: list["PalabraOCR"]) -> list[list["PalabraOCR"]]:
    """Agrupa palabras en filas visuales según su posición Y real en la
    imagen — no según el block_num/par_num/line_num que asigna
    Tesseract internamente a partir de su propio análisis de layout.

    Encontrado corriendo una boleta real (Líder, 13 productos, columna
    vacía y ancha entre el nombre del producto y el precio — ver
    README, "Cuarta ronda de boletas reales"): con ese layout, la
    estrategia de PSM que terminó ganando por puntaje detectó nombre y
    precio como bloques de texto SEPARADOS (el mismo tipo de análisis
    pensado para columnas de diario) y los devolvió en su propio orden
    interno — primero los 13 nombres, recién al final los 13 precios —
    no en el orden real de lectura de la boleta. Agrupar por line_num
    (como se hacía antes) hereda ese error porque confía en la misma
    segmentación de Tesseract que falló ahí.

    Agrupar por posición Y real es inmune a ese problema puntual: dos
    palabras de la misma fila visual comparten una posición vertical
    similar sin importar en qué "bloque" las haya clasificado
    Tesseract. No soluciona errores de lectura de caracteres (una "S"
    leída como "5" sigue siendo un error de lectura, esté donde esté
    posicionada) — solo el orden en el que se ensambla el texto ya
    reconocido, que es exactamente lo que colapsó en Líder.

    El agrupamiento compara cada palabra contra el promedio (no el
    rango) de la fila que va formando, para no ir "arrastrando" el
    límite de la fila palabra a palabra hacia filas vecinas.
    """
    if not palabras:
        return []

    alturas = [p.alto for p in palabras if p.alto > 0]
    alto_mediano = statistics.median(alturas) if alturas else 20.0
    umbral_y = max(alto_mediano * _TOLERANCIA_FILA_FRACCION_ALTO, _TOLERANCIA_FILA_MINIMA_PX)

    ordenadas = sorted(palabras, key=lambda p: (p.y + p.alto / 2, p.x))

    filas: list[list[PalabraOCR]] = [[ordenadas[0]]]
    centro_fila = ordenadas[0].y + ordenadas[0].alto / 2

    for palabra in ordenadas[1:]:
        centro_palabra = palabra.y + palabra.alto / 2
        if abs(centro_palabra - centro_fila) <= umbral_y:
            filas[-1].append(palabra)
            centro_fila = sum(p.y + p.alto / 2 for p in filas[-1]) / len(filas[-1])
        else:
            filas.append([palabra])
            centro_fila = centro_palabra

    for fila in filas:
        fila.sort(key=lambda p: p.x)

    return filas


def _ejecutar_tesseract(imagen: np.ndarray, psm: int, settings: Settings) -> ResultadoOCR:
    """Ejecuta pytesseract con un PSM específico y devuelve el resultado
    estructurado. Puede lanzar excepciones de pytesseract; el llamador
    decide cómo manejarlas."""
    config = construir_config(psm, settings)

    datos = pytesseract.image_to_data(
        imagen,
        lang=settings.TESSERACT_LANG,
        config=config,
        output_type=Output.DICT,
    )

    palabras: list[PalabraOCR] = []

    total_elementos = len(datos.get("text", []))
    for i in range(total_elementos):
        texto = datos["text"][i].strip()
        confianza = float(datos["conf"][i])

        # conf == -1 identifica elementos contenedores (bloque/línea), no
        # palabras reales reconocidas; se descartan junto con texto vacío.
        if not texto or confianza < 0:
            continue

        palabras.append(
            PalabraOCR(
                texto=texto,
                confianza=confianza,
                x=datos["left"][i],
                y=datos["top"][i],
                ancho=datos["width"][i],
                alto=datos["height"][i],
                # linea/bloque quedan como metadata informativa de
                # Tesseract; el orden real del texto ya NO se arma a
                # partir de ellos (ver _agrupar_en_filas).
                linea=datos["line_num"][i],
                bloque=datos["block_num"][i],
            )
        )

    texto_completo = "\n".join(_reconstruir_linea(fila) for fila in _agrupar_en_filas(palabras))
    confianza_promedio = (
        sum(p.confianza for p in palabras) / len(palabras) if palabras else 0.0
    )

    return ResultadoOCR(
        texto_completo=texto_completo,
        palabras=palabras,
        confianza_promedio=confianza_promedio,
        psm_utilizado=psm,
        puntaje=_puntaje_palabras(palabras),
    )


# Confianza mínima para que una palabra "cuente" en el puntaje de una
# estrategia. Sin este filtro, una lectura con muchas palabras basura de
# confianza muy baja (ruido mal interpretado como texto) podía superar
# en puntaje a una lectura limpia con menos palabras pero correctas.
_UMBRAL_PALABRA_CONFIABLE = 40.0


def _puntaje_palabras(palabras: list["PalabraOCR"]) -> float:
    """Suma la confianza de las palabras razonablemente confiables,
    ponderada por su longitud.

    Se pondera por longitud porque fragmentos de 1-2 caracteres suelen
    ser ruido mal interpretado como texto, incluso cuando Tesseract les
    asigna una confianza individual alta (esto ocurre sobre todo en
    imágenes binarizadas con artefactos). Esto es un filtro adicional a
    la confianza reportada por Tesseract, no un reemplazo: detectar con
    certeza una lectura completamente sin sentido (que no produce ningún
    campo reconocible) es responsabilidad de las validaciones semánticas
    de la Etapa 6 y de los reintentos de la Etapa 7, no de este módulo.

    Se calcula una sola vez por resultado (en `_ejecutar_tesseract`, ver
    `ResultadoOCR.puntaje`) y se reusa tanto para elegir el mejor PSM
    dentro de una variante (`_es_mejor`) como para decidir, desde la
    Etapa 7, si vale la pena seguir probando otras variantes de imagen.
    """
    return sum(
        p.confianza * len(p.texto)
        for p in palabras
        if p.confianza >= _UMBRAL_PALABRA_CONFIABLE and len(p.texto) >= 2
    )


def _es_mejor(candidato: ResultadoOCR, actual: ResultadoOCR) -> bool:
    """Compara dos resultados de OCR para decidir cuál conservar."""
    return candidato.puntaje > actual.puntaje


def reconocer_texto(
    imagen: np.ndarray,
    settings: Settings,
    estrategias_psm: list[int] | None = None,
) -> ResultadoOCR:
    """Ejecuta el OCR probando distintos modos PSM y se queda con el de
    mayor puntaje, tal como pide la sección "OCR" del spec ("debe
    intentar maximizar la precisión antes de aceptar una lectura").

    Lanza MotorOCRError si Tesseract no está instalado o si todas las
    estrategias probadas fallan con una excepción.
    """
    psms_a_probar = estrategias_psm or PSM_ESTRATEGIAS

    mejor_resultado: ResultadoOCR | None = None
    for psm in psms_a_probar:
        try:
            resultado = _ejecutar_tesseract(imagen, psm, settings)
        except pytesseract.TesseractNotFoundError as exc:
            raise MotorOCRError(
                "Tesseract OCR no está instalado o no se encontró el binario. "
                "Revisa TESSERACT_CMD o instala tesseract-ocr y tesseract-ocr-spa."
            ) from exc
        except Exception:
            # Esta estrategia en particular falló; se continúa probando
            # las siguientes en vez de abortar todo el reconocimiento.
            continue

        if mejor_resultado is None or _es_mejor(resultado, mejor_resultado):
            mejor_resultado = resultado

    if mejor_resultado is None:
        raise MotorOCRError(
            f"Las {len(psms_a_probar)} estrategias de OCR probadas fallaron."
        )

    return mejor_resultado

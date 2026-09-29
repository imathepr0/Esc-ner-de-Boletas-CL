"""
Orquestación del pipeline completo de preprocesamiento de imagen.

Combina, en un orden pensado para fotos de boletas chilenas tomadas con
teléfono, todos los pasos de la sección "Preprocesamiento de imágenes"
de las especificaciones: detección y recorte del documento, corrección
de rotación e inclinación, medición de nitidez y reescalado, escala de
grises, reducción de sombras, mejora de contraste, eliminación de ruido,
nitidez (realce) y binarización.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.image_processing.document_detection import detectar_y_recortar_documento
from app.image_processing.enhancement import (
    aumentar_nitidez,
    binarizar,
    convertir_a_grises,
    eliminar_ruido,
    mejorar_contraste,
    reducir_sombras,
)
from app.image_processing.orientation import corregir_inclinacion, corregir_rotacion_cardinal
from app.image_processing.quality import calcular_nitidez

# Tope al factor de ampliación que aplica `_escalar_si_es_necesario`: por
# más pequeño que venga el documento recortado, agrandarlo más de esto
# ya no reconstruye información real, solo agranda el ruido/artefactos.
_FACTOR_ESCALA_MAXIMO = 3.0


@dataclass
class ResultadoPreprocesamiento:
    """Salida del pipeline de preprocesamiento.

    `imagen_binaria` es el resultado final listo para OCR. `imagen_gris`
    se conserva (mejorada pero sin binarizar) como alternativa: el motor
    de OCR (Etapa 4) puede preferir una u otra según el caso.

    `nitidez` y `factor_escala_aplicado` tienen valor por defecto a
    propósito: cualquier código que construya este dataclass directamente
    (p. ej. pruebas que arman un resultado a mano) sigue funcionando sin
    tener que conocer estos 2 campos agregados después.
    """

    imagen_binaria: np.ndarray
    imagen_gris: np.ndarray
    documento_detectado: bool
    rotacion_cardinal_aplicada: int
    angulo_inclinacion_aplicado: float
    nitidez: float = 0.0
    factor_escala_aplicado: float = 1.0
    # Documento recortado/enderezado/reescalado a color, SIN las mejoras
    # de contraste/ruido/nitidez pensadas para texto (ver preprocesar_imagen).
    # Puede ser None si se construye este dataclass directamente sin
    # pasar por preprocesar_imagen (ej. algunas pruebas).
    imagen_recortada: np.ndarray | None = None


def _redimensionar_si_necesario(imagen: np.ndarray, dimension_maxima: int) -> np.ndarray:
    """Reduce el tamaño de imágenes muy grandes para acelerar el resto
    del pipeline, sin afectar la legibilidad del texto (rendimiento)."""
    alto, ancho = imagen.shape[:2]
    lado_mayor = max(alto, ancho)
    if lado_mayor <= dimension_maxima:
        return imagen

    factor = dimension_maxima / lado_mayor
    nuevo_tamano = (int(ancho * factor), int(alto * factor))
    return cv2.resize(imagen, nuevo_tamano, interpolation=cv2.INTER_AREA)


def _escalar_si_es_necesario(
    imagen: np.ndarray, ancho_objetivo: int
) -> tuple[np.ndarray, float]:
    """Reescala HACIA ARRIBA el documento ya recortado si su lado MENOR
    es más chico que `ancho_objetivo`, con interpolación Lanczos (la de
    mejor calidad de OpenCV para agrandar imágenes: preserva mejor los
    bordes finos que bicúbica o lineal).

    Pensado para boletas fotografiadas a poca resolución o desde lejos,
    donde el texto queda con pocos píxeles de alto por carácter — causa
    raíz identificada de números mal leídos por el OCR (ver README,
    sección "Detección de borrosidad y reescalado"). El factor de escala
    tiene un tope (`_FACTOR_ESCALA_MAXIMO`) para no amplificar en exceso
    imágenes extremadamente pequeñas.

    Devuelve (imagen_resultante, factor_aplicado); el factor es 1.0 si
    no hizo falta escalar.
    """
    alto, ancho = imagen.shape[:2]
    lado_menor = min(alto, ancho)
    if lado_menor <= 0 or lado_menor >= ancho_objetivo:
        return imagen, 1.0

    factor = min(ancho_objetivo / lado_menor, _FACTOR_ESCALA_MAXIMO)
    nuevo_tamano = (max(int(ancho * factor), 1), max(int(alto * factor), 1))
    escalada = cv2.resize(imagen, nuevo_tamano, interpolation=cv2.INTER_LANCZOS4)
    return escalada, factor


def preprocesar_imagen(
    imagen_bgr: np.ndarray,
    dimension_maxima: int = 2000,
    ancho_objetivo_escalado: int = 1200,
) -> ResultadoPreprocesamiento:
    """Ejecuta el pipeline completo de preprocesamiento sobre una imagen
    BGR (formato OpenCV) ya validada y decodificada.

    Cada paso está diseñado para degradarse con gracia: si un paso no
    logra mejorar la imagen (por ejemplo, no se detecta un documento
    claro), se continúa con la mejor versión disponible en vez de fallar.
    """
    imagen = _redimensionar_si_necesario(imagen_bgr, dimension_maxima)

    imagen, documento_detectado = detectar_y_recortar_documento(imagen)

    imagen, rotacion_aplicada = corregir_rotacion_cardinal(imagen)

    imagen, angulo_inclinacion = corregir_inclinacion(imagen)

    # La nitidez se mide ACÁ: sobre el documento ya recortado y
    # enderezado, pero todavía SIN reescalar. Medirla después de
    # reescalar mezclaría la nitidez real de la foto con un artefacto de
    # la interpolación usada para agrandarla (una imagen chica pero
    # nítida, agrandada con Lanczos, puede medir "menos nítida" que la
    # misma imagen sin agrandar — no porque la foto sea peor, sino por
    # cómo la interpolación reparte los bordes entre más píxeles). Medir
    # antes de escalar evita ese sesgo y refleja la calidad real de la
    # foto capturada.
    nitidez = calcular_nitidez(imagen)

    imagen, factor_escala = _escalar_si_es_necesario(imagen, ancho_objetivo_escalado)

    # Se conserva ACÁ (recortado, enderezado, reescalado, pero todavía a
    # color y SIN las mejoras pensadas para texto) para reusar en el
    # timbre electrónico: CLAHE/filtro bilateral/enfoque ayudan a leer
    # letras pero degradan el patrón de un código de barras lo
    # suficiente como para que deje de decodificar (confirmado en esta
    # sesión: el mismo recorte decodificaba antes de esas mejoras y
    # dejaba de hacerlo después) — ver pipeline_service.py.
    imagen_recortada = imagen

    gris = convertir_a_grises(imagen)
    gris = reducir_sombras(gris)
    gris = mejorar_contraste(gris)
    gris = eliminar_ruido(gris)
    gris = aumentar_nitidez(gris)

    binaria = binarizar(gris)

    return ResultadoPreprocesamiento(
        imagen_binaria=binaria,
        imagen_gris=gris,
        documento_detectado=documento_detectado,
        rotacion_cardinal_aplicada=rotacion_aplicada,
        angulo_inclinacion_aplicado=angulo_inclinacion,
        nitidez=nitidez,
        factor_escala_aplicado=factor_escala,
        imagen_recortada=imagen_recortada,
    )


def rotar_resultado_preprocesamiento(
    preprocesado: ResultadoPreprocesamiento, grados: int
) -> ResultadoPreprocesamiento:
    """Aplica una rotación cardinal adicional (90 o 270) a un resultado
    de preprocesamiento ya calculado.

    Usada por la Etapa 7 cuando `estimar_rotacion_via_osd` contradice la
    corrección de eje que hizo esta etapa con la heurística rápida (ver
    README, "Quinta ronda: una foto real expone un bug en la corrección
    de rotación") — evita rehacer detección de documento, inclinación,
    nitidez y escalado desde cero: solo rota lo que ya depende de la
    orientación (`imagen_gris`, `imagen_binaria`, `imagen_recortada`) y
    deja el resto del resultado igual.
    """
    if grados not in (90, 270):
        raise ValueError(f"rotar_resultado_preprocesamiento solo acepta 90 o 270, recibió {grados}")

    rotacion_cv2 = cv2.ROTATE_90_CLOCKWISE if grados == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE

    return replace(
        preprocesado,
        imagen_gris=cv2.rotate(preprocesado.imagen_gris, rotacion_cv2),
        imagen_binaria=cv2.rotate(preprocesado.imagen_binaria, rotacion_cv2),
        imagen_recortada=(
            cv2.rotate(preprocesado.imagen_recortada, rotacion_cv2)
            if preprocesado.imagen_recortada is not None
            else None
        ),
        rotacion_cardinal_aplicada=(preprocesado.rotacion_cardinal_aplicada + grados) % 360,
    )

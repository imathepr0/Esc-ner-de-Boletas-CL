"""
Decodificación del Timbre Electrónico (TED) que traen las boletas
electrónicas chilenas: un código de barras PDF417 con datos
estructurados (RUT emisor, tipo de documento, folio, fecha de emisión,
monto total) firmados por el SII.

Esta es una mejora de "mejor esfuerzo" sobre el flujo principal basado
en OCR (Etapas 4-7): cuando el timbre decodifica, sus datos son más
precisos que el texto reconocido por OCR (vienen de un código de
barras, no de reconocer letras en una foto) y se usan para reemplazar
fecha/total con confianza 100. Cuando no decodifica —el caso más común
en fotos de la boleta completa, donde el código queda muy chico— el
resto del pipeline funciona exactamente igual que si este módulo no
existiera. No es IA: es lectura de código de barras + parseo de texto
con expresiones regulares.

Ver `barcode_decoder.py` (localizar y decodificar el PDF417) y
`ted_parser.py` (interpretar el texto XML-like ya decodificado).
"""

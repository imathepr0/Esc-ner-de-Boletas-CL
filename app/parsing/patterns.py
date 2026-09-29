"""
Patrones y palabras clave usados por el parser para reconocer campos en
el texto reconocido por el OCR. Reglas generales, no ligadas a un
formato de boleta específico (ver "Parser" en las especificaciones):
no asumen posiciones fijas ni un único tipo de documento.
"""
import re

# --- Fecha: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YY, etc. ---
PATRON_FECHA = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")

# --- Núcleo de un monto en formato chileno: punto como separador de
# miles (el estándar), coma también aceptada como separador de miles
# alternativo — el OCR confunde frecuentemente "." con "," en fotos
# reales (encontrado corriendo fotos reales, ver README: "TOTAL 4,250"
# se interpretaba como 4,25 en vez de 4.250, perdiendo el último
# dígito). El lookahead negativo al final de la parte decimal evita el
# caso contrario: un monto de 2 decimales genuino (ej. "4,25") no debe
# confundirse con el inicio de un grupo de miles si viene un tercer
# dígito pegado justo después (esto es lo que causaba el bug).
_NUCLEO_MONTO = r"\d{1,3}(?:[.,]\d{3})+|\d+"
PATRON_MONTO = re.compile(rf"\$?\s?({_NUCLEO_MONTO})(?:[.,](\d{{2}})(?!\d))?")
PATRON_NUMERO_EN_LINEA = re.compile(rf"{_NUCLEO_MONTO}(?:[.,]\d{{2}}(?!\d))?")

# --- Cantidad al inicio de una línea de producto (ej. "2 Cafe...", "3x Empanada") ---
PATRON_CANTIDAD_INICIAL = re.compile(
    r"^\s*(\d{1,3})\s*[xX]?\s+(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])"
)

# --- Palabras clave de encabezado/pie que NO son líneas de producto ---
PALABRAS_CLAVE_NO_PRODUCTO = (
    "TOTAL", "SUBTOTAL", "NETO", "IVA", "VUELTO", "CAMBIO", "PROPINA",
    "FECHA", "HORA", "RUT", "BOLETA", "FOLIO", "FACTURA",
    "MEDIO DE PAGO", "FORMA DE PAGO", "PAGO",
    "GRACIAS", "CAJERO", "VENDEDOR", "SUCURSAL", "DIRECCION",
    "TELEFONO", "CLIENTE", "ATENDIDO", "MESA", "GARZON",
    "CLP", "PESOS", "PESO",
    # Líneas de "detalle de pago" (ej. "TARJETA DEBITO $5.970",
    # "EFECTIVO $742") se confundían con líneas de producto reales:
    # encontrado corriendo fotos reales (ver README). No se agregó
    # "TARJETA" sola porque algunos comercios sí venden tarjetas
    # (regalo, SIM) como producto.
    "DEBITO", "DÉBITO", "CREDITO", "CRÉDITO", "EFECTIVO",
    # Mismo problema que arriba pero en inglés: algunos POS imprimen el
    # medio de pago así (ej. "Debit $22.990" en una boleta de tienda de
    # ropa, encontrada corriendo fotos reales) — sin esto, la línea se
    # leía como un producto con el total completo como precio.
    "DEBIT", "CREDIT",
    # Texto de programas de puntos/fidelización (ej. "PODRIAS HABER
    # ACUMULADO : 14" de Puntos Cencosud) también se confundía con
    # producto — encontrado corriendo fotos reales. Cencosud opera
    # muchas cadenas chilenas (Jumbo, Santa Isabel, Easy, Paris, entre
    # otras), así que este texto de pie de boleta es razonablemente
    # frecuente.
    "PUNTOS", "ACUMULADO",
)

# Etiquetas alternativas a "TOTAL" que algunos sistemas de punto de
# venta usan para el mismo concepto (el monto final de la boleta) — ej.
# "MONTO BOLETA ELECTRONICA $6.500" en vez de "TOTAL $6.500". Encontrado
# corriendo fotos reales: sin esto, extraer_total ni siquiera miraba esa
# línea porque no contenía la palabra "TOTAL" (ver README).
ETIQUETAS_TOTAL_ALTERNATIVAS = ("MONTO BOLETA",)

# --- Etiqueta "RUT"/"R.U.T." tolerante a puntuación entre letras: el OCR
# de boletas reales a veces lee "R.U.T." con comas en vez de puntos, o
# con/sin espacios ("R.U,T.", "R U T"), lo que rompe el chequeo de
# substring simple de "RUT" de PALABRAS_CLAVE_NO_PRODUCTO. Se usa junto
# con esa lista (no en reemplazo) en _es_encabezado_o_pie.
PATRON_ETIQUETA_RUT = re.compile(r"R[.,]?\s?U[.,]?\s?T[.,]?\b")

# --- Métodos de pago reconocidos (clave = como aparece en la boleta) ---
# Nota: "CREDITO"/"DEBITO" ya cubren por substring frases más largas como
# "TARJETA DE CREDITO" (no hace falta una clave aparte para esas). Las
# marcas de tarjeta (VISA, MASTERCARD) sí necesitan su propia clave: los
# comprobantes Transbank suelen imprimir solo la marca, sin la palabra
# "crédito"/"débito" cerca, y la marca por sí sola no permite saber cuál
# de las dos es — se usa "Tarjeta" como categoría genérica en ese caso.
METODOS_PAGO = {
    "EFECTIVO": "Efectivo",
    "DEBITO": "Débito",
    "DÉBITO": "Débito",
    "CREDITO": "Crédito",
    "CRÉDITO": "Crédito",
    "REDCOMPRA": "Débito",
    "TRANSFERENCIA": "Transferencia",
    "TRANSBANK": "Débito/Crédito",
    "CHEQUE": "Cheque",
    "VISA": "Tarjeta",
    "MASTERCARD": "Tarjeta",
    # Mismo medio de pago, etiqueta en inglés — encontrado corriendo
    # fotos reales (ver PALABRAS_CLAVE_NO_PRODUCTO más arriba).
    "DEBIT": "Débito",
    "CREDIT": "Crédito",
}

# --- Sufijos de razón social chilenos (sin puntos: se comparan por token) ---
SUFIJOS_RAZON_SOCIAL = {"SA", "LTDA", "LIMITADA", "SPA", "EIRL"}

PATRON_SOLO_SEPARADORES = re.compile(r"^[\s\-=_*.]+$")
PATRON_SOLO_NUMEROS = re.compile(r"^[\d\s.,\-/:]+$")

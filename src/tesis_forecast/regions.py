"""
Lista unica de regiones usada por todos los pipelines de modelado.

Extraido de la celda 49 del notebook legacy (diccionario ARCHIVOS_REGIONES),
que ya incluia las 8 regiones (BCA incluida). Se centraliza aqui para que
cualquier modelo que se extraiga despues use la misma lista, en vez de que
cada pipeline mantenga su propia copia (eso fue lo que causo que algunos
pipelines del notebook legacy excluyeran BCA por error de copiado).
"""

REGIONS_ALL = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]

"""
Carga y recorte inicial de IGAE y Temperaturas.

Extraido verbatim de las celdas 1, 3-18 y 23-25 del notebook legacy. Los
indices posicionales (iloc[6], iloc[7], iloc[10], iloc[15], iloc[118:],
iloc[:-10]) son exactamente los mismos que en el notebook original: son
fragiles frente a cambios en el formato de IGAE_2.xlsx, pero no se tocan
aqui porque el encargo de esta migracion es preservar el comportamiento,
no corregirlo.

El orden de recorte (primero cabeza y luego cola, o al reves segun la
serie) tambien se preserva tal cual estaba en el notebook original, aunque
el resultado final es equivalente en ambos ordenes.
"""

import pandas as pd


def cargar_igae(ruta: str = "IGAE_2.xlsx") -> pd.DataFrame:
    """Celda 4."""
    return pd.read_excel(ruta, sheet_name=0)


def cargar_temperaturas(ruta: str = "Temperaturas promedio.csv") -> pd.Series:
    """Celda 3."""
    return pd.read_csv(ruta, skiprows=2)


def extraer_igae_2002(igae_df: pd.DataFrame) -> pd.Series:
    """Celdas 5, 7, 8."""
    igae_serie = igae_df.iloc[6]
    igae_2002 = igae_serie.iloc[118:]
    igae_2002 = igae_2002.iloc[:-10]
    return igae_2002


def extraer_primarias(igae_df: pd.DataFrame) -> pd.Series:
    """Celdas 10, 11, 12."""
    primarias = igae_df.iloc[7]
    primarias = primarias[:-10]
    primarias = primarias[118:]
    return primarias


def extraer_secundarias(igae_df: pd.DataFrame) -> pd.Series:
    """Celdas 13, 14, 15."""
    secundarias = igae_df.iloc[10]
    secundarias = secundarias[118:]
    secundarias = secundarias[:-10]
    return secundarias


def extraer_terciarias(igae_df: pd.DataFrame) -> pd.Series:
    """Celdas 16, 17, 18."""
    terciarias = igae_df.iloc[15]
    terciarias = terciarias[118:]
    terciarias = terciarias[:-10]
    return terciarias


def extraer_temperaturas_serie(temperaturas_df: pd.DataFrame) -> pd.Series:
    """Celdas 23, 24, 25."""
    temperaturas_serie = temperaturas_df.iloc[0]
    temperaturas_serie = temperaturas_serie[1:]
    temperaturas_serie = temperaturas_serie.dropna()
    return temperaturas_serie

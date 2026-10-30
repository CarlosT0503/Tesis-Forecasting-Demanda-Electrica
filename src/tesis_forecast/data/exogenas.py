"""
Limpieza y expansion horaria de las exogenas globales (Temperatura, IGAE,
Primarias, Secundarias, Terciarias).

drop_every_13th: extraido verbatim de la celda 20.
expandir_exogena: extraido verbatim de la celda 29 (mismas constantes
FECHA_INICIO, FECHA_CORTE_INICIO, FECHA_CORTE_FIN, mismo algoritmo de
expansion mensual -> horaria).

cargar_exogenas_horarias(): equivale a encadenar las celdas 20, 21, 29 y 30,
pero devolviendo un diccionario en vez de dejar las 5 series como variables
sueltas en el namespace global del notebook. Ese diccionario es lo que el
pipeline de XGBoost recibe como parametro explicito (en vez de leerlo de
`globals()`, que es lo que hacia la celda 49 original).
"""

import pandas as pd

from .loaders import (
    cargar_igae,
    cargar_temperaturas,
    extraer_igae_2002,
    extraer_primarias,
    extraer_secundarias,
    extraer_terciarias,
    extraer_temperaturas_serie,
)

FECHA_INICIO = "2002-01-01"
FECHA_CORTE_INICIO = "2019-01-01"
FECHA_CORTE_FIN = "2026-03-01"


def drop_every_13th(series: pd.Series) -> pd.Series:
    """
    Elimina cada 13o valor (13, 26, 39, ...) de una Series,
    devolviendo la Series corrida sin esos elementos.
    """
    mask = [(i + 1) % 13 != 0 for i in range(len(series))]
    return series.iloc[mask].reset_index(drop=True)


def expandir_exogena(serie_original, nombre: str = "serie") -> pd.DataFrame:
    serie_original = pd.to_numeric(pd.Series(serie_original), errors="coerce")

    fechas = pd.date_range(start=FECHA_INICIO, periods=len(serie_original), freq="MS")

    df = pd.DataFrame({"fecha_mensual": fechas, "valor": serie_original.values})

    df = df[
        (df["fecha_mensual"] >= FECHA_CORTE_INICIO)
        & (df["fecha_mensual"] <= FECHA_CORTE_FIN)
    ].copy()

    df = df.dropna(subset=["valor"])

    filas = []
    for _, row in df.iterrows():
        fecha_mes = row["fecha_mensual"]
        valor = row["valor"]

        for dia in range(fecha_mes.days_in_month):
            fecha_actual = fecha_mes + pd.Timedelta(days=dia)

            for hora in range(1, 25):
                filas.append({"fecha": fecha_actual.normalize(), "hora": hora, "valor": valor})

    df_horario = pd.DataFrame(filas)

    print(
        f"{nombre:15s} | "
        f"{df_horario['fecha'].min().date()} -> {df_horario['fecha'].max().date()} | "
        f"{len(df_horario):,} filas"
    )

    return df_horario


def cargar_exogenas_horarias(
    data_dir: str = ".",
    archivo_igae: str = "IGAE_2.xlsx",
    archivo_temperaturas: str = "Temperaturas promedio.csv",
) -> dict:
    """
    Reproduce las celdas 1-31 del notebook legacy y devuelve las 5 series
    horarias (Temperaturas_H, Primarias_H, Secundarias_H, Terciarias_H,
    IGAE_H) como un diccionario, con las mismas claves que usaba
    EXOG_SOURCE_MAP en la celda 49.
    """
    import os

    igae_df = cargar_igae(os.path.join(data_dir, archivo_igae))
    temperaturas_df = cargar_temperaturas(os.path.join(data_dir, archivo_temperaturas))

    igae_2002 = extraer_igae_2002(igae_df)
    primarias = extraer_primarias(igae_df)
    secundarias = extraer_secundarias(igae_df)
    terciarias = extraer_terciarias(igae_df)
    temperaturas_serie = extraer_temperaturas_serie(temperaturas_df)

    primarias_clean = drop_every_13th(primarias)
    secundarias_clean = drop_every_13th(secundarias)
    terciarias_clean = drop_every_13th(terciarias)
    igae_2002_clean = drop_every_13th(igae_2002)

    return {
        "Temperaturas_H": expandir_exogena(temperaturas_serie, "Temperaturas"),
        "Primarias_H": expandir_exogena(primarias_clean, "Primarias"),
        "Secundarias_H": expandir_exogena(secundarias_clean, "Secundarias"),
        "Terciarias_H": expandir_exogena(terciarias_clean, "Terciarias"),
        "IGAE_H": expandir_exogena(igae_2002_clean, "IGAE"),
    }

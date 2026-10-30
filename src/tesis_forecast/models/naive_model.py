"""
Pipeline Naive (demanda, univariado).

Extraido de la celda 45 del notebook legacy ("Modelos univariados"), bloque
NAIVE dentro de `evaluar_serie`: `forecast_naive(train, horizon)` = repetir
el ultimo valor observado del train para todo el horizonte. Sin
hiperparametros, sin Optuna, sin exogenas.

Cambios mecanicos (mismo patron que los demas modulos migrados):
  1. `ejecutar_pipeline()` -> `run(...)`.
  2. RESULTS_* (listas globales) -> `_ResultsAccumulator` local. Nota: en
     la celda 45 `RESULTS_SERIES` ya se llenaba con un append POR FILA
     (tanto para "real" como para "prediccion", nunca un bloque con arreglo
     completo) -- el mismo patron seguro que usan XGBoost/LightGBM/LSTM
     directa, sin el riesgo que tuvo FCNN.
  3. `cargar_regiones()`/`ARCHIVOS_REGIONES` (7 regiones, sin BCA) ->
     parametro `regions_all` (se usa la lista centralizada de 8 regiones,
     que incluye BCA).
  4. La celda original evaluaba DEMANDA y GENERACION para cada region. Por
     consistencia con el resto del sistema (target unico = Demanda, ya
     decidido para todos los modelos migrados), aqui solo se evalua
     Demanda. La formula de `forecast_naive` no cambia.
  5. Split train/test: la celda original usa una formula DINAMICA
     (`test_size = max(720, 10% de la serie)`, acotado a `len(serie)//3`;
     `train` = todo lo anterior al test). Esa formula dinamica es el
     comportamiento POR DEFECTO aqui (`train_hours`/`forecast_horizon` en
     None, que en el RUN_NAME se muestran como "auto"). Si se especifican
     ambos como enteros, actuan como ventana fija (override), una
     posibilidad que no existia en la celda 45 -- se agrega solo para
     uniformar la interfaz con `ExperimentConfig`/`run_matrix()`, no
     porque la celda original lo permitiera.

Sin exogenas: `EXOG_COLS_DEFAULT`/`EXOG_CATALOGO` estan vacios a proposito;
el runner rechaza cualquier exogena que se intente pasar.

Sin tuning: no hay `trials.csv` ni `config_usada.csv` (no hay ningun
hiperparametro que registrar).
"""

import os

import numpy as np
import pandas as pd

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..metrics import calcular_metricas

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

MIN_OBS = 24 * 30  # igual a la celda 45

EXOG_COLS_DEFAULT = []
EXOG_CATALOGO = []

# "auto" = split dinamico exacto de la celda 45 (ver docstring). Si se pasa
# un entero explicito en ExperimentConfig, se usa como ventana fija.
TRAIN_LAST_HOURS_DEFAULT = "auto"
FORECAST_HORIZON_DEFAULT = "auto"
N_TRIALS_OPTUNA_DEFAULT = None  # sin uso, no hay tuning


# =========================================================
# BASELINE (copia exacta de la celda 45)
# =========================================================

def forecast_naive(train, horizon):
    return np.repeat(train[-1], horizon)


# =========================================================
# CARGAR REGIONES
# =========================================================

def cargar_regiones(regions_all, data_dir):
    regiones = {}

    for region in regions_all:
        archivo = os.path.join(data_dir, f"{region}_long.csv")

        if not os.path.exists(archivo):
            print(f"No encontre {archivo}, salto {region}")
            continue

        df = pd.read_csv(archivo)
        df.columns = df.columns.astype(str).str.strip()

        regiones[region] = df
        print(f"OK {region}: {archivo} cargado con shape {df.shape}")

    return regiones


# =========================================================
# EXTRAER SERIE HORARIA
# =========================================================

def extraer_serie_horaria(df, columna, nombre_serie):
    if columna not in df.columns:
        raise ValueError(f"No existe columna {columna} en {nombre_serie}. Columnas: {list(df.columns)}")
    if COL_FECHA not in df.columns:
        raise ValueError(f"No existe columna '{COL_FECHA}' en {nombre_serie}")
    if COL_HORA not in df.columns:
        raise ValueError(f"No existe columna '{COL_HORA}' en {nombre_serie}")

    aux = df[[COL_FECHA, COL_HORA, columna]].copy()

    aux[COL_FECHA] = pd.to_datetime(aux[COL_FECHA], errors="coerce")
    aux[COL_HORA] = pd.to_numeric(aux[COL_HORA], errors="coerce")
    aux[columna] = pd.to_numeric(aux[columna], errors="coerce")

    aux = aux.dropna(subset=[COL_FECHA, COL_HORA, columna])

    aux["hora_0_23"] = aux[COL_HORA].astype(int) - 1
    aux["datetime"] = aux[COL_FECHA] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    aux = aux.sort_values("datetime")

    return aux[columna].values.astype(float), aux["datetime"].values


# =========================================================
# SPLIT TRAIN / TEST
# =========================================================

def _resolver_split(serie, forecast_horizon, train_hours):
    """
    Por defecto ("auto"/None) reproduce exactamente la formula de la celda
    45: test_size = max(720h, 10% de la serie), acotado a len(serie)//3;
    train = todo lo anterior al test. Si forecast_horizon/train_hours se
    pasan como enteros, se usan como ventana fija (capacidad nueva, no
    existia en la celda 45 -- ver docstring del modulo).
    """
    if forecast_horizon in (None, "auto"):
        test_size = max(24 * 30, int(len(serie) * 0.10))
        test_size = min(test_size, len(serie) // 3)
    else:
        test_size = min(int(forecast_horizon), len(serie) // 3)

    train = serie[:-test_size]
    test = serie[-test_size:]

    if train_hours not in (None, "auto"):
        train = train[-int(train_hours):]

    return train, test, test_size


# =========================================================
# ACUMULADOR / GUARDADO
# =========================================================

class _ResultsAccumulator:
    def __init__(self):
        self.series = []
        self.metrics = []


def guardar_predicciones(resultados, nombre_serie, fechas_test, pred, modelo):
    for j, pred_val in enumerate(pred):
        if j < len(fechas_test):
            resultados.series.append({
                "serie": nombre_serie,
                "fecha": fechas_test[j],
                "tipo": "prediccion",
                "subset": "test",
                "modelo": modelo,
                "valor": pred_val,
            })


def guardar_metricas(resultados, nombre_serie, modelo, tuneado, metricas, horizonte_usado):
    resultados.metrics.append({
        "serie": nombre_serie,
        "modelo": modelo,
        "tuneado": tuneado,
        "horizonte_usado": horizonte_usado,
        "MAPE": metricas["MAPE"],
        "sMAPE": metricas["sMAPE"],
        "MAE": metricas["MAE"],
        "RMSE": metricas["RMSE"],
    })


def _region_de_serie(nombre_serie):
    return nombre_serie.split("_")[0]


def _guardar_avance_csv(resultados, output_dir):
    if resultados.series:
        df_series = pd.DataFrame(resultados.series)
        df_series["fecha"] = pd.to_datetime(df_series["fecha"], errors="coerce")
        df_series["region"] = df_series["serie"].map(_region_de_serie)
        df_series = df_series.sort_values(["serie", "fecha", "modelo"])
        df_series.to_csv(os.path.join(output_dir, "series.csv"), index=False, encoding="utf-8-sig")
        print(f"\nOK Series guardadas (avance): {len(df_series):,} registros")

    if resultados.metrics:
        df_metrics = pd.DataFrame(resultados.metrics)
        df_metrics["region"] = df_metrics["serie"].map(_region_de_serie)
        df_metrics = df_metrics.sort_values(["serie", "MAPE"])
        df_metrics.to_csv(os.path.join(output_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Metricas guardadas (avance): {len(df_metrics):,} registros")


# =========================================================
# EVALUAR SERIE
# =========================================================

def evaluar_serie(nombre_serie, serie, fechas, forecast_horizon, train_hours, resultados):
    if len(serie) < MIN_OBS:
        print(f"   Serie insuficiente: {nombre_serie}")
        return

    for j, fecha in enumerate(fechas):
        resultados.series.append({
            "serie": nombre_serie,
            "fecha": fecha,
            "tipo": "real",
            "subset": "completo",
            "modelo": "real",
            "valor": serie[j],
        })

    train, test, test_size = _resolver_split(serie, forecast_horizon, train_hours)
    fechas_test = fechas[-test_size:]
    horizon = len(test)

    print("\n   Split general")
    print(f"      Train: {len(train)} obs")
    print(f"      Test:  {len(test)} obs")

    try:
        pred = forecast_naive(train, horizon)
        metricas = calcular_metricas(test, pred)

        if metricas:
            guardar_metricas(resultados, nombre_serie, "Naive", False, metricas, "serie_completa")
            guardar_predicciones(resultados, nombre_serie, fechas_test, pred, "Naive")

            print(f"      Naive: MAPE={metricas['MAPE']:.2f}%")

    except Exception as e:
        print(f"      Naive Error: {str(e)[:80]}")


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run(
    exogenas_globales: dict,
    regions_all: list,
    train_hours=TRAIN_LAST_HOURS_DEFAULT,
    forecast_horizon=FORECAST_HORIZON_DEFAULT,
    exog_cols: list = None,
    optuna_n_trials=N_TRIALS_OPTUNA_DEFAULT,
    data_dir: str = "/content",
    output_dir: str = ".",
):
    """
    Equivalente al bloque NAIVE de evaluar_serie() en la celda 45,
    parametrizado. Devuelve (series_df, metricas_df, trials_df,
    config_usada_df); trials_df/config_usada_df siempre vacios (no hay
    tuning ni hiperparametros en un baseline Naive).
    """
    if exog_cols:
        raise ValueError("El modelo Naive es univariado: no acepta exogenas.")

    resultados = _ResultsAccumulator()

    # Checkpoint por region: sin trials/config_usada (Naive no tunea nada),
    # forecast_horizon dinamico ("auto") salvo override explicito.
    forecast_horizon_fijo = forecast_horizon if isinstance(forecast_horizon, int) else None
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all, forecast_horizon=forecast_horizon_fijo,
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print("PIPELINE NAIVE DEMANDA")
    print("=" * 80)
    print(f"Directorio de salida: {output_dir}")
    print(f"Train: {train_hours}")
    print(f"Forecast horizon: {forecast_horizon}")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        print("\n" + "=" * 80)
        print(f"Serie: {region}_DEMANDA")
        print("=" * 80)

        nombre_serie = f"{region}_DEMANDA"
        serie, fechas = extraer_serie_horaria(df, COL_DEMANDA, nombre_serie)

        print(f"Serie completa: {len(serie):,} observaciones")
        print(f"Rango: {fechas[0]} a {fechas[-1]}")

        evaluar_serie(nombre_serie, serie, fechas, forecast_horizon, train_hours, resultados)

        print("\nGuardando avance...")
        _guardar_avance_csv(resultados, output_dir)

    series_df = pd.DataFrame(resultados.series)
    if len(series_df) > 0:
        series_df["fecha"] = pd.to_datetime(series_df["fecha"], errors="coerce")
        series_df["region"] = series_df["serie"].map(_region_de_serie)
        series_df = series_df.sort_values(["serie", "fecha", "modelo"])

    metricas_df = pd.DataFrame(resultados.metrics)
    if len(metricas_df) > 0:
        metricas_df["region"] = metricas_df["serie"].map(_region_de_serie)
        metricas_df = metricas_df.sort_values(["serie", "MAPE"])

    trials_df = pd.DataFrame()
    config_usada_df = pd.DataFrame()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO")
    print("=" * 80)

    if len(metricas_df) > 0:
        print("\nResultados:")
        print(metricas_df.sort_values("MAPE").to_string(index=False))

    return series_df, metricas_df, trials_df, config_usada_df

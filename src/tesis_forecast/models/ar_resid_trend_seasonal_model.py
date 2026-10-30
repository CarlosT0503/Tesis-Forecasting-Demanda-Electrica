"""
Pipeline AR sobre residuos + Tendencia + Estacionalidad (demanda, univariado)
-- COMBINACION NUEVA, NO extraccion de un pipeline completo de legacy.

*** No existe en el notebook legacy un pipeline que descomponga la serie via
STL y modele el residuo con AR SIN combinarlo tambien con una LSTM para la
tendencia (esa es la arquitectura completa del Ensemble, celda 60). La celda
60 usa AR sobre el residuo STL (`seleccionar_ar_por_aic`/`forecast_ar_resid`)
pero la tendencia ahi viene de una LSTM tuneada con Optuna, no de una
regresion lineal simple. ***

Este modulo toma exactamente las piezas de tendencia/estacionalidad de
`naive_trend_seasonal_model.py` (que a su vez son copia exacta de la celda
64: `descomponer_stl`, `forecast_tendencia_lineal`,
`forecast_estacionalidad_repetida`) y sustituye el residuo=0 de ese modelo
por un AR ajustado sobre el residuo STL, usando exactamente la misma logica
de `seleccionar_ar_por_aic` de la celda 60 (y de `ar_model.py`, que ya la
reutiliza para la serie cruda). Combinacion nueva construida a partir de dos
piezas legacy independientes, confirmada explicitamente por el usuario el
2026-08-09:

  "AR sobre residuos + tendencia + estacionalidad: misma tendencia y
  estacionalidad del punto 2 + AR sobre el residuo usando la logica de la
  celda 60."

pred_final = tendencia_forecast + estacionalidad_forecast + residuo_pronosticado(AR)

Decisiones tomadas para esta combinacion (autorizadas explicitamente):
  - Univariado, sin exogenas (igual que AR standalone y Naive_Trend_Seasonal).
  - Train/test: misma ventana FIJA que Naive_Trend_Seasonal (default
    3600h/168h) -- no la formula dinamica de AR standalone. Se prefirio asi
    por consistencia con el resto de la familia STL (Naive_Trend_Seasonal,
    LSTM_Resid) que comparte esta misma tendencia/estacionalidad; la STL con
    period=168 necesita esa ventana estable, igual que se justifico en
    naive_trend_seasonal_model.py.
  - Metricas: se reutiliza la variante de la celda 64 (mascara `isfinite`,
    con guarda contra entrada vacia), la misma familia usada en
    Naive_Trend_Seasonal -- por consistencia dentro de esta sub-familia de
    combinaciones nuevas basadas en STL, no la de `metrics.py` compartida
    que usa AR standalone.
  - `trials.csv` contiene el barrido de lags AR (lag, AIC, BIC) sobre el
    RESIDUO STL del set de entrenamiento -- misma estructura que
    `ar_model.py`, pero aqui el AR se ajusta sobre `resid`, no sobre la
    serie cruda.
  - `config_usada.csv` registra el lag optimo elegido por region.
  - MAX_LAG_AR=168 y `trend="c"` en AutoReg, identicos a la celda 60 y a
    `ar_model.py`.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.ar_model import AutoReg

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

TRAIN_LAST_HOURS_DEFAULT = 24 * 30 * 5   # 3600 horas, igual a Naive_Trend_Seasonal
FORECAST_HORIZON_DEFAULT = 24 * 7        # 168 horas

STL_PERIOD = 168
MAX_LAG_AR = 168  # identico a MAX_LAG_AR de la celda 60 / ar_model.py

EXOG_COLS_DEFAULT = []
EXOG_CATALOGO = []
N_TRIALS_OPTUNA_DEFAULT = None  # sin uso, no hay tuning Optuna (AR por AIC)

NOMBRE_MODELO = "AR_Resid_Trend_Seasonal"


# =========================================================
# METRICAS (copia exacta de la celda 64 / naive_trend_seasonal_model.py)
# =========================================================

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (denom != 0)
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def calcular_metricas(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan}

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


# =========================================================
# STL + TENDENCIA + ESTACIONALIDAD (copia exacta de la celda 64)
# =========================================================

def descomponer_stl(train):
    stl = STL(pd.Series(train).astype(float), period=STL_PERIOD, robust=True)
    res = stl.fit()
    return np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid), res


def forecast_tendencia_lineal(trend_train, horizon):
    trend_train = np.asarray(trend_train, dtype=float)
    x = np.arange(len(trend_train)).reshape(-1, 1)

    modelo = LinearRegression()
    modelo.fit(x, trend_train)

    x_future = np.arange(len(trend_train), len(trend_train) + horizon).reshape(-1, 1)

    return modelo.predict(x_future)


def forecast_estacionalidad_repetida(seasonal_train, horizon, period=168):
    ultimos = np.asarray(seasonal_train[-period:], dtype=float)
    reps = int(np.ceil(horizon / period))
    return np.tile(ultimos, reps)[:horizon]


# =========================================================
# AR sobre residuo (copia exacta de seleccionar_ar_por_aic, celda 60 / ar_model.py)
# =========================================================

def seleccionar_ar_por_aic(y, max_lag=168):
    y = pd.Series(y).astype(float).dropna().to_numpy()

    resultados = []
    mejor_modelo, mejor_lag, mejor_aic = None, None, np.inf

    for lag in range(1, max_lag + 1):
        try:
            model = AutoReg(y, lags=lag, trend="c", old_names=False)
            fitted = model.fit()

            resultados.append({"lag": lag, "AIC": fitted.aic, "BIC": fitted.bic})

            if np.isfinite(fitted.aic) and fitted.aic < mejor_aic:
                mejor_aic = fitted.aic
                mejor_lag = lag
                mejor_modelo = fitted

        except Exception as e:
            resultados.append({"lag": lag, "AIC": np.nan, "BIC": np.nan, "error": str(e)[:150]})

    if mejor_modelo is None:
        raise RuntimeError("No se pudo ajustar ningun AR valido sobre el residuo.")

    return mejor_modelo, mejor_lag, pd.DataFrame(resultados)


def forecast_ar_resid(resid_train, horizon):
    modelo, lag_optimo, df_lags = seleccionar_ar_por_aic(resid_train, max_lag=MAX_LAG_AR)

    pred = modelo.predict(start=len(resid_train), end=len(resid_train) + horizon - 1, dynamic=False)

    return np.asarray(pred), lag_optimo, df_lags


# =========================================================
# CARGAR REGIONES / EXTRAER SERIE (mismo patron que naive_trend_seasonal_model.py)
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
        print(f"OK {region}: {archivo} {df.shape}")

    return regiones


def convertir_hora_0_23(serie_hora):
    """
    Hora viene conceptualmente en 1..24 -> hora_0_23 = Hora - 1, SIEMPRE.

    FIX (bug de desfase +1h, confirmado por auditoria): esta funcion antes
    solo restaba 1 si TODA la columna caia en [1,24]; una sola fila fuera
    de rango dejaba la columna COMPLETA sin ajustar, produciendo un
    desfase sistematico de 1 hora en la columna 'fecha' de series.csv
    respecto al resto del proyecto. Este modelo es univariado (sin
    exogenas) y usa train/test indexados por POSICION, no por datetime, asi
    que el bug nunca afecto los valores de MAE/RMSE/MAPE/sMAPE ya
    calculados -- solo la etiqueta de fecha. Ahora es incondicional e
    igual a xgboost_model.py/naive_model.py/etc.
    """
    return pd.to_numeric(serie_hora, errors="coerce").astype(float) - 1


def extraer_serie_horaria(df, columna):
    aux = df[[COL_FECHA, COL_HORA, columna]].copy()

    aux[COL_FECHA] = pd.to_datetime(aux[COL_FECHA], errors="coerce")
    aux[COL_HORA] = pd.to_numeric(aux[COL_HORA], errors="coerce")
    aux[columna] = pd.to_numeric(aux[columna], errors="coerce")

    aux = aux.dropna(subset=[COL_FECHA, COL_HORA, columna])

    aux["hora_0_23"] = convertir_hora_0_23(aux[COL_HORA])
    aux = aux.dropna(subset=["hora_0_23"])
    aux["hora_0_23"] = aux["hora_0_23"].astype(int)

    aux["datetime"] = aux[COL_FECHA] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    aux = aux.sort_values("datetime").drop_duplicates("datetime", keep="last")

    return aux[columna].to_numpy(dtype=float), aux["datetime"].to_numpy()


# =========================================================
# ACUMULADOR / GUARDADO
# =========================================================

class _ResultsAccumulator:
    def __init__(self):
        self.series = []
        self.metrics = []
        self.trials = []
        self.config_usada = []


def _region_de_serie(nombre_serie):
    return nombre_serie.split("_")[0]


def _construir_df_series(bloques):
    """
    `resultados.series` mezcla bloques escalares (una fila por prediccion) y
    un bloque "real" por region con `fecha`/`valor` como arreglo completo --
    el mismo patron que causo un bug en fcnn_model.py (ver su docstring).
    `np.atleast_1d` normaliza ambos casos antes de construir cada bloque,
    sin tocar ningun valor ni el orden de las filas.
    """
    frames = []
    for bloque in bloques:
        fechas_arr = np.atleast_1d(bloque["fecha"])
        valores_arr = np.atleast_1d(bloque["valor"])
        frames.append(pd.DataFrame({
            "serie": bloque["serie"],
            "fecha": pd.to_datetime(fechas_arr, errors="coerce"),
            "tipo": bloque["tipo"],
            "subset": bloque["subset"],
            "modelo": bloque["modelo"],
            "valor": valores_arr,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _guardar_avance_csv(resultados, output_dir):
    if resultados.series:
        df_series = _construir_df_series(resultados.series)
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

    if resultados.trials:
        df_trials = pd.concat(resultados.trials, ignore_index=True)
        df_trials.to_csv(os.path.join(output_dir, "trials.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Trials (barrido de lags AR sobre residuo) guardados (avance): {len(df_trials):,} registros")

    if resultados.config_usada:
        df_config = pd.DataFrame(resultados.config_usada)
        df_config["region"] = df_config["serie"].map(_region_de_serie)
        df_config.to_csv(os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config):,} registros")


# =========================================================
# EVALUAR REGION
# =========================================================

def evaluar_region(region, df, train_hours, forecast_horizon, resultados):
    nombre_serie = f"{region}_DEMANDA"

    serie, fechas = extraer_serie_horaria(df, COL_DEMANDA)

    requeridas = train_hours + forecast_horizon
    if len(serie) < requeridas:
        print(f"AVISO: {nombre_serie} tiene {len(serie):,} horas; se requieren al menos {requeridas:,}.")
        return

    serie_reciente = serie[-requeridas:]
    fechas_recientes = fechas[-requeridas:]

    train = serie_reciente[:-forecast_horizon]
    test = serie_reciente[-forecast_horizon:]
    fechas_test = fechas_recientes[-forecast_horizon:]

    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")

    try:
        trend, seasonal, resid, stl_res = descomponer_stl(train)
        del stl_res

        trend_forecast = forecast_tendencia_lineal(trend, horizon=len(test))
        seasonal_forecast = forecast_estacionalidad_repetida(seasonal, horizon=len(test), period=STL_PERIOD)

        print("   Seleccionando orden AR sobre residuo por AIC (barrido 1-168)...")
        resid_forecast, lag_optimo, df_lags = forecast_ar_resid(resid, horizon=len(test))
        print(f"      AR (residuo) lag optimo: {lag_optimo}")

        pred_final = trend_forecast + seasonal_forecast + resid_forecast

        metricas = calcular_metricas(test, pred_final)
        print(f"      {NOMBRE_MODELO} MAPE={metricas['MAPE']:.2f}%")

        df_lags = df_lags.copy()
        df_lags["serie"] = nombre_serie
        df_lags["modelo"] = NOMBRE_MODELO
        resultados.trials.append(df_lags)

        resultados.config_usada.append({
            "serie": nombre_serie,
            "modelo": NOMBRE_MODELO,
            "parametros": str({"lag_resid": lag_optimo, "trend": "c", "stl_period": STL_PERIOD}),
            "train_horas": len(train),
        })

        resultados.series.append({
            "serie": nombre_serie, "fecha": fechas, "tipo": "real",
            "subset": "completo", "modelo": "real", "valor": serie,
        })

        for j, val in enumerate(pred_final):
            resultados.series.append({
                "serie": nombre_serie, "fecha": fechas_test[j], "tipo": "prediccion",
                "subset": "test", "modelo": NOMBRE_MODELO, "valor": val,
            })

        resultados.metrics.append({
            "serie": nombre_serie, "modelo": NOMBRE_MODELO,
            "MAE": metricas["MAE"], "RMSE": metricas["RMSE"],
            "MAPE": metricas["MAPE"], "sMAPE": metricas["sMAPE"],
        })

    except Exception as e:
        print(f"Error en {nombre_serie}: {type(e).__name__}: {e}")


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run(
    exogenas_globales: dict,
    regions_all: list,
    train_hours: int = TRAIN_LAST_HOURS_DEFAULT,
    forecast_horizon: int = FORECAST_HORIZON_DEFAULT,
    exog_cols: list = None,
    optuna_n_trials=N_TRIALS_OPTUNA_DEFAULT,
    data_dir: str = "/content",
    output_dir: str = ".",
):
    """
    Pipeline AR_Resid_Trend_Seasonal nuevo (ver docstring del modulo).
    Devuelve (series_df, metricas_df, trials_df, config_usada_df); trials_df
    contiene el barrido de lags AR sobre el residuo STL (no trials Optuna).
    """
    if exog_cols:
        raise ValueError(f"El modelo {NOMBRE_MODELO} es univariado: no acepta exogenas.")

    resultados = _ResultsAccumulator()

    # Checkpoint por region: trials.csv aqui es el barrido de lags AR sobre
    # el residuo (MAX_LAG_AR=168 filas exactas), no trials de Optuna --
    # se exige el conteo exacto para detectar un barrido cortado a la mitad.
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all, forecast_horizon=forecast_horizon,
        requiere_trials=True, requiere_config_usada=True, trials_esperados=MAX_LAG_AR,
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print(f"PIPELINE {NOMBRE_MODELO.upper()} DEMANDA (combinacion nueva, ver docstring)")
    print("=" * 80)
    print(f"Train: {train_hours} h")
    print(f"Test: {forecast_horizon} h")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        print("\n" + "=" * 80)
        print(f"Serie: {region}_DEMANDA")
        print("=" * 80)

        evaluar_region(region, df, train_hours, forecast_horizon, resultados)

        print("\nGuardando avance...")
        _guardar_avance_csv(resultados, output_dir)

    series_df = pd.DataFrame()
    if resultados.series:
        series_df = _construir_df_series(resultados.series)
        series_df["region"] = series_df["serie"].map(_region_de_serie)
        series_df = series_df.sort_values(["serie", "fecha", "modelo"])

    metricas_df = pd.DataFrame(resultados.metrics)
    if len(metricas_df) > 0:
        metricas_df["region"] = metricas_df["serie"].map(_region_de_serie)
        metricas_df = metricas_df.sort_values(["serie", "MAPE"])

    trials_df = pd.concat(resultados.trials, ignore_index=True) if resultados.trials else pd.DataFrame()

    config_usada_df = pd.DataFrame(resultados.config_usada)
    if len(config_usada_df) > 0:
        config_usada_df["region"] = config_usada_df["serie"].map(_region_de_serie)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO")
    print("=" * 80)

    if len(metricas_df) > 0:
        print("\nResultados por region:")
        print(metricas_df.sort_values("MAPE").to_string(index=False))

    return series_df, metricas_df, trials_df, config_usada_df

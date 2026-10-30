"""
Pipeline AR (demanda, univariado) -- COMBINACION NUEVA, NO extraccion de un
pipeline completo de legacy.

*** No existe en el notebook legacy un modelo AR aplicado directamente a la
serie cruda. La celda 45 (baseline univariado) dice explicitamente en su
encabezado "Sin ARIMA / SARIMA". La UNICA implementacion de AR en todo el
notebook es `seleccionar_ar_por_aic`/`forecast_ar_resid` en la celda 60
(Ensemble), y ahi SIEMPRE se aplica al residuo de una descomposicion STL,
nunca a la serie cruda. ***

Este modulo reutiliza esa funcion de ajuste de AR **exactamente igual**
(AutoReg con `trend="c"`, orden elegido por AIC barriendo lags 1-168), pero
la aplica DIRECTO a la serie de Demanda sin ninguna descomposicion STL --
una combinacion nueva construida a partir de una pieza legacy, confirmada
explicitamente por el usuario el 2026-08-09.

Decisiones tomadas para esta combinacion (autorizadas explicitamente):
  - Univariado, sin exogenas (igual que Naive/Naive_Trend).
  - Split train/test: se reutiliza la MISMA formula dinamica de la celda 45
    (`test_size = max(720h, 10% de la serie)`, acotado a `len//3`) que ya
    se uso para Naive/Naive_Trend -- por consistencia dentro de la familia
    de baselines univariados nuevos, no porque la celda 60 la usara (la
    celda 60 usa una ventana fija de 3600h/168h, pero esa ventana viene
    acoplada a la descomposicion STL que aqui no existe).
  - Metricas: se reutiliza el `metrics.py` compartido (formula identica a
    XGBoost/Naive/Naive_Trend), no la variante con mascara `isfinite` de la
    celda 60 -- por la misma razon de consistencia dentro de esta familia.
  - `trials.csv` aqui NO son trials de Optuna (AR no usa Optuna, el orden
    se elige por AIC) sino la tabla de barrido de lags (lag, AIC, BIC) que
    ya producia `seleccionar_ar_por_aic` en la celda 60 -- se expone en el
    mismo archivo por consistencia de nombre con los demas modelos, no
    porque sea literalmente lo mismo que un trial de Optuna.
  - `config_usada.csv` registra el lag optimo elegido por region.

MAX_LAG_AR=168 y `trend="c"` en AutoReg son identicos a la celda 60.
"""

import os

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..metrics import calcular_metricas

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

MIN_OBS = 24 * 30

MAX_LAG_AR = 168  # identico a MAX_LAG_AR de la celda 60

EXOG_COLS_DEFAULT = []
EXOG_CATALOGO = []

TRAIN_LAST_HOURS_DEFAULT = "auto"
FORECAST_HORIZON_DEFAULT = "auto"
N_TRIALS_OPTUNA_DEFAULT = None  # sin uso, AR no usa Optuna (orden por AIC)


# =========================================================
# AR (copia exacta de seleccionar_ar_por_aic / forecast_ar_resid, celda 60)
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
        raise RuntimeError("No se pudo ajustar ningun AR valido.")

    return mejor_modelo, mejor_lag, pd.DataFrame(resultados)


def forecast_ar(serie_train, horizon):
    modelo, lag_optimo, df_lags = seleccionar_ar_por_aic(serie_train, max_lag=MAX_LAG_AR)

    pred = modelo.predict(start=len(serie_train), end=len(serie_train) + horizon - 1, dynamic=False)

    return np.asarray(pred), lag_optimo, df_lags


# =========================================================
# CARGAR REGIONES / EXTRAER SERIE (mismo patron que naive_model.py)
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
# SPLIT TRAIN / TEST (misma formula dinamica que naive_model.py)
# =========================================================

def _resolver_split(serie, forecast_horizon, train_hours):
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
        self.trials = []
        self.config_usada = []


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

    if resultados.trials:
        df_trials = pd.concat(resultados.trials, ignore_index=True)
        df_trials.to_csv(os.path.join(output_dir, "trials.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Trials (barrido de lags AR) guardados (avance): {len(df_trials):,} registros")

    if resultados.config_usada:
        df_config = pd.DataFrame(resultados.config_usada)
        df_config["region"] = df_config["serie"].map(_region_de_serie)
        df_config.to_csv(os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config):,} registros")


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
        print("   Seleccionando orden AR por AIC (barrido 1-168)...")

        pred, lag_optimo, df_lags = forecast_ar(train, horizon)

        print(f"      AR lag optimo: {lag_optimo}")

        df_lags = df_lags.copy()
        df_lags["serie"] = nombre_serie
        df_lags["modelo"] = "AR"
        resultados.trials.append(df_lags)

        resultados.config_usada.append({
            "serie": nombre_serie,
            "modelo": "AR",
            "parametros": str({"lag": lag_optimo, "trend": "c"}),
            "horizonte_usado": "serie_completa",
            "train_horas": len(train),
        })

        metricas = calcular_metricas(test, pred)

        if metricas:
            guardar_metricas(resultados, nombre_serie, "AR", False, metricas, "serie_completa")
            guardar_predicciones(resultados, nombre_serie, fechas_test, pred, "AR")

            print(f"      AR: MAPE={metricas['MAPE']:.2f}%")

    except Exception as e:
        print(f"      AR Error: {str(e)[:120]}")


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
    Pipeline AR nuevo (ver docstring del modulo). Devuelve (series_df,
    metricas_df, trials_df, config_usada_df). trials_df contiene el barrido
    de lags por AIC/BIC, no trials de Optuna.
    """
    if exog_cols:
        raise ValueError("El modelo AR (standalone) es univariado: no acepta exogenas.")

    resultados = _ResultsAccumulator()

    # Checkpoint por region: el split es dinamico ("auto") salvo override
    # explicito, asi que solo se exige un forecast_horizon fijo cuando el
    # llamador lo fijo; si no, basta con que la region tenga al menos una
    # prediccion (ver checkpoint.region_es_completa). trials.csv aqui es el
    # barrido de lags por AIC (MAX_LAG_AR=168 filas exactas), no trials de
    # Optuna -- se exige el conteo exacto para detectar un barrido cortado
    # a la mitad.
    forecast_horizon_fijo = forecast_horizon if isinstance(forecast_horizon, int) else None
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all,
        forecast_horizon=forecast_horizon_fijo,
        requiere_trials=True, requiere_config_usada=True,
        trials_esperados=MAX_LAG_AR,
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print("PIPELINE AR DEMANDA (combinacion nueva, ver docstring)")
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

    trials_df = pd.concat(resultados.trials, ignore_index=True) if resultados.trials else pd.DataFrame()

    config_usada_df = pd.DataFrame(resultados.config_usada)
    if len(config_usada_df) > 0:
        config_usada_df["region"] = config_usada_df["serie"].map(_region_de_serie)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO")
    print("=" * 80)

    if len(metricas_df) > 0:
        print("\nResultados:")
        print(metricas_df.sort_values("MAPE").to_string(index=False))

    return series_df, metricas_df, trials_df, config_usada_df

"""
Pipeline LightGBM multivariado (demanda, horizonte de 1 semana).

*** ADAPTADO / ESTANDARIZADO A PARTIR DE LA CELDA 46 DEL NOTEBOOK LEGACY. ***
*** NO ES UNA EXTRACCION EXACTA DE UN PIPELINE VIGENTE. ***

A diferencia de XGBoost, LSTM, SARIMAX, FCNN y Ensemble -- que sí tienen una
seccion propia y dedicada en el notebook legacy bajo el patron de "2 semanas
train / 1 semana test" -- LightGBM solo aparecia dentro de la celda 46
("Modelos univariados" / prototipo multivariado), mezclado junto con Naive,
XGBoost y LSTM, usando un esquema de train/test por PORCENTAJE de la serie
completa (no un horizonte fijo de 168h) y sin las exogenas electricas por
region (Generacion/Importacion/Exportacion).

Por instruccion explicita del usuario, este modulo:
  - SI conserva de la celda 46: el espacio de busqueda de Optuna de
    LightGBM (`objective_lightgbm`, lineas 403-443 de la celda 46) y el uso
    de `LGBMRegressor`. Estos hiperparametros NO se tocaron ni se
    reemplazaron por los de XGBoost.
  - NO conserva de la celda 46: el esquema de train/test por porcentaje,
    la ausencia de exogenas electricas, y las features de calendario
    (hour/dayofweek/month seno-coseno) que `create_feature_df_multivar`
    agregaba en la celda 46.
  - En su lugar, ADAPTA el resto del pipeline (ventana train/test fija,
    catalogo completo de 8 exogenas, tratamiento futuro conocido/estimado,
    construccion de features por lags+rolling, guardado, RUN_NAME) para que
    sea comparable con `xgboost_model.py`. La mayor parte de este bloque
    "marco" viene de ahi (que a su vez es fiel a la celda 49), no de la
    celda 46.

REDISENO 2026-08-09 -- VENTANA DE LAGS ADAPTATIVA (rediseño explícitamente
autorizado, no aplica a XGBoost):

La primera version de este modulo copiaba literal el `WINDOW_DEFAULT=168`
fijo de XGBoost para las features de lags. Eso tiene un defecto real e
independiente del smoke test que lo revelo: `create_feature_df` genera una
columna `lag_168` vía `.shift(168)`, y si la serie de entrenamiento
disponible en ese punto tiene <=168 filas, esa columna queda enteramente
NaN y `.dropna()` elimina TODAS las filas -- el ajuste queda entrenando
sobre un DataFrame vacio. Con `train_hours=336` (default) y
`forecast_horizon=168` (default), el presupuesto de filas que ve cada
trial de Optuna durante el tuning es `train_hours - forecast_horizon =
168` filas -- exactamente igual al `window`, por lo que **todos los
trials de Optuna retornaban `inf` de forma silenciosa** (el guard
`if len(df_train) < 50: return float("inf")` lo capturaba sin lanzar
excepcion) y el tuning nunca comparaba hiperparametros de verdad. El
ajuste final SI funcionaba (usa el `train_hours` completo, no el reducido
por validacion), asi que el pipeline igual producia metricas -- pero con
hiperparametros esencialmente arbitrarios, no tuneados. Con un
`train_hours` mas chico (como en un smoke test rapido, `train_hours=48`),
incluso el ajuste final se queda sin filas y el pipeline no produce
ninguna metrica para esa serie (sin lanzar excepcion tampoco -- solo
`calcular_metricas` devuelve `None` sobre predicciones NaN, y esa serie
simplemente no aparece en `metricas.csv`).

Correccion: `_resolver_window(train_hours, forecast_horizon)` deriva el
`window` del presupuesto real de filas disponibles para el tuning
(`train_hours - forecast_horizon`), dejando siempre un margen >= 50 filas
utilizables, con piso de 24h (un dia, para no perder estacionalidad
diaria) y techo de 168h (una semana, igual que XGBoost, para no perder
comparabilidad cuando el presupuesto lo permite). Con el default vigente
(336h train / 168h horizonte) esto da `window=84` en vez de 168 -- MENOS
lag history que XGBoost, pero con un tuning que **si compara
hiperparametros de verdad** en vez de retornar `inf` siempre. Con
presupuestos grandes (ej. `train_hours=1000`), el `window` sube hasta el
techo de 168h, igualandose a XGBoost. Es una decision nueva y deliberada,
no una preservacion de la celda 46 (que no tenia este problema porque
usaba un split de validacion completamente distinto, por porcentaje).

Segunda parte del mismo defecto, misma correccion: `create_feature_df`
tambien tenia columnas `rolling_mean_168` / `rolling_std_168` con ventana
de 168h FIJA (copiada de XGBoost), independiente del `window` de lags. Aun
reduciendo el window de lags, estas dos columnas seguian exigiendo 168
filas previas y `.dropna()` seguia vaciando el DataFrame en cualquier
escenario con presupuesto menor a 168. Ahora la ventana "larga" de rolling
usa el mismo `window` resuelto (no un numero fijo) y la ventana "corta"
usa `min(24, window)` -- todas las features (lags + ambos rolling) exigen
exactamente `window` filas de historia, ni una mas, asi que el numero de
filas utilizables tras el dropna() es siempre `len(y) - window`, verificado
empiricamente (ver commit). Los nombres de columnas cambiaron de
`rolling_mean_24/168` a `rolling_mean_corta/larga` para no sugerir un
numero de horas fijo que ya no aplica.

Ver docs/MODELOS_MIGRADOS.md para el detalle linea por linea de que viene
de donde.
"""

import os
import gc

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMRegressor

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..metrics import mape, smape, calcular_metricas

# =========================================================
# CONFIG POR DEFECTO
#
# FORECAST_HORIZON_DEFAULT, TRAIN_LAST_HOURS_DEFAULT, EXOG_COLS_DEFAULT y
# todo el tratamiento futuro de exogenas: ADAPTADOS para igualar el marco
# vigente de xgboost_model.py (no vienen de la celda 46).
#
# N_TRIALS_OPTUNA_DEFAULT y WINDOW_DEFAULT: SI coinciden con la celda 46
# (N_TRIALS_OPTUNA=10, WINDOW_DEFAULT=168), se conservan por ser iguales.
# =========================================================

FORECAST_HORIZON_DEFAULT = 24 * 7    # 168 horas -- adaptado, igual a XGBoost
TRAIN_LAST_HOURS_DEFAULT = 24 * 14   # 336 horas = 2 semanas -- adaptado, igual a XGBoost
N_TRIALS_OPTUNA_DEFAULT = 10         # igual a la celda 46 (N_TRIALS_OPTUNA)

WINDOW_DEFAULT = 168                 # techo del window (una semana) -- ver _resolver_window()
WINDOW_MINIMO = 24                   # piso del window (un dia)
FILAS_MINIMAS_TUNING = 50            # mismo umbral que el guard de objective_lightgbm

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

# Catalogo completo de exogenas -- adaptado: la celda 46 solo tenia las 5
# globales (sin Generacion/Importacion/Exportacion). Se extiende aqui al
# catalogo completo de XGBoost para que sea comparable.
EXOG_COLS_DEFAULT = [
    "Temperatura",
    "Primarias",
    "Secundarias",
    "Terciarias",
    "IGAE",
    "Generacion",
    "Importacion",
    "Exportacion",
]
EXOG_CATALOGO = list(EXOG_COLS_DEFAULT)

EXOG_SOURCE_MAP = {
    "Temperatura": "Temperaturas_H",
    "Primarias": "Primarias_H",
    "Secundarias": "Secundarias_H",
    "Terciarias": "Terciarias_H",
    "IGAE": "IGAE_H",
}

EXOG_CONOCIDAS_FUTURO = ["Temperatura", "IGAE"]

EXOG_NO_CONOCIDAS_FUTURO = [
    "Primarias",
    "Secundarias",
    "Terciarias",
    "Generacion",
    "Importacion",
    "Exportacion",
]

LAG_SEMANA_1 = 24 * 7    # 168
LAG_SEMANA_2 = 24 * 14   # 336


# =========================================================
# UTILIDADES (copiadas del marco de xgboost_model.py)
# =========================================================

def cleanup():
    gc.collect()


def _hora_a_0_23(hora):
    hora = pd.to_numeric(hora, errors="coerce")
    if hora.dropna().empty:
        return hora
    if hora.min() >= 1 and hora.max() <= 24:
        return hora.astype(float) - 1
    return hora.astype(float)


def _normalizar_exogena(df, nombre_variable):
    aux = df.copy()
    aux.columns = aux.columns.astype(str).str.strip()

    cols = {c.lower(): c for c in aux.columns}

    for requerida in ["fecha", "hora", "valor"]:
        if requerida not in cols:
            raise ValueError(f"No existe columna {requerida} en {nombre_variable}")

    aux = aux[[cols["fecha"], cols["hora"], cols["valor"]]].copy()

    aux[cols["fecha"]] = pd.to_datetime(aux[cols["fecha"]], errors="coerce")
    aux["hora_0_23"] = _hora_a_0_23(aux[cols["hora"]])
    aux[nombre_variable] = pd.to_numeric(aux[cols["valor"]], errors="coerce")

    aux = aux.dropna(subset=[cols["fecha"], "hora_0_23", nombre_variable])
    aux["hora_0_23"] = aux["hora_0_23"].astype(int)

    if not aux["hora_0_23"].between(0, 23).all():
        raise ValueError(f"La columna hora de {nombre_variable} debe estar en 0-23 o 1-24")

    aux["datetime"] = aux[cols["fecha"]] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    return (
        aux[["datetime", nombre_variable]]
        .groupby("datetime", as_index=False)
        .mean()
        .sort_values("datetime")
    )


# =========================================================
# CONSTRUIR EXOGENAS POR REGION
# =========================================================

def merge_exogenas(region, exogenas_globales, exog_cols, data_dir):
    exog_df = None

    for nombre_variable, nombre_df in EXOG_SOURCE_MAP.items():
        if nombre_variable not in exog_cols:
            continue

        if nombre_df not in exogenas_globales:
            raise ValueError(f"No existe el DataFrame global {nombre_df}.")

        aux = _normalizar_exogena(exogenas_globales[nombre_df], nombre_variable)

        exog_df = aux if exog_df is None else exog_df.merge(aux, on="datetime", how="outer")

    # Exogenas especificas de la region: se leen desde data_dir
    #   {data_dir}/{region}_GEN.csv, _IMP.csv, _EXP.csv
    region_exog_map = {
        "Generacion": f"{region}_GEN.csv",
        "Importacion": f"{region}_IMP.csv",
        "Exportacion": f"{region}_EXP.csv",
    }

    for nombre_variable, nombre_archivo in region_exog_map.items():
        if nombre_variable not in exog_cols:
            continue

        ruta_exog = os.path.join(data_dir, nombre_archivo)

        if not os.path.exists(ruta_exog):
            raise FileNotFoundError(f"No existe el archivo {ruta_exog}")

        df_exog_region = pd.read_csv(ruta_exog)
        aux = _normalizar_exogena(df_exog_region, nombre_variable)

        exog_df = aux if exog_df is None else exog_df.merge(aux, on="datetime", how="outer")

    if exog_df is None:
        raise ValueError("No hay ninguna exogena activa en exog_cols.")

    exog_df = exog_df.sort_values("datetime").reset_index(drop=True)
    exog_df[exog_cols] = exog_df[exog_cols].ffill().bfill()

    print(f"\nExogenas cargadas para {region}:")
    for col in exog_cols:
        print(f"   {col:15s} | {exog_df[col].notna().sum():,} valores")

    return exog_df[["datetime"] + exog_cols]


# =========================================================
# ALINEAR EXOGENAS A LA SERIE OBJETIVO
# =========================================================

def alinear_exogenas_a_fechas(fechas, exogenas_df, exog_cols):
    if exogenas_df is None:
        raise ValueError("exogenas_df no puede ser None")

    base = pd.DataFrame({"datetime": pd.to_datetime(fechas, errors="coerce")})

    aux = exogenas_df.copy()
    aux["datetime"] = pd.to_datetime(aux["datetime"], errors="coerce")
    aux = aux.dropna(subset=["datetime"]).sort_values("datetime")

    out = base.merge(aux[["datetime"] + exog_cols], on="datetime", how="left")

    out[exog_cols] = out[exog_cols].ffill().bfill()

    if out[exog_cols].isna().any().any():
        raise ValueError("No hay exogenas suficientes para las fechas de la serie")

    return out[exog_cols].astype(float).reset_index(drop=True)


# =========================================================
# CONSTRUIR EXOGENAS DEL HORIZONTE
# SIN UTILIZAR EL FUTURO REAL DE LAS VARIABLES OPERATIVAS
# =========================================================

def construir_future_exog(exog_serie, train_end, horizon, exog_cols):
    """
    Para Temperatura e IGAE: utiliza el valor correspondiente al horizonte.

    Para las demas exogenas: NO utiliza el valor real futuro. Cada hora se
    estima con promedio(misma hora hace 1 semana, misma hora hace 2 semanas).

    Este tratamiento es identico al de XGBoost -- adaptado por instruccion
    explicita, la celda 46 no distinguia conocidas/no-conocidas.
    """
    future = exog_serie.iloc[train_end:train_end + horizon].copy().reset_index(drop=True)

    for variable in EXOG_NO_CONOCIDAS_FUTURO:
        if variable not in exog_cols:
            continue

        valores_estimados = []
        for h in range(horizon):
            indice_futuro = train_end + h
            idx_semana_1 = indice_futuro - LAG_SEMANA_1
            idx_semana_2 = indice_futuro - LAG_SEMANA_2

            if idx_semana_2 < 0:
                raise ValueError(f"No existen 2 semanas previas para estimar {variable}")

            valor_1 = float(exog_serie.iloc[idx_semana_1][variable])
            valor_2 = float(exog_serie.iloc[idx_semana_2][variable])
            estimado = float(np.mean([valor_1, valor_2]))

            valores_estimados.append(estimado)

        future[variable] = valores_estimados

    return future[exog_cols].astype(float).reset_index(drop=True)


def validar_horizonte(modelo, pred, forecast_horizon):
    if len(pred) != forecast_horizon:
        raise ValueError(f"{modelo}: prediccion con {len(pred)} horas; se esperaban {forecast_horizon}")


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
        raise ValueError(f"No existe columna {columna} en {nombre_serie}")
    if COL_FECHA not in df.columns:
        raise ValueError(f"No existe columna '{COL_FECHA}' en {nombre_serie}")
    if COL_HORA not in df.columns:
        raise ValueError(f"No existe columna '{COL_HORA}' en {nombre_serie}")

    aux = df[[COL_FECHA, COL_HORA, columna]].copy()

    aux[COL_FECHA] = pd.to_datetime(aux[COL_FECHA], errors="coerce")
    aux[COL_HORA] = pd.to_numeric(aux[COL_HORA], errors="coerce")
    aux[columna] = pd.to_numeric(aux[columna], errors="coerce")

    aux = aux.dropna(subset=[COL_FECHA, COL_HORA])

    aux["hora_0_23"] = aux[COL_HORA].astype(int) - 1
    aux["datetime"] = aux[COL_FECHA] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    aux = aux.sort_values("datetime")

    return aux[columna].values.astype(float), aux["datetime"].values


# =========================================================
# FEATURES HORARIAS
#
# Adaptado: usa lags + rolling + trend (igual que XGBoost en espiritu), SIN
# las features de calendario (hour/dayofweek/month seno-coseno) que la
# celda 46 si agregaba en create_feature_df_multivar. Se prioriza
# consistencia con el marco vigente sobre replicar ese detalle del
# prototipo.
#
# REDISENO 2026-08-09: en la version anterior, "rolling_mean_168" /
# "rolling_std_168" usaban una ventana de 168h FIJA, independiente del
# `window` de lags -- eso significaba que aunque `_resolver_window()`
# redujera el window de lags, estas dos columnas seguian exigiendo 168
# filas previas para no ser NaN, y `.dropna()` seguia vaciando el
# DataFrame en cualquier escenario con presupuesto de filas menor a 168.
# Ahora la ventana "larga" de rolling usa el mismo `window` resuelto (no
# un numero fijo), y la ventana "corta" usa `min(24, window)` (24h = un
# dia; como WINDOW_MINIMO ya es 24, en la practica esto siempre da 24).
# Con esto, TODAS las features (lags + ambos rolling) requieren exactamente
# `window` filas de historia, ni una mas -- el numero de filas utilizables
# tras el dropna() es exactamente `len(y) - window`, coincidiendo con la
# aritmetica que `_resolver_window()` asume.
# =========================================================

def create_feature_df(y, window, exog_cols, exog=None):
    df = pd.DataFrame({"y": np.asarray(y, dtype=float)})

    if exog is not None:
        exog_df = pd.DataFrame(exog).reset_index(drop=True)

        if len(exog_df) != len(df):
            raise ValueError("La longitud de exog debe coincidir con y")

        for col in exog_cols:
            if col not in exog_df.columns:
                raise ValueError(f"Falta variable exogena {col}")
            df[col] = pd.to_numeric(exog_df[col], errors="coerce")

    for lag in range(1, window + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    rolling_corta = min(24, window)
    rolling_larga = window

    y_past = df["y"].shift(1)
    df["rolling_mean_corta"] = y_past.rolling(rolling_corta).mean()
    df["rolling_std_corta"] = y_past.rolling(rolling_corta).std()
    df["rolling_mean_larga"] = y_past.rolling(rolling_larga).mean()
    df["rolling_std_larga"] = y_past.rolling(rolling_larga).std()
    df["trend"] = np.arange(len(df))

    return df.dropna()


def create_features_from_history(hist_y, window, exog_cols, exog_row=None):
    features = {}

    if exog_row is not None:
        exog_values = exog_row.to_dict() if isinstance(exog_row, pd.Series) else dict(exog_row)

        for col in exog_cols:
            if col not in exog_values:
                raise ValueError(f"Falta variable exogena futura {col}")
            features[col] = float(exog_values[col])

    for lag in range(1, window + 1):
        features[f"lag_{lag}"] = hist_y[-lag] if len(hist_y) >= lag else hist_y[0]

    rolling_corta = min(24, window)
    rolling_larga = window

    features["rolling_mean_corta"] = np.mean(hist_y[-rolling_corta:])
    features["rolling_std_corta"] = np.std(hist_y[-rolling_corta:])
    features["rolling_mean_larga"] = np.mean(hist_y[-rolling_larga:])
    features["rolling_std_larga"] = np.std(hist_y[-rolling_larga:])
    features["trend"] = len(hist_y)

    return pd.DataFrame([features])


# =========================================================
# LIGHTGBM - OBJECTIVE OPTUNA
#
# Espacio de busqueda IDENTICO a objective_lightgbm en la celda 46
# (lineas 403-415): n_estimators, max_depth, learning_rate, num_leaves,
# subsample, colsample_bytree, reg_alpha, reg_lambda, random_state=42,
# verbose=-1. NO se copiaron los hiperparametros de XGBoost.
# =========================================================

def objective_lightgbm(trial, train_y, val_y, window, exog_cols, train_exog=None, val_exog=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25),
        "num_leaves": trial.suggest_int("num_leaves", 16, 80),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "verbose": -1,
    }

    df_train = create_feature_df(train_y, window, exog_cols, train_exog)

    if len(df_train) < 50:
        return float("inf")

    X_train = df_train.drop(columns=["y"])
    y_train = df_train["y"]

    val_context_y = np.concatenate([train_y, val_y])
    val_context_exog = pd.concat(
        [pd.DataFrame(train_exog), pd.DataFrame(val_exog)], ignore_index=True
    )

    df_val_all = create_feature_df(val_context_y, window, exog_cols, val_context_exog)

    if len(df_val_all) < len(val_y):
        return float("inf")

    X_val = df_val_all.drop(columns=["y"]).iloc[-len(val_y):]
    y_val = df_val_all["y"].iloc[-len(val_y):]

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    score = smape(y_val.values, preds)

    return score if not np.isnan(score) else float("inf")


# =========================================================
# OPTUNA
# =========================================================

def tune_lightgbm(train_y, val_y, nombre_serie, window, exog_cols, optuna_db, n_trials, train_exog=None, val_exog=None):
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"{nombre_serie}_lightgbm_2semanas",
        storage=f"sqlite:///{optuna_db}",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective_lightgbm(trial, train_y, val_y, window, exog_cols, train_exog, val_exog),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return study.best_params, study.trials_dataframe()


# =========================================================
# FORECAST RECURSIVO
#
# Estructura identica al forecast recursivo de XGBoost (adaptado); usa
# LGBMRegressor con los mejores parametros encontrados por Optuna.
# =========================================================

def forecast_lightgbm_tuned(train_y, horizon, best_params, window, exog_cols, train_exog=None, future_exog=None):
    try:
        if future_exog is None or len(future_exog) != horizon:
            raise ValueError("future_exog debe tener exactamente horizon filas")

        df_train = create_feature_df(train_y, window, exog_cols, train_exog)
        X_train = df_train.drop(columns=["y"])
        y_train = df_train["y"]

        model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        preds = []
        hist = list(train_y)

        future_exog = pd.DataFrame(future_exog).reset_index(drop=True)

        for step in range(horizon):
            X_future = create_features_from_history(hist, window, exog_cols, future_exog.iloc[step])
            X_future = X_future.reindex(columns=X_train.columns)

            pred = model.predict(X_future)[0]
            preds.append(pred)

            hist.append(pred)

        return np.array(preds)

    except Exception as e:
        print(f"         Error forecast LightGBM: {str(e)[:120]}")
        return np.full(horizon, np.nan)


# =========================================================
# ACUMULADOR DE RESULTADOS
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
    """Guardado incremental por region, mismo patron que xgboost_model.py."""

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
        print(f"OK Trials Optuna guardados (avance): {len(df_trials):,} registros")

    if resultados.config_usada:
        df_config_usada = pd.DataFrame(resultados.config_usada)
        df_config_usada["region"] = df_config_usada["serie"].map(_region_de_serie)
        df_config_usada.to_csv(os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config_usada):,} registros")


# =========================================================
# IMPRIMIR RESULTADOS
# =========================================================

def imprimir_resultados(df_metricas):
    if df_metricas is None or len(df_metricas) == 0:
        print("\nNo hay resultados para mostrar")
        return

    print("\n" + "=" * 100)
    print("MEJOR MODELO POR SERIE")
    print("=" * 100)

    mejor_por_serie = df_metricas.loc[df_metricas.groupby("serie")["MAPE"].idxmin()]
    print(
        mejor_por_serie[["serie", "modelo", "horizonte_usado", "MAPE", "sMAPE", "MAE", "RMSE"]]
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("RANKING GLOBAL DE MODELOS")
    print("=" * 100)

    ranking = df_metricas.groupby("modelo")["MAPE"].agg(["mean", "std"]).round(2).sort_values("mean")
    print(ranking.to_string())


# =========================================================
# VENTANA DE LAGS ADAPTATIVA (rediseno, ver docstring del modulo)
# =========================================================

def _resolver_window(train_hours, forecast_horizon):
    """
    Deriva el `window` de lags a partir del presupuesto real de filas que
    tendra cada trial de Optuna (`train_hours - forecast_horizon`, el
    tamano de `tune_train_y` en _preparar_serie), en vez de usar un valor
    fijo de 168h como XGBoost.

    Sin esto, con window=168 fijo, cualquier configuracion donde
    `train_hours - forecast_horizon <= 168` deja el DataFrame de features
    de tuning vacio tras el dropna() (la columna lag_168 queda enteramente
    NaN) y Optuna nunca compara hiperparametros de verdad -- ver docstring
    del modulo para el detalle completo.

    Garantiza:
      - Al menos FILAS_MINIMAS_TUNING (50) filas utilizables para el fit
        de cada trial, siempre que el presupuesto lo permita.
      - Nunca por debajo de WINDOW_MINIMO (24h, un dia -- preserva algo de
        estacionalidad diaria).
      - Nunca por encima de WINDOW_DEFAULT (168h, una semana -- el mismo
        techo que usa XGBoost, para no perder comparabilidad cuando el
        presupuesto de filas es generoso).
    """
    presupuesto_tuning = max(train_hours - forecast_horizon, 1)
    candidato = presupuesto_tuning // 2

    return min(WINDOW_DEFAULT, max(WINDOW_MINIMO, candidato))


# =========================================================
# PREPARAR TRAIN / TEST
# =========================================================

def _preparar_serie(nombre_serie, serie, fechas, exogenas_df, exog_cols, train_hours, forecast_horizon):
    required_hours = train_hours + forecast_horizon

    if len(serie) < required_hours:
        print(f"   Serie insuficiente: {nombre_serie}")
        return None

    start = len(serie) - required_hours
    serie_reciente = serie[start:]
    fechas_recientes = fechas[start:]

    exog_completa = alinear_exogenas_a_fechas(fechas, exogenas_df, exog_cols)

    exog_inicio = max(0, start - LAG_SEMANA_2)
    exog_contexto = exog_completa.iloc[exog_inicio:].reset_index(drop=True)

    offset_train = start - exog_inicio

    train = serie_reciente[:train_hours]
    test = serie_reciente[train_hours:]
    fechas_test = fechas_recientes[train_hours:]

    train_exog = exog_contexto.iloc[offset_train:offset_train + train_hours].reset_index(drop=True)

    train_end_contexto = offset_train + train_hours

    test_exog = construir_future_exog(
        exog_serie=exog_contexto,
        train_end=train_end_contexto,
        horizon=forecast_horizon,
        exog_cols=exog_cols,
    )

    window_resuelto = _resolver_window(train_hours, forecast_horizon)

    print("\n   Split general fijo")
    print(f"      Train: {len(train):,} obs")
    print(f"      Test: {len(test):,} obs")
    print(f"      Horizonte: {forecast_horizon} horas")
    print(f"      Window de lags (adaptativo): {window_resuelto} horas")

    print("\n   Exogenas activas:")
    for col in exog_cols:
        if col in EXOG_CONOCIDAS_FUTURO:
            print(f"      {col}: valor del horizonte")
        else:
            print(f"      {col}: estimada con t-168 y t-336")

    val_size = forecast_horizon

    tune_train_y = train[:-val_size]
    tune_val_y = train[-val_size:]

    tune_train_exog = train_exog.iloc[:-val_size].reset_index(drop=True)

    val_start_contexto = train_end_contexto - val_size

    tune_val_exog = construir_future_exog(
        exog_serie=exog_contexto,
        train_end=val_start_contexto,
        horizon=val_size,
        exog_cols=exog_cols,
    )

    return {
        "train": train,
        "test": test,
        "train_exog": train_exog,
        "test_exog": test_exog,
        "fechas_test": fechas_test,
        "horizon": forecast_horizon,
        "horizonte_usado": f"{forecast_horizon}_horas",
        "tune_train_y": tune_train_y,
        "tune_val_y": tune_val_y,
        "window_tune": window_resuelto,
        "train_exog_tune": tune_train_exog,
        "val_exog_tune": tune_val_exog,
    }


# =========================================================
# EVALUAR SERIE
# =========================================================

def evaluar_serie(nombre_serie, serie, fechas, exogenas_df, exog_cols, train_hours, forecast_horizon,
                   optuna_db, n_trials, resultados):
    contexto = _preparar_serie(nombre_serie, serie, fechas, exogenas_df, exog_cols, train_hours, forecast_horizon)

    if contexto is None:
        return

    try:
        print("\n      LightGBM tuning...")

        best_params, trials_df = tune_lightgbm(
            contexto["tune_train_y"],
            contexto["tune_val_y"],
            nombre_serie,
            contexto["window_tune"],
            exog_cols,
            optuna_db,
            n_trials,
            contexto["train_exog_tune"],
            contexto["val_exog_tune"],
        )

        trials_df["serie"] = nombre_serie
        trials_df["modelo"] = "LightGBM"
        resultados.trials.append(trials_df)

        resultados.config_usada.append({
            "serie": nombre_serie,
            "modelo": "LightGBM",
            "parametros": str(best_params),
            "horizonte_usado": contexto["horizonte_usado"],
            "train_horas": train_hours,
            "exogenas": str(exog_cols),
        })

        pred = forecast_lightgbm_tuned(
            contexto["train"],
            contexto["horizon"],
            best_params,
            contexto["window_tune"],
            exog_cols,
            contexto["train_exog"],
            contexto["test_exog"],
        )

        validar_horizonte("LightGBM_Tuned", pred, forecast_horizon)

        metricas = calcular_metricas(contexto["test"], pred)

        if metricas:
            guardar_metricas(resultados, nombre_serie, "LightGBM_Tuned", True, metricas, contexto["horizonte_usado"])
            guardar_predicciones(resultados, nombre_serie, contexto["fechas_test"], pred, "LightGBM_Tuned")

            print(f"      LightGBM_Tuned: MAPE={metricas['MAPE']:.2f}%")

    except Exception as e:
        print(f"      Error evaluando {nombre_serie}: {type(e).__name__}: {e}")

    finally:
        cleanup()


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run(
    exogenas_globales: dict,
    regions_all: list,
    train_hours: int = TRAIN_LAST_HOURS_DEFAULT,
    forecast_horizon: int = FORECAST_HORIZON_DEFAULT,
    exog_cols: list = None,
    optuna_n_trials: int = N_TRIALS_OPTUNA_DEFAULT,
    data_dir: str = "/content",
    output_dir: str = ".",
):
    """
    Pipeline LightGBM adaptado al marco vigente (ver docstring del modulo).

    Devuelve (series_df, metricas_df, trials_df, config_usada_df).
    """
    exog_cols = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)

    resultados = _ResultsAccumulator()
    optuna_db = os.path.join(output_dir, "optuna_lightgbm.db")

    # Checkpoint por region -- ver el mismo bloque en xgboost_model.py.
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all, forecast_horizon=forecast_horizon,
        requiere_trials=True, requiere_config_usada=True,
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print("PIPELINE LIGHTGBM DEMANDA (ADAPTADO DESDE CELDA 46)")
    print("=" * 80)
    print(f"Directorio de salida: {output_dir}")
    print(f"Train usado: {train_hours} horas")
    print(f"Test usado: {forecast_horizon} horas")

    print("\nExogenas activas:")
    for exog in exog_cols:
        print(f"   - {exog}")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        print("\n" + "=" * 80)
        print(f"Serie: {region}_DEMANDA")
        print("=" * 80)

        exogenas_df = merge_exogenas(region, exogenas_globales, exog_cols, data_dir)
        print(f"Exogenas construidas: {len(exogenas_df):,} filas")

        nombre_serie = f"{region}_DEMANDA"
        serie, fechas = extraer_serie_horaria(df, COL_DEMANDA, nombre_serie)

        print(f"Serie completa: {len(serie):,} observaciones")
        print(f"Rango: {fechas[0]} a {fechas[-1]}")

        evaluar_serie(
            nombre_serie, serie, fechas, exogenas_df, exog_cols,
            train_hours, forecast_horizon, optuna_db, optuna_n_trials, resultados,
        )

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

    imprimir_resultados(metricas_df if len(metricas_df) > 0 else None)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO")
    print("=" * 80)

    return series_df, metricas_df, trials_df, config_usada_df

"""
Pipeline LSTM directa multi-horizonte (demanda, horizonte de 1 semana).

Extraido de la celda 58 del notebook legacy ("LSTM pero con 60 epocas y mas
Optuna"), la version vigente confirmada (la celda 55, "LSTM: No funciono!",
es la version legacy/fallida y NO se migra). Arquitectura (LSTM historica +
rama de exogenas futuras + Dense(168) en una sola pasada), espacio de
busqueda de Optuna, EPOCHS_LSTM=60/PATIENCE_LSTM=8, tratamiento de exogenas
(Temperatura/IGAE = valor del horizonte; Generacion/Importacion/Exportacion
= UN SOLO lag de 168h, NO promedio de dos lags como XGBoost) y formulas de
metricas son IDENTICOS al original.

Cambios mecanicos (no de logica), mismo tipo de adaptacion que en
xgboost_model.py:

  1. `globals()` (Temperaturas_H, IGAE_H) y el diccionario global `series`
     (para GEN/IMP/EXP por region) se sustituyen por `exogenas_globales`
     (parametro) y lectura directa de `{region}_GEN/IMP/EXP.csv` desde
     `data_dir`.
  2. `ejecutar_pipeline()` pasa a ser `run(...)`.
  3. RESULTS_SERIES/RESULTS_METRICS/RESULTS_TRIALS/RESULTS_CONFIG_USADA
     (listas globales) pasan a `_ResultsAccumulator` local a cada llamada.
  4. `DRIVE_OUTPUT_DIR`/`SAVE_PREFIX`/`OPTUNA_DB` fijos se sustituyen por
     `output_dir` (carpeta del experimento) y nombres de archivo fijos.
  5. `DATA_DIR` fijo se sustituye por el parametro `data_dir`.

IMPORTANTE -- esta celda SI tenia proteccion por region mas robusta que
XGBoost: en la celda original, `ejecutar_pipeline()` envuelve
merge_exogenas + extraer_serie_horaria + evaluar_serie + guardado en un
solo try/except por region (XGBoost solo protegia evaluar_serie). Esa
proteccion se preserva identica aqui.

Metricas (mape/smape/calcular_metricas): usan mascara isfinite, DISTINTAS
de las de XGBoost (que no la tienen) -- por eso este modulo trae su propia
copia en vez de importar metrics.py.

Todo lo demas es una copia literal de la celda 58.
"""

import os
import gc

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..atomic_io import escribir_csv_atomico
from ..errores_log import registrar_error

# =========================================================
# CONFIG POR DEFECTO (identica a la celda 58)
# =========================================================

FORECAST_HORIZON_DEFAULT = 24 * 7          # 168 horas
TRAIN_LAST_HOURS_DEFAULT = 24 * 30 * 3     # 2160 horas = 3 meses

WINDOW_OPTIONS = [24, 48, 168]             # ventanas que prueba Optuna

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

# Catalogo -- coincide con los nombres canonicos (esta celda ya usaba
# "Temperatura" singular, igual que XGBoost; no requiere traduccion).
EXOG_COLS_DEFAULT = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
EXOG_CATALOGO = list(EXOG_COLS_DEFAULT)

EXOG_SOURCE_MAP = {
    "Temperatura": "Temperaturas_H",
    "IGAE": "IGAE_H",
}

EXOG_CONOCIDAS_FUTURO = ["Temperatura", "IGAE"]

# Exogenas electricas: para el horizonte se usan con UN SOLO lag de 168h
# (GEN_t_futuro <- GEN_(t-168)), a diferencia de XGBoost que promedia dos
# lags (t-168, t-336).
EXOG_LAG_SEMANAL = ["Generacion", "Importacion", "Exportacion"]
LAG_EXOG_FUTURO = 168

N_TRIALS_LSTM_DEFAULT = 10
EPOCHS_LSTM = 60
PATIENCE_LSTM = 8


# =========================================================
# UTILIDADES / METRICAS (copia exacta de la celda 58)
# =========================================================

def cleanup():
    K.clear_session()
    gc.collect()


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

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (denominator != 0)
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calcular_metricas(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return None

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


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

    aux["datetime"] = aux[cols["fecha"]] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    return (
        aux[["datetime", nombre_variable]]
        .groupby("datetime", as_index=False)
        .mean()
        .sort_values("datetime")
    )


# =========================================================
# EXOGENAS DE CADA REGION
# =========================================================

def merge_exogenas(region, exogenas_globales, exog_cols, data_dir):
    dfs = []

    for nombre_variable, nombre_df in EXOG_SOURCE_MAP.items():
        if nombre_variable not in exog_cols:
            continue

        if nombre_df not in exogenas_globales:
            raise ValueError(f"No existe {nombre_df}")

        dfs.append(_normalizar_exogena(exogenas_globales[nombre_df], nombre_variable))

    region_exog_map = {
        "Generacion": f"{region}_GEN.csv",
        "Importacion": f"{region}_IMP.csv",
        "Exportacion": f"{region}_EXP.csv",
    }

    for variable, nombre_archivo in region_exog_map.items():
        if variable not in exog_cols:
            continue

        ruta = os.path.join(data_dir, nombre_archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No existe el archivo {ruta}")

        dfs.append(_normalizar_exogena(pd.read_csv(ruta), variable))

    if len(dfs) == 0:
        raise ValueError("No hay exogenas activas.")

    exog_df = dfs[0]
    for aux in dfs[1:]:
        exog_df = exog_df.merge(aux, on="datetime", how="outer")

    exog_df = exog_df.sort_values("datetime").reset_index(drop=True)
    exog_df[exog_cols] = exog_df[exog_cols].ffill().bfill()

    return exog_df[["datetime"] + exog_cols]


# =========================================================
# ALINEAR EXOGENAS
# =========================================================

def alinear_exogenas_a_fechas(fechas, exogenas_df, exog_cols):
    base = pd.DataFrame({"datetime": pd.to_datetime(fechas, errors="coerce")})

    aux = exogenas_df.copy()
    aux["datetime"] = pd.to_datetime(aux["datetime"], errors="coerce")
    aux = aux.dropna(subset=["datetime"]).sort_values("datetime")

    out = base.merge(aux[["datetime"] + exog_cols], on="datetime", how="left")
    out[exog_cols] = out[exog_cols].ffill().bfill()

    if out[exog_cols].isna().any().any():
        raise ValueError("No hay exogenas suficientes.")

    return out[exog_cols].astype(float).reset_index(drop=True)


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
        print(f"OK {region}: {archivo} {df.shape}")

    return regiones


# =========================================================
# EXTRAER DEMANDA
# =========================================================

def extraer_serie_horaria(df, columna, nombre_serie):
    if columna not in df.columns:
        raise ValueError(f"No existe {columna} en {nombre_serie}")

    aux = df[[COL_FECHA, COL_HORA, columna]].copy()

    aux[COL_FECHA] = pd.to_datetime(aux[COL_FECHA], errors="coerce")
    aux[COL_HORA] = pd.to_numeric(aux[COL_HORA], errors="coerce")
    aux[columna] = pd.to_numeric(aux[columna], errors="coerce")

    aux = aux.dropna(subset=[COL_FECHA, COL_HORA])

    aux["hora_0_23"] = aux[COL_HORA].astype(int) - 1
    aux["datetime"] = aux[COL_FECHA] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    aux = aux.sort_values("datetime").drop_duplicates("datetime", keep="last")

    return aux[columna].to_numpy(dtype=float), aux["datetime"].to_numpy()


# =========================================================
# BLOQUE DE EXOGENAS FUTURAS
#
# Temp/IGAE: valor de esa hora. GEN/IMP/EXP: valor de esa hora - 168h
# (un solo lag, no promedio de dos como XGBoost).
# =========================================================

def construir_future_exog_directa(exog, origin, horizon, exog_cols):
    filas = []

    for h in range(horizon):
        idx_future = origin + h
        row = {}

        for col in exog_cols:
            if col in EXOG_CONOCIDAS_FUTURO:
                row[col] = exog.iloc[idx_future][col]

            elif col in EXOG_LAG_SEMANAL:
                idx_lag = idx_future - LAG_EXOG_FUTURO
                if idx_lag < 0:
                    raise ValueError(f"No existe lag 168 para {col}")
                row[col] = exog.iloc[idx_lag][col]

            else:
                raise ValueError(f"No se definio tratamiento futuro para {col}")

        filas.append(row)

    return pd.DataFrame(filas)[exog_cols].astype(float)


# =========================================================
# ESCALADO
# =========================================================

def escalar_historia(y, exog, scaler_y, scaler_exog, exog_cols):
    y_scaled = scaler_y.transform(np.asarray(y).reshape(-1, 1))
    exog_scaled = scaler_exog.transform(pd.DataFrame(exog)[exog_cols])
    return np.concatenate([y_scaled, exog_scaled], axis=1)


# =========================================================
# CREAR DATASET DIRECTO
# =========================================================

def crear_dataset_directo(y, exog, window, scaler_y, scaler_exog, exog_cols, forecast_horizon):
    y = np.asarray(y, dtype=float)
    exog = pd.DataFrame(exog).reset_index(drop=True)

    hist_scaled = escalar_historia(y, exog, scaler_y, scaler_exog, exog_cols)

    X_hist, X_future, Y = [], [], []

    inicio = max(window, LAG_EXOG_FUTURO)
    ultimo_origin = len(y) - forecast_horizon

    for origin in range(inicio, ultimo_origin + 1):
        hist = hist_scaled[origin - window:origin]

        future_exog = construir_future_exog_directa(exog, origin, forecast_horizon, exog_cols)
        future_exog_scaled = scaler_exog.transform(future_exog[exog_cols])

        target = y[origin:origin + forecast_horizon]
        target_scaled = scaler_y.transform(target.reshape(-1, 1)).ravel()

        X_hist.append(hist)
        X_future.append(future_exog_scaled)
        Y.append(target_scaled)

    return np.asarray(X_hist), np.asarray(X_future), np.asarray(Y)


# =========================================================
# MODELO LSTM DIRECTO
# =========================================================

def construir_modelo_lstm_directo(window, n_features_hist, n_exog, units, dropout, learning_rate, forecast_horizon):
    hist_input = Input(shape=(window, n_features_hist), name="historia")
    hist_encoded = LSTM(units, dropout=dropout, name="lstm")(hist_input)

    future_input = Input(shape=(forecast_horizon, n_exog), name="exogenas_futuras")
    future_flat = Flatten(name="flatten_exogenas")(future_input)

    combinado = Concatenate(name="fusion")([hist_encoded, future_flat])

    output = Dense(forecast_horizon, name="demanda_horizonte")(combinado)

    model = Model(inputs=[hist_input, future_input], outputs=output)
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return model


# =========================================================
# OPTUNA
# =========================================================

def objective_lstm(trial, train_y, train_exog, exog_cols, forecast_horizon):
    params = {
        "units": trial.suggest_categorical("units", [32, 64, 128]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "window": trial.suggest_categorical("window", WINDOW_OPTIONS),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.001, 0.003]),
    }

    window = params["window"]

    # HOLDOUT DE forecast_horizon DENTRO DEL TRAIN
    val_horizon = forecast_horizon

    core_y = train_y[:-val_horizon]
    core_exog = train_exog.iloc[:-val_horizon].reset_index(drop=True)

    scaler_y = StandardScaler()
    scaler_exog = StandardScaler()

    scaler_y.fit(np.asarray(core_y).reshape(-1, 1))
    scaler_exog.fit(core_exog[exog_cols])

    X_hist, X_future, Y = crear_dataset_directo(
        core_y, core_exog, window, scaler_y, scaler_exog, exog_cols, forecast_horizon
    )

    if len(X_hist) < 100:
        cleanup()
        return float("inf")

    model = construir_modelo_lstm_directo(
        window=window,
        n_features_hist=1 + len(exog_cols),
        n_exog=len(exog_cols),
        units=params["units"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        forecast_horizon=forecast_horizon,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE_LSTM, restore_best_weights=True)

    model.fit(
        [X_hist, X_future], Y,
        validation_split=0.2,
        epochs=EPOCHS_LSTM,
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0,
        shuffle=False,
    )

    # VALIDACION REAL DE forecast_horizon HORAS
    origin = len(core_y)

    all_y = np.asarray(train_y, dtype=float)
    all_exog = train_exog.reset_index(drop=True)

    hist_raw = all_y[origin - window:origin]
    hist_exog_raw = all_exog.iloc[origin - window:origin].reset_index(drop=True)

    hist_scaled = escalar_historia(hist_raw, hist_exog_raw, scaler_y, scaler_exog, exog_cols)

    future_exog = construir_future_exog_directa(all_exog, origin, forecast_horizon, exog_cols)
    future_scaled = scaler_exog.transform(future_exog[exog_cols])

    pred_scaled = model.predict(
        [hist_scaled.reshape(1, window, -1), future_scaled.reshape(1, forecast_horizon, len(exog_cols))],
        verbose=0,
    )[0]

    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    val_real = all_y[origin:origin + forecast_horizon]

    score = smape(val_real, pred)

    cleanup()

    if np.isnan(score):
        return float("inf")

    return score


def tune_lstm(train_y, train_exog, nombre_serie, exog_cols, forecast_horizon, optuna_db, n_trials):
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"{nombre_serie}_lstm_directa",
        storage=f"sqlite:///{optuna_db}",
        load_if_exists=True,
    )

    # Al reanudar una region cortada a mitad, el estudio (en el .db) ya trae intentos completos: solo se corren
    # los que faltan para llegar a `n_trials` (mismo patron que FCNN/LSTM residual/Ensemble STL), de modo que el
    # presupuesto de Optuna de cada region sigue siendo exactamente `n_trials`.
    pendientes = _intentos_pendientes(study, n_trials)
    print(f"      Trials de Optuna restantes: {pendientes} de {n_trials}")

    if pendientes > 0:
        study.optimize(
            lambda trial: objective_lstm(trial, train_y, train_exog, exog_cols, forecast_horizon),
            n_trials=pendientes,
            show_progress_bar=False,
        )

    return study.best_params, study.trials_dataframe()


def _intentos_pendientes(study, n_trials):
    """Intentos de Optuna que faltan para completar el presupuesto `n_trials` (solo cuentan los COMPLETE)."""
    completos = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    return max(0, n_trials - completos)


# =========================================================
# FORECAST FINAL DIRECTO
# =========================================================

def forecast_lstm_directo(train_y, train_exog, full_exog, test_origin, best_params, exog_cols, forecast_horizon):
    try:
        window = best_params["window"]

        scaler_y = StandardScaler()
        scaler_exog = StandardScaler()

        scaler_y.fit(np.asarray(train_y).reshape(-1, 1))
        scaler_exog.fit(train_exog[exog_cols])

        X_hist, X_future, Y = crear_dataset_directo(
            train_y, train_exog, window, scaler_y, scaler_exog, exog_cols, forecast_horizon
        )

        if len(X_hist) < 100:
            raise ValueError("Muy pocas secuencias para entrenamiento directo")

        model = construir_modelo_lstm_directo(
            window=window,
            n_features_hist=1 + len(exog_cols),
            n_exog=len(exog_cols),
            units=best_params["units"],
            dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"],
            forecast_horizon=forecast_horizon,
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE_LSTM, restore_best_weights=True)

        model.fit(
            [X_hist, X_future], Y,
            validation_split=0.2,
            epochs=EPOCHS_LSTM,
            batch_size=best_params["batch_size"],
            callbacks=[early_stop],
            verbose=0,
            shuffle=False,
        )

        hist_y = train_y[-window:]
        hist_exog = train_exog.iloc[-window:].reset_index(drop=True)

        hist_scaled = escalar_historia(hist_y, hist_exog, scaler_y, scaler_exog, exog_cols)

        future_exog = construir_future_exog_directa(full_exog, test_origin, forecast_horizon, exog_cols)
        future_scaled = scaler_exog.transform(future_exog[exog_cols])

        pred_scaled = model.predict(
            [hist_scaled.reshape(1, window, -1), future_scaled.reshape(1, forecast_horizon, len(exog_cols))],
            verbose=0,
        )[0]

        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        return pred

    except Exception as e:
        print(f"         Error forecast LSTM directa: {type(e).__name__}: {str(e)[:150]}")
        return np.full(forecast_horizon, np.nan)

    finally:
        cleanup()


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
        resultados.series.append({
            "serie": nombre_serie,
            "fecha": fechas_test[j],
            "tipo": "prediccion",
            "subset": "test",
            "modelo": modelo,
            "valor": pred_val,
        })


def guardar_metricas(resultados, nombre_serie, modelo, metricas, horizonte_usado):
    resultados.metrics.append({
        "serie": nombre_serie,
        "modelo": modelo,
        "tuneado": True,
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
        escribir_csv_atomico(df_series, os.path.join(output_dir, "series.csv"), index=False, encoding="utf-8-sig")
        print(f"\nOK Series guardadas (avance): {len(df_series):,} registros")

    if resultados.metrics:
        df_metrics = pd.DataFrame(resultados.metrics)
        df_metrics["region"] = df_metrics["serie"].map(_region_de_serie)
        df_metrics = df_metrics.sort_values(["serie", "MAPE"])
        escribir_csv_atomico(df_metrics, os.path.join(output_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Metricas guardadas (avance): {len(df_metrics):,} registros")

    if resultados.trials:
        df_trials = pd.concat(resultados.trials, ignore_index=True)
        escribir_csv_atomico(df_trials, os.path.join(output_dir, "trials.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Trials Optuna guardados (avance): {len(df_trials):,} registros")

    if resultados.config_usada:
        df_config = pd.DataFrame(resultados.config_usada)
        df_config["region"] = df_config["serie"].map(_region_de_serie)
        escribir_csv_atomico(df_config, os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config):,} registros")


# =========================================================
# EVALUAR UNA SERIE
# =========================================================

def evaluar_serie(nombre_serie, serie, fechas, exogenas_df, exog_cols, train_hours, forecast_horizon,
                   optuna_db, n_trials, resultados):
    required_hours = train_hours + forecast_horizon

    if len(serie) < required_hours:
        print(f"Serie insuficiente: {nombre_serie}")
        return

    exog_serie = alinear_exogenas_a_fechas(fechas, exogenas_df, exog_cols)

    train_start = len(serie) - forecast_horizon - train_hours
    test_origin = len(serie) - forecast_horizon

    train = serie[train_start:test_origin]
    test = serie[test_origin:]
    fechas_test = fechas[test_origin:]

    train_exog = exog_serie.iloc[train_start:test_origin].reset_index(drop=True)

    print("\nSplit:")
    print(f"   Train: {len(train):,} h")
    print(f"   Test:  {len(test):,} h")
    print("   Estrategia: DIRECTA MULTI-HORIZONTE")

    print("\nExogenas:")
    for col in exog_cols:
        if col in EXOG_CONOCIDAS_FUTURO:
            print(f"   {col}: valor del horizonte")
        else:
            print(f"   {col}: lag 168h")

    print("\nLSTM directa tuning...")
    best_params, trials_df = tune_lstm(train, train_exog, nombre_serie, exog_cols, forecast_horizon, optuna_db, n_trials)

    print("\nMejores parametros:")
    print(best_params)

    trials_df["serie"] = nombre_serie
    trials_df["modelo"] = "LSTM_Directa"
    resultados.trials.append(trials_df)

    resultados.config_usada.append({
        "serie": nombre_serie,
        "modelo": "LSTM_Directa",
        "parametros": str(best_params),
        "horizonte_usado": f"{forecast_horizon}_horas_directas",
        "train_horas": train_hours,
        "exogenas": str(exog_cols),
    })

    pred = forecast_lstm_directo(train, train_exog, exog_serie, test_origin, best_params, exog_cols, forecast_horizon)

    print(f"\nPredicciones validas: {np.isfinite(pred).sum()}/{len(pred)}")

    metricas = calcular_metricas(test, pred)

    if metricas is None:
        print("No fue posible calcular metricas.")
        return

    guardar_metricas(resultados, nombre_serie, "LSTM_Directa", metricas, f"{forecast_horizon}_horas_directas")
    guardar_predicciones(resultados, nombre_serie, fechas_test, pred, "LSTM_Directa")

    print("\nLSTM_Directa:")
    print(f"   MAPE:  {metricas['MAPE']:.2f}%")
    print(f"   sMAPE: {metricas['sMAPE']:.2f}%")
    print(f"   MAE:   {metricas['MAE']:.2f}")
    print(f"   RMSE:  {metricas['RMSE']:.2f}")


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run(
    exogenas_globales: dict,
    regions_all: list,
    train_hours: int = TRAIN_LAST_HOURS_DEFAULT,
    forecast_horizon: int = FORECAST_HORIZON_DEFAULT,
    exog_cols: list = None,
    optuna_n_trials: int = N_TRIALS_LSTM_DEFAULT,
    data_dir: str = "/content",
    output_dir: str = ".",
):
    """
    Equivalente a ejecutar_pipeline() en la celda 58. Devuelve
    (series_df, metricas_df, trials_df, config_usada_df).
    """
    exog_cols = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)

    resultados = _ResultsAccumulator()
    optuna_db = os.path.join(output_dir, "optuna_lstm_directa.db")

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
    print("PIPELINE LSTM DIRECTA MULTI-HORIZONTE")
    print("=" * 80)
    print(f"Train: {train_hours} h")
    print(f"Forecast: {forecast_horizon} h")
    print("Estrategia: DIRECTA MULTI-HORIZONTE")

    print("\nExogenas activas:")
    for col in exog_cols:
        print(f"   - {col}")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        print("\n" + "=" * 80)
        print(f"Serie: {region}_DEMANDA")
        print("=" * 80)

        try:
            exogenas_df = merge_exogenas(region, exogenas_globales, exog_cols, data_dir)

            nombre_serie = f"{region}_DEMANDA"
            serie, fechas = extraer_serie_horaria(df, COL_DEMANDA, nombre_serie)

            print(f"Serie completa: {len(serie):,} observaciones")

            evaluar_serie(
                nombre_serie, serie, fechas, exogenas_df, exog_cols,
                train_hours, forecast_horizon, optuna_db, optuna_n_trials, resultados,
            )

            print("\nGuardando avance...")
            _guardar_avance_csv(resultados, output_dir)

        except Exception as e:
            print(f"Error {region}: {type(e).__name__}: {e}")
            registrar_error(region, e)

        finally:
            cleanup()

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

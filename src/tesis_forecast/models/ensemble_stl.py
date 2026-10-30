"""
Pipeline Ensemble STL (demanda, horizonte de 1 semana): descomposicion STL
del train en tendencia + estacionalidad + residuo, cada componente
pronosticado por un submodelo distinto, y suma de los tres para el
pronostico final.

Extraido de la celda 60 del notebook legacy ("Ensamble"):
  - Tendencia -> LSTM (con exogenas), tuneado con Optuna.
  - Estacionalidad -> FCNN (con exogenas), tuneada con Optuna.
  - Residuo -> AR univariado, orden elegido por AIC (barrido 1-168).
  - pred_final = trend_pred + seasonal_pred + resid_pred.

Por instruccion explicita de una conversacion previa: el Ensemble
CONSERVA esta arquitectura tal cual -- NO se convierte en un ensamble de
XGBoost/LSTM directa/SARIMAX/FCNN ya entrenados. Los submodelos LSTM y FCNN
de este archivo son independientes de `lstm_direct.py` y `fcnn_model.py`
(arquitecturas distintas, ver mas abajo), se entrenan de cero sobre cada
componente STL.

Diferencias de arquitectura frente a `lstm_direct.py` / `fcnn_model.py`
(preservadas, NO unificadas):
  - El LSTM aqui NO usa secuencias reales de shape (window, features): usa
    `crear_ventanas()` que aplana la ventana de 168h + exogenas en un solo
    vector, y `reshape_lstm_features()` le da forma (n, 1, features) --
    LSTM con un solo timestep "pseudo-secuencial", 1 o 2 capas.
  - El objetivo de Optuna para LSTM y FCNN aqui es **MAE**, no sMAPE (que
    es lo que usan XGBoost/LSTM directa/FCNN standalone).
  - `tunear_lstm`/`tunear_fcnn` SI cuentan trials ya completados en el
    estudio y solo corren los que faltan (`remaining = N_TRIALS -
    completados`), igual que `fcnn_model.py`.
  - `calcular_metricas` aqui NO tiene guarda contra entrada vacia (a
    diferencia de XGBoost/LSTM directa/SARIMAX/FCNN, que devuelven
    None/NaN) -- si no hay datos validos, lanza una excepcion igual que en
    la celda original. No se agrego el guard: hacerlo habria sido cambiar
    el comportamiento del pipeline original.

Cambios mecanicos (no de logica):
  1. `globals()` (Temperaturas_H, IGAE_H) y el diccionario global `series`
     se sustituyen por `exogenas_globales` (parametro) y lectura directa
     de `{region}_GEN/IMP/EXP.csv` desde `data_dir`.
  2. El script de nivel de modulo se envuelve en `run(...)`.
  3. `np.random.seed(SEED)` / `tf.random.set_seed(SEED)` se movieron de
     nivel de modulo a el inicio de `run()`.
  4. `OUTPUT_DIR`/`INCREMENTAL_DIR` fijos se sustituyen por `output_dir`;
     el guardado "por serie" en subcarpetas se reemplaza por el mismo
     patron de acumulador + series.csv/metricas.csv en la raiz de la
     carpeta del experimento que usan los demas modelos migrados.
  5. metricas.csv / config_usada.csv: la tabla de metricas original
     (MAE/RMSE/MAPE/sMAPE + trend_model/seasonal_model/resid_model/
     exogenas/lag_electricas/resid_lag_optimo/trend_best_val_MAE/
     seasonal_best_val_MAE/trend_params/seasonal_params/train_horas/
     test_horas/window/trials) se separa en metricas.csv (desempeno) y
     config_usada.csv (el resto). Ningun valor se descarta.
  6. trials.csv: la celda original NO exportaba `study.trials_dataframe()`
     de LSTM/FCNN a CSV, ni la tabla `df_lags_resid` (AIC/BIC por lag del
     AR) mas alla del archivo separado `_AR_resid_lags_AIC.csv` por serie.
     Aqui se combinan los tres (trials de LSTM, trials de FCNN, y el
     barrido de lags del AR) en un solo trials.csv con una columna
     `modelo` para distinguirlos -- son los mismos datos que ya existian
     en la celda original (en el .db de Optuna y en el CSV separado), solo
     expuestos en el mismo formato que usan los demas modelos migrados.
  7. EXOG_NAMES usaba "Temperaturas" (plural) y sufijos "_lag168". El
     catalogo externo de `ExperimentConfig.exogenas` es el mismo para
     todos los modelos; este modulo traduce internamente (ver
     CANONICAL_TO_INTERNAL, identico al de fcnn_model.py).

Todo lo demas es una copia literal de la celda 60.
"""

import os
import gc
import json

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.ar_model import AutoReg

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..atomic_io import escribir_csv_atomico
from ..errores_log import registrar_error

# =========================================================
# CONFIG POR DEFECTO (identica a la celda 60)
# =========================================================

TRAIN_LAST_HOURS_DEFAULT = 24 * 30 * 5   # 3600 horas = 5 meses
FORECAST_HORIZON_DEFAULT = 24 * 7        # 168 horas

WINDOW = 168
STL_PERIOD = 168
MAX_LAG_AR = 168

N_TRIALS_DEFAULT = 5
EPOCHS = 60
SEED = 42

ELECTRIC_LAG = 168

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

EXOG_COLS_DEFAULT = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
EXOG_CATALOGO = list(EXOG_COLS_DEFAULT)

CANONICAL_TO_INTERNAL = {
    "Temperatura": "Temperaturas",
    "IGAE": "IGAE",
    "Generacion": "Generacion_lag168",
    "Importacion": "Importacion_lag168",
    "Exportacion": "Exportacion_lag168",
}

NOMBRE_MODELO_FINAL = "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid_EXOG_ALL_Lag168"


# =========================================================
# METRICAS (copia exacta de la celda 60 -- SIN guarda contra
# entrada vacia, a diferencia de los demas modelos)
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

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def convertir_hora_0_23(serie_hora):
    """
    Hora viene conceptualmente en 1..24 -> hora_0_23 = Hora - 1, SIEMPRE.

    FIX (bug de desfase +1h, confirmado por auditoria): esta funcion antes
    solo restaba 1 si TODA la columna caia en [1,24]; una sola fila fuera
    de rango dejaba la columna COMPLETA sin ajustar, produciendo un
    desfase sistematico de 1 hora entre los timestamps de este modulo
    (demanda Y exogenas, ambas pasan por aca) y los de xgboost_model.py/
    lightgbm_model.py/lstm_direct.py/sarimax_model.py/naive_model.py/
    naive_trend_model.py/ar_model.py (que siempre restan 1, sin
    condicion). Ahora es incondicional e igual a esos modulos -- demanda
    y exogenas quedan garantizadas en la MISMA convencion horaria.
    """
    return pd.to_numeric(serie_hora, errors="coerce").astype(float) - 1


# =========================================================
# LECTURA DE DEMANDA
# =========================================================

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
# PREPARAR EXOGENA HORARIA
# =========================================================

def preparar_exogena_horaria(df, nombre):
    aux = df.copy()
    aux.columns = aux.columns.astype(str).str.strip()

    cols_lower = {c.lower(): c for c in aux.columns}

    for requerida in ["fecha", "hora", "valor"]:
        if requerida not in cols_lower:
            raise ValueError(f"{nombre}: no tiene columna {requerida}")

    fecha_col = cols_lower["fecha"]
    hora_col = cols_lower["hora"]
    valor_col = cols_lower["valor"]

    aux[fecha_col] = pd.to_datetime(aux[fecha_col], errors="coerce")
    aux[hora_col] = pd.to_numeric(aux[hora_col], errors="coerce")
    aux[valor_col] = pd.to_numeric(aux[valor_col], errors="coerce")

    aux = aux.dropna(subset=[fecha_col, hora_col, valor_col])

    aux["hora_0_23"] = convertir_hora_0_23(aux[hora_col])
    aux = aux.dropna(subset=["hora_0_23"])
    aux["hora_0_23"] = aux["hora_0_23"].astype(int)

    aux["datetime"] = aux[fecha_col] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    return (
        aux[["datetime", valor_col]]
        .rename(columns={valor_col: nombre})
        .groupby("datetime", as_index=False)
        .mean()
        .sort_values("datetime")
    )


def preparar_exogena_lag168(df, nombre):
    """X_lag168[t] = X[t-168], vía desplazar el timestamp 168h hacia adelante."""
    aux = preparar_exogena_horaria(df, nombre)
    aux["datetime"] = aux["datetime"] + pd.to_timedelta(ELECTRIC_LAG, unit="h")
    return aux


# =========================================================
# MATRIZ EXOGENA POR REGION
# =========================================================

def construir_matriz_exogena_region(region, exogenas_globales, exog_names, data_dir):
    dfs = []

    if "Temperaturas" in exog_names:
        if "Temperaturas_H" not in exogenas_globales:
            raise ValueError("No existe Temperaturas_H")
        dfs.append(preparar_exogena_horaria(exogenas_globales["Temperaturas_H"], "Temperaturas"))

    if "IGAE" in exog_names:
        if "IGAE_H" not in exogenas_globales:
            raise ValueError("No existe IGAE_H")
        dfs.append(preparar_exogena_horaria(exogenas_globales["IGAE_H"], "IGAE"))

    region_exog_map = {
        "Generacion_lag168": f"{region}_GEN.csv",
        "Importacion_lag168": f"{region}_IMP.csv",
        "Exportacion_lag168": f"{region}_EXP.csv",
    }

    for variable, nombre_archivo in region_exog_map.items():
        if variable not in exog_names:
            continue

        ruta = os.path.join(data_dir, nombre_archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No existe el archivo {ruta}")

        dfs.append(preparar_exogena_lag168(pd.read_csv(ruta), variable))

    if len(dfs) == 0:
        raise ValueError("No hay exogenas activas.")

    exog = dfs[0]
    for df_next in dfs[1:]:
        exog = exog.merge(df_next, on="datetime", how="outer")

    exog = exog.sort_values("datetime").reset_index(drop=True)
    exog[exog_names] = exog[exog_names].ffill().bfill()

    print(f"\nExogenas ensemble {region}:")
    for col in exog_names:
        print(f"   {col:25s} | {exog[col].notna().sum():,}")
    print(f"Rango: {exog['datetime'].min()} -> {exog['datetime'].max()}")

    return exog


def alinear_exogenas_con_fechas(fechas, exog_region, exog_names):
    base = pd.DataFrame({"datetime": pd.to_datetime(fechas)})

    X = base.merge(exog_region[["datetime"] + exog_names], on="datetime", how="left")
    X[exog_names] = X[exog_names].ffill().bfill()

    if X[exog_names].isna().any().any():
        faltantes = X[exog_names].isna().sum().to_dict()
        raise ValueError(f"Exogenas faltantes: {faltantes}")

    return X[exog_names].astype(float).reset_index(drop=True)


# =========================================================
# VENTANAS (features aplanadas, no secuencias reales)
# =========================================================

def crear_ventanas(y, window, exog=None):
    """
    Para predecir componente[t]: componente[t-window:t] + (si hay exogenas)
    exog[t-window:t] + exog[t]. Las electricas ya vienen como lag168.
    """
    X, Y = [], []

    y = np.asarray(y, dtype=float)
    exog_values = np.asarray(exog, dtype=float) if exog is not None else None

    for i in range(window, len(y)):
        y_window = y[i - window:i]

        if exog_values is not None:
            exog_window = exog_values[i - window:i].reshape(-1)
            exog_actual = exog_values[i].reshape(-1)
            features = np.concatenate([y_window, exog_window, exog_actual])
        else:
            features = y_window

        X.append(features)
        Y.append(y[i])

    return np.asarray(X), np.asarray(Y)


def reshape_lstm_features(X):
    return X.reshape(X.shape[0], 1, X.shape[1])


# =========================================================
# LSTM PARA TENDENCIA
# =========================================================

def construir_lstm(params, input_dim):
    model = Sequential()
    model.add(Input(shape=(1, input_dim)))

    if params["n_layers"] == 1:
        model.add(LSTM(params["units"]))
    else:
        model.add(LSTM(params["units"], return_sequences=True))
        model.add(Dropout(params["dropout"]))
        model.add(LSTM(params["units_2"]))

    model.add(Dropout(params["dropout"]))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="mse")

    return model


def tunear_lstm(train_component, nombre_serie, exog_train, n_trials, output_dir, forecast_horizon):
    X, y = crear_ventanas(train_component, WINDOW, exog=exog_train)

    val_size = forecast_horizon
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = reshape_lstm_features(scaler_x.fit_transform(X_train))
    X_val_s = reshape_lstm_features(scaler_x.transform(X_val))

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    def objective(trial):
        tf.keras.backend.clear_session()

        params = {
            "n_layers": trial.suggest_categorical("n_layers", [1, 2]),
            "units": trial.suggest_categorical("units", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        params["units_2"] = trial.suggest_categorical("units_2", [32, 64, 128]) if params["n_layers"] == 2 else 0

        model = construir_lstm(params, input_dim=X_train_s.shape[2])
        early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

        model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=EPOCHS, batch_size=params["batch_size"], verbose=0, callbacks=[early],
        )

        pred_s = model.predict(X_val_s, verbose=0)
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        return mean_absolute_error(y_val, pred)

    db_path = os.path.join(output_dir, f"{nombre_serie}_LSTM_trend_EXOG_ALL_optuna.db")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{nombre_serie}_LSTM_trend_EXOG_ALL",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    print(f"      LSTM trend trials completos: {completed}")
    print(f"      LSTM trend trials restantes: {remaining}")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    return study.best_params, study.best_value, study.trials_dataframe()


def entrenar_lstm_final(train_component, params, exog_train, forecast_horizon):
    X, y = crear_ventanas(train_component, WINDOW, exog=exog_train)

    val_size = forecast_horizon
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = reshape_lstm_features(scaler_x.fit_transform(X_train))
    X_val_s = reshape_lstm_features(scaler_x.transform(X_val))

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    params = params.copy()
    params["units_2"] = params.get("units_2", 0)

    model = construir_lstm(params, input_dim=X_train_s.shape[2])
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS, batch_size=params["batch_size"], verbose=0, callbacks=[early],
    )

    del X, y, X_train, y_train, X_val, y_val, X_train_s, X_val_s, y_train_s, y_val_s
    gc.collect()

    return model, scaler_x, scaler_y


def forecast_recursivo_lstm(model, train_component, horizon, scaler_x, scaler_y, exog_hist=None, exog_future=None):
    historial = list(np.asarray(train_component, dtype=float))
    preds = []

    if exog_hist is not None:
        exog_total = pd.concat([exog_hist.reset_index(drop=True), exog_future.reset_index(drop=True)], ignore_index=True)
        exog_values = np.asarray(exog_total, dtype=float)
    else:
        exog_values = None

    for step in range(horizon):
        y_window = np.asarray(historial[-WINDOW:], dtype=float)

        if exog_values is not None:
            end_idx = len(train_component) + step
            exog_window = exog_values[end_idx - WINDOW:end_idx].reshape(-1)
            exog_actual = exog_values[end_idx].reshape(-1)
            x = np.concatenate([y_window, exog_window, exog_actual]).reshape(1, -1)
        else:
            x = y_window.reshape(1, -1)

        x_s = reshape_lstm_features(scaler_x.transform(x))
        pred_s = model.predict(x_s, verbose=0)
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1))[0, 0]

        preds.append(pred)
        historial.append(pred)

    return np.asarray(preds)


# =========================================================
# FCNN PARA ESTACIONALIDAD
# =========================================================

def construir_fcnn(params, input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(params["units_1"], activation="relu"))
    model.add(Dropout(params["dropout"]))

    if params["n_layers"] == 2:
        model.add(Dense(params["units_2"], activation="relu"))
        model.add(Dropout(params["dropout"]))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="mse")

    return model


def tunear_fcnn(train_component, nombre_serie, exog_train, n_trials, output_dir, forecast_horizon):
    X, y = crear_ventanas(train_component, WINDOW, exog=exog_train)

    val_size = forecast_horizon
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    def objective(trial):
        tf.keras.backend.clear_session()

        params = {
            "n_layers": trial.suggest_categorical("n_layers", [1, 2]),
            "units_1": trial.suggest_categorical("units_1", [32, 64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        params["units_2"] = trial.suggest_categorical("units_2", [16, 32, 64, 128]) if params["n_layers"] == 2 else 0

        model = construir_fcnn(params, input_dim=X_train_s.shape[1])
        early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

        model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=EPOCHS, batch_size=params["batch_size"], verbose=0, callbacks=[early],
        )

        pred_s = model.predict(X_val_s, verbose=0)
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        return mean_absolute_error(y_val, pred)

    db_path = os.path.join(output_dir, f"{nombre_serie}_FCNN_seasonal_EXOG_ALL_optuna.db")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{nombre_serie}_FCNN_seasonal_EXOG_ALL",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    print(f"      FCNN seasonal trials completos: {completed}")
    print(f"      FCNN seasonal trials restantes: {remaining}")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    return study.best_params, study.best_value, study.trials_dataframe()


def entrenar_fcnn_final(train_component, params, exog_train, forecast_horizon):
    X, y = crear_ventanas(train_component, WINDOW, exog=exog_train)

    val_size = forecast_horizon
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    params = params.copy()
    params["units_2"] = params.get("units_2", 0)

    model = construir_fcnn(params, input_dim=X_train_s.shape[1])
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS, batch_size=params["batch_size"], verbose=0, callbacks=[early],
    )

    del X, y, X_train, y_train, X_val, y_val, X_train_s, X_val_s, y_train_s, y_val_s
    gc.collect()

    return model, scaler_x, scaler_y


def forecast_recursivo_fcnn(model, train_component, horizon, scaler_x, scaler_y, exog_hist=None, exog_future=None):
    historial = list(np.asarray(train_component, dtype=float))
    preds = []

    if exog_hist is not None:
        exog_total = pd.concat([exog_hist.reset_index(drop=True), exog_future.reset_index(drop=True)], ignore_index=True)
        exog_values = np.asarray(exog_total, dtype=float)
    else:
        exog_values = None

    for step in range(horizon):
        y_window = np.asarray(historial[-WINDOW:], dtype=float)

        if exog_values is not None:
            end_idx = len(train_component) + step
            exog_window = exog_values[end_idx - WINDOW:end_idx].reshape(-1)
            exog_actual = exog_values[end_idx].reshape(-1)
            x = np.concatenate([y_window, exog_window, exog_actual]).reshape(1, -1)
        else:
            x = y_window.reshape(1, -1)

        x_s = scaler_x.transform(x)
        pred_s = model.predict(x_s, verbose=0)
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1))[0, 0]

        preds.append(pred)
        historial.append(pred)

    return np.asarray(preds)


# =========================================================
# AR RESIDUO
# =========================================================

def seleccionar_ar_por_aic(y, max_lag=168):
    y = pd.Series(y).astype(float).dropna().to_numpy()

    resultados = []
    mejor_modelo, mejor_lag, mejor_aic = None, None, np.inf

    for lag in range(1, max_lag + 1):
        try:
            # Sin `old_names`: en statsmodels >= 0.15 ese argumento ya no existe (TypeError en TODOS los candidatos), y en las
            # versiones que lo aceptan su valor por defecto ya es False, por lo que el ajuste es identico.
            model = AutoReg(y, lags=lag, trend="c")
            fitted = model.fit()

            resultados.append({"lag": lag, "AIC": fitted.aic, "BIC": fitted.bic})

            if np.isfinite(fitted.aic) and fitted.aic < mejor_aic:
                mejor_aic = fitted.aic
                mejor_lag = lag
                mejor_modelo = fitted

        except Exception as e:
            resultados.append({"lag": lag, "AIC": np.nan, "BIC": np.nan, "error": f"{type(e).__name__}: {str(e)[:140]}"})

    if mejor_modelo is None:
        raise RuntimeError(_explicar_fallo_ar(resultados, len(y), max_lag))

    return mejor_modelo, mejor_lag, pd.DataFrame(resultados)


def _explicar_fallo_ar(resultados, n_obs, max_lag):
    """Mensaje con la CAUSA cuando ningun candidato AR resulta valido (antes solo decia que ninguno lo era)."""
    df = pd.DataFrame(resultados)
    errores = df["error"].dropna() if "error" in df.columns else pd.Series(dtype=str)
    no_finitos = int(((df["error"].isna() if "error" in df.columns else True) & ~np.isfinite(df["AIC"])).sum()) if len(df) else 0
    partes = [f"No se pudo ajustar ningun AR valido (residuo con {n_obs} observaciones, {len(df)} de {max_lag} candidatos evaluados)."]
    if len(errores):
        partes.append(f"{len(errores)} candidato(s) lanzaron excepcion; causas (conteo): " + "; ".join(
            f"{k} x{v}" for k, v in errores.value_counts().head(3).items()))
    if no_finitos:
        partes.append(f"{no_finitos} candidato(s) dieron AIC no finito (residuo constante o con valores no finitos).")
    return " ".join(partes)


def forecast_ar_resid(resid, horizon):
    modelo, lag_optimo, df_lags = seleccionar_ar_por_aic(resid, max_lag=MAX_LAG_AR)

    pred = modelo.predict(start=len(resid), end=len(resid) + horizon - 1, dynamic=False)

    return np.asarray(pred), modelo, lag_optimo, df_lags


def descomponer_stl(train):
    stl = STL(pd.Series(train).astype(float), period=STL_PERIOD, robust=True)
    res = stl.fit()
    return np.asarray(res.trend), np.asarray(res.seasonal), np.asarray(res.resid), res


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


def _guardar_avance_csv(resultados, output_dir):
    if resultados.series:
        frames = []
        for bloque in resultados.series:
            frames.append(pd.DataFrame({
                "serie": bloque["serie"],
                "fecha": pd.to_datetime(bloque["fecha"], errors="coerce"),
                "tipo": bloque["tipo"],
                "subset": bloque["subset"],
                "modelo": bloque["modelo"],
                "valor": bloque["valor"],
            }))
        df_series = pd.concat(frames, ignore_index=True)
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
        print(f"OK Trials guardados (avance): {len(df_trials):,} registros")

    if resultados.config_usada:
        df_config = pd.DataFrame(resultados.config_usada)
        df_config["region"] = df_config["serie"].map(_region_de_serie)
        escribir_csv_atomico(df_config, os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config):,} registros")


# =========================================================
# EVALUAR UNA REGION
# =========================================================

def evaluar_region(region, df, exog_region, exog_names, train_hours, forecast_horizon, n_trials, output_dir, resultados):
    nombre_serie = f"{region}_DEMANDA"

    print("\n" + "=" * 80)
    print(f"Evaluando ensemble por componentes EXOG: {nombre_serie}")
    print("=" * 80)

    serie, fechas = extraer_serie_horaria(df, COL_DEMANDA)
    X_exog = alinear_exogenas_con_fechas(fechas, exog_region, exog_names)

    requeridas = train_hours + forecast_horizon
    if len(serie) < requeridas:
        print(f"AVISO: Serie insuficiente: {nombre_serie}")
        return

    serie_reciente = serie[-requeridas:]
    fechas_recientes = fechas[-requeridas:]
    X_reciente = X_exog.iloc[-requeridas:].reset_index(drop=True)

    train = serie_reciente[:-forecast_horizon]
    test = serie_reciente[-forecast_horizon:]
    fechas_test = fechas_recientes[-forecast_horizon:]

    X_train = X_reciente.iloc[:-forecast_horizon].reset_index(drop=True)
    X_test = X_reciente.iloc[-forecast_horizon:].reset_index(drop=True)

    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")
    print(f"Exog train: {X_train.shape}")
    print(f"Exog test:  {X_test.shape}")

    print("\nDefinicion exogenas:")
    for col in exog_names:
        if col in ["Temperaturas", "IGAE"]:
            print(f"   {col}: contemporanea")
        else:
            print(f"   {col}: lag {ELECTRIC_LAG}h")

    # STL
    trend, seasonal, resid, stl_res = descomponer_stl(train)
    del stl_res
    gc.collect()

    # LSTM TENDENCIA + EXOG
    print("\n   Optuna LSTM para tendencia con exogenas...")

    params_trend, best_mae_trend, trials_trend = tunear_lstm(
        trend, nombre_serie, X_train, n_trials, output_dir, forecast_horizon
    )
    print(f"      Mejor MAE trend validacion: {best_mae_trend:.4f}")
    print(f"      Params trend: {params_trend}")

    lstm_trend, sx_trend, sy_trend = entrenar_lstm_final(trend, params_trend, X_train, forecast_horizon)

    trend_pred = forecast_recursivo_lstm(
        lstm_trend, trend, horizon=len(test), scaler_x=sx_trend, scaler_y=sy_trend,
        exog_hist=X_train, exog_future=X_test,
    )

    del lstm_trend, sx_trend, sy_trend
    tf.keras.backend.clear_session()
    gc.collect()

    # FCNN ESTACIONALIDAD + EXOG
    print("\n   Optuna FCNN para estacionalidad con exogenas...")

    params_seasonal, best_mae_seasonal, trials_seasonal = tunear_fcnn(
        seasonal, nombre_serie, X_train, n_trials, output_dir, forecast_horizon
    )
    print(f"      Mejor MAE seasonal validacion: {best_mae_seasonal:.4f}")
    print(f"      Params seasonal: {params_seasonal}")

    fcnn_season, sx_season, sy_season = entrenar_fcnn_final(seasonal, params_seasonal, X_train, forecast_horizon)

    seasonal_pred = forecast_recursivo_fcnn(
        fcnn_season, seasonal, horizon=len(test), scaler_x=sx_season, scaler_y=sy_season,
        exog_hist=X_train, exog_future=X_test,
    )

    del fcnn_season, sx_season, sy_season
    tf.keras.backend.clear_session()
    gc.collect()

    # AR RESIDUO
    print("\n   Ajustando AR sobre residuo por AIC...")

    resid_pred, ar_resid, lag_resid, df_lags_resid = forecast_ar_resid(resid, horizon=len(test))
    del ar_resid
    gc.collect()

    print(f"      AR resid lag optimo: {lag_resid}")

    # RECONSTRUCCION FINAL
    pred_final = trend_pred + seasonal_pred + resid_pred
    metricas = calcular_metricas(test, pred_final)

    print(
        f"\n   ENSEMBLE EXOG final: MAPE={metricas['MAPE']:.2f}% | "
        f"sMAPE={metricas['sMAPE']:.2f}% | MAE={metricas['MAE']:.2f} | RMSE={metricas['RMSE']:.2f}"
    )

    # GUARDAR: series (real, final, y 3 componentes)
    resultados.series.append({
        "serie": nombre_serie, "fecha": fechas, "tipo": "real",
        "subset": "completo", "modelo": "real", "valor": serie,
    })
    resultados.series.append({
        "serie": nombre_serie, "fecha": fechas_test, "tipo": "prediccion",
        "subset": "test", "modelo": NOMBRE_MODELO_FINAL, "valor": pred_final,
    })
    resultados.series.append({
        "serie": nombre_serie, "fecha": fechas_test, "tipo": "componente_pred",
        "subset": "test", "modelo": "LSTM_trend_EXOG_ALL_Lag168", "valor": trend_pred,
    })
    resultados.series.append({
        "serie": nombre_serie, "fecha": fechas_test, "tipo": "componente_pred",
        "subset": "test", "modelo": "FCNN_seasonal_EXOG_ALL_Lag168", "valor": seasonal_pred,
    })
    resultados.series.append({
        "serie": nombre_serie, "fecha": fechas_test, "tipo": "componente_pred",
        "subset": "test", "modelo": "AR_resid", "valor": resid_pred,
    })

    resultados.metrics.append({
        "serie": nombre_serie, "modelo": NOMBRE_MODELO_FINAL,
        "MAE": metricas["MAE"], "RMSE": metricas["RMSE"], "MAPE": metricas["MAPE"], "sMAPE": metricas["sMAPE"],
    })

    resultados.config_usada.append({
        "serie": nombre_serie,
        "modelo": NOMBRE_MODELO_FINAL,
        "trend_model": "LSTM_EXOG",
        "seasonal_model": "FCNN_EXOG",
        "resid_model": "AR_AIC",
        "exogenas": ",".join(exog_names),
        "lag_electricas": ELECTRIC_LAG,
        "resid_lag_optimo": lag_resid,
        "trend_best_val_MAE": best_mae_trend,
        "seasonal_best_val_MAE": best_mae_seasonal,
        "trend_params": json.dumps(params_trend),
        "seasonal_params": json.dumps(params_seasonal),
        "train_horas": train_hours,
        "test_horas": forecast_horizon,
        "window": WINDOW,
        "trials": n_trials,
    })

    trials_trend["serie"] = nombre_serie
    trials_trend["modelo"] = "LSTM_trend"
    resultados.trials.append(trials_trend)

    trials_seasonal["serie"] = nombre_serie
    trials_seasonal["modelo"] = "FCNN_seasonal"
    resultados.trials.append(trials_seasonal)

    df_lags_resid = df_lags_resid.copy()
    df_lags_resid["serie"] = nombre_serie
    df_lags_resid["modelo"] = "AR_resid"
    resultados.trials.append(df_lags_resid)

    tf.keras.backend.clear_session()
    gc.collect()


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run(
    exogenas_globales: dict,
    regions_all: list,
    train_hours: int = TRAIN_LAST_HOURS_DEFAULT,
    forecast_horizon: int = FORECAST_HORIZON_DEFAULT,
    exog_cols: list = None,
    optuna_n_trials: int = N_TRIALS_DEFAULT,
    data_dir: str = "/content",
    output_dir: str = ".",
):
    """
    Equivalente al script de nivel de modulo de la celda 60. Devuelve
    (series_df, metricas_df, trials_df, config_usada_df).
    """
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    exog_cols_canonico = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)
    exog_names = [CANONICAL_TO_INTERNAL[c] for c in exog_cols_canonico]

    resultados = _ResultsAccumulator()

    # Checkpoint por region: series.csv aqui se reconstruye por bloques
    # (fecha/valor como arreglo) porque `_guardar_avance_csv` de este
    # modulo no tiene la guarda `np.atleast_1d` -- ver el mismo comentario
    # en sarimax_model.py. Esto tambien preserva las filas
    # "componente_pred" (LSTM_trend/FCNN_seasonal/AR_resid) sin tratamiento
    # especial: se reconstruyen como cualquier otro bloque agrupado por
    # (serie, tipo, subset, modelo).
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all, forecast_horizon=forecast_horizon,
        requiere_trials=True, requiere_config_usada=True, formato_series="bloques",
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print("ENSEMBLE STL + LSTM + FCNN + AR CON EXOGENAS GENERALIZADAS")
    print("=" * 80)
    print(f"Train: {train_hours} h")
    print(f"Test: {forecast_horizon} h")
    print(f"Window: {WINDOW} h")

    print("\nExogenas activas:")
    for exog in exog_names:
        print(f"   - {exog}")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        try:
            exog_region = construir_matriz_exogena_region(region, exogenas_globales, exog_names, data_dir)

            evaluar_region(
                region, df, exog_region, exog_names, train_hours, forecast_horizon,
                optuna_n_trials, output_dir, resultados,
            )

        except Exception as e:
            print(f"Error general en {region}: {type(e).__name__}: {e}")
            registrar_error(region, e)

        print("\nGuardando avance...")
        _guardar_avance_csv(resultados, output_dir)

    series_df = pd.DataFrame()
    if resultados.series:
        frames = []
        for bloque in resultados.series:
            frames.append(pd.DataFrame({
                "serie": bloque["serie"],
                "fecha": pd.to_datetime(bloque["fecha"], errors="coerce"),
                "tipo": bloque["tipo"],
                "subset": bloque["subset"],
                "modelo": bloque["modelo"],
                "valor": bloque["valor"],
            }))
        series_df = pd.concat(frames, ignore_index=True)
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

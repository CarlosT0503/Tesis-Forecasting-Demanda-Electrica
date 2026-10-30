"""
Pipeline LSTM multivariada sobre residuos + Tendencia + Estacionalidad
(demanda) -- COMBINACION NUEVA, NO extraccion de un pipeline completo de
legacy.

*** No existe en el notebook legacy un pipeline que combine tendencia lineal
simple + estacionalidad repetida simple (celda 64) con un residuo modelado
por LSTM multivariada usando el catalogo/tratamiento de exogenas de la LSTM
directa (celda 58). La celda 60 (Ensemble) SI usa una LSTM para modelar un
componente STL, pero (a) esa LSTM modela la TENDENCIA, no el residuo, (b) la
estacionalidad ahi la modela una FCNN (no una formula de estacionalidad
repetida), y (c) las exogenas ahi se preparan con
`preparar_exogena_lag168`/`construir_matriz_exogena_region`, un tratamiento
distinto (desplaza el timestamp 168h) al de la celda 58. ***

Este modulo combina piezas de DOS celdas legacy independientes:

  1. Tendencia + estacionalidad: copia EXACTA de
     `descomponer_stl`/`forecast_tendencia_lineal`/
     `forecast_estacionalidad_repetida` de la celda 64, identica a como se
     usan en `naive_trend_seasonal_model.py` y
     `ar_resid_trend_seasonal_model.py`.

  2. Modelo del residuo: la arquitectura LSTM-por-componente-STL de la
     celda 60 (`crear_ventanas`, `reshape_lstm_features`, `construir_lstm`,
     `tunear_lstm`, `entrenar_lstm_final`, `forecast_recursivo_lstm`),
     copiada EXACTA de `ensemble_stl.py`, mismas funciones, mismo `WINDOW`
     =168, mismo objetivo de Optuna (MAE), mismo patron de trials
     acumulados (`remaining = n_trials - completados`). Ahi donde el
     Ensemble original la usa para modelar la TENDENCIA, aqui se retarget al
     RESIDUO (`train_component=resid` en vez de `trend`) -- las funciones
     son genericas sobre que componente reciben, no requieren cambios.

  3. Catalogo y tratamiento de exogenas: copia EXACTA de `lstm_direct.py`
     (celda 58) -- `EXOG_COLS_DEFAULT`, `EXOG_SOURCE_MAP`,
     `EXOG_CONOCIDAS_FUTURO`, `EXOG_LAG_SEMANAL`, `LAG_EXOG_FUTURO=168`,
     `merge_exogenas`, `alinear_exogenas_a_fechas`,
     `construir_future_exog_directa` -- NO el tratamiento propio del
     Ensemble (`preparar_exogena_lag168`, que desplaza el timestamp). Esta
     sustitucion fue instruida explicitamente por el usuario el 2026-08-09:
     "usa el mismo conjunto y tratamiento de exógenas que la LSTM
     multivariada vigente salvo que exista una razón técnica concreta".

Prediccion final: tendencia_forecast + estacionalidad_forecast +
residuo_pronosticado(LSTM).

Decisiones tecnicas tomadas para esta combinacion (autorizadas explicitamente
por las instrucciones de autonomia; se documentan aqui por ser no triviales):

  - Extraccion de la serie de demanda (`extraer_serie_horaria`): se usa la
    version de `naive_trend_seasonal_model.py` (con `convertir_hora_0_23`,
    que acepta tanto Hora 1-24 como 0-23, y hace
    `drop_duplicates(keep="last")`), NO la version mas simple de
    `lstm_direct.py` (que asume Hora siempre 1-24) -- por consistencia con
    el resto de la familia STL de esta tarea (#2, #3) y porque es la version
    mas robusta de las dos disponibles.

  - Alineacion de exogenas: se construye el bloque de exogenas alineado a la
    serie COMPLETA reciente (`alinear_exogenas_a_fechas` sobre toda la
    ventana train+test), y el bloque futuro que ve la LSTM del residuo
    (`exog_future` en `forecast_recursivo_lstm`) se construye con
    `construir_future_exog_directa` (misma funcion de la celda 58): Temperatura
    e IGAE usan el valor contemporaneo del horizonte (exogenas "conocidas a
    futuro", igual que en LSTM directa/XGBoost), Generacion/Importacion/
    Exportacion usan el valor de esa hora - 168h. El bloque historico
    (`exog_hist`, usado durante entrenamiento y como contexto de la ventana
    recursiva) usa los valores observados contemporaneos, sin ningun lag --
    igual que hace `ensemble_stl.py` con su propio `exog_train`. Es decir: el
    tratamiento especial (conocida-a-futuro vs. lag168) solo aplica al tramo
    futuro que la LSTM aun no observo, exactamente como en `lstm_direct.py`.

  - Metricas: se reutiliza la variante de la celda 64 (mascara `isfinite`,
    con guarda contra entrada vacia), la misma familia usada en
    `naive_trend_seasonal_model.py` / `ar_resid_trend_seasonal_model.py` --
    por consistencia dentro de esta sub-familia de combinaciones nuevas
    basadas en STL, NO la variante sin guarda de `ensemble_stl.py` (de donde
    viene la arquitectura LSTM) ni la de `lstm_direct.py` (de donde viene el
    catalogo de exogenas).

  - `series.csv`: solo se guardan las filas "real" y la prediccion FINAL
    combinada (tendencia+estacionalidad+residuo) -- NO se guardan filas de
    componente por separado (`componente_pred`), a diferencia de
    `ensemble_stl.py`. Se prefirio asi por consistencia con
    `naive_trend_seasonal_model.py`/`ar_resid_trend_seasonal_model.py`
    (que tampoco desglosan componentes), no porque el desglose fuera
    incorrecto.

  - N_TRIALS_DEFAULT=5 (no 10): se hereda de `ensemble_stl.py`, de donde
    viene la maquinaria de tuning/entrenamiento de la LSTM que aqui se
    reutiliza sin cambios, no del `N_TRIALS_LSTM_DEFAULT=10` de
    `lstm_direct.py` (de donde solo se toma el catalogo de exogenas, no su
    logica de tuning).

  - Train/test: ventana FIJA 3600h/168h, igual que el resto de la familia
    STL de esta tarea (#2, #3) -- NO la ventana dinamica de AR standalone ni
    la ventana de 2160h de `lstm_direct.py`.

  - Univariado NO: a diferencia de los otros 3 modelos nuevos de esta tarea,
    este SI requiere exogenas (son el input de la LSTM del residuo); si
    `exog_cols` resuelve a una lista vacia, `merge_exogenas` (copiado de la
    celda 58) ya lanza `ValueError: No hay exogenas activas.` -- no se anadio
    ninguna validacion adicional.

LIMITACION CONOCIDA: este modulo requiere tensorflow y optuna, ninguno de
los dos instalado en el entorno local de desarrollo. No pudo ejecutarse un
smoke test real end-to-end localmente (a diferencia de los otros 3 modelos
de esta tarea) -- solo se verifico que compila (`py_compile`) y se revisaron
manualmente las formas/dimensiones de los arreglos pasados entre funciones.
Debe correrse el smoke test en Colab, igual que se hizo con LSTM directa,
FCNN, Ensemble y LightGBM.
"""

import os
import gc

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf

from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..atomic_io import escribir_csv_atomico
from ..errores_log import registrar_error

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

TRAIN_LAST_HOURS_DEFAULT = 24 * 30 * 5   # 3600 horas, igual al resto de la familia STL
FORECAST_HORIZON_DEFAULT = 24 * 7        # 168 horas

STL_PERIOD = 168
WINDOW = 168          # identico a ensemble_stl.py
EPOCHS = 60            # identico a ensemble_stl.py
SEED = 42              # identico a ensemble_stl.py
N_TRIALS_DEFAULT = 5   # identico a ensemble_stl.py (no el 10 de lstm_direct.py)

# Catalogo y tratamiento de exogenas -- copia exacta de lstm_direct.py (celda 58)
EXOG_COLS_DEFAULT = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
EXOG_CATALOGO = list(EXOG_COLS_DEFAULT)

EXOG_SOURCE_MAP = {
    "Temperatura": "Temperaturas_H",
    "IGAE": "IGAE_H",
}

EXOG_CONOCIDAS_FUTURO = ["Temperatura", "IGAE"]
EXOG_LAG_SEMANAL = ["Generacion", "Importacion", "Exportacion"]
LAG_EXOG_FUTURO = 168

NOMBRE_MODELO = "LSTM_Resid_Trend_Seasonal"


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
# CARGAR REGIONES / EXTRAER SERIE (version robusta, igual a
# naive_trend_seasonal_model.py / ar_resid_trend_seasonal_model.py)
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
    desfase sistematico de 1 hora respecto a la demanda de xgboost_model.py/
    lightgbm_model.py/lstm_direct.py/sarimax_model.py/naive_model.py/
    naive_trend_model.py/ar_model.py (que siempre restan 1, sin
    condicion). Ahora es incondicional e igual a esos modulos. Ver tambien
    `_hora_a_0_23()` mas abajo (exogenas) -- ambas funciones deben quedar
    en la MISMA convencion para que `alinear_exogenas_a_fechas()` no
    desalinee demanda vs exogenas.
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
# EXOGENAS (copia exacta de lstm_direct.py / celda 58)
# =========================================================

def _hora_a_0_23(hora):
    """
    Hora viene conceptualmente en 1..24 -> resta 1, SIEMPRE (mismo fix que
    `convertir_hora_0_23()` arriba, ver su docstring). Aplica a las
    exogenas normalizadas por `_normalizar_exogena()` -- demanda y
    exogenas quedan garantizadas en la MISMA convencion horaria, para que
    el merge por 'datetime' en `alinear_exogenas_a_fechas()` no las
    desalinee.
    """
    return pd.to_numeric(hora, errors="coerce").astype(float) - 1


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
# VENTANAS + LSTM POR COMPONENTE (copia exacta de ensemble_stl.py,
# retargeted aqui al RESIDUO en vez de a la tendencia)
# =========================================================

def crear_ventanas(y, window, exog=None):
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

    db_path = os.path.join(output_dir, f"{nombre_serie}_LSTM_resid_EXOG_ALL_optuna.db")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{nombre_serie}_LSTM_resid_EXOG_ALL",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    print(f"      LSTM resid trials completos: {completed}")
    print(f"      LSTM resid trials restantes: {remaining}")

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
    `np.atleast_1d` normaliza ambos casos antes de construir cada bloque.
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
        print(f"OK Trials (LSTM residuo) guardados (avance): {len(df_trials):,} registros")

    if resultados.config_usada:
        df_config = pd.DataFrame(resultados.config_usada)
        df_config["region"] = df_config["serie"].map(_region_de_serie)
        escribir_csv_atomico(df_config, os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config):,} registros")


# =========================================================
# EVALUAR REGION
# =========================================================

def evaluar_region(region, df, exogenas_globales, exog_cols, train_hours, forecast_horizon, n_trials, data_dir, output_dir, resultados):
    nombre_serie = f"{region}_DEMANDA"

    print("\n" + "=" * 80)
    print(f"Serie: {nombre_serie}")
    print("=" * 80)

    serie, fechas = extraer_serie_horaria(df, COL_DEMANDA)

    exogenas_df = merge_exogenas(region, exogenas_globales, exog_cols, data_dir)
    exog_full = alinear_exogenas_a_fechas(fechas, exogenas_df, exog_cols)

    requeridas = train_hours + forecast_horizon
    if len(serie) < requeridas:
        print(f"AVISO: {nombre_serie} tiene {len(serie):,} horas; se requieren al menos {requeridas:,}.")
        return

    serie_reciente = serie[-requeridas:]
    fechas_recientes = fechas[-requeridas:]
    exog_reciente = exog_full.iloc[-requeridas:].reset_index(drop=True)

    train = serie_reciente[:-forecast_horizon]
    test = serie_reciente[-forecast_horizon:]
    fechas_test = fechas_recientes[-forecast_horizon:]

    X_train = exog_reciente.iloc[:-forecast_horizon].reset_index(drop=True)

    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")
    print(f"Exog train: {X_train.shape}")

    try:
        trend, seasonal, resid, stl_res = descomponer_stl(train)
        del stl_res
        gc.collect()

        trend_forecast = forecast_tendencia_lineal(trend, horizon=len(test))
        seasonal_forecast = forecast_estacionalidad_repetida(seasonal, horizon=len(test), period=STL_PERIOD)

        X_test_future = construir_future_exog_directa(exog_reciente, origin=len(train), horizon=len(test), exog_cols=exog_cols)

        print("\n   Optuna LSTM para residuo con exogenas...")
        params_resid, best_mae_resid, trials_resid = tunear_lstm(
            resid, nombre_serie, X_train, n_trials, output_dir, len(test)
        )
        print(f"      Mejor MAE resid validacion: {best_mae_resid:.4f}")
        print(f"      Params resid: {params_resid}")

        lstm_resid, sx_resid, sy_resid = entrenar_lstm_final(resid, params_resid, X_train, len(test))

        resid_forecast = forecast_recursivo_lstm(
            lstm_resid, resid, horizon=len(test), scaler_x=sx_resid, scaler_y=sy_resid,
            exog_hist=X_train, exog_future=X_test_future,
        )

        del lstm_resid, sx_resid, sy_resid
        tf.keras.backend.clear_session()
        gc.collect()

        pred_final = trend_forecast + seasonal_forecast + resid_forecast

        metricas = calcular_metricas(test, pred_final)
        print(f"      {NOMBRE_MODELO} MAPE={metricas['MAPE']:.2f}%")

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

        resultados.config_usada.append({
            "serie": nombre_serie,
            "modelo": NOMBRE_MODELO,
            "resid_model": "LSTM_EXOG",
            "resid_params": str(params_resid),
            "resid_best_val_MAE": best_mae_resid,
            "exogenas": ",".join(exog_cols),
            "lag_electricas": LAG_EXOG_FUTURO,
            "window": WINDOW,
            "stl_period": STL_PERIOD,
            "train_horas": len(train),
            "trials": n_trials,
        })

        trials_resid = trials_resid.copy()
        trials_resid["serie"] = nombre_serie
        trials_resid["modelo"] = "LSTM_resid"
        resultados.trials.append(trials_resid)

    except Exception as e:
        print(f"Error en {nombre_serie}: {type(e).__name__}: {e}")
        registrar_error(nombre_serie, e)

    finally:
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
    Pipeline LSTM_Resid_Trend_Seasonal nuevo (ver docstring del modulo).
    Devuelve (series_df, metricas_df, trials_df, config_usada_df).
    """
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    exog_cols = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)

    resultados = _ResultsAccumulator()

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
    print(f"PIPELINE {NOMBRE_MODELO.upper()} DEMANDA (combinacion nueva, ver docstring)")
    print("=" * 80)
    print(f"Train: {train_hours} h")
    print(f"Test: {forecast_horizon} h")
    print(f"Window LSTM: {WINDOW} h")

    print("\nExogenas activas:")
    for exog in exog_cols:
        print(f"   - {exog}")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        evaluar_region(region, df, exogenas_globales, exog_cols, train_hours, forecast_horizon, optuna_n_trials, data_dir, output_dir, resultados)

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

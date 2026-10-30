"""
Pipeline FCNN multivariada (demanda, horizonte de 1 semana).

Extraido de la celda 64 del notebook legacy ("Redes multivariadas" /
"PIPELINE FCNN MULTIVARIADA + FCNN MULTIVARIADA SOBRE RESIDUOS STL"). Esta
celda produce DOS modelos por region, ambos preservados aqui:

  1. `FCNN_Multivariada_EXOG_Lag168`: FCNN recursiva directa sobre la
     demanda (arquitectura, espacio de Optuna, forecast recursivo
     identicos al original).
  2. `STL_FCNN_Multivariada_Residuos_EXOG_Lag168`: descomposicion STL
     sobre el train; tendencia -> regresion lineal; estacionalidad ->
     patron repetido del ultimo periodo; residuo -> la MISMA arquitectura
     FCNN de (1) pero entrenada sobre los residuos; reconstruccion final =
     tendencia + estacionalidad + residuo.

Arquitectura (Dense 1-2 capas + Dropout), espacio de busqueda de Optuna,
EPOCHS=60, ventana de 168h, y el tratamiento de exogenas (Temperaturas/IGAE
contemporaneas, Generacion/Importacion/Exportacion pre-desplazadas 168h por
`preparar_exogena_lag168`, que corre el timestamp +168h en vez de calcular
un promedio de dos lags como XGBoost) son IDENTICOS al original.

Cambios mecanicos (no de logica), mismo tipo de adaptacion que en los
demas modulos:

  1. `globals()` (Temperaturas_H, IGAE_H) y el diccionario global `series`
     se sustituyen por `exogenas_globales` (parametro) y lectura directa
     de `{region}_GEN/IMP/EXP.csv` desde `data_dir`.
  2. El script de nivel de modulo se envuelve en `run(...)`.
  3. `OUTPUT_DIR`/`INCREMENTAL_DIR` fijos se sustituyen por `output_dir`.
     La celda original guardaba en subcarpetas por serie
     (`INCREMENTAL_DIR/<serie>/<serie>_FCNN_multivariada_series.csv`); aqui
     se usa el mismo patron de acumulador + series.csv/metricas.csv en la
     raiz de la carpeta del experimento que usan los demas modelos.
  4. `np.random.seed(SEED)` / `tf.random.set_seed(SEED)` se movieron de
     nivel de modulo (efecto secundario al importar) a el inicio de
     `run()`, para no alterar el estado aleatorio global solo por importar
     este archivo.
  5. `tunear_fcnn_optuna` SI capturaba resume por conteo de trials
     completados (`remaining = max(0, N_TRIALS - completados)`) -- se
     preserva identico, es distinto del resto de los modelos (que no
     cuentan trials previos).
  6. metricas.csv / config_usada.csv: la celda original guardaba todo en
     una sola tabla de metricas muy ancha (MAE, RMSE, MAPE, sMAPE,
     best_val_sMAPE, params, exogenas, train_horas, test_horas, window,
     stl_period, trials, lag_electricas). Aqui se separa en metricas.csv
     (desempeno: MAE/RMSE/MAPE/sMAPE) y config_usada.csv (el resto) --
     mismo criterio de separacion que se aplico a XGBoost. Ningun valor
     se descarta, solo se reparte en dos archivos.
  7. trials.csv: la celda original NO exportaba `study.trials_dataframe()`
     a CSV (solo quedaba en el .db de Optuna). Aqui SI se exporta -- son
     los mismos trials que ya existian en el estudio, solo expuestos en el
     mismo formato que usan los demas modelos migrados, sin cambiar ningun
     valor calculado.
  8. EXOG_NAMES usaba "Temperaturas" (plural) y sufijos "_lag168" en la
     celda original. El catalogo externo de `ExperimentConfig.exogenas` es
     el mismo para todos los modelos ("Temperatura", "Generacion", etc.);
     este modulo traduce internamente (ver CANONICAL_TO_INTERNAL).

Todo lo demas es una copia literal de la celda 64.
"""

import os
import gc
import json

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.seasonal import STL

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..atomic_io import escribir_csv_atomico
from ..errores_log import registrar_error

# =========================================================
# CONFIG POR DEFECTO (identica a la celda 64)
# =========================================================

TRAIN_LAST_HOURS_DEFAULT = 24 * 30 * 5   # 3600 horas = 5 meses
FORECAST_HORIZON_DEFAULT = 24 * 7        # 168 horas

WINDOW = 168
STL_PERIOD = 168

N_TRIALS_DEFAULT = 5
EPOCHS = 60
SEED = 42

ELECTRIC_LAG = 168

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

# Catalogo EXTERNO (canonico). Internamente se traduce a los nombres de la
# celda 64 (ver CANONICAL_TO_INTERNAL) -- OJO: las electricas ya incluyen
# el lag168 en el nombre interno, es una caracteristica del modelo, no
# solo una etiqueta.
EXOG_COLS_DEFAULT = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
EXOG_CATALOGO = list(EXOG_COLS_DEFAULT)

CANONICAL_TO_INTERNAL = {
    "Temperatura": "Temperaturas",
    "IGAE": "IGAE",
    "Generacion": "Generacion_lag168",
    "Importacion": "Importacion_lag168",
    "Exportacion": "Exportacion_lag168",
}

MODELO_DIRECTA = "FCNN_Multivariada_EXOG_Lag168"
MODELO_STL_RESIDUOS = "STL_FCNN_Multivariada_Residuos_EXOG_Lag168"


# =========================================================
# METRICAS (copia exacta de la celda 64)
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
    """Espera: fecha, hora/Hora, valor. Devuelve: datetime, <nombre>."""
    aux = df.copy()
    aux.columns = aux.columns.astype(str).str.strip()

    cols_lower = {c.lower(): c for c in aux.columns}

    for requerida in ["fecha", "hora", "valor"]:
        if requerida not in cols_lower:
            raise ValueError(f"{nombre}: no existe columna {requerida}")

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


def preparar_exogena_lag168(df, nombre_salida):
    """
    Convierte una serie electrica X_t en X_lag168[t] = X[t-168], desplazando
    el timestamp 168h hacia adelante -- la fila de tiempo t contiene
    exclusivamente informacion conocida en t-168.
    """
    aux = preparar_exogena_horaria(df, nombre_salida)
    aux["datetime"] = aux["datetime"] + pd.to_timedelta(ELECTRIC_LAG, unit="h")
    return aux


# =========================================================
# CONSTRUIR MATRIZ EXOGENA POR REGION
# =========================================================

def construir_matriz_exogena_region(region, exogenas_globales, exog_names, data_dir):
    """Temperaturas: contemporanea. IGAE: contemporaneo. Gen/Imp/Exp: lag168."""
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
        raise ValueError("No hay ninguna exogena activa.")

    exog = dfs[0]
    for df_next in dfs[1:]:
        exog = exog.merge(df_next, on="datetime", how="outer")

    exog = exog.sort_values("datetime").reset_index(drop=True)
    exog[exog_names] = exog[exog_names].ffill().bfill()

    print(f"\nExogenas {region}:")
    for col in exog_names:
        print(f"   {col:25s} | {exog[col].notna().sum():,} valores")
    print(f"Rango: {exog['datetime'].min()} -> {exog['datetime'].max()}")

    return exog


# =========================================================
# ALINEAR EXOGENAS CON DEMANDA
# =========================================================

def alinear_exogenas_con_fechas(fechas, exog_region, exog_names):
    base = pd.DataFrame({"datetime": pd.to_datetime(fechas)})

    X = base.merge(exog_region[["datetime"] + exog_names], on="datetime", how="left")
    X[exog_names] = X[exog_names].ffill().bfill()

    if X[exog_names].isna().any().any():
        faltantes = X[exog_names].isna().sum().to_dict()
        raise ValueError(f"No fue posible completar las exogenas: {faltantes}")

    return X[exog_names].astype(float).reset_index(drop=True)


# =========================================================
# DATASET SUPERVISADO MULTIVARIADO
# =========================================================

def crear_ventanas_multivariadas(y, exog, window):
    """
    Para predecir y[t]: y[t-window:t], exog[t-window:t], exog[t]. Las
    columnas electricas ya vienen transformadas a lag168 (Generacion_lag168[t]
    = Generacion[t-168]), mientras Temperaturas[t]/IGAE[t] son contemporaneas.
    """
    y = np.asarray(y, dtype=float)
    exog = np.asarray(exog, dtype=float)

    if len(y) != len(exog):
        raise ValueError(f"Longitudes incompatibles: y={len(y)}, exog={len(exog)}")

    X, Y = [], []

    for i in range(window, len(y)):
        y_lags = y[i - window:i].reshape(-1, 1)
        exog_lags = exog[i - window:i]

        ventana = np.concatenate([y_lags, exog_lags], axis=1).ravel()
        exog_actual = exog[i].ravel()

        features = np.concatenate([ventana, exog_actual])

        X.append(features)
        Y.append(y[i])

    return np.asarray(X), np.asarray(Y)


# =========================================================
# FORECAST RECURSIVO
# =========================================================

def forecast_recursivo_fcnn_multivariada(model, y_train, exog_train, exog_future, horizon, window, scaler_x, scaler_y):
    historial_y = list(np.asarray(y_train, dtype=float))
    historial_exog = [row.copy() for row in np.asarray(exog_train, dtype=float)]

    exog_future = np.asarray(exog_future, dtype=float)
    preds = []

    if len(exog_future) < horizon:
        raise ValueError("exog_future tiene menos filas que el horizonte solicitado")

    for paso in range(horizon):
        y_lags = np.asarray(historial_y[-window:], dtype=float).reshape(-1, 1)
        exog_lags = np.asarray(historial_exog[-window:], dtype=float)

        ventana = np.concatenate([y_lags, exog_lags], axis=1).ravel()
        exog_actual = exog_future[paso].ravel()

        x = np.concatenate([ventana, exog_actual]).reshape(1, -1)
        x_scaled = scaler_x.transform(x)

        pred_scaled = model.predict(x_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

        preds.append(pred)

        historial_y.append(pred)
        historial_exog.append(exog_actual.copy())

    return np.asarray(preds)


# =========================================================
# MODELO FCNN -- ARQUITECTURA ORIGINAL
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


# =========================================================
# OPTUNA
# =========================================================

def tunear_fcnn_optuna(train_y, train_exog, nombre_serie, modelo_nombre, n_trials, output_dir, forecast_horizon):
    X, y = crear_ventanas_multivariadas(train_y, train_exog, WINDOW)

    if len(X) <= forecast_horizon:
        raise ValueError("No hay suficientes ventanas para separar entrenamiento y validacion.")

    val_size = forecast_horizon

    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    input_dim = X_train.shape[1]

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

        model = construir_fcnn(params, input_dim=input_dim)
        early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

        model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=EPOCHS,
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=[early],
        )

        pred_val_s = model.predict(X_val_s, verbose=0)
        pred_val = scaler_y.inverse_transform(pred_val_s.reshape(-1, 1)).ravel()

        return smape(y_val, pred_val)

    db_path = os.path.join(output_dir, f"{nombre_serie}_{modelo_nombre}_optuna.db")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{nombre_serie}_{modelo_nombre}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    print(f"      Trials completos: {completed}")
    print(f"      Trials restantes: {remaining}")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    return study.best_params, study.best_value, study.trials_dataframe()


# =========================================================
# ENTRENAR FCNN FINAL
# =========================================================

def entrenar_fcnn_final(train_y, train_exog, params, forecast_horizon):
    X, y = crear_ventanas_multivariadas(train_y, train_exog, WINDOW)

    if len(X) <= forecast_horizon:
        raise ValueError("No hay suficientes ventanas para separar entrenamiento y validacion.")

    val_size = forecast_horizon

    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)

    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    final_params = {
        "n_layers": params["n_layers"],
        "units_1": params["units_1"],
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "units_2": params.get("units_2", 0),
    }

    model = construir_fcnn(final_params, input_dim=X_train.shape[1])
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS,
        batch_size=final_params["batch_size"],
        verbose=0,
        callbacks=[early],
    )

    return model, scaler_x, scaler_y, history


# =========================================================
# STL
# =========================================================

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


def _construir_df_series(bloques):
    """
    Construye el DataFrame de series a partir de `resultados.series`.

    `resultados.series` mezcla dos formas de bloque: filas individuales por
    prediccion (valores escalares, uno por hora -- ver `_guardar_bloque`) y
    un bloque "real" por region con `fecha`/`valor` como arreglo completo de
    la serie. `pd.DataFrame({...})` construido directamente sobre un bloque
    de puros escalares falla con "If using all scalar values, you must pass
    an index"; `np.atleast_1d` normaliza ambos casos a arreglo antes de
    construir cada bloque, sin cambiar ni un valor ni el orden de las filas
    resultantes.
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


def _guardar_bloque(resultados, nombre_serie, modelo, pred, fechas_test, metricas, best_val_smape, params,
                     exog_names, train_hours, forecast_horizon):
    for j, val in enumerate(pred):
        resultados.series.append({
            "serie": nombre_serie, "fecha": fechas_test[j], "tipo": "prediccion",
            "subset": "test", "modelo": modelo, "valor": val,
        })

    resultados.metrics.append({
        "serie": nombre_serie, "modelo": modelo,
        "MAE": metricas["MAE"], "RMSE": metricas["RMSE"], "MAPE": metricas["MAPE"], "sMAPE": metricas["sMAPE"],
    })

    resultados.config_usada.append({
        "serie": nombre_serie,
        "modelo": modelo,
        "parametros": json.dumps(params),
        "best_val_sMAPE": best_val_smape,
        "horizonte_usado": f"{forecast_horizon}_horas",
        "train_horas": train_hours,
        "test_horas": forecast_horizon,
        "window": WINDOW,
        "stl_period": STL_PERIOD,
        "lag_electricas": ELECTRIC_LAG,
        "exogenas": ", ".join(exog_names),
    })


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
        print(f"OK Trials Optuna guardados (avance): {len(df_trials):,} registros")

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
    print(f"Evaluando {nombre_serie}")
    print("=" * 80)

    serie, fechas = extraer_serie_horaria(df, COL_DEMANDA)
    X_exog = alinear_exogenas_con_fechas(fechas, exog_region, exog_names)

    requeridas = train_hours + forecast_horizon
    if len(serie) < requeridas:
        print(f"AVISO: {nombre_serie} tiene {len(serie):,} horas; se requieren al menos {requeridas:,}.")
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
    print(f"Exogenas train: {X_train.shape}")
    print(f"Exogenas test:  {X_test.shape}")

    print("\nDefinicion exogenas:")
    for col in exog_names:
        if col in ["Temperaturas", "IGAE"]:
            print(f"   {col}: contemporanea")
        else:
            print(f"   {col}: lag {ELECTRIC_LAG}h")

    huvo_resultado = False

    # 1. FCNN MULTIVARIADA SOBRE DEMANDA
    try:
        print("\n   Optuna FCNN multivariada sobre demanda...")

        best_params, best_smape, trials_df = tunear_fcnn_optuna(
            train_y=train, train_exog=X_train, nombre_serie=nombre_serie,
            modelo_nombre="FCNN_multivariada", n_trials=n_trials, output_dir=output_dir,
            forecast_horizon=forecast_horizon,
        )

        print(f"      Mejor sMAPE validacion: {best_smape:.4f}%")
        print(f"      Params: {best_params}")

        model, scaler_x, scaler_y, history = entrenar_fcnn_final(train, X_train, best_params, forecast_horizon)

        pred = forecast_recursivo_fcnn_multivariada(
            model=model, y_train=train, exog_train=X_train, exog_future=X_test,
            horizon=len(test), window=WINDOW, scaler_x=scaler_x, scaler_y=scaler_y,
        )

        metricas = calcular_metricas(test, pred)
        print(f"      FCNN multivariada MAPE={metricas['MAPE']:.2f}%")

        trials_df["serie"] = nombre_serie
        trials_df["modelo"] = MODELO_DIRECTA
        resultados.trials.append(trials_df)

        _guardar_bloque(resultados, nombre_serie, MODELO_DIRECTA, pred, fechas_test, metricas, best_smape,
                         best_params, exog_names, train_hours, forecast_horizon)
        huvo_resultado = True

        del model, scaler_x, scaler_y, history
        tf.keras.backend.clear_session()
        gc.collect()

    except Exception as e:
        print(f"Error FCNN multivariada en {nombre_serie}: {type(e).__name__}: {e}")
        registrar_error(nombre_serie, e)

    # 2. STL + FCNN MULTIVARIADA SOBRE RESIDUOS
    try:
        print("\n   STL + Optuna FCNN multivariada sobre residuos...")

        trend, seasonal, resid, stl_res = descomponer_stl(train)

        trend_forecast = forecast_tendencia_lineal(trend, horizon=len(test))
        seasonal_forecast = forecast_estacionalidad_repetida(seasonal, horizon=len(test), period=STL_PERIOD)

        best_params_res, best_smape_res, trials_df_res = tunear_fcnn_optuna(
            train_y=resid, train_exog=X_train, nombre_serie=nombre_serie,
            modelo_nombre="FCNN_multivariada_residuos_STL", n_trials=n_trials, output_dir=output_dir,
            forecast_horizon=forecast_horizon,
        )

        print(f"      Mejor sMAPE residuos validacion: {best_smape_res:.4f}%")
        print(f"      Params residuos: {best_params_res}")

        model_res, scaler_x_res, scaler_y_res, history_res = entrenar_fcnn_final(
            resid, X_train, best_params_res, forecast_horizon
        )

        resid_pred = forecast_recursivo_fcnn_multivariada(
            model=model_res, y_train=resid, exog_train=X_train, exog_future=X_test,
            horizon=len(test), window=WINDOW, scaler_x=scaler_x_res, scaler_y=scaler_y_res,
        )

        pred_residuos = trend_forecast + seasonal_forecast + resid_pred

        metricas_res = calcular_metricas(test, pred_residuos)
        print(f"      STL + FCNN multivariada residuos MAPE={metricas_res['MAPE']:.2f}%")

        trials_df_res["serie"] = nombre_serie
        trials_df_res["modelo"] = MODELO_STL_RESIDUOS
        resultados.trials.append(trials_df_res)

        _guardar_bloque(resultados, nombre_serie, MODELO_STL_RESIDUOS, pred_residuos, fechas_test, metricas_res,
                         best_smape_res, best_params_res, exog_names, train_hours, forecast_horizon)
        huvo_resultado = True

        del model_res, scaler_x_res, scaler_y_res, history_res, stl_res
        tf.keras.backend.clear_session()
        gc.collect()

    except Exception as e:
        print(f"Error STL + FCNN multivariada sobre residuos en {nombre_serie}: {type(e).__name__}: {e}")
        registrar_error(nombre_serie, e)

    if huvo_resultado:
        resultados.series.append({
            "serie": nombre_serie, "fecha": fechas, "tipo": "real",
            "subset": "completo", "modelo": "real", "valor": serie,
        })


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
    Equivalente al script de nivel de modulo de la celda 64. Devuelve
    (series_df, metricas_df, trials_df, config_usada_df) con 2 filas por
    region en metricas_df/config_usada_df (una por cada una de las dos
    estrategias: directa y STL-residuos).
    """
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    exog_cols_canonico = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)
    exog_names = [CANONICAL_TO_INTERNAL[c] for c in exog_cols_canonico]

    resultados = _ResultsAccumulator()

    # Checkpoint por region: FCNN produce DOS modelos por region (directa y
    # STL-residuos), asi que se exigen exactamente 2 filas de metricas/
    # config_usada por region para considerarla completa -- si solo una de
    # las dos estrategias termino, la region se reintenta COMPLETA (las dos
    # estrategias comparten el mismo bloque "real" de series.csv, que solo
    # se guarda si `huvo_resultado`, ver evaluar_region).
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all, forecast_horizon=forecast_horizon,
        n_modelos_esperados=2, requiere_trials=True, requiere_config_usada=True,
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print("FCNN MULTIVARIADA + STL FCNN RESIDUOS")
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
        print(metricas_df.sort_values(["serie", "MAPE"]).to_string(index=False))

        print("\nPromedio por modelo:")
        print(metricas_df.groupby("modelo")["MAPE"].agg(["mean", "std"]).sort_values("mean").to_string())

    return series_df, metricas_df, trials_df, config_usada_df

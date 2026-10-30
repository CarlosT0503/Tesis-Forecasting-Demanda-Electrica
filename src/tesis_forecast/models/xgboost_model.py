"""
Pipeline XGBoost multivariado (demanda, horizonte de 1 semana).

Extraido de la celda 49 del notebook legacy ("Ahora si multivariado a 1
semana XGBoost"). La arquitectura del modelo, el espacio de busqueda de
Optuna, el tratamiento de exogenas conocidas/no-conocidas en el horizonte,
el forecast recursivo y las formulas de metricas son IDENTICOS al original.

Cambios mecanicos (no de logica) respecto a la celda original, todos
documentados en el checklist de equivalencia entregado junto con este
modulo:

  1. `merge_exogenas` recibe `exogenas_globales` (dict) como parametro en
     vez de leer `Temperaturas_H`, `IGAE_H`, etc. de `globals()`.
  2. Los acumuladores RESULTS_SERIES / RESULTS_METRICS / RESULTS_TRIALS /
     RESULTS_CONFIG_USADA dejan de ser listas globales del modulo y pasan a
     vivir dentro de un objeto `_ResultsAccumulator` local a cada llamada de
     `run()`. Esto es necesario para que dos experimentos (dos llamadas a
     `run()`) en la misma sesion de Python no contaminen sus resultados
     entre si; en el notebook original esto no era un riesgo porque cada
     sesion de Colab ejecutaba la celda una sola vez.
  3. `DRIVE_OUTPUT_DIR` / `SAVE_PREFIX` / `OPTUNA_DB` fijos se sustituyen por
     un parametro `output_dir` (la carpeta del experimento, ya resuelta por
     el runner) y nombres de archivo fijos (series.csv, metricas.csv,
     trials.csv, config_usada.csv) en vez del prefijo
     `xgboost_demanda_horario_2semanas_*`.
  4. La ruta hardcodeada "/content" (para *_GEN.csv, *_IMP.csv, *_EXP.csv) y
     la ruta relativa implicita de `ARCHIVOS_REGIONES` (para *_long.csv) se
     unifican bajo un unico parametro `data_dir`, con default "/content"
     para reproducir el comportamiento por defecto de Colab.
  5. `ejecutar_pipeline()` (que se autoejecutaba al correr la celda) pasa a
     ser `run(...)`, una funcion que el runner llama explicitamente y que
     devuelve (series_df, metricas_df, trials_df, config_usada_df) en vez de
     solo imprimir y guardar en Drive.
  6. Se eliminaron las constantes `MIN_OBS`, `VAL_GAP`, `VAL_SPLITS`: en la
     celda original estaban definidas pero nunca se usaban en el resto del
     pipeline (codigo muerto). No afecta ningun calculo.

Todo lo demas -- nombres de funciones, formulas, constantes de negocio,
orden de operaciones -- es una copia literal de la celda 49.
"""

import os
import gc

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador
from ..metrics import mape, smape, calcular_metricas

# =========================================================
# CONFIG POR DEFECTO (identica a la celda 49)
# =========================================================

FORECAST_HORIZON_DEFAULT = 24 * 7    # 168 horas
TRAIN_LAST_HOURS_DEFAULT = 24 * 14   # 336 horas = 2 semanas
N_TRIALS_OPTUNA_DEFAULT = 10

WINDOW_DEFAULT = 168                 # una semana de lags, fijo (no configurable)

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

# Catalogo completo de exogenas que este modelo sabe tratar, en el mismo
# orden que EXOG_COLS en la celda original (orden por defecto = todas).
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
# UTILIDADES
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

    # -----------------------------------------------------
    # EXOGENAS GLOBALES (Temperatura, Primarias, Secundarias,
    # Terciarias, IGAE) -- vienen del diccionario exogenas_globales,
    # no de globals() como en la celda original.
    # -----------------------------------------------------

    for nombre_variable, nombre_df in EXOG_SOURCE_MAP.items():
        if nombre_variable not in exog_cols:
            continue

        if nombre_df not in exogenas_globales:
            raise ValueError(f"No existe el DataFrame global {nombre_df}.")

        aux = _normalizar_exogena(exogenas_globales[nombre_df], nombre_variable)

        exog_df = aux if exog_df is None else exog_df.merge(aux, on="datetime", how="outer")

    # -----------------------------------------------------
    # EXOGENAS ESPECIFICAS DE LA REGION: se leen desde data_dir
    #   {data_dir}/{region}_GEN.csv
    #   {data_dir}/{region}_IMP.csv
    #   {data_dir}/{region}_EXP.csv
    # -----------------------------------------------------

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

    # Completar huecos de las exogenas.
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

    # Solo las exogenas se rellenan. La demanda nunca se modifica aqui.
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

    # LAGS DE DEMANDA
    for lag in range(1, window + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # ROLLING FEATURES
    y_past = df["y"].shift(1)
    df["rolling_mean_24"] = y_past.rolling(24).mean()
    df["rolling_std_24"] = y_past.rolling(24).std()
    df["rolling_mean_168"] = y_past.rolling(168).mean()
    df["rolling_std_168"] = y_past.rolling(168).std()
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

    # LAGS
    for lag in range(1, window + 1):
        features[f"lag_{lag}"] = hist_y[-lag] if len(hist_y) >= lag else hist_y[0]

    # ROLLING
    features["rolling_mean_24"] = np.mean(hist_y[-24:])
    features["rolling_std_24"] = np.std(hist_y[-24:])
    features["rolling_mean_168"] = np.mean(hist_y[-168:])
    features["rolling_std_168"] = np.std(hist_y[-168:])
    features["trend"] = len(hist_y)

    return pd.DataFrame([features])


# =========================================================
# XGBOOST - OBJECTIVE OPTUNA
# =========================================================

def objective_xgboost(trial, train_y, val_y, window, exog_cols, train_exog=None, val_exog=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
    }

    df_train = create_feature_df(train_y, window, exog_cols, train_exog)

    if len(df_train) < 50:
        return float("inf")

    X_train = df_train.drop(columns=["y"])
    y_train = df_train["y"]

    # VALIDACION
    val_context_y = np.concatenate([train_y, val_y])
    val_context_exog = pd.concat(
        [pd.DataFrame(train_exog), pd.DataFrame(val_exog)], ignore_index=True
    )

    df_val_all = create_feature_df(val_context_y, window, exog_cols, val_context_exog)

    if len(df_val_all) < len(val_y):
        return float("inf")

    X_val = df_val_all.drop(columns=["y"]).iloc[-len(val_y):]
    y_val = df_val_all["y"].iloc[-len(val_y):]

    model = XGBRegressor(**params, objective="reg:squarederror", n_jobs=1)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    score = smape(y_val.values, preds)

    return score if not np.isnan(score) else float("inf")


# =========================================================
# OPTUNA
# =========================================================

def tune_xgboost(train_y, val_y, nombre_serie, window, exog_cols, optuna_db, n_trials, train_exog=None, val_exog=None):
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"{nombre_serie}_xgboost_2semanas",
        storage=f"sqlite:///{optuna_db}",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective_xgboost(trial, train_y, val_y, window, exog_cols, train_exog, val_exog),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return study.best_params, study.trials_dataframe()


# =========================================================
# FORECAST RECURSIVO
# =========================================================

def forecast_xgboost_tuned(train_y, horizon, best_params, window, exog_cols, train_exog=None, future_exog=None):
    try:
        if future_exog is None or len(future_exog) != horizon:
            raise ValueError("future_exog debe tener exactamente horizon filas")

        df_train = create_feature_df(train_y, window, exog_cols, train_exog)
        X_train = df_train.drop(columns=["y"])
        y_train = df_train["y"]

        model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=42, n_jobs=1)
        model.fit(X_train, y_train)

        preds = []
        hist = list(train_y)

        future_exog = pd.DataFrame(future_exog).reset_index(drop=True)

        for step in range(horizon):
            X_future = create_features_from_history(hist, window, exog_cols, future_exog.iloc[step])
            X_future = X_future.reindex(columns=X_train.columns)

            pred = model.predict(X_future)[0]
            preds.append(pred)

            # Forecast recursivo: la prediccion pasa a formar parte de la historia.
            hist.append(pred)

        return np.array(preds)

    except Exception as e:
        print(f"         Error forecast XGBoost: {str(e)[:120]}")
        return np.full(horizon, np.nan)


# =========================================================
# ACUMULADOR DE RESULTADOS (reemplaza las listas globales
# RESULTS_SERIES / RESULTS_METRICS / RESULTS_TRIALS /
# RESULTS_CONFIG_USADA de la celda original)
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
    """Guardado incremental (equivalente a guardar_todos_csv() en la celda
    original), llamado despues de cada region para no perder avance si la
    sesion de Colab se cae a medio pipeline. Escribe con los mismos nombres
    finales (series.csv, metricas.csv, trials.csv, config_usada.csv) dentro
    de la carpeta del experimento."""

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
# PREPARAR TRAIN / TEST
# =========================================================

def _preparar_serie(nombre_serie, serie, fechas, exogenas_df, exog_cols, train_hours, forecast_horizon):
    required_hours = train_hours + forecast_horizon

    if len(serie) < required_hours:
        print(f"   Serie insuficiente: {nombre_serie}")
        return None

    # IMPORTANTE: se conserva SOLO: train_hours horas train + forecast_horizon horas test
    start = len(serie) - required_hours
    serie_reciente = serie[start:]
    fechas_recientes = fechas[start:]

    # ALINEAR EXOGENAS A TODO EL PERIODO ORIGINAL
    exog_completa = alinear_exogenas_a_fechas(fechas, exogenas_df, exog_cols)

    # Necesitamos conservar tambien historia anterior para poder obtener
    # t-336 al construir las exogenas futuras.
    exog_inicio = max(0, start - LAG_SEMANA_2)
    exog_contexto = exog_completa.iloc[exog_inicio:].reset_index(drop=True)

    # Posicion donde empieza nuestro train dentro de exog_contexto
    offset_train = start - exog_inicio

    # TRAIN / TEST OBJETIVO
    train = serie_reciente[:train_hours]
    test = serie_reciente[train_hours:]
    fechas_test = fechas_recientes[train_hours:]

    # TRAIN EXOG
    train_exog = exog_contexto.iloc[offset_train:offset_train + train_hours].reset_index(drop=True)

    # FUTURE EXOG: train_end_contexto es la posicion donde comienza el test
    # dentro de exog_contexto.
    train_end_contexto = offset_train + train_hours

    test_exog = construir_future_exog(
        exog_serie=exog_contexto,
        train_end=train_end_contexto,
        horizon=forecast_horizon,
        exog_cols=exog_cols,
    )

    # DIAGNOSTICOS
    print("\n   Split general fijo")
    print(f"      Train: {len(train):,} obs")
    print(f"      Test: {len(test):,} obs")
    print(f"      Horizonte: {forecast_horizon} horas")

    print("\n   Exogenas activas:")
    for col in exog_cols:
        if col in EXOG_CONOCIDAS_FUTURO:
            print(f"      {col}: valor del horizonte")
        else:
            print(f"      {col}: estimada con t-168 y t-336")

    # VALIDACION: primeras horas para train, ultimas forecast_horizon para
    # validacion (mantiene orden temporal).
    val_size = forecast_horizon

    tune_train_y = train[:-val_size]
    tune_val_y = train[-val_size:]

    tune_train_exog = train_exog.iloc[:-val_size].reset_index(drop=True)

    # EXOG VALIDACION SIN LEAKAGE: las variables operativas tampoco deben
    # ver los valores reales en la validacion.
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
        "window_tune": WINDOW_DEFAULT,
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
        print("\n      XGBoost tuning...")

        best_params, trials_df = tune_xgboost(
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
        trials_df["modelo"] = "XGBoost"
        resultados.trials.append(trials_df)

        resultados.config_usada.append({
            "serie": nombre_serie,
            "modelo": "XGBoost",
            "parametros": str(best_params),
            "horizonte_usado": contexto["horizonte_usado"],
            "train_horas": train_hours,
            "exogenas": str(exog_cols),
        })

        # FORECAST FINAL
        pred = forecast_xgboost_tuned(
            contexto["train"],
            contexto["horizon"],
            best_params,
            contexto["window_tune"],
            exog_cols,
            contexto["train_exog"],
            contexto["test_exog"],
        )

        validar_horizonte("XGBoost_Tuned_2Semanas", pred, forecast_horizon)

        metricas = calcular_metricas(contexto["test"], pred)

        if metricas:
            guardar_metricas(resultados, nombre_serie, "XGBoost_Tuned_2Semanas", True, metricas, contexto["horizonte_usado"])
            guardar_predicciones(resultados, nombre_serie, contexto["fechas_test"], pred, "XGBoost_Tuned_2Semanas")

            print(f"      XGBoost_Tuned_2Semanas: MAPE={metricas['MAPE']:.2f}%")

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
    Equivalente a ejecutar_pipeline() en la celda 49, pero parametrizado y
    devolviendo los resultados en vez de asumir que ya existen variables
    globales / una carpeta de Drive fija.

    Devuelve (series_df, metricas_df, trials_df, config_usada_df). Este
    ultimo (mejores hiperparametros de Optuna por region) se guarda como
    archivo independiente config_usada.csv, separado de metricas.csv: uno
    describe desempeno, el otro los hiperparametros que lo produjeron.
    """
    exog_cols = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)

    resultados = _ResultsAccumulator()
    optuna_db = os.path.join(output_dir, "optuna_xgboost.db")

    # Checkpoint por region: si una region ya tiene metricas validas, el
    # numero exacto de predicciones esperado (forecast_horizon), sus trials
    # de Optuna y su config_usada, se salta -- no se vuelve a tunear ni a
    # entrenar. Los trials de Optuna de xgboost/lightgbm/lstm_direct tienen
    # ADEMAS su propio resume interno (via el .db de Optuna,
    # load_if_exists=True); este checkpoint es a nivel de REGION COMPLETA
    # (metricas + prediccion final ya guardadas), un nivel por encima.
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
    print("PIPELINE XGBOOST DEMANDA")
    print("=" * 80)
    print(f"Directorio de salida: {output_dir}")
    print(f"Train usado: {train_hours} horas")
    print(f"Test usado: {forecast_horizon} horas")

    print("\nExogenas activas:")
    for exog in exog_cols:
        print(f"   - {exog}")

    # CARGAR REGIONES
    regiones = cargar_regiones(regiones_pendientes, data_dir)

    # PROCESAR
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

    # FINAL
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

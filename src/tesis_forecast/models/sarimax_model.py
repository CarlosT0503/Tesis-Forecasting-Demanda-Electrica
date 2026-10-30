"""
Pipeline SARIMAX (demanda, horizonte de 1 semana).

Extraido de la celda 62 del notebook legacy ("SARIMAX"). El orden fijo
SARIMA_ORDER=(1,1,1), SARIMA_SEASONAL_ORDER=(1,0,1,168), la llamada a
statsmodels.SARIMAX, y el tratamiento de exogenas conocidas/no-conocidas
en el horizonte son IDENTICOS al original. A diferencia de XGBoost/LSTM,
este modelo NO usa Optuna -- el orden es fijo, no hay busqueda.

Cambios mecanicos (no de logica), mismo tipo de adaptacion que en
xgboost_model.py / lstm_direct.py:

  1. `globals()` (Temperaturas_H, IGAE_H) y el diccionario global `series`
     (para GEN/IMP/EXP por region) se sustituyen por `exogenas_globales`
     (parametro) y lectura directa de `{region}_GEN/IMP/EXP.csv` desde
     `data_dir`, respectivamente.
  2. El script de nivel de modulo (celda 62 NO tenia una funcion
     `ejecutar_pipeline`, era un for-loop directo al final de la celda) se
     envuelve en `run(...)`.
  3. El guardado "por serie" en subcarpetas individuales
     (`INCREMENTAL_DIR/<serie>/<serie>_SARIMAX_series.csv`) se reemplaza
     por el mismo patron de acumulador + guardado incremental de
     series.csv/metricas.csv en la carpeta del experimento que usan los
     demas modelos migrados -- incluye el mismo contenido (filas "real" +
     predicciones + AIC/BIC/order en metricas), solo cambia DONDE se
     escribe.
  4. No hay Optuna, por lo que no hay trials.csv. Se genera igual un
     `config_usada.csv` (con el order/seasonal_order fijo, para consistencia
     de esquema con los demas modelos, aunque aqui no representa nada
     "tuneado").
  5. EXOG_NAMES usaba "Temperaturas" (plural) en la celda original. El
     catalogo externo de `ExperimentConfig.exogenas` es ahora el mismo
     para todos los modelos ("Temperatura", singular, igual que XGBoost);
     este modulo traduce internamente "Temperatura" -> "Temperaturas" al
     entrar a `run()`, sin tocar el resto de la logica (que sigue usando
     "Temperaturas" como en la celda original).

Todo lo demas -- formulas de metricas, arquitectura SARIMAX, construccion
de exogenas futuras -- es una copia literal de la celda 62.
"""

import os
import gc

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..checkpoint import cargar_checkpoint_regiones, precargar_en_acumulador

# =========================================================
# CONFIG POR DEFECTO (identica a la celda 62)
# =========================================================

FORECAST_HORIZON_DEFAULT = 24 * 7        # 168 horas (celda 62: TEST_HORIZON)
TRAIN_LAST_HOURS_DEFAULT = 24 * 30 * 2   # 1440 horas = 2 meses

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 0, 1, 168)

LAG_SEMANA_1 = 24 * 7     # 168
LAG_SEMANA_2 = 24 * 14    # 336

# Catalogo EXTERNO (canonico, igual para todos los modelos). Internamente
# se traduce a los nombres que usaba la celda 62 (ver CANONICAL_TO_INTERNAL).
EXOG_COLS_DEFAULT = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
EXOG_CATALOGO = list(EXOG_COLS_DEFAULT)

# Traduccion nombre canonico -> nombre interno de la celda 62.
CANONICAL_TO_INTERNAL = {
    "Temperatura": "Temperaturas",
    "IGAE": "IGAE",
    "Generacion": "Generacion",
    "Importacion": "Importacion",
    "Exportacion": "Exportacion",
}
INTERNAL_TO_CANONICAL = {v: k for k, v in CANONICAL_TO_INTERNAL.items()}

# Fuente de las exogenas globales (mismo mapeo que XGBoost para las que
# aplican aqui).
EXOG_SOURCE_MAP = {
    "Temperaturas": "Temperaturas_H",
    "IGAE": "IGAE_H",
}

EXOG_CONOCIDAS_FUTURO = ["Temperaturas", "IGAE"]

EXOG_NO_CONOCIDAS_FUTURO = ["Generacion", "Importacion", "Exportacion"]


# =========================================================
# METRICAS (copia exacta de la celda 62 -- distintas de XGBoost:
# usan mascara isfinite, y calcular_metricas devuelve un dict de NaN en
# vez de None cuando no hay datos validos)
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
# LEER DEMANDA
# =========================================================

def extraer_serie_horaria(df, columna):
    aux = df[[COL_FECHA, COL_HORA, columna]].copy()

    aux[COL_FECHA] = pd.to_datetime(aux[COL_FECHA], errors="coerce")
    aux[COL_HORA] = pd.to_numeric(aux[COL_HORA], errors="coerce")
    aux[columna] = pd.to_numeric(aux[columna], errors="coerce")

    aux = aux.dropna(subset=[COL_FECHA, COL_HORA, columna])

    aux["hora_0_23"] = aux[COL_HORA].astype(int) - 1
    aux["datetime"] = aux[COL_FECHA] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    aux = aux.sort_values("datetime").drop_duplicates("datetime", keep="last")

    return aux[columna].to_numpy(dtype=float), aux["datetime"].to_numpy()


# =========================================================
# NORMALIZAR EXOGENA
# =========================================================

def preparar_exogena_horaria(df, nombre):
    aux = df.copy()
    aux.columns = aux.columns.astype(str).str.strip()

    col_hora = "hora" if "hora" in aux.columns else "Hora"

    if "fecha" not in aux.columns:
        raise ValueError(f"{nombre} no tiene columna fecha")
    if col_hora not in aux.columns:
        raise ValueError(f"{nombre} no tiene columna hora/Hora")
    if "valor" not in aux.columns:
        raise ValueError(f"{nombre} no tiene columna valor")

    aux["fecha"] = pd.to_datetime(aux["fecha"], errors="coerce")
    aux[col_hora] = pd.to_numeric(aux[col_hora], errors="coerce")
    aux["valor"] = pd.to_numeric(aux["valor"], errors="coerce")

    aux = aux.dropna(subset=["fecha", col_hora, "valor"])

    hora = aux[col_hora]

    # Acepta 1-24 o 0-23
    if hora.min() >= 1 and hora.max() <= 24:
        aux["hora_0_23"] = hora.astype(int) - 1
    else:
        aux["hora_0_23"] = hora.astype(int)

    aux["datetime"] = aux["fecha"] + pd.to_timedelta(aux["hora_0_23"], unit="h")

    return (
        aux[["datetime", "valor"]]
        .rename(columns={"valor": nombre})
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
    )


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
        "Generacion": f"{region}_GEN.csv",
        "Importacion": f"{region}_IMP.csv",
        "Exportacion": f"{region}_EXP.csv",
    }

    for variable, nombre_archivo in region_exog_map.items():
        if variable not in exog_names:
            continue

        ruta = os.path.join(data_dir, nombre_archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No existe el archivo {ruta}")

        dfs.append(preparar_exogena_horaria(pd.read_csv(ruta), variable))

    if len(dfs) == 0:
        raise ValueError("No hay ninguna exogena activa.")

    exog = dfs[0]
    for df_next in dfs[1:]:
        exog = exog.merge(df_next, on="datetime", how="outer")

    exog = exog.sort_values("datetime").reset_index(drop=True)
    exog[exog_names] = exog[exog_names].ffill().bfill()

    print(f"\nExogenas {region}:")
    print(", ".join(exog_names))
    print(f"Filas disponibles: {len(exog):,}")

    return exog


# =========================================================
# ALINEAR EXOGENAS CON DEMANDA
# =========================================================

def alinear_exogenas_con_fechas(fechas, exog_global, exog_names):
    base = pd.DataFrame({"datetime": pd.to_datetime(fechas)})

    X = base.merge(exog_global[["datetime"] + exog_names], on="datetime", how="left")
    X[exog_names] = X[exog_names].ffill().bfill()

    if X[exog_names].isna().any().any():
        raise ValueError("Hay valores faltantes en las exogenas.")

    return X[exog_names].astype(float).reset_index(drop=True)


# =========================================================
# EXOGENAS FUTURAS
# =========================================================

def construir_exogenas_futuras(X_completo, train_end, horizon, exog_names):
    """
    Temperaturas e IGAE: usan el valor correspondiente al horizonte.
    Generacion/Importacion/Exportacion: NO usan el valor real futuro,
    se estiman con promedio(t-168, t-336).
    """
    X_future = X_completo.iloc[train_end:train_end + horizon].copy().reset_index(drop=True)

    for variable in EXOG_NO_CONOCIDAS_FUTURO:
        if variable not in exog_names:
            continue

        estimados = []
        for h in range(horizon):
            indice_futuro = train_end + h
            idx_1 = indice_futuro - LAG_SEMANA_1
            idx_2 = indice_futuro - LAG_SEMANA_2

            if idx_2 < 0:
                raise ValueError(f"No existen 2 semanas anteriores para estimar {variable}")

            valor_1 = float(X_completo.iloc[idx_1][variable])
            valor_2 = float(X_completo.iloc[idx_2][variable])

            estimados.append((valor_1 + valor_2) / 2)

        X_future[variable] = estimados

    return X_future[exog_names].astype(float).reset_index(drop=True)


# =========================================================
# SARIMAX
# =========================================================

def entrenar_predecir_sarimax(train, horizon, order, seasonal_order=(0, 0, 0, 0), X_train=None, X_test=None):
    y_train = np.asarray(train, dtype=float)

    X_train_arr = np.asarray(X_train, dtype=float) if X_train is not None else None
    X_test_arr = np.asarray(X_test, dtype=float) if X_test is not None else None

    model = SARIMAX(
        endog=y_train,
        exog=X_train_arr,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )

    try:
        fitted = model.fit(disp=False, maxiter=100, low_memory=True)
    except TypeError:
        fitted = model.fit(disp=False, maxiter=100)

    pred = fitted.predict(start=len(y_train), end=len(y_train) + horizon - 1, exog=X_test_arr)

    aic = fitted.aic
    bic = fitted.bic

    pred = np.asarray(pred)

    try:
        converged = fitted.mle_retvals.get("converged", None)
        iterations = fitted.mle_retvals.get("iterations", None)
        print(f"      Converged: {converged}")
        print(f"      Iterations: {iterations}")
    except Exception:
        pass

    del fitted
    del model
    gc.collect()

    return pred, aic, bic


# =========================================================
# EVALUAR SERIE
# =========================================================

def evaluar_serie(nombre_serie, serie, fechas, exog_global, exog_names, train_hours, forecast_horizon):
    X_exog = alinear_exogenas_con_fechas(fechas, exog_global, exog_names)

    requeridas = train_hours + forecast_horizon
    if len(serie) < requeridas:
        raise ValueError(f"{nombre_serie}: serie insuficiente.")

    train_start = len(serie) - requeridas
    train_end = len(serie) - forecast_horizon

    train = serie[train_start:train_end]
    test = serie[train_end:]
    fechas_test = fechas[train_end:]

    X_train = X_exog.iloc[train_start:train_end].reset_index(drop=True)
    X_test = construir_exogenas_futuras(X_exog, train_end, forecast_horizon, exog_names)

    print(f"Train demanda: {len(train):,}")
    print(f"Train exogenas: {len(X_train):,}")
    print(f"Test demanda: {len(test):,}")
    print(f"Test exogenas: {len(X_test):,}")

    print("\nTratamiento de exogenas:")
    for variable in exog_names:
        if variable in EXOG_CONOCIDAS_FUTURO:
            print(f"   {variable}: valor del horizonte")
        else:
            print(f"   {variable}: promedio t-168 y t-336")

    print(f"\n   Ajustando SARIMAX{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}...")

    pred, aic, bic = entrenar_predecir_sarimax(
        train=train,
        horizon=len(test),
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        X_train=X_train,
        X_test=X_test,
    )

    metricas = calcular_metricas(test, pred)
    print(f"      SARIMAX MAPE={metricas['MAPE']:.2f}%")

    return {
        "serie": serie,
        "fechas": fechas,
        "fechas_test": fechas_test,
        "pred": pred,
        "metricas": metricas,
        "aic": aic,
        "bic": bic,
    }


# =========================================================
# ACUMULADOR / GUARDADO
# =========================================================

class _ResultsAccumulator:
    def __init__(self):
        self.series = []
        self.metrics = []
        self.config_usada = []


NOMBRE_MODELO = "SARIMAX_1_1_1__1_0_1_168_EXOG_2M"


def _region_de_serie(nombre_serie):
    return nombre_serie.split("_")[0]


def guardar_resultado_serie(resultados, nombre_serie, resultado, exog_names, train_hours, forecast_horizon):
    resultados.series.append({
        "serie": nombre_serie,
        "fecha": pd.Series(resultado["fechas"]),
        "tipo": "real",
        "subset": "completo",
        "modelo": "real",
        "valor": resultado["serie"],
    })

    resultados.series.append({
        "serie": nombre_serie,
        "fecha": pd.Series(resultado["fechas_test"]),
        "tipo": "prediccion",
        "subset": "test",
        "modelo": NOMBRE_MODELO,
        "valor": resultado["pred"],
    })

    resultados.metrics.append({
        "serie": nombre_serie,
        "modelo": NOMBRE_MODELO,
        "order": str(SARIMA_ORDER),
        "seasonal_order": str(SARIMA_SEASONAL_ORDER),
        "AIC": resultado["aic"],
        "BIC": resultado["bic"],
        "MAPE": resultado["metricas"]["MAPE"],
        "sMAPE": resultado["metricas"]["sMAPE"],
        "MAE": resultado["metricas"]["MAE"],
        "RMSE": resultado["metricas"]["RMSE"],
        "train_horas": train_hours,
        "test_horas": forecast_horizon,
        "exogenas": str(exog_names),
        "metodo_exog_futuras": "Temp/IGAE horizonte; GEN/IMP/EXP promedio t-168,t-336",
    })

    resultados.config_usada.append({
        "serie": nombre_serie,
        "modelo": NOMBRE_MODELO,
        "parametros": str({"order": SARIMA_ORDER, "seasonal_order": SARIMA_SEASONAL_ORDER}),
        "horizonte_usado": f"{forecast_horizon}_horas",
        "train_horas": train_hours,
        "exogenas": str(exog_names),
    })


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
        df_series.to_csv(os.path.join(output_dir, "series.csv"), index=False, encoding="utf-8-sig")
        print(f"\nOK Series guardadas (avance): {len(df_series):,} registros")

    if resultados.metrics:
        df_metrics = pd.DataFrame(resultados.metrics)
        df_metrics["region"] = df_metrics["serie"].map(_region_de_serie)
        df_metrics = df_metrics.sort_values(["serie", "MAPE"])
        df_metrics.to_csv(os.path.join(output_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Metricas guardadas (avance): {len(df_metrics):,} registros")

    if resultados.config_usada:
        df_config = pd.DataFrame(resultados.config_usada)
        df_config["region"] = df_config["serie"].map(_region_de_serie)
        df_config.to_csv(os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
        print(f"OK Config usada guardada (avance): {len(df_config):,} registros")


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
# PIPELINE PRINCIPAL
# =========================================================

def run(
    exogenas_globales: dict,
    regions_all: list,
    train_hours: int = TRAIN_LAST_HOURS_DEFAULT,
    forecast_horizon: int = FORECAST_HORIZON_DEFAULT,
    exog_cols: list = None,
    optuna_n_trials: int = None,   # sin uso -- SARIMAX no tunea, se acepta para firma uniforme con el runner
    data_dir: str = "/content",
    output_dir: str = ".",
):
    """
    Equivalente al script de nivel de modulo de la celda 62 (no tenia
    ejecutar_pipeline()), parametrizado. Devuelve
    (series_df, metricas_df, trials_df, config_usada_df); trials_df va
    siempre vacio (SARIMAX no usa Optuna).

    `exog_cols` se recibe en nombres CANONICOS (iguales a los de XGBoost,
    ej. "Temperatura") y se traduce aqui a los nombres internos que usaba
    la celda 62 (ej. "Temperaturas").
    """
    exog_cols_canonico = list(exog_cols) if exog_cols is not None else list(EXOG_COLS_DEFAULT)
    exog_names = [CANONICAL_TO_INTERNAL[c] for c in exog_cols_canonico]

    resultados = _ResultsAccumulator()

    # Checkpoint por region: SARIMAX no usa Optuna (sin trials.csv), pero
    # SI genera config_usada.csv (order/seasonal_order fijos). `series.csv`
    # aqui se reconstruye por bloques (fecha/valor como arreglo), no fila
    # por fila -- `_guardar_avance_csv` de este modulo no tiene la guarda
    # `np.atleast_1d` que si tienen FCNN/naive_trend_seasonal/etc., asi que
    # precargar filas escalares sueltas repetiria el mismo bug que tuvo
    # FCNN. Ver checkpoint.py.
    regiones_completas, previos = cargar_checkpoint_regiones(
        output_dir, regions_all, forecast_horizon=forecast_horizon,
        requiere_config_usada=True, formato_series="bloques",
    )
    precargar_en_acumulador(resultados, previos)

    regiones_pendientes = [r for r in regions_all if r not in regiones_completas]
    if regiones_completas:
        print(f"Checkpoint: {len(regiones_completas)} region(es) ya completas, se saltan: {sorted(regiones_completas)}")
    if not regiones_pendientes:
        print("Todas las regiones ya estan completas segun el checkpoint.")

    print("=" * 80)
    print("PIPELINE SARIMAX DEMANDA")
    print("=" * 80)
    print(f"Directorio de salida: {output_dir}")
    print(f"Train usado: {train_hours} horas")
    print(f"Test usado: {forecast_horizon} horas")

    print("\nExogenas activas:")
    for exog in exog_names:
        print(f"   - {exog}")

    regiones = cargar_regiones(regiones_pendientes, data_dir)

    for region, df in regiones.items():
        print("\n" + "=" * 80)
        print(f"Serie: {region}_DEMANDA")
        print("=" * 80)

        try:
            exog_region = construir_matriz_exogena_region(region, exogenas_globales, exog_names, data_dir)

            nombre_serie = f"{region}_DEMANDA"
            serie, fechas = extraer_serie_horaria(df, COL_DEMANDA)

            resultado = evaluar_serie(nombre_serie, serie, fechas, exog_region, exog_names, train_hours, forecast_horizon)
            guardar_resultado_serie(resultados, nombre_serie, resultado, exog_names, train_hours, forecast_horizon)

        except Exception as e:
            print(f"Error en region {region}: {type(e).__name__}: {e}")

        finally:
            gc.collect()

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

    config_usada_df = pd.DataFrame(resultados.config_usada)
    if len(config_usada_df) > 0:
        config_usada_df["region"] = config_usada_df["serie"].map(_region_de_serie)

    trials_df = pd.DataFrame()   # SARIMAX no usa Optuna

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO")
    print("=" * 80)

    if len(metricas_df) > 0:
        print("\nResultados por region:")
        print(metricas_df.sort_values(["serie", "MAPE"]).to_string(index=False))

    return series_df, metricas_df, trials_df, config_usada_df

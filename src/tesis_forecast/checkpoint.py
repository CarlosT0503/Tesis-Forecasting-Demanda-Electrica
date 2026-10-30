"""
Resume/checkpoint por region, generico sobre cualquier modelo.

Motivacion (ver docs/CHECKPOINT_RESUME.md para el detalle completo): antes
de este modulo, `run_experiment(..., overwrite=True)` borraba la carpeta
COMPLETA de un experimento incompleto (`shutil.rmtree(run_dir)`) y volvia a
correr las 8 regiones desde cero, incluidas las que ya habian terminado con
exito. En Colab, donde una sesion puede caerse a medio pipeline, esto
desperdicia horas de computo ya hecho (Optuna, LSTM/FCNN entrenadas, etc.)
cada vez que hay que reanudar.

Este modulo NO ejecuta ningun modelo ni sabe nada de XGBoost/LSTM/SARIMAX
en particular. Solo sabe leer los 4 CSV de salida (series/metricas/trials/
config_usada) que TODOS los modelos ya escriben con el mismo esquema de
columnas, decidir que regiones ya estan completas y validas, y devolver esos
datos ya filtrados en el formato que cada modelo necesita para pre-sembrar
su propio `_ResultsAccumulator` antes de correr solo las regiones que
faltan. Cada modelo sigue siendo responsable de:
  - decidir cuantos "modelos" distintos produce por region (la mayoria: 1;
    FCNN: 2, una fila de metricas por estrategia);
  - decidir si tiene trials.csv/config_usada.csv (algunos, como Naive, no
    tunean nada y nunca los generan);
  - filtrar `regions_all` a las regiones pendientes y correr su loop de
    siempre, sin ningun cambio de metodologia.

Dos formatos de "series" precargada, porque los modelos migrados usan dos
convenciones distintas para acumular `resultados.series` (ver
docs/CHECKPOINT_RESUME.md):

  - "filas": una fila por registro (fecha/valor escalares). Es el formato
    nativo de xgboost/lightgbm/lstm_direct/naive/naive_trend/ar, y tambien
    funciona sin problema en los modelos que usan `_construir_df_series`
    con `np.atleast_1d` (fcnn, naive_trend_seasonal,
    ar_resid_trend_seasonal, lstm_resid) porque esa funcion normaliza
    escalares a arreglos de 1 elemento.
  - "bloques": un bloque por (serie, tipo, subset, modelo) con fecha/valor
    como arreglo. Necesario para sarimax/ensemble_stl, que construyen su
    DataFrame de series con `pd.DataFrame({...bloque...})` SIN ninguna
    normalizacion -- pasarles una fila escalar sacaria el mismo error que
    tuvo FCNN ("If using all scalar values, you must pass an index").
"""

import os

import numpy as np
import pandas as pd


def _leer_csv_si_existe(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    return df if len(df.columns) > 0 else None


def _coerce_booleanos(df, columnas=("tuneado",)):
    """
    pandas relee "True"/"False" (como se escriben en el CSV) como texto,
    no bool. Sin esto, al mezclar filas precargadas (texto) con filas
    recien calculadas (bool real) en la misma corrida, la columna queda
    con tipos mixtos y comparaciones como `df["tuneado"] == True` fallan
    silenciosamente para las filas precargadas.
    """
    if df is None:
        return df
    for col in columnas:
        if col in df.columns and df[col].dtype == object:
            mapeado = df[col].map({"True": True, "False": False})
            df[col] = mapeado.where(mapeado.notna(), df[col])
    return df


def region_es_completa(
    nombre_serie,
    metricas_df,
    series_df,
    trials_df,
    config_usada_df,
    forecast_horizon=None,
    n_modelos_esperados=1,
    requiere_trials=False,
    requiere_config_usada=False,
    trials_esperados=None,
):
    """
    Una region se considera completa solo si:
      1. Tiene exactamente `n_modelos_esperados` fila(s) de metricas, todas
         con MAPE/sMAPE/MAE/RMSE finitos (no NaN/inf).
      2. Tiene predicciones (`tipo == "prediccion"`): si `forecast_horizon`
         es un entero conocido, exactamente
         `forecast_horizon * n_modelos_esperados` filas; si es None (split
         dinamico "auto", donde el conteo esperado varia por region segun
         el largo de la serie y no se puede saber sin recargar los datos
         crudos), basta con que haya al menos una.
      3. Si `requiere_trials`: al menos una fila de trials para esa serie
         (y, si se pasa `trials_esperados`, exactamente esa cantidad).
      4. Si `requiere_config_usada`: exactamente `n_modelos_esperados`
         fila(s) de config_usada para esa serie.
    """
    m = metricas_df[metricas_df["serie"] == nombre_serie] if metricas_df is not None and len(metricas_df) else pd.DataFrame()
    if len(m) != n_modelos_esperados:
        return False

    for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
        if col not in m.columns:
            return False
        valores = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            return False

    s_pred = pd.DataFrame()
    if series_df is not None and len(series_df) and "tipo" in series_df.columns:
        s_pred = series_df[(series_df["serie"] == nombre_serie) & (series_df["tipo"] == "prediccion")]

    if isinstance(forecast_horizon, int):
        if len(s_pred) != forecast_horizon * n_modelos_esperados:
            return False
    else:
        if len(s_pred) == 0:
            return False

    if requiere_trials:
        t = trials_df[trials_df["serie"] == nombre_serie] if trials_df is not None and len(trials_df) else pd.DataFrame()
        if len(t) == 0:
            return False
        if trials_esperados is not None and len(t) != trials_esperados:
            return False

    if requiere_config_usada:
        c = config_usada_df[config_usada_df["serie"] == nombre_serie] if config_usada_df is not None and len(config_usada_df) else pd.DataFrame()
        if len(c) != n_modelos_esperados:
            return False

    return True


def _precargar_series_filas(series_df_filtrado):
    if series_df_filtrado is None or len(series_df_filtrado) == 0:
        return []
    cols = [c for c in ["serie", "fecha", "tipo", "subset", "modelo", "valor"] if c in series_df_filtrado.columns]
    return series_df_filtrado[cols].to_dict("records")


def _precargar_series_bloques(series_df_filtrado):
    if series_df_filtrado is None or len(series_df_filtrado) == 0:
        return []

    bloques = []
    cols_group = ["serie", "tipo", "subset", "modelo"]
    for clave, grupo in series_df_filtrado.groupby(cols_group, sort=False):
        serie_, tipo_, subset_, modelo_ = clave
        grupo_ordenado = grupo.sort_values("fecha")
        bloques.append({
            "serie": serie_,
            "fecha": grupo_ordenado["fecha"].to_numpy(),
            "tipo": tipo_,
            "subset": subset_,
            "modelo": modelo_,
            "valor": grupo_ordenado["valor"].to_numpy(),
        })
    return bloques


def cargar_checkpoint_regiones(
    output_dir: str,
    regions_all: list,
    forecast_horizon=None,
    n_modelos_esperados: int = 1,
    requiere_trials: bool = False,
    requiere_config_usada: bool = False,
    trials_esperados=None,
    formato_series: str = "filas",
):
    """
    Lee series.csv/metricas.csv/trials.csv/config_usada.csv de `output_dir`
    (si existen) y determina, para cada region de `regions_all`, si ya tiene
    un resultado completo y valido (ver `region_es_completa`).

    Devuelve `(regiones_completas, previos)`:
      - `regiones_completas`: set de nombres de region (ej. {"BCA", "CEN"})
        que ya estan listas y deben saltarse.
      - `previos`: dict con los datos de esas regiones, ya filtrados y en el
        formato que el `_ResultsAccumulator` del modelo necesita:
          - `previos["series"]`: lista de dicts (formato "filas" o
            "bloques" segun `formato_series`), para
            `resultados.series.extend(...)`.
          - `previos["metrics"]`: lista de dicts, para
            `resultados.metrics.extend(...)`.
          - `previos["trials_df"]`: un DataFrame (o None si no hay nada que
            precargar), para `resultados.trials.append(...)` si el modelo
            tiene ese atributo.
          - `previos["config_usada"]`: lista de dicts, para
            `resultados.config_usada.extend(...)`.

    Si algun CSV no existe o no se puede leer, se trata como checkpoint
    vacio para ESE archivo (no como error) -- una region solo cuenta como
    completa si TODA la evidencia requerida esta presente y es valida; si
    falta o esta corrupta, la region simplemente se vuelve a calcular desde
    cero (esto es lo que garantiza que una region parcial/corrupta se
    reemplace en vez de mezclarse con datos previos inconsistentes).
    """
    series_df = _leer_csv_si_existe(os.path.join(output_dir, "series.csv"))
    metricas_df = _coerce_booleanos(_leer_csv_si_existe(os.path.join(output_dir, "metricas.csv")))
    trials_df = _leer_csv_si_existe(os.path.join(output_dir, "trials.csv"))
    config_usada_df = _leer_csv_si_existe(os.path.join(output_dir, "config_usada.csv"))

    regiones_completas = set()
    for region in regions_all:
        nombre_serie = f"{region}_DEMANDA"
        if region_es_completa(
            nombre_serie, metricas_df, series_df, trials_df, config_usada_df,
            forecast_horizon=forecast_horizon,
            n_modelos_esperados=n_modelos_esperados,
            requiere_trials=requiere_trials,
            requiere_config_usada=requiere_config_usada,
            trials_esperados=trials_esperados,
        ):
            regiones_completas.add(region)

    vacio = {"series": [], "metrics": [], "trials_df": None, "config_usada": []}
    if not regiones_completas:
        return regiones_completas, vacio

    nombres_completos = {f"{r}_DEMANDA" for r in regiones_completas}

    series_prev = []
    if series_df is not None and len(series_df):
        subset_series = series_df[series_df["serie"].isin(nombres_completos)]
        if formato_series == "bloques":
            series_prev = _precargar_series_bloques(subset_series)
        else:
            series_prev = _precargar_series_filas(subset_series)

    metrics_prev = []
    if metricas_df is not None and len(metricas_df):
        subset_metricas = metricas_df[metricas_df["serie"].isin(nombres_completos)]
        cols = [c for c in subset_metricas.columns if c != "region"]
        metrics_prev = subset_metricas[cols].to_dict("records")

    trials_prev_df = None
    if trials_df is not None and len(trials_df):
        subset_trials = trials_df[trials_df["serie"].isin(nombres_completos)].reset_index(drop=True)
        if len(subset_trials) > 0:
            trials_prev_df = subset_trials

    config_prev = []
    if config_usada_df is not None and len(config_usada_df):
        subset_config = config_usada_df[config_usada_df["serie"].isin(nombres_completos)]
        cols = [c for c in subset_config.columns if c != "region"]
        config_prev = subset_config[cols].to_dict("records")

    return regiones_completas, {
        "series": series_prev,
        "metrics": metrics_prev,
        "trials_df": trials_prev_df,
        "config_usada": config_prev,
    }


def precargar_en_acumulador(resultados, previos):
    """
    Siembra `resultados` (un `_ResultsAccumulator` de cualquier modelo) con
    los datos de `previos` (el segundo elemento que devuelve
    `cargar_checkpoint_regiones`). Defensivo ante modelos que no tienen
    `.trials`/`.config_usada` (ej. Naive, Naive_Trend, Naive_Trend_Seasonal,
    que nunca tunean nada y no definen esos atributos en su acumulador).
    """
    resultados.series.extend(previos["series"])
    resultados.metrics.extend(previos["metrics"])

    if hasattr(resultados, "trials") and previos["trials_df"] is not None:
        resultados.trials.append(previos["trials_df"])

    if hasattr(resultados, "config_usada"):
        resultados.config_usada.extend(previos["config_usada"])

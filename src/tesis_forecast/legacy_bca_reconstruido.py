"""
Completar BCA (la unica region que falta en `Legacy_Univariados/`, que ya
tiene CEN/NES/NOR/NTE/OCC/ORI/PEN) con 8 estructuras de modelado
UNIVARIADAS -- CAMBIO DE CRITERIO explicito (2026-08-18): este modulo YA
NO intenta reproducir/aproximar el pipeline legacy perdido de las otras 7
regiones. En su lugar, usa directamente las ARQUITECTURAS ACTUALES del
repo (`ar_model.py`, `sarimax_model.py`, `ar_resid_trend_seasonal_model.py`,
`fcnn_model.py`, `ensemble_stl.py`), sin ninguna exogena, bajo la misma
ventana temporal que el bloque legacy -- sin afirmar en ningun momento
equivalencia metodologica con el script legacy original (que no esta
disponible en este repo; ver la comparacion metodologica hecha aparte para
`STL_AR_residuos_AIC` como ejemplo de por que esa afirmacion no se puede
sostener).

Las 8 estructuras que se mantienen (mismos nombres que antes, por
continuidad de codigo/tests, pero SIN pretension de fidelidad legacy):
  - ARIMA(1,1,1)
  - SARIMA(1,1,1)(1,0,1,168)
  - AR por AIC
  - STL + AR sobre residuos
  - FCNN univariada
  - STL + FCNN sobre residuos
  - LSTM univariada
  - Ensemble STL univariado (LSTM tendencia + FCNN estacionalidad + AR residuo)

Cada fila de salida (metricas) queda marcada EXPLICITAMENTE con:
  - origen      = "bca_univariado_reconstruido"
  - metodologia = "arquitectura_actual_sin_exogenas"
  - train_hours = 384
  - forecast_horizon = 168
para que nunca se confunda con una fila del legacy historico (que no tiene
estas columnas) ni se trate como si fuera la misma metodologia.

Config usada para BCA:
  - train_hours = 384 (16 dias)
  - forecast_horizon = 168 (7 dias)
  - ventana de test: se DERIVA de las ultimas `forecast_horizon` horas de
    `BCA_long.csv` (nunca se hardcodea) -- objetivo esperado 2026-05-17
    00:00 .. 2026-05-23 23:00, la misma ventana que el bloque legacy.
  - cero variables exogenas en las 8 estructuras.

Salida: `metricas_bca_reconstruido.csv` / `series_bca_reconstruido.csv` +
`metadata_reconstruccion.json` en
`Pipeline_Resultados/Legacy_Univariados_BCA_Reconstruido/` -- NUNCA en
`Legacy_Univariados/` (los 2 CSV originales de las 7 regiones no se
tocan). El esquema de `metricas_bca_reconstruido.csv` YA NO es identico al
de `metricas_global.csv` (se le agregaron las 4 columnas de arriba a
proposito, como marca de origen) -- integrarlo al consolidado requiere un
cambio explicito en `aggregator.py`, todavia no hecho.

Estrategias que requieren tensorflow/optuna (FCNN univariada, STL+FCNN
residuos, LSTM univariada, Ensemble STL univariado) se importan de forma
perezosa/opcional -- este modulo es importable sin esas dependencias (para
poder testear las 4 estructuras livianas localmente); intentar EJECUTAR
una estructura pesada sin tensorflow/optuna instalados lanza un error
claro, no una falla silenciosa.
"""

import gc
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .models import ar_model, sarimax_model, ar_resid_trend_seasonal_model
from .metrics import calcular_metricas as calcular_metricas_livianas

try:
    from .models import fcnn_model, ensemble_stl
    MODELOS_PESADOS_DISPONIBLES = True
    _IMPORT_ERROR_PESADOS = None
except ImportError as e:
    fcnn_model = None
    ensemble_stl = None
    MODELOS_PESADOS_DISPONIBLES = False
    _IMPORT_ERROR_PESADOS = e


REGION_BCA = "BCA"
TRAIN_HOURS_BCA = 384
FORECAST_HORIZON_BCA = 168

# Marca de origen/metodologia -- EXPLICITAMENTE distinta de "legacy_univariado"
# (el origen que usa aggregator.py para las 7 regiones historicas). Nunca se
# afirma equivalencia con el pipeline legacy perdido -- ver docstring del modulo.
ORIGEN_BCA = "bca_univariado_reconstruido"
METODOLOGIA_BCA = "arquitectura_actual_sin_exogenas"

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"
NOMBRE_SERIE_BCA = f"{REGION_BCA}_DEMANDA"


# =========================================================
# LAS 8 ESTRUCTURAS: arquitectura actual, sin exogenas, sin pretension de
# fidelidad legacy (ver docstring del modulo)
# =========================================================
#
# "disponible_arquitectura_actual" responde una pregunta distinta a la que
# respondia el campo "reproducible" antes de este cambio de criterio: YA NO
# es "¿esto reproduce fielmente el script legacy perdido?" (pregunta que no
# se puede responder sin ese codigo) sino "¿esta estructura de modelado
# existe hoy en el repo y se puede aplicar sin exogenas?" -- las 8 lo
# cumplen.
#
# Las 4 "livianas" (sin tensorflow/optuna) llaman funciones YA EXISTENTES
# casi sin envoltorio (sarimax_model.entrenar_predecir_sarimax ya acepta
# X_train=None/X_test=None de forma nativa; ar_model ya es univariado por
# diseno).
#
# Las 4 "pesadas" componen unicamente piezas ya probadas:
#   - LSTM univariada y el Ensemble univariado usan
#     ensemble_stl.construir_lstm/tunear_lstm/entrenar_lstm_final/
#     forecast_recursivo_lstm/crear_ventanas con exog=None -- soporte
#     NATIVO de esas funciones (`exog: Optional = None` ya en su firma).
#   - FCNN univariada y STL+FCNN residuos usan
#     fcnn_model.crear_ventanas_multivariadas/tunear_fcnn_optuna/
#     entrenar_fcnn_final/forecast_recursivo_fcnn_multivariada con un
#     DataFrame de exogenas de 0 columnas (mismo n de filas que y).
#     Esto NO se probo end-to-end localmente (fcnn_model.py requiere
#     tensorflow/optuna, no instalados aqui) -- mismo tipo de limitacion ya
#     documentada en lstm_resid_model.py ("requiere smoke test en Colab").
ESTRATEGIAS_BCA = {
    "ARIMA_1_1_1": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": False,
        "fuente": "sarimax_model.entrenar_predecir_sarimax(order=(1,1,1), seasonal_order=(0,0,0,0), X_train=None, X_test=None)",
        "nota": "ARIMA(1,1,1) puro -- sin estacionalidad, sin exogenas. No se afirma que sea el mismo ARIMA_1_1_1 del legacy historico (no hay codigo fuente de ese script).",
    },
    "SARIMA_1_1_1__1_0_1_168": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": False,
        "fuente": "sarimax_model.entrenar_predecir_sarimax(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER, X_train=None, X_test=None)",
        "nota": "SARIMA(1,1,1)(1,0,1,168) -- mismas constantes que sarimax_model.py vigente, sin exogenas. No se afirma equivalencia con el script legacy perdido.",
    },
    "AR_AIC": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": False,
        "fuente": "ar_model.forecast_ar / seleccionar_ar_por_aic (directo sobre demanda)",
        "nota": "AR con orden elegido por AIC (barrido 1-168), arquitectura actual de ar_model.py, univariada por diseno.",
    },
    "STL_AR_residuos_AIC": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": False,
        "fuente": "ar_resid_trend_seasonal_model.py completo (STL + tendencia lineal + estacionalidad repetida + AR-AIC sobre residuo)",
        "nota": (
            "Estructura 'STL + AR sobre residuo' con la arquitectura ACTUAL del repo "
            "(ar_resid_trend_seasonal_model.py es una combinacion nueva del repo, no una "
            "extraccion legacy -- ver comparacion metodologica aparte). Ya NO se afirma "
            "equivalencia exacta con la estrategia legacy STL_AR_residuos_AIC de las otras "
            "7 regiones -- son estructuras conceptualmente similares, nada mas."
        ),
    },
    "FCNN_Individual": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": True,
        "fuente": "fcnn_model.crear_ventanas_multivariadas/tunear_fcnn_optuna/entrenar_fcnn_final/forecast_recursivo_fcnn_multivariada con exog de 0 columnas",
        "nota": "FCNN univariada -- misma arquitectura/espacio Optuna que fcnn_model.py, sin ninguna exogena. Sin smoke test local (requiere tensorflow/optuna).",
    },
    "STL_FCNN_residuos": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": True,
        "fuente": "descomponer_stl + FCNN (0 columnas de exogena) sobre el residuo + reconstruccion trend+seasonal",
        "nota": "STL + FCNN univariada sobre residuo -- misma arquitectura que la variante 'residuos' de fcnn_model.py, sin exogenas. Sin smoke test local.",
    },
    "LSTM_Individual": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": True,
        "fuente": "ensemble_stl.construir_lstm/tunear_lstm/entrenar_lstm_final/forecast_recursivo_lstm con exog=None (soporte nativo)",
        "nota": "LSTM univariada -- misma arquitectura LSTM-sobre-ventana-aplanada de ensemble_stl.py, aplicada directo a la demanda cruda. Sin smoke test local.",
    },
    "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid": {
        "disponible_arquitectura_actual": True,
        "requiere_pesados": True,
        "fuente": "ensemble_stl.py completo (LSTM tendencia + FCNN estacionalidad + AR residuo) con exog=None en LSTM/FCNN",
        "nota": "Ensemble STL univariado -- misma arquitectura del Ensemble moderno, sin exogenas. Sin smoke test local.",
    },
}

assert len(ESTRATEGIAS_BCA) == 8, f"se esperaban exactamente 8 estructuras BCA, hay {len(ESTRATEGIAS_BCA)}"
assert all(v["disponible_arquitectura_actual"] for v in ESTRATEGIAS_BCA.values()), (
    "si alguna estructura deja de estar disponible con la arquitectura actual, debe quedar "
    "disponible_arquitectura_actual=False y documentarse que falta -- no se inventa una implementacion"
)


# =========================================================
# EXTRAER SERIE Y SPLIT (convencion horaria CORREGIDA -- Hora-1 incondicional)
# =========================================================

def extraer_serie_bca(df: pd.DataFrame):
    """Reutiliza ar_model.extraer_serie_horaria (Hora-1 incondicional, la convencion correcta) -- ningun codigo nuevo."""
    return ar_model.extraer_serie_horaria(df, COL_DEMANDA, NOMBRE_SERIE_BCA)


def _split_train_test(serie, fechas, train_hours: int, forecast_horizon: int):
    requeridas = train_hours + forecast_horizon
    if len(serie) < requeridas:
        raise ValueError(
            f"Serie BCA insuficiente: {len(serie)} horas disponibles, se requieren al menos {requeridas} "
            f"(train_hours={train_hours} + forecast_horizon={forecast_horizon})."
        )

    serie_reciente = serie[-requeridas:]
    fechas_recientes = fechas[-requeridas:]

    train = serie_reciente[:-forecast_horizon]
    test = serie_reciente[-forecast_horizon:]
    fechas_test = fechas_recientes[-forecast_horizon:]

    return train, test, fechas_test


def _exog_vacio(n: int) -> pd.DataFrame:
    """DataFrame de 0 columnas y n filas -- mecanicamente equivalente a 'sin exogenas' para las funciones de fcnn_model.py (ver ESTRATEGIAS_BCA)."""
    return pd.DataFrame(index=range(n))


# =========================================================
# ESTRATEGIAS LIVIANAS (sin tensorflow/optuna)
# =========================================================

def _ejecutar_arima_1_1_1(train, test, horizon):
    pred, aic, bic = sarimax_model.entrenar_predecir_sarimax(
        train=train, horizon=horizon, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
        X_train=None, X_test=None,
    )
    metricas = calcular_metricas_livianas(test, pred)
    config_extra = {"order": "(1, 1, 1)", "seasonal_order": "(0, 0, 0, 0)", "aic": aic, "bic": bic}
    return pred, metricas, config_extra, None


def _ejecutar_sarima(train, test, horizon):
    pred, aic, bic = sarimax_model.entrenar_predecir_sarimax(
        train=train, horizon=horizon,
        order=sarimax_model.SARIMA_ORDER, seasonal_order=sarimax_model.SARIMA_SEASONAL_ORDER,
        X_train=None, X_test=None,
    )
    metricas = calcular_metricas_livianas(test, pred)
    config_extra = {
        "order": str(sarimax_model.SARIMA_ORDER), "seasonal_order": str(sarimax_model.SARIMA_SEASONAL_ORDER),
        "aic": aic, "bic": bic,
    }
    return pred, metricas, config_extra, None


def _ejecutar_ar_aic(train, test, horizon):
    pred, lag_optimo, df_lags = ar_model.forecast_ar(train, horizon)
    metricas = calcular_metricas_livianas(test, pred)
    config_extra = {"lag_optimo": lag_optimo, "trend": "c", "max_lag": ar_model.MAX_LAG_AR}
    return pred, metricas, config_extra, df_lags


def _ejecutar_stl_ar_residuos_aic(train, test, horizon):
    m = ar_resid_trend_seasonal_model
    trend, seasonal, resid, stl_res = m.descomponer_stl(train)
    del stl_res

    trend_forecast = m.forecast_tendencia_lineal(trend, horizon=horizon)
    seasonal_forecast = m.forecast_estacionalidad_repetida(seasonal, horizon=horizon, period=m.STL_PERIOD)

    resid_forecast, lag_optimo, df_lags = m.forecast_ar_resid(resid, horizon=horizon)
    pred = trend_forecast + seasonal_forecast + resid_forecast

    metricas = m.calcular_metricas(test, pred)
    config_extra = {"lag_resid_optimo": lag_optimo, "trend": "c", "stl_period": m.STL_PERIOD, "max_lag": m.MAX_LAG_AR}
    return pred, metricas, config_extra, df_lags


# =========================================================
# ESTRATEGIAS PESADAS (requieren tensorflow/optuna)
# =========================================================

def _requerir_pesados(nombre_estrategia):
    if not MODELOS_PESADOS_DISPONIBLES:
        raise ImportError(
            f"La estrategia '{nombre_estrategia}' requiere tensorflow/optuna (fcnn_model.py/"
            f"ensemble_stl.py no se pudieron importar en este entorno: {_IMPORT_ERROR_PESADOS}). "
            "Correr en Colab, donde ambos estan instalados."
        )


def _ejecutar_fcnn_individual(train, test, horizon, nombre_serie, n_trials, output_dir):
    _requerir_pesados("FCNN_Individual")
    m = fcnn_model

    exog_train = _exog_vacio(len(train))
    exog_test = _exog_vacio(horizon)

    best_params, best_smape, trials_df = m.tunear_fcnn_optuna(
        train_y=train, train_exog=exog_train, nombre_serie=nombre_serie,
        modelo_nombre="FCNN_Individual_BCA", n_trials=n_trials, output_dir=output_dir,
        forecast_horizon=horizon,
    )
    model, scaler_x, scaler_y, _ = m.entrenar_fcnn_final(train, exog_train, best_params, horizon)
    pred = m.forecast_recursivo_fcnn_multivariada(
        model=model, y_train=train, exog_train=exog_train, exog_future=exog_test,
        horizon=horizon, window=m.WINDOW, scaler_x=scaler_x, scaler_y=scaler_y,
    )
    metricas = m.calcular_metricas(test, pred)

    del model, scaler_x, scaler_y
    gc.collect()

    config_extra = {"parametros": json.dumps(best_params), "best_val_sMAPE": best_smape, "window": m.WINDOW}
    return pred, metricas, config_extra, trials_df


def _ejecutar_stl_fcnn_residuos(train, test, horizon, nombre_serie, n_trials, output_dir):
    _requerir_pesados("STL_FCNN_residuos")
    m = fcnn_model

    trend, seasonal, resid, stl_res = m.descomponer_stl(train)
    del stl_res

    trend_forecast = m.forecast_tendencia_lineal(trend, horizon=horizon)
    seasonal_forecast = m.forecast_estacionalidad_repetida(seasonal, horizon=horizon, period=m.STL_PERIOD)

    exog_train = _exog_vacio(len(resid))
    exog_test = _exog_vacio(horizon)

    best_params, best_smape, trials_df = m.tunear_fcnn_optuna(
        train_y=resid, train_exog=exog_train, nombre_serie=nombre_serie,
        modelo_nombre="STL_FCNN_residuos_BCA", n_trials=n_trials, output_dir=output_dir,
        forecast_horizon=horizon,
    )
    model, scaler_x, scaler_y, _ = m.entrenar_fcnn_final(resid, exog_train, best_params, horizon)
    resid_forecast = m.forecast_recursivo_fcnn_multivariada(
        model=model, y_train=resid, exog_train=exog_train, exog_future=exog_test,
        horizon=horizon, window=m.WINDOW, scaler_x=scaler_x, scaler_y=scaler_y,
    )
    pred = trend_forecast + seasonal_forecast + resid_forecast
    metricas = m.calcular_metricas(test, pred)

    del model, scaler_x, scaler_y
    gc.collect()

    config_extra = {"parametros": json.dumps(best_params), "best_val_sMAPE": best_smape, "window": m.WINDOW, "stl_period": m.STL_PERIOD}
    return pred, metricas, config_extra, trials_df


def _ejecutar_lstm_individual(train, test, horizon, nombre_serie, n_trials, output_dir):
    _requerir_pesados("LSTM_Individual")
    e = ensemble_stl

    params, best_mae, trials_df = e.tunear_lstm(train, nombre_serie, exog_train=None, n_trials=n_trials, output_dir=output_dir, forecast_horizon=horizon)
    model, scaler_x, scaler_y = e.entrenar_lstm_final(train, params, exog_train=None, forecast_horizon=horizon)
    pred = e.forecast_recursivo_lstm(model, train, horizon=horizon, scaler_x=scaler_x, scaler_y=scaler_y, exog_hist=None, exog_future=None)
    metricas = e.calcular_metricas(test, pred)

    del model, scaler_x, scaler_y
    gc.collect()

    config_extra = {"parametros": json.dumps(params), "best_val_MAE": best_mae, "window": e.WINDOW}
    return pred, metricas, config_extra, trials_df


def _ejecutar_ensemble_univariado(train, test, horizon, nombre_serie, n_trials, output_dir):
    _requerir_pesados("ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid")
    e = ensemble_stl

    trend, seasonal, resid, stl_res = e.descomponer_stl(train)
    del stl_res

    params_trend, best_mae_trend, trials_trend = e.tunear_lstm(trend, nombre_serie, exog_train=None, n_trials=n_trials, output_dir=output_dir, forecast_horizon=horizon)
    lstm_trend, sx_trend, sy_trend = e.entrenar_lstm_final(trend, params_trend, exog_train=None, forecast_horizon=horizon)
    trend_pred = e.forecast_recursivo_lstm(lstm_trend, trend, horizon=horizon, scaler_x=sx_trend, scaler_y=sy_trend, exog_hist=None, exog_future=None)
    del lstm_trend, sx_trend, sy_trend
    gc.collect()

    params_seasonal, best_mae_seasonal, trials_seasonal = e.tunear_fcnn(seasonal, nombre_serie, exog_train=None, n_trials=n_trials, output_dir=output_dir, forecast_horizon=horizon)
    fcnn_season, sx_season, sy_season = e.entrenar_fcnn_final(seasonal, params_seasonal, exog_train=None, forecast_horizon=horizon)
    seasonal_pred = e.forecast_recursivo_fcnn(fcnn_season, seasonal, horizon=horizon, scaler_x=sx_season, scaler_y=sy_season, exog_hist=None, exog_future=None)
    del fcnn_season, sx_season, sy_season
    gc.collect()

    resid_pred, ar_resid_modelo, lag_resid, df_lags_resid = e.forecast_ar_resid(resid, horizon=horizon)
    del ar_resid_modelo
    gc.collect()

    pred = trend_pred + seasonal_pred + resid_pred
    metricas = e.calcular_metricas(test, pred)

    trials_trend = trials_trend.copy(); trials_trend["modelo"] = "LSTM_trend"
    trials_seasonal = trials_seasonal.copy(); trials_seasonal["modelo"] = "FCNN_seasonal"
    df_lags_resid = df_lags_resid.copy(); df_lags_resid["modelo"] = "AR_resid"
    trials_df = pd.concat([trials_trend, trials_seasonal, df_lags_resid], ignore_index=True, sort=False)

    config_extra = {
        "trend_params": json.dumps(params_trend), "seasonal_params": json.dumps(params_seasonal),
        "resid_lag_optimo": lag_resid, "trend_best_val_MAE": best_mae_trend, "seasonal_best_val_MAE": best_mae_seasonal,
        "stl_period": e.STL_PERIOD, "window": e.WINDOW,
    }
    return pred, metricas, config_extra, trials_df


_DISPATCH_LIVIANAS = {
    "ARIMA_1_1_1": lambda train, test, horizon, **kw: _ejecutar_arima_1_1_1(train, test, horizon),
    "SARIMA_1_1_1__1_0_1_168": lambda train, test, horizon, **kw: _ejecutar_sarima(train, test, horizon),
    "AR_AIC": lambda train, test, horizon, **kw: _ejecutar_ar_aic(train, test, horizon),
    "STL_AR_residuos_AIC": lambda train, test, horizon, **kw: _ejecutar_stl_ar_residuos_aic(train, test, horizon),
}

_DISPATCH_PESADAS = {
    "FCNN_Individual": _ejecutar_fcnn_individual,
    "STL_FCNN_residuos": _ejecutar_stl_fcnn_residuos,
    "LSTM_Individual": _ejecutar_lstm_individual,
    "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid": _ejecutar_ensemble_univariado,
}


def ejecutar_estrategia_bca(nombre_estrategia, train, test, horizon, nombre_serie=NOMBRE_SERIE_BCA, n_trials=5, output_dir="."):
    """Dispatcher unico: devuelve (pred, metricas, config_extra, trials_df_o_None)."""
    if nombre_estrategia in _DISPATCH_LIVIANAS:
        return _DISPATCH_LIVIANAS[nombre_estrategia](train, test, horizon)
    if nombre_estrategia in _DISPATCH_PESADAS:
        return _DISPATCH_PESADAS[nombre_estrategia](train, test, horizon, nombre_serie, n_trials, output_dir)
    raise ValueError(f"Estrategia BCA desconocida: {nombre_estrategia!r}. Validas: {list(ESTRATEGIAS_BCA)}")


# =========================================================
# ORQUESTACION COMPLETA
# =========================================================

def run_bca_reconstruido(
    data_dir: str,
    output_dir: str,
    train_hours: int = TRAIN_HOURS_BCA,
    forecast_horizon: int = FORECAST_HORIZON_BCA,
    optuna_n_trials: int = 5,
    estrategias: Optional[list] = None,
):
    """
    Corre las estructuras BCA solicitadas (default: las 8) y escribe
    `metricas_bca_reconstruido.csv`/`series_bca_reconstruido.csv`/
    `metadata_reconstruccion.json` en `output_dir` (pensado para
    `Pipeline_Resultados/Legacy_Univariados_BCA_Reconstruido/`).

    Cada fila de `metricas_bca_reconstruido.csv` queda marcada con
    `origen=ORIGEN_BCA` ("bca_univariado_reconstruido"),
    `metodologia=METODOLOGIA_BCA` ("arquitectura_actual_sin_exogenas"),
    `train_hours` y `forecast_horizon` -- para que nunca se confunda con
    una fila de `metricas_global.csv` (legacy historico de las otras 7
    regiones), que no tiene estas columnas.

    Solo lee `{data_dir}/BCA_long.csv`; NUNCA toca `Legacy_Univariados/`
    (los CSV de las otras 7 regiones) ni ninguna carpeta de
    `Pipeline_Resultados/<RUN_NAME>/`.
    """
    estrategias = list(estrategias) if estrategias is not None else list(ESTRATEGIAS_BCA)
    desconocidas = [e for e in estrategias if e not in ESTRATEGIAS_BCA]
    if desconocidas:
        raise ValueError(f"Estrategias no reconocidas: {desconocidas}. Validas: {list(ESTRATEGIAS_BCA)}")

    archivo_bca = os.path.join(data_dir, f"{REGION_BCA}_long.csv")
    if not os.path.exists(archivo_bca):
        raise FileNotFoundError(f"No existe {archivo_bca}")

    df_bca = pd.read_csv(archivo_bca)
    df_bca.columns = df_bca.columns.astype(str).str.strip()

    serie, fechas = extraer_serie_bca(df_bca)
    train, test, fechas_test = _split_train_test(serie, fechas, train_hours, forecast_horizon)

    series_rows, metric_rows = [], []
    trials_frames = []
    errores = {}

    for nombre_estrategia in estrategias:
        print(f"\n{'=' * 80}\nBCA reconstruido -- {nombre_estrategia}\n{'=' * 80}")
        try:
            pred, metricas, config_extra, trials_df = ejecutar_estrategia_bca(
                nombre_estrategia, train, test, forecast_horizon,
                nombre_serie=NOMBRE_SERIE_BCA, n_trials=optuna_n_trials, output_dir=output_dir,
            )
        except Exception as e:
            print(f"Error en {nombre_estrategia}: {type(e).__name__}: {e}")
            errores[nombre_estrategia] = f"{type(e).__name__}: {e}"
            continue

        for j, val in enumerate(pred):
            series_rows.append({
                "serie": NOMBRE_SERIE_BCA, "fecha": fechas_test[j], "tipo": "prediccion",
                "subset": "test", "modelo": nombre_estrategia, "valor": val,
            })

        metric_rows.append({
            "serie": NOMBRE_SERIE_BCA, "modelo": nombre_estrategia,
            "MAE": metricas["MAE"], "RMSE": metricas["RMSE"], "MAPE": metricas["MAPE"], "sMAPE": metricas["sMAPE"],
            "origen": ORIGEN_BCA, "metodologia": METODOLOGIA_BCA,
            "train_hours": train_hours, "forecast_horizon": forecast_horizon,
        })

        if trials_df is not None and len(trials_df) > 0:
            trials_df = trials_df.copy()
            trials_df["estrategia"] = nombre_estrategia
            trials_frames.append(trials_df)

        print(f"{nombre_estrategia}: MAPE={metricas['MAPE']:.2f}%")

    if metric_rows:
        # 'real' una sola vez -- serie completa, misma convencion que el resto del proyecto.
        for j, fecha in enumerate(fechas):
            series_rows.append({
                "serie": NOMBRE_SERIE_BCA, "fecha": fecha, "tipo": "real",
                "subset": "completo", "modelo": "real", "valor": serie[j],
            })

    series_df = pd.DataFrame(series_rows)
    if len(series_df) > 0:
        series_df["fecha"] = pd.to_datetime(series_df["fecha"], errors="coerce")
        series_df = series_df.sort_values(["serie", "fecha", "modelo"])

    metricas_df = pd.DataFrame(metric_rows)

    os.makedirs(output_dir, exist_ok=True)

    metricas_path = os.path.join(output_dir, "metricas_bca_reconstruido.csv")
    series_path = os.path.join(output_dir, "series_bca_reconstruido.csv")
    metadata_path = os.path.join(output_dir, "metadata_reconstruccion.json")

    metricas_df.to_csv(metricas_path, index=False, encoding="utf-8-sig")
    series_df.to_csv(series_path, index=False, encoding="utf-8-sig")

    metadata = {
        "origen": ORIGEN_BCA,
        "metodologia": METODOLOGIA_BCA,
        "region": REGION_BCA,
        "train_hours": train_hours,
        "forecast_horizon": forecast_horizon,
        "fecha_test_inicio": str(pd.Timestamp(fechas_test[0])) if len(fechas_test) else None,
        "fecha_test_fin": str(pd.Timestamp(fechas_test[-1])) if len(fechas_test) else None,
        "estrategias_solicitadas": estrategias,
        "estrategias_completadas": [r["modelo"] for r in metric_rows],
        "estrategias_con_error": errores,
        "aviso": (
            "Reconstruccion con la ARQUITECTURA ACTUAL del repo, sin exogenas -- NO afirma "
            "equivalencia metodologica con el pipeline legacy perdido de las otras 7 regiones "
            "de Legacy_Univariados/. Ver docstring de legacy_bca_reconstruido.py y "
            "ESTRATEGIAS_BCA[*]['nota'] para el detalle de cada estructura."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nEscrito: {metricas_path}")
    print(f"Escrito: {series_path}")
    print(f"Escrito: {metadata_path}")

    if errores:
        print(f"\nAVISO: {len(errores)} estrategia(s) con error: {list(errores)}")

    trials_completos_df = pd.concat(trials_frames, ignore_index=True, sort=False) if trials_frames else pd.DataFrame()
    if len(trials_completos_df) > 0:
        trials_path = os.path.join(output_dir, "trials_bca_reconstruido.csv")
        trials_completos_df.to_csv(trials_path, index=False, encoding="utf-8-sig")
        print(f"Escrito: {trials_path}")

    return metricas_df, series_df, metadata

"""
Tests del agregador de resultados (src/tesis_forecast/aggregator.py).

Fixtures 100% sinteticas -- ninguna corrida real, ningun modelo se
entrena. Cada fixture escribe a mano una carpeta
`Pipeline_Resultados/<RUN_NAME>/` con `config.json`/`metricas.csv`/
`series.csv` (y `trials.csv`/`config_usada.csv` donde aplica) siguiendo
EXACTAMENTE el esquema que ya escriben los modulos reales (columnas
confirmadas leyendo el codigo fuente, no de memoria -- ver
`_escribir_run_1a_estilo_xgboost`/`_escribir_run_1b_univariado`/etc. mas
abajo para el detalle).

13 escenarios cubiertos, uno por funcion `test_*`, en el mismo orden en
que se pidieron:
  1. clasificacion 1A
  2. clasificacion 1B
  3. clasificacion individual
  4. clasificacion temp_igae (el bugfix central de esta tarea: NO debe
     confundirse con 1A/1B solo por el modelo)
  5. FCNN con dos estrategias conserva ambas
  6. deduplicacion de 'real'
  7. componente_pred excluido
  8. ventanas de prediccion distintas
  9. conflicto de valores reales reportado
  10. corrida incompleta excluida
  11. idempotencia
  12. Consolidado/ no se redescubre como run
  13. esquema consistente aun si los modelos tienen columnas distintas

Solo depende de pandas/numpy -- corre en cualquier entorno.

Uso:
    python tests/test_aggregator.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast import aggregator as agg
from tesis_forecast.config import build_run_name
from tesis_forecast.regions import REGIONS_ALL

REGIONES_TEST = ["BCA", "CEN"]

# 7 regiones, SIN BCA -- mismo patron confirmado empiricamente para
# Legacy_Univariados (falta BCA en las corridas legacy).
REGIONES_LEGACY = [r for r in REGIONS_ALL if r != "BCA"]


# =========================================================
# FIXTURES: construyen carpetas de corrida sinteticas
# =========================================================

def _config_json(run_dir, run_name, modelo, exogenas, train_hours=336, forecast_horizon=4,
                  optuna_n_trials=10, seed=42, notas=""):
    cfg = {
        "modelo": modelo,
        "exogenas": exogenas,
        "train_hours": train_hours,
        "forecast_horizon": forecast_horizon,
        "optuna_n_trials": optuna_n_trials,
        "seed": seed,
        "notas": notas,
        "run_name": run_name,
        "git_commit": "abc1234",
        "generated_at": "2026-08-09T00:00:00+00:00",
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)


def _pred_fechas(forecast_horizon, offset_horas=0):
    base = pd.Timestamp("2024-06-01") + pd.Timedelta(hours=offset_horas)
    return pd.date_range(base, periods=forecast_horizon, freq="h")


def _escribir_run_1a_estilo_xgboost(pipeline_dir, run_name, modelo, exogenas, regiones,
                                     train_hours=336, forecast_horizon=4, mape_base=2.0):
    """
    Estilo xgboost/lightgbm/lstm_direct: SIN filas 'real' en series.csv
    (confirmado leyendo xgboost_model.py/lightgbm_model.py/lstm_direct.py
    -- solo guardan predicciones), CON trials.csv/config_usada.csv, y
    columnas 'tuneado'/'horizonte_usado' en metricas.csv. Sirve tambien
    para lstm_resid/otros modelos con exogenas: solo la forma de los CSV
    importa aqui, no el modelo puntual (clasificar_familia depende de
    config.json, no del formato de metricas.csv).
    """
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _config_json(run_dir, run_name, modelo, exogenas, train_hours, forecast_horizon)

    series_rows, metric_rows, trials_rows, config_rows = [], [], [], []
    for i, region in enumerate(regiones):
        nombre_serie = f"{region}_DEMANDA"
        fechas = _pred_fechas(forecast_horizon)
        for h, fecha in enumerate(fechas):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": f"{modelo}_Tuned", "valor": 100.0 + h, "region": region,
            })
        metric_rows.append({
            "serie": nombre_serie, "modelo": f"{modelo}_Tuned", "tuneado": True,
            "horizonte_usado": f"{forecast_horizon}_horas",
            "MAPE": mape_base + i, "sMAPE": mape_base + i, "MAE": 1.0, "RMSE": 1.5, "region": region,
        })
        for lag in range(3):
            trials_rows.append({"serie": nombre_serie, "modelo": modelo, "number": lag, "value": 1.0, "region": region})
        config_rows.append({"serie": nombre_serie, "modelo": modelo, "parametros": "{}", "region": region})

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(trials_rows).to_csv(os.path.join(run_dir, "trials.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(config_rows).to_csv(os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_sarimax(pipeline_dir, run_name, exogenas, regiones, train_hours=1440, forecast_horizon=4,
                           mape_base=1.5, offset_horas=0, valor_real_base=100.0):
    """
    Estilo sarimax: SI guarda 'real' (serie completa, aqui simplificada a
    train_hours+forecast_horizon horas), tiene su PROPIA columna
    'exogenas' en metricas.csv (nombres internos) -- justo el caso de
    colision que debe resolver `_evitar_colision_metadata`. Sin
    trials.csv (SARIMAX no tunea), con config_usada.csv.
    """
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _config_json(run_dir, run_name, "sarimax", exogenas, train_hours, forecast_horizon)

    series_rows, metric_rows, config_rows = [], [], []
    n_real = train_hours + forecast_horizon
    for i, region in enumerate(regiones):
        nombre_serie = f"{region}_DEMANDA"
        fechas_reales = _pred_fechas(n_real, offset_horas=offset_horas - train_hours)
        for h, fecha in enumerate(fechas_reales):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "real",
                "subset": "completo", "modelo": "real", "valor": valor_real_base + h, "region": region,
            })
        fechas_test = _pred_fechas(forecast_horizon, offset_horas=offset_horas)
        for h, fecha in enumerate(fechas_test):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": "SARIMAX_1_1_1", "valor": 90.0 + h, "region": region,
            })
        metric_rows.append({
            "serie": nombre_serie, "modelo": "SARIMAX_1_1_1", "order": "(1, 1, 1)",
            "AIC": 123.4, "BIC": 130.0,
            "MAPE": mape_base + i, "sMAPE": mape_base + i, "MAE": 1.0, "RMSE": 1.2,
            "exogenas": str(["Temperaturas", "IGAE"]),  # nombres internos -- debe colisionar y renombrarse
            "region": region,
        })
        config_rows.append({"serie": nombre_serie, "modelo": "SARIMAX_1_1_1", "parametros": "{}", "region": region})

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(config_rows).to_csv(os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_1b_univariado(pipeline_dir, run_name, modelo, regiones, forecast_horizon=4,
                                 mape_base=5.0, offset_horas=0, valor_real_base=100.0, train_hours=20):
    """
    Estilo naive/naive_trend/naive_trend_seasonal/ar/ar_resid_trend_seasonal:
    univariado (exogenas=[]), SI guarda 'real' (serie completa), sin
    trials/config_usada (naive/naive_trend/naive_trend_seasonal) por
    simplicidad -- el punto de esta fixture es family=1B + dedup de real.
    """
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _config_json(run_dir, run_name, modelo, [], train_hours, forecast_horizon)

    series_rows, metric_rows = [], []
    n_real = train_hours + forecast_horizon
    for i, region in enumerate(regiones):
        nombre_serie = f"{region}_DEMANDA"
        fechas_reales = _pred_fechas(n_real, offset_horas=offset_horas - train_hours)
        for h, fecha in enumerate(fechas_reales):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "real",
                "subset": "completo", "modelo": "real", "valor": valor_real_base + h, "region": region,
            })
        fechas_test = _pred_fechas(forecast_horizon, offset_horas=offset_horas)
        for h, fecha in enumerate(fechas_test):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": modelo, "valor": 80.0 + h, "region": region,
            })
        metric_rows.append({
            "serie": nombre_serie, "modelo": modelo, "tuneado": False, "horizonte_usado": "serie_completa",
            "MAPE": mape_base + i, "sMAPE": mape_base + i, "MAE": 1.0, "RMSE": 1.2, "region": region,
        })

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_ensemble_con_componentes(pipeline_dir, run_name, exogenas, regiones, forecast_horizon=4,
                                            train_hours=20, mape_base=3.0, offset_horas=0):
    """Estilo ensemble_stl: real + prediccion final + 3 filas 'componente_pred' por hora -- estas ultimas deben quedar excluidas de series_master."""
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _config_json(run_dir, run_name, "ensemble_stl", exogenas, train_hours, forecast_horizon)

    series_rows, metric_rows, trials_rows, config_rows = [], [], [], []
    n_real = train_hours + forecast_horizon
    for i, region in enumerate(regiones):
        nombre_serie = f"{region}_DEMANDA"
        fechas_reales = _pred_fechas(n_real, offset_horas=offset_horas - train_hours)
        for h, fecha in enumerate(fechas_reales):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "real",
                "subset": "completo", "modelo": "real", "valor": 100.0 + h, "region": region,
            })
        fechas_test = _pred_fechas(forecast_horizon, offset_horas=offset_horas)
        for h, fecha in enumerate(fechas_test):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": "ENSEMBLE_STL_FINAL", "valor": 95.0 + h, "region": region,
            })
            for comp in ["LSTM_trend", "FCNN_seasonal", "AR_resid"]:
                series_rows.append({
                    "serie": nombre_serie, "fecha": fecha, "tipo": "componente_pred",
                    "subset": "test", "modelo": comp, "valor": 30.0 + h, "region": region,
                })
        metric_rows.append({
            "serie": nombre_serie, "modelo": "ENSEMBLE_STL_FINAL",
            "MAPE": mape_base + i, "sMAPE": mape_base + i, "MAE": 1.0, "RMSE": 1.3, "region": region,
        })
        trials_rows.append({"serie": nombre_serie, "modelo": "LSTM_trend", "number": 0, "value": 1.0, "region": region})
        config_rows.append({"serie": nombre_serie, "modelo": "ENSEMBLE_STL_FINAL", "parametros": "{}", "region": region})

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(trials_rows).to_csv(os.path.join(run_dir, "trials.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(config_rows).to_csv(os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_fcnn_2_estrategias(pipeline_dir, run_name, exogenas, regiones, forecast_horizon=4, train_hours=20):
    """Estilo fcnn: 2 filas de metricas por region (directa + STL-residuos)."""
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _config_json(run_dir, run_name, "fcnn", exogenas, train_hours, forecast_horizon)

    series_rows, metric_rows = [], []
    for region in regiones:
        nombre_serie = f"{region}_DEMANDA"
        fechas_test = _pred_fechas(forecast_horizon)
        for modelo_estrategia, base in [("FCNN_Multivariada_EXOG_Lag168", 70.0), ("STL_FCNN_Multivariada_Residuos_EXOG_Lag168", 75.0)]:
            for h, fecha in enumerate(fechas_test):
                series_rows.append({
                    "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                    "subset": "test", "modelo": modelo_estrategia, "valor": base + h, "region": region,
                })
            metric_rows.append({
                "serie": nombre_serie, "modelo": modelo_estrategia,
                "MAPE": 4.0, "sMAPE": 4.0, "MAE": 1.0, "RMSE": 1.1, "region": region,
            })
        fechas_reales = _pred_fechas(train_hours + forecast_horizon, offset_horas=-train_hours)
        for h, fecha in enumerate(fechas_reales):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "real",
                "subset": "completo", "modelo": "real", "valor": 100.0 + h, "region": region,
            })

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_incompleto(pipeline_dir, run_name, regiones_presentes):
    """Le falta una region -- validar_resultado() debe marcarla incompleta."""
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _config_json(run_dir, run_name, "naive", [], "auto", "auto")

    metric_rows = []
    series_rows = []
    for region in regiones_presentes:
        nombre_serie = f"{region}_DEMANDA"
        metric_rows.append({"serie": nombre_serie, "modelo": "Naive", "MAPE": 9.0, "sMAPE": 9.0, "MAE": 1.0, "RMSE": 1.0, "region": region})
        series_rows.append({"serie": nombre_serie, "fecha": _pred_fechas(1)[0], "tipo": "prediccion", "subset": "test", "modelo": "Naive", "valor": 1.0, "region": region})

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    return run_dir


# =========================================================
# 1-4. CLASIFICACION DE FAMILIA
# =========================================================

def test_clasificacion_1a():
    """Todo modelo en FAMILIA_1A_MODELOS, con el catalogo COMPLETO de exogenas (no 1, no exactamente Temp+IGAE), clasifica como '1A'."""
    catalogo_completo = ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"]
    for modelo in sorted(agg.FAMILIA_1A_MODELOS):
        familia, exog_ind = agg.clasificar_familia(modelo, catalogo_completo)
        assert familia == "1A", f"{modelo}: esperaba '1A', obtuve {familia!r}"
        assert exog_ind is None

    print("OK test_clasificacion_1a")


def test_clasificacion_1b():
    """Todo modelo en FAMILIA_1B_MODELOS, univariado ([]) o multivariado (catalogo != Temp+IGAE), clasifica como '1B'."""
    for modelo in sorted(agg.FAMILIA_1B_MODELOS):
        familia, exog_ind = agg.clasificar_familia(modelo, [])
        assert familia == "1B", f"{modelo} (univariado): esperaba '1B', obtuve {familia!r}"
        assert exog_ind is None

    # lstm_resid es el unico 1B multivariado -- catalogo completo (no Temp+IGAE) debe seguir siendo 1B
    familia, _ = agg.clasificar_familia("lstm_resid", ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"])
    assert familia == "1B"

    print("OK test_clasificacion_1b")


def test_clasificacion_individual():
    """len(exogenas) == 1 clasifica como 'individual' sin importar el modelo, y exogena_individual es esa unica exogena."""
    for modelo in list(agg.FAMILIA_1A_MODELOS) + list(agg.FAMILIA_1B_MODELOS) + ["modelo_inventado"]:
        for exog in ["Temperatura", "IGAE", "Generacion"]:
            familia, exog_ind = agg.clasificar_familia(modelo, [exog])
            assert familia == "individual", f"{modelo}+{[exog]}: esperaba 'individual', obtuve {familia!r}"
            assert exog_ind == exog

    # len==1 gana incluso si esa unica exogena fuera 'Temperatura' o 'IGAE' -- no se confunde con temp_igae (que exige las 2)
    assert agg.clasificar_familia("xgboost", ["Temperatura"]) == ("individual", "Temperatura")
    assert agg.clasificar_familia("xgboost", ["IGAE"]) == ("individual", "IGAE")

    # modelo no reconocido y sin exogenas -> unknown, nunca se inventa una familia
    assert agg.clasificar_familia("modelo_inventado", []) == ("unknown", None)

    print("OK test_clasificacion_individual")


def test_clasificacion_temp_igae():
    """
    EL BUGFIX CENTRAL: set(exogenas) == {'Temperatura','IGAE'} debe
    clasificar como 'temp_igae' -- NUNCA como '1A' (por el modelo estar en
    FAMILIA_1A_MODELOS) ni como '1B' (idem para lstm_resid). El chequeo de
    temp_igae debe evaluarse ANTES que el chequeo por modelo.
    """
    # Caso critico 1: xgboost es 1A por modelo, pero con Temp+IGAE debe ser temp_igae, no 1A
    assert agg.clasificar_familia("xgboost", ["Temperatura", "IGAE"]) == ("temp_igae", None)
    # mismo caso con las 2 exogenas en el otro orden -- el set() no depende del orden
    assert agg.clasificar_familia("xgboost", ["IGAE", "Temperatura"]) == ("temp_igae", None)

    # Caso critico 2: lstm_resid es 1B por modelo, pero con Temp+IGAE debe ser temp_igae, no 1B
    assert agg.clasificar_familia("lstm_resid", ["Temperatura", "IGAE"]) == ("temp_igae", None)

    # Las 7 configs reales de Fase 3 (mismos RUN_NAME que test_fase3_temp_igae.py) -- ninguna es 1A/1B/individual
    modelos_multivariados_fase3 = ["xgboost", "lightgbm", "lstm_direct", "sarimax", "fcnn", "ensemble_stl", "lstm_resid"]
    for modelo in modelos_multivariados_fase3:
        familia, exog_ind = agg.clasificar_familia(modelo, ["Temperatura", "IGAE"])
        assert familia == "temp_igae", f"{modelo}+Temp+IGAE: esperaba 'temp_igae', obtuve {familia!r}"
        assert exog_ind is None

    # Un conjunto de 2 exogenas que NO sea exactamente {Temperatura, IGAE} no debe entrar en esta regla
    familia, _ = agg.clasificar_familia("xgboost", ["Temperatura", "Generacion"])
    assert familia != "temp_igae"

    # end-to-end: una corrida real de Fase 3 en disco debe clasificar temp_igae via descubrir_runs_completos
    tmp = tempfile.mkdtemp(prefix="agg_test_temp_igae_e2e_")
    try:
        run_name_xgb = build_run_name("xgboost", 336, 168, ["Temperatura", "IGAE"])
        assert run_name_xgb == "XGBoost_train336h_fh168h_Temp-IGAE"
        _escribir_run_1a_estilo_xgboost(tmp, run_name_xgb, "xgboost", ["Temperatura", "IGAE"], REGIONES_TEST,
                                         train_hours=336, forecast_horizon=168)
        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(resultado.runs) == 1
        assert resultado.runs[0].familia_experimento == "temp_igae"
        assert resultado.runs[0].exogena_individual is None

        metricas_df = agg.construir_metricas_master(resultado.runs)
        assert set(metricas_df["familia_experimento"]) == {"temp_igae"}
        assert set(metricas_df["modelo"]) == {"xgboost"}

        print("OK test_clasificacion_temp_igae")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# 5. FCNN CON DOS ESTRATEGIAS
# =========================================================

def test_fcnn_dos_estrategias_preservadas():
    tmp = tempfile.mkdtemp(prefix="agg_test_fcnn_")
    try:
        _escribir_run_fcnn_2_estrategias(
            tmp, "FCNN_train20h_fh4h_baseline",
            ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"], REGIONES_TEST,
        )
        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(resultado.runs) == 1

        metricas_df = agg.construir_metricas_master(resultado.runs)
        # 2 regiones x 2 estrategias = 4 filas, nunca promediadas/recalculadas
        assert len(metricas_df) == 4

        fcnn_bca = metricas_df[metricas_df["region"] == "BCA"]
        assert len(fcnn_bca) == 2
        assert set(fcnn_bca["modelo_estrategia"]) == {"FCNN_Multivariada_EXOG_Lag168", "STL_FCNN_Multivariada_Residuos_EXOG_Lag168"}
        # modelo canonico es el mismo para ambas filas -- 'modelo_estrategia' es lo que las distingue
        assert set(fcnn_bca["modelo"]) == {"fcnn"}
        assert set(fcnn_bca["familia_experimento"]) == {"1A"}

        series_df = agg.construir_series_master(resultado.runs, verbose=False)
        preds_fcnn = series_df[(series_df["serie_tipo"] == "prediccion") & (series_df["region"] == "BCA")]
        assert set(preds_fcnn["modelo_estrategia"]) == {"FCNN_Multivariada_EXOG_Lag168", "STL_FCNN_Multivariada_Residuos_EXOG_Lag168"}

        print("OK test_fcnn_dos_estrategias_preservadas")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# 6-7. SERIES_MASTER: dedup de real + exclusion de componente_pred
# =========================================================

def test_real_deduplicado():
    tmp = tempfile.mkdtemp(prefix="agg_test_real_dedup_")
    try:
        # Dos corridas 1B univariadas, MISMA ventana de test -- su 'real' debe deduplicarse.
        _escribir_run_1b_univariado(tmp, "Naive_auto", "naive", REGIONES_TEST, forecast_horizon=4, offset_horas=0, valor_real_base=100.0)
        _escribir_run_1b_univariado(tmp, "Naive_Trend_auto", "naive_trend", REGIONES_TEST, forecast_horizon=4, offset_horas=0, valor_real_base=100.0)

        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(resultado.runs) == 2

        series_df = agg.construir_series_master(resultado.runs, verbose=False)

        n_real = (series_df["serie_tipo"] == "real").sum()
        # deduplicado por (region, timestamp): 2 regiones x 4 horas = 8, NO 2 x 2 x 4 = 16
        assert n_real == 2 * 4, f"esperaba 8 filas 'real' deduplicadas, hubo {n_real}"

        reales = series_df[series_df["serie_tipo"] == "real"]
        assert reales["run_name"].isna().all()
        assert reales["modelo"].isna().all()
        assert (reales["modelo_estrategia"] == "real").all()
        # nunca deben aparecer duplicados en (region, timestamp) dentro de 'real'
        assert not reales.duplicated(subset=["region", "timestamp"]).any()

        print("OK test_real_deduplicado")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_componente_pred_excluido():
    tmp = tempfile.mkdtemp(prefix="agg_test_componente_pred_")
    try:
        _escribir_run_ensemble_con_componentes(
            tmp, "Ensemble_STL_train20h_fh4h_baseline",
            ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"],
            REGIONES_TEST, forecast_horizon=4, offset_horas=0,
        )
        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(resultado.runs) == 1

        series_df = agg.construir_series_master(resultado.runs, verbose=False)

        assert "componente_pred" not in set(series_df["serie_tipo"])
        # ninguna fila de LSTM_trend/FCNN_seasonal/AR_resid (los componentes internos) debe sobrevivir
        assert not set(series_df["modelo_estrategia"]) & {"LSTM_trend", "FCNN_seasonal", "AR_resid"}

        n_pred = (series_df["serie_tipo"] == "prediccion").sum()
        # solo la prediccion FINAL: 2 regiones x 4 horas = 8 (no 8 x 4 = 32, que incluiria los componentes)
        assert n_pred == 2 * 4, f"esperaba 8 filas de prediccion (solo la final), hubo {n_pred}"
        assert set(series_df[series_df["serie_tipo"] == "prediccion"]["modelo_estrategia"]) == {"ENSEMBLE_STL_FINAL"}

        print("OK test_componente_pred_excluido")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# 8. VENTANAS DE PREDICCION DISTINTAS
# =========================================================

def test_ventanas_prediccion_distintas():
    """
    Dos corridas con ventanas de test QUE NO SE SOLAPAN para la misma
    region: el 'real' de cada una debe conservarse por separado (segun su
    propio horizonte comparable), sin intentar alinearlas a la fuerza ni
    perder ninguna.
    """
    tmp = tempfile.mkdtemp(prefix="agg_test_ventanas_")
    try:
        _escribir_run_1b_univariado(tmp, "Naive_auto", "naive", REGIONES_TEST, forecast_horizon=4, offset_horas=0, valor_real_base=100.0)
        # Ventana de test bien distinta (1000h despues) -- no deberia solaparse
        _escribir_run_1b_univariado(tmp, "AR_auto", "ar", REGIONES_TEST, forecast_horizon=3, offset_horas=1000, valor_real_base=200.0)

        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(resultado.runs) == 2

        buf = io.StringIO()
        with redirect_stdout(buf):
            series_df = agg.construir_series_master(resultado.runs, verbose=True)
        salida = buf.getvalue()
        assert "rangos DISTINTOS de test" in salida, "el reporte de ventanas distintas no se imprimio"

        n_real = (series_df["serie_tipo"] == "real").sum()
        # Sin solape -> union simple: 2 regiones x (4 + 3) horas = 14, ninguna se pierde ni se fuerza a coincidir
        assert n_real == 2 * (4 + 3), f"esperaba 14 filas 'real' (sin solape), hubo {n_real}"

        n_pred = (series_df["serie_tipo"] == "prediccion").sum()
        assert n_pred == 2 * (4 + 3)

        print("OK test_ventanas_prediccion_distintas")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# 9. CONFLICTO DE VALORES REALES
# =========================================================

def test_conflicto_valores_reales_reportado():
    tmp = tempfile.mkdtemp(prefix="agg_test_conflicto_")
    try:
        _escribir_run_1b_univariado(tmp, "Naive_auto", "naive", REGIONES_TEST[:1], forecast_horizon=2, offset_horas=0, valor_real_base=100.0)
        # Mismo horizonte, pero con datos "reales" distintos (simula snapshot de datos distinto)
        _escribir_run_1b_univariado(tmp, "Naive_Trend_auto", "naive_trend", REGIONES_TEST[:1], forecast_horizon=2, offset_horas=0, valor_real_base=999.0)

        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST[:1])

        buf = io.StringIO()
        with redirect_stdout(buf):
            series_df = agg.construir_series_master(resultado.runs, verbose=True)
        salida = buf.getvalue()
        # El conflicto debe reportarse explicitamente -- no promediarse ni descartarse en silencio
        assert "valores 'real' distintos entre corridas" in salida, "el conflicto de valores reales no se reporto"
        assert "2 (region, timestamp)" in salida, f"esperaba reportar 2 conflictos, salida: {salida!r}"

        reales = series_df[series_df["serie_tipo"] == "real"]
        assert len(reales) == 2  # 1 region x 2 horas, deduplicado (no 4)
        # Se conserva consistentemente el valor de UNA sola corrida (la que se
        # proceso primero) para las 2 horas -- no se promedian ni se mezclan.
        assert (reales["valor"] < 200).all() or (reales["valor"] > 200).all(), (
            f"se mezclaron valores de ambas corridas en vez de conservar una sola: {reales['valor'].tolist()}"
        )

        print("OK test_conflicto_valores_reales_reportado")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# 10. CORRIDA INCOMPLETA EXCLUIDA
# =========================================================

def test_corrida_incompleta_excluida():
    tmp = tempfile.mkdtemp(prefix="agg_test_incompleta_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp, "XGBoost_train336h_fh4h_Temp-IGAE", "xgboost",
                                         ["Temperatura", "IGAE"], REGIONES_TEST)
        _escribir_run_incompleto(tmp, "Naive_incompleta", [REGIONES_TEST[0]])  # falta CEN

        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)

        assert len(resultado.runs) == 1, f"esperaba 1 corrida completa, hubo {len(resultado.runs)}"
        assert resultado.runs[0].run_name == "XGBoost_train336h_fh4h_Temp-IGAE"
        assert resultado.n_incompletos == 1, f"esperaba 1 incompleta detectada, hubo {resultado.n_incompletos}"
        assert resultado.incompletos[0][0] == "Naive_incompleta"

        # Nunca se toca/borra la carpeta incompleta -- sigue en disco, intacta
        assert os.path.isdir(os.path.join(tmp, "Naive_incompleta"))

        # reporte_consolidacion: aparece como excluida, con razon
        reporte_df = agg.construir_reporte_consolidacion(resultado)
        fila_incompleta = reporte_df[reporte_df["run_name"] == "Naive_incompleta"].iloc[0]
        assert fila_incompleta["estado"] == "incompleto"
        assert fila_incompleta["incluido"] == False
        assert fila_incompleta["razon"]  # no vacia

        fila_completa = reporte_df[reporte_df["run_name"] == "XGBoost_train336h_fh4h_Temp-IGAE"].iloc[0]
        assert fila_completa["estado"] == "completo"
        assert fila_completa["incluido"] == True

        print("OK test_corrida_incompleta_excluida")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# 11. IDEMPOTENCIA
# =========================================================

def test_idempotencia():
    """
    Simula el caso real: matrices 1A/1B ya completas + una corrida
    'individual' + una corrida todavia incompleta en progreso en otro
    Colab. `consolidar_resultados` debe incluir solo lo completo, y
    correrlo dos veces seguidas sobre EXACTAMENTE los mismos resultados
    debe producir contenido logicamente identico -- sin duplicar filas ni
    depender de que metricas_master.csv/series_master.csv ya existan.
    """
    tmp_pipeline = tempfile.mkdtemp(prefix="agg_test_idem_pipeline_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp_pipeline, "XGBoost_train336h_fh4h_baseline", "xgboost",
                                         ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
                                         REGIONES_TEST)
        _escribir_run_1b_univariado(tmp_pipeline, "Naive_auto", "naive", REGIONES_TEST)
        _escribir_run_1a_estilo_xgboost(tmp_pipeline, "XGBoost_train336h_fh4h_Temp", "xgboost", ["Temperatura"], REGIONES_TEST)
        _escribir_run_incompleto(tmp_pipeline, "XGBoost_train336h_fh4h_IGAE", [REGIONES_TEST[0]])

        # 1ra ejecucion: ni metricas_master.csv ni series_master.csv existen todavia
        metricas_df, series_df, descubrimiento = agg.consolidar_resultados(
            tmp_pipeline, regiones_esperadas=REGIONES_TEST,
        )

        assert len(descubrimiento.runs) == 3
        assert descubrimiento.n_incompletos == 1
        assert set(metricas_df["familia_experimento"]) == {"1A", "1B", "individual"}
        assert (metricas_df[metricas_df["familia_experimento"] == "individual"]["exogena_individual"] == "Temperatura").all()

        output_dir = os.path.join(tmp_pipeline, agg.NOMBRE_CARPETA_CONSOLIDADO)
        assert os.path.exists(os.path.join(output_dir, "metricas_master.csv"))
        assert os.path.exists(os.path.join(output_dir, "series_master.csv"))
        assert os.path.exists(os.path.join(output_dir, "reporte_consolidacion.csv"))

        # No se toco ni se borro NADA dentro de Pipeline_Resultados (las 4 carpetas de corrida siguen ahi, + Consolidado)
        assert set(os.listdir(tmp_pipeline)) == {
            "XGBoost_train336h_fh4h_baseline", "Naive_auto", "XGBoost_train336h_fh4h_Temp",
            "XGBoost_train336h_fh4h_IGAE", agg.NOMBRE_CARPETA_CONSOLIDADO,
        }

        # 2da ejecucion: metricas_master.csv/series_master.csv YA existen -- no debe acumular ni duplicar
        metricas_df2, series_df2, descubrimiento2 = agg.consolidar_resultados(
            tmp_pipeline, regiones_esperadas=REGIONES_TEST,
        )
        assert len(descubrimiento2.runs) == 3, "una 2da corrida no deberia inflar el numero de runs descubiertos"

        pd.testing.assert_frame_equal(
            metricas_df.sort_values(["run_name", "region"]).reset_index(drop=True),
            metricas_df2.sort_values(["run_name", "region"]).reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(series_df.reset_index(drop=True), series_df2.reset_index(drop=True))

        # 3ra ejecucion leyendo el CSV escrito en disco (no el DataFrame en memoria) -- confirma que no hay acumulacion fisica
        metricas_desde_disco = pd.read_csv(os.path.join(output_dir, "metricas_master.csv"), encoding="utf-8-sig")
        assert len(metricas_desde_disco) == len(metricas_df)

        print("OK test_idempotencia")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


# =========================================================
# 12. Consolidado/ NO SE REDESCUBRE COMO RUN
# =========================================================

def test_consolidado_no_se_redescubre_como_run():
    tmp_pipeline = tempfile.mkdtemp(prefix="agg_test_consolidado_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp_pipeline, "XGBoost_train336h_fh4h_baseline", "xgboost",
                                         ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
                                         REGIONES_TEST)

        # 1ra consolidacion: crea Pipeline_Resultados/Consolidado/ con 3 CSV (sin config.json)
        metricas_df, series_df, descubrimiento = agg.consolidar_resultados(
            tmp_pipeline, regiones_esperadas=REGIONES_TEST,
        )
        assert len(descubrimiento.runs) == 1

        output_dir = os.path.join(tmp_pipeline, agg.NOMBRE_CARPETA_CONSOLIDADO)
        assert os.path.isdir(output_dir)
        assert not os.path.exists(os.path.join(output_dir, "config.json"))

        # Descubrir de nuevo (independiente de consolidar_resultados): Consolidado/ NUNCA debe aparecer
        # ni como corrida completa, ni como incompleta, ni como "sin config.json"
        resultado2 = agg.descubrir_runs_completos(tmp_pipeline, regiones_esperadas=REGIONES_TEST)
        assert len(resultado2.runs) == 1, "Consolidado/ no deberia sumar corridas"
        assert all(r.run_name != agg.NOMBRE_CARPETA_CONSOLIDADO for r in resultado2.runs)
        assert all(nombre != agg.NOMBRE_CARPETA_CONSOLIDADO for nombre, _ in resultado2.incompletos)
        # Aunque Consolidado/ no tiene config.json, no debe contarse como "sin_config" tampoco
        assert resultado2.n_sin_config == 0, (
            f"Consolidado/ se conto como 'sin config.json' (n_sin_config={resultado2.n_sin_config}) -- deberia excluirse antes de llegar a ese chequeo"
        )
        assert all(d.nombre != agg.NOMBRE_CARPETA_CONSOLIDADO for d in resultado2.descartados)

        # 2da consolidacion sobre el mismo pipeline (Consolidado/ ya existe con contenido del paso anterior):
        # debe seguir viendo exactamente 1 corrida, nunca redescubrir su propia salida.
        metricas_df2, series_df2, descubrimiento2 = agg.consolidar_resultados(
            tmp_pipeline, regiones_esperadas=REGIONES_TEST,
        )
        assert len(descubrimiento2.runs) == 1
        pd.testing.assert_frame_equal(
            metricas_df.reset_index(drop=True), metricas_df2.reset_index(drop=True),
        )

        print("OK test_consolidado_no_se_redescubre_como_run")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


# =========================================================
# 13. ESQUEMA CONSISTENTE CON COLUMNAS HETEROGENEAS
# =========================================================

def test_esquema_consistente_columnas_heterogeneas():
    tmp = tempfile.mkdtemp(prefix="agg_test_esquema_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp, "XGBoost_train336h_fh4h_baseline", "xgboost",
                                         ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
                                         REGIONES_TEST, mape_base=2.0)
        _escribir_run_sarimax(tmp, "SARIMAX_train1440h_fh4h_baseline",
                               ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"],
                               REGIONES_TEST, mape_base=1.5)
        _escribir_run_1b_univariado(tmp, "Naive_auto", "naive", REGIONES_TEST, mape_base=5.0)

        resultado = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(resultado.runs) == 3

        metricas_df = agg.construir_metricas_master(resultado.runs)

        # Columnas EXCLUSIVAS de un modelo deben sobrevivir via union, no descartarse ni forzar un esquema comun
        for col_exclusiva in ["AIC", "BIC", "order"]:  # solo sarimax
            assert col_exclusiva in metricas_df.columns, f"falta columna exclusiva de sarimax: {col_exclusiva}"
        for col_exclusiva in ["tuneado", "horizonte_usado"]:  # xgboost y naive la tienen, sarimax no
            assert col_exclusiva in metricas_df.columns

        # Donde no aplica, debe quedar NaN -- nunca 0/'' inventado
        fila_sarimax = metricas_df[metricas_df["run_name"] == "SARIMAX_train1440h_fh4h_baseline"].iloc[0]
        fila_xgb = metricas_df[metricas_df["run_name"] == "XGBoost_train336h_fh4h_baseline"].iloc[0]
        assert pd.isna(fila_xgb["AIC"]) and pd.isna(fila_xgb["BIC"]), "xgboost no deberia tener AIC/BIC (son de sarimax)"
        assert pd.isna(fila_sarimax["tuneado"]), "sarimax no deberia tener 'tuneado' (es de xgboost/naive)"

        # Colision de 'exogenas' en sarimax: la original (nombres internos) debe sobrevivir renombrada, sin pisar la canonica
        assert fila_sarimax["exogenas"] == "Temperatura|IGAE|Generacion|Importacion|Exportacion"
        assert "Temperaturas" in fila_sarimax["exogenas_original"]

        # Las columnas comunes a los 3 (MAPE/sMAPE/MAE/RMSE/region + las 13 de metadata) siguen presentes en todas las filas
        for col in ["MAE", "RMSE", "MAPE", "sMAPE", "region"] + agg._COLUMNAS_METADATA:
            assert col in metricas_df.columns, f"falta columna comun {col}"
            assert metricas_df[col].notna().all() or col in ("exogena_individual", "notas", "metodologia"), f"columna comun {col} con NaN inesperado"

        # El esquema (conjunto de columnas) debe ser el MISMO al reconstruir dos veces seguidas
        metricas_df2 = agg.construir_metricas_master(resultado.runs)
        assert set(metricas_df.columns) == set(metricas_df2.columns)
        assert list(metricas_df.columns) == list(metricas_df2.columns)

        print("OK test_esquema_consistente_columnas_heterogeneas")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# FIXTURES: Legacy_Univariados/ sintetico (metricas_global.csv/series_global.csv)
# =========================================================

def _escribir_legacy_univariado(
    pipeline_dir, estrategias, regiones=None, forecast_horizon=4,
    incluir_real=True, incluir_componente_pred=False,
    valor_real_base=100.0, offset_horas=0, valor_real_por_region=None,
):
    """
    Escribe Pipeline_Resultados/Legacy_Univariados/{metricas_global.csv,series_global.csv}
    sinteticos, con el MISMO esquema que usa el resto del proyecto para
    series.csv/metricas.csv (serie/modelo/fecha/tipo/subset/valor, sin
    columna 'region' explicita -- se deriva de 'serie', igual que
    `_region_de_serie()` en cada modulo de modelo).
    """
    regiones = regiones if regiones is not None else REGIONES_LEGACY
    legacy_dir = os.path.join(pipeline_dir, "Legacy_Univariados")
    os.makedirs(legacy_dir, exist_ok=True)

    metric_rows, series_rows = [], []

    for region in regiones:
        nombre_serie = f"{region}_DEMANDA"
        real_base = (valor_real_por_region or {}).get(region, valor_real_base)

        if incluir_real:
            fechas_reales = _pred_fechas(20 + forecast_horizon, offset_horas=offset_horas - 20)
            for h, fecha in enumerate(fechas_reales):
                series_rows.append({
                    "serie": nombre_serie, "fecha": fecha, "tipo": "real",
                    "subset": "completo", "modelo": "real", "valor": real_base + h,
                })

        fechas_test = _pred_fechas(forecast_horizon, offset_horas=offset_horas)
        for i, estrategia in enumerate(estrategias):
            for h, fecha in enumerate(fechas_test):
                series_rows.append({
                    "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                    "subset": "test", "modelo": estrategia, "valor": 50.0 + i + h,
                })
            metric_rows.append({
                "serie": nombre_serie, "modelo": estrategia,
                "MAPE": 3.0 + i, "sMAPE": 3.0 + i, "MAE": 1.0, "RMSE": 1.2,
            })

            if incluir_componente_pred and i == 0:
                for comp in ["LSTMtrend", "FCNNseason", "ARresid"]:
                    for h, fecha in enumerate(fechas_test):
                        series_rows.append({
                            "serie": nombre_serie, "fecha": fecha, "tipo": "componente_pred",
                            "subset": "test", "modelo": comp, "valor": 10.0 + h,
                        })

    metricas_path = os.path.join(legacy_dir, "metricas_global.csv")
    series_path = os.path.join(legacy_dir, "series_global.csv")
    pd.DataFrame(metric_rows).to_csv(metricas_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(series_rows).to_csv(series_path, index=False, encoding="utf-8-sig")
    return metricas_path, series_path


ESTRATEGIAS_LEGACY_EJEMPLO = [
    "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid",
    "ARIMA_1_1_1",
    "SARIMA_1_1_1__1_0_1_168",
    "FCNN_Individual",
    "STL_FCNN_residuos",
    "AR_AIC",
    "STL_AR_residuos_AIC",
    "LSTM_Individual",
]


# =========================================================
# LEGACY UNIVARIADO -- 14 escenarios
# =========================================================

def test_legacy_metricas_valido_se_integra():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_metricas_")
    try:
        metricas_path, series_path = _escribir_legacy_univariado(tmp, ESTRATEGIAS_LEGACY_EJEMPLO)

        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")

        resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)

        assert len(resultado.metricas_df) == len(ESTRATEGIAS_LEGACY_EJEMPLO) * len(REGIONES_LEGACY)
        assert set(resultado.metricas_df["familia_experimento"]) == {"univariado"}
        assert set(resultado.metricas_df["modelo_estrategia"]) == set(ESTRATEGIAS_LEGACY_EJEMPLO)
        assert set(resultado.metricas_df["modelo"]) == set(agg.MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO.values())

        # MAE/RMSE/MAPE/sMAPE preservados tal cual (sin recalcular)
        fila = resultado.metricas_df[
            (resultado.metricas_df["modelo_estrategia"] == "ARIMA_1_1_1") & (resultado.metricas_df["region"] == "CEN")
        ].iloc[0]
        assert fila["MAPE"] == 3.0 + ESTRATEGIAS_LEGACY_EJEMPLO.index("ARIMA_1_1_1")
        assert fila["run_dir"] == metricas_path
        assert fila["exogenas"] == ""
        assert fila["exogena_individual"] is None
        assert fila["train_hours"] is None

        print("OK test_legacy_metricas_valido_se_integra")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_series_valido_se_integra():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_series_")
    try:
        metricas_path, series_path = _escribir_legacy_univariado(tmp, ESTRATEGIAS_LEGACY_EJEMPLO, forecast_horizon=4)

        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")
        resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)

        # 8 estrategias x 7 regiones x 4 horas
        assert len(resultado.pred_df) == len(ESTRATEGIAS_LEGACY_EJEMPLO) * len(REGIONES_LEGACY) * 4

        series_df = agg.construir_series_master([], verbose=False, extra_pred_frames=[resultado.pred_df], extra_real_frames=[resultado.real_df])
        preds = series_df[series_df["serie_tipo"] == "prediccion"]
        assert set(preds["familia_experimento"]) == {"univariado"}
        assert set(preds["run_name"]) == {agg.build_run_name_legacy_univariado(e) for e in ESTRATEGIAS_LEGACY_EJEMPLO}
        assert set(preds["modelo"]) == set(agg.MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO.values())

        print("OK test_legacy_series_valido_se_integra")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_7_regiones_no_se_fuerzan_a_8():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_7regiones_")
    try:
        metricas_path, series_path = _escribir_legacy_univariado(tmp, ["AR_AIC"], regiones=REGIONES_LEGACY)

        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")

        validacion = agg.validar_legacy_univariado(legacy_metricas_df, legacy_series_df)
        assert validacion.ok
        assert "BCA" not in validacion.regiones_metricas
        assert "BCA" not in validacion.regiones_series
        assert len(validacion.regiones_metricas) == 7
        assert set(validacion.regiones_metricas) == set(REGIONES_LEGACY)

        resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)
        assert set(resultado.metricas_df["region"]) == set(REGIONES_LEGACY)
        assert len(resultado.metricas_df) == 7  # 1 estrategia x 7 regiones, no 8

        print("OK test_legacy_7_regiones_no_se_fuerzan_a_8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_componente_pred_excluido():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_componente_")
    try:
        metricas_path, series_path = _escribir_legacy_univariado(
            tmp, ["AR_AIC", "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid"],
            regiones=REGIONES_LEGACY[:2], forecast_horizon=4, incluir_componente_pred=True,
        )
        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")

        validacion = agg.validar_legacy_univariado(legacy_metricas_df, legacy_series_df)
        assert validacion.tiene_componente_pred

        resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)
        assert "componente_pred" not in set(resultado.pred_df.get("serie_tipo", pd.Series(dtype=object)))
        assert not set(resultado.pred_df["modelo_estrategia"]) & {"LSTMtrend", "FCNNseason", "ARresid"}
        # Solo las 2 estrategias reales x 2 regiones x 4 horas -- sin los componentes
        assert len(resultado.pred_df) == 2 * 2 * 4

        print("OK test_legacy_componente_pred_excluido")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_real_deduplicado_contra_moderno():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_real_dedup_")
    try:
        # Corrida moderna 1B con region CEN, offset/horizonte que se solapan con la legacy
        _escribir_run_1b_univariado(tmp, "Naive_auto", "naive", ["CEN"], forecast_horizon=4, offset_horas=0, valor_real_base=100.0)
        # Legacy con la MISMA region/ventana/valor real -- debe deduplicarse, no duplicarse
        metricas_path, series_path = _escribir_legacy_univariado(
            tmp, ["AR_AIC"], regiones=["CEN"], forecast_horizon=4, offset_horas=0, valor_real_base=100.0,
        )

        resultado_moderno = agg.descubrir_runs_completos(tmp, regiones_esperadas=["CEN"])
        assert len(resultado_moderno.runs) == 1

        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")
        legacy_resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)

        series_df = agg.construir_series_master(
            resultado_moderno.runs, verbose=False,
            extra_pred_frames=[legacy_resultado.pred_df], extra_real_frames=[legacy_resultado.real_df],
        )

        reales = series_df[series_df["serie_tipo"] == "real"]
        # 1 region x 4 horas, deduplicado entre moderno Y legacy -- no 2x4=8
        assert len(reales) == 4, f"esperaba 4 filas 'real' deduplicadas (moderno+legacy), hubo {len(reales)}"
        assert not reales.duplicated(subset=["region", "timestamp"]).any()

        print("OK test_legacy_real_deduplicado_contra_moderno")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_conflicto_real_reportado():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_conflicto_")
    try:
        # Mismo horizonte, pero con "real" DISTINTO entre moderno y legacy
        _escribir_run_1b_univariado(tmp, "Naive_auto", "naive", ["CEN"], forecast_horizon=4, offset_horas=0, valor_real_base=100.0)
        metricas_path, series_path = _escribir_legacy_univariado(
            tmp, ["AR_AIC"], regiones=["CEN"], forecast_horizon=4, offset_horas=0, valor_real_base=999.0,
        )

        resultado_moderno = agg.descubrir_runs_completos(tmp, regiones_esperadas=["CEN"])
        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")
        legacy_resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)

        buf = io.StringIO()
        with redirect_stdout(buf):
            series_df = agg.construir_series_master(
                resultado_moderno.runs, verbose=True,
                extra_pred_frames=[legacy_resultado.pred_df], extra_real_frames=[legacy_resultado.real_df],
            )
        salida = buf.getvalue()
        assert "valores 'real' distintos entre corridas" in salida, "el conflicto real legacy/moderno no se reporto"

        reales = series_df[series_df["serie_tipo"] == "real"]
        assert len(reales) == 4  # deduplicado, no promediado -- una sola fuente por hora
        assert (reales["valor"] < 200).all() or (reales["valor"] > 200).all()

        print("OK test_legacy_conflicto_real_reportado")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_run_name_estable():
    nombre1 = agg.build_run_name_legacy_univariado("SARIMA_1_1_1__1_0_1_168")
    nombre2 = agg.build_run_name_legacy_univariado("SARIMA_1_1_1__1_0_1_168")
    assert nombre1 == nombre2 == "Legacy_Univariado__SARIMA_1_1_1__1_0_1_168"

    # Unico por estrategia, no depende del orden de llamada
    nombres = {agg.build_run_name_legacy_univariado(e) for e in ESTRATEGIAS_LEGACY_EJEMPLO}
    assert len(nombres) == len(ESTRATEGIAS_LEGACY_EJEMPLO)

    # No colisiona con RUN_NAMEs modernos reales (todos empiezan con MODEL_LABELS, nunca con el prefijo legacy)
    nombres_modernos = {
        build_run_name("xgboost", 336, 168, ["Temperatura"]),
        build_run_name("sarimax", 1440, 168, ["Temperatura", "IGAE"]),
        build_run_name("naive", "auto", "auto", []),
    }
    assert not (nombres & nombres_modernos)
    assert all(n.startswith("Legacy_Univariado__") for n in nombres)

    print("OK test_legacy_run_name_estable")


def test_legacy_mapping_estrategia_modelo_canonico():
    esperado = {
        "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid": "ensemble_stl_univariado",
        "ARIMA_1_1_1": "arima",
        "SARIMA_1_1_1__1_0_1_168": "sarima",
        "FCNN_Individual": "fcnn_univariada",
        "STL_FCNN_residuos": "fcnn_residuos_stl",
        "AR_AIC": "ar_aic",
        "STL_AR_residuos_AIC": "ar_residuos_stl",
        "LSTM_Individual": "lstm_univariada",
    }
    assert agg.MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO == esperado

    # Ningun canonico legacy colisiona con un canonico moderno (1A/1B)
    canonicos_legacy = set(esperado.values())
    canonicos_modernos = agg.FAMILIA_1A_MODELOS | agg.FAMILIA_1B_MODELOS
    assert not (canonicos_legacy & canonicos_modernos), (
        f"canonico(s) legacy colisionan con canonicos modernos: {canonicos_legacy & canonicos_modernos}"
    )

    print("OK test_legacy_mapping_estrategia_modelo_canonico")


def test_legacy_estrategia_desconocida_no_se_inventa():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_desconocida_")
    try:
        estrategias = ["AR_AIC", "MODELO_RARISIMO_NO_MAPEADO"]
        metricas_path, series_path = _escribir_legacy_univariado(tmp, estrategias, regiones=REGIONES_LEGACY[:2], forecast_horizon=4)

        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")

        validacion = agg.validar_legacy_univariado(legacy_metricas_df, legacy_series_df)
        assert validacion.estrategias_sin_mapeo == ["MODELO_RARISIMO_NO_MAPEADO"]
        assert any("sin mapeo" in p.lower() for p in validacion.problemas)

        buf = io.StringIO()
        with redirect_stdout(buf):
            resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=True)
        salida = buf.getvalue()
        assert "MODELO_RARISIMO_NO_MAPEADO" in salida  # se reporta explicitamente

        # Se incluye igual (no se pierden resultados reales), pero con modelo=None -- nunca se inventa un id
        fila_desconocida = resultado.metricas_df[resultado.metricas_df["modelo_estrategia"] == "MODELO_RARISIMO_NO_MAPEADO"]
        assert len(fila_desconocida) == 2  # 2 regiones, no se excluyo
        assert fila_desconocida["modelo"].isna().all()
        assert fila_desconocida["familia_experimento"].eq("univariado").all()

        assert "MODELO_RARISIMO_NO_MAPEADO" in resultado.estrategias_sin_mapeo

        print("OK test_legacy_estrategia_desconocida_no_se_inventa")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_no_se_clasifica_como_1a_1b_individual_temp_igae():
    tmp = tempfile.mkdtemp(prefix="agg_test_legacy_familia_")
    try:
        metricas_path, series_path = _escribir_legacy_univariado(tmp, ESTRATEGIAS_LEGACY_EJEMPLO, regiones=REGIONES_LEGACY[:2], forecast_horizon=4)
        legacy_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(series_path, encoding="utf-8-sig")

        resultado = agg.procesar_legacy_univariado(legacy_metricas_df, legacy_series_df, metricas_path, series_path, verbose=False)

        familias = set(resultado.metricas_df["familia_experimento"])
        assert familias == {"univariado"}
        assert not familias & {"1A", "1B", "individual", "temp_igae", "unknown"}

        # Los canonicos legacy tampoco estan en las listas 1A/1B (clasificar_familia() nunca se llamo para estas filas)
        canonicos_presentes = set(resultado.metricas_df["modelo"].dropna())
        assert not (canonicos_presentes & (agg.FAMILIA_1A_MODELOS | agg.FAMILIA_1B_MODELOS))

        print("OK test_legacy_no_se_clasifica_como_1a_1b_individual_temp_igae")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_idempotencia_con_legacy():
    tmp_pipeline = tempfile.mkdtemp(prefix="agg_test_idem_legacy_pipeline_")
    try:
        _escribir_run_1a_estilo_xgboost(
            tmp_pipeline, "XGBoost_train336h_fh4h_baseline", "xgboost",
            ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
            REGIONES_TEST,
        )
        metricas_path, series_path = _escribir_legacy_univariado(tmp_pipeline, ESTRATEGIAS_LEGACY_EJEMPLO, forecast_horizon=4)

        metricas_df, series_df, descubrimiento = agg.consolidar_resultados(
            tmp_pipeline, regiones_esperadas=REGIONES_TEST,
            legacy_metricas_path=metricas_path, legacy_series_path=series_path,
        )

        assert "univariado" in set(metricas_df["familia_experimento"])
        assert len(metricas_df[metricas_df["familia_experimento"] == "univariado"]) == len(ESTRATEGIAS_LEGACY_EJEMPLO) * len(REGIONES_LEGACY)

        metricas_df2, series_df2, descubrimiento2 = agg.consolidar_resultados(
            tmp_pipeline, regiones_esperadas=REGIONES_TEST,
            legacy_metricas_path=metricas_path, legacy_series_path=series_path,
        )

        pd.testing.assert_frame_equal(
            metricas_df.sort_values(["run_name", "region"]).reset_index(drop=True),
            metricas_df2.sort_values(["run_name", "region"]).reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(series_df.reset_index(drop=True), series_df2.reset_index(drop=True))

        print("OK test_idempotencia_con_legacy")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


def test_legacy_univariados_no_se_descubre_como_run_moderno():
    tmp_pipeline = tempfile.mkdtemp(prefix="agg_test_legacy_no_redescubierto_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp_pipeline, "XGBoost_train336h_fh4h_baseline", "xgboost",
                                         ["Temperatura", "IGAE"], REGIONES_TEST)
        _escribir_legacy_univariado(tmp_pipeline, ["AR_AIC"], regiones=REGIONES_LEGACY[:1], forecast_horizon=4)

        resultado = agg.descubrir_runs_completos(tmp_pipeline, regiones_esperadas=REGIONES_TEST)

        assert len(resultado.runs) == 1  # solo la corrida moderna
        assert all(r.run_name != "Legacy_Univariados" for r in resultado.runs)
        assert all(d.nombre != "Legacy_Univariados" for d in resultado.descartados)
        assert resultado.n_sin_config == 0, (
            f"Legacy_Univariados/ se conto como 'sin config.json' (n_sin_config={resultado.n_sin_config}) -- deberia excluirse antes"
        )

        print("OK test_legacy_univariados_no_se_descubre_como_run_moderno")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


def test_legacy_archivo_faltante_error_claro():
    tmp_pipeline = tempfile.mkdtemp(prefix="agg_test_legacy_archivo_faltante_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp_pipeline, "XGBoost_train336h_fh4h_baseline", "xgboost",
                                         ["Temperatura", "IGAE"], REGIONES_TEST)
        metricas_path, series_path = _escribir_legacy_univariado(tmp_pipeline, ["AR_AIC"], regiones=REGIONES_LEGACY[:1], forecast_horizon=4)

        # Solo un path -- debe fallar de inmediato, sin escribir nada
        output_dir = os.path.join(tmp_pipeline, "Consolidado")
        try:
            agg.consolidar_resultados(
                tmp_pipeline, regiones_esperadas=REGIONES_TEST,
                legacy_metricas_path=metricas_path, legacy_series_path=None,
            )
            assert False, "esperaba ValueError por pasar solo un path legacy"
        except ValueError as e:
            assert "AMBOS" in str(e) or "ambos" in str(e).lower()
        assert not os.path.exists(output_dir), "no deberia haber escrito nada si la integracion legacy fallo"

        # Path legacy que no existe -- error claro, no crea Consolidado/ a medias
        try:
            agg.consolidar_resultados(
                tmp_pipeline, regiones_esperadas=REGIONES_TEST,
                legacy_metricas_path=os.path.join(tmp_pipeline, "NO_EXISTE.csv"),
                legacy_series_path=series_path,
            )
            assert False, "esperaba FileNotFoundError por legacy_metricas_path inexistente"
        except FileNotFoundError:
            pass
        assert not os.path.exists(output_dir), "no deberia haber escrito nada si el archivo legacy no existe"

        print("OK test_legacy_archivo_faltante_error_claro")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


def test_no_regresion_consolidado_moderno_sin_legacy():
    """Sin pasar legacy_metricas_path/legacy_series_path, consolidar_resultados debe comportarse EXACTAMENTE igual que antes de esta tarea."""
    tmp_pipeline = tempfile.mkdtemp(prefix="agg_test_no_regresion_")
    try:
        _escribir_run_1a_estilo_xgboost(
            tmp_pipeline, "XGBoost_train336h_fh4h_baseline", "xgboost",
            ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
            REGIONES_TEST,
        )
        _escribir_run_1b_univariado(tmp_pipeline, "Naive_auto", "naive", REGIONES_TEST)

        metricas_df, series_df, descubrimiento = agg.consolidar_resultados(tmp_pipeline, regiones_esperadas=REGIONES_TEST)

        assert "univariado" not in set(metricas_df["familia_experimento"])
        assert set(metricas_df["familia_experimento"]) == {"1A", "1B"}

        reporte_df = agg.construir_reporte_consolidacion(descubrimiento)
        assert set(reporte_df["origen"]) == {"moderno"}

        # Las columnas nuevas (test_start/test_end/n_predicciones) siguen presentes pero
        # no rompen nada -- metricas_master conserva todas las columnas de siempre
        for col in ["run_name", "modelo", "familia_experimento", "region", "MAE", "RMSE", "MAPE", "sMAPE"]:
            assert col in metricas_df.columns

        print("OK test_no_regresion_consolidado_moderno_sin_legacy")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


def main():
    test_clasificacion_1a()
    test_clasificacion_1b()
    test_clasificacion_individual()
    test_clasificacion_temp_igae()
    test_fcnn_dos_estrategias_preservadas()
    test_real_deduplicado()
    test_componente_pred_excluido()
    test_ventanas_prediccion_distintas()
    test_conflicto_valores_reales_reportado()
    test_corrida_incompleta_excluida()
    test_idempotencia()
    test_consolidado_no_se_redescubre_como_run()
    test_esquema_consistente_columnas_heterogeneas()

    print("\nTODOS LOS TESTS DEL AGREGADOR PASARON (13/13 escenarios)")

    test_legacy_metricas_valido_se_integra()
    test_legacy_series_valido_se_integra()
    test_legacy_7_regiones_no_se_fuerzan_a_8()
    test_legacy_componente_pred_excluido()
    test_legacy_real_deduplicado_contra_moderno()
    test_legacy_conflicto_real_reportado()
    test_legacy_run_name_estable()
    test_legacy_mapping_estrategia_modelo_canonico()
    test_legacy_estrategia_desconocida_no_se_inventa()
    test_legacy_no_se_clasifica_como_1a_1b_individual_temp_igae()
    test_idempotencia_con_legacy()
    test_legacy_univariados_no_se_descubre_como_run_moderno()
    test_legacy_archivo_faltante_error_claro()
    test_no_regresion_consolidado_moderno_sin_legacy()

    print("\nTODOS LOS TESTS DE LEGACY UNIVARIADO PASARON (14/14 escenarios)")


if __name__ == "__main__":
    main()

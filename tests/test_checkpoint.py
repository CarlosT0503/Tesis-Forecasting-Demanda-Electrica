"""
Tests del checkpoint/resume por region (src/tesis_forecast/checkpoint.py).

Dos partes:

  1. Tests unitarios de `checkpoint.py` contra CSV sinteticos escritos a
     mano (rapidos, deterministicos, sin entrenar ningun modelo) -- cubren
     los 5 escenarios pedidos a nivel de la logica de checkpoint pura.
  2. Tests end-to-end usando `naive_trend_seasonal_model.py` (Family B: usa
     `_construir_df_series` con `np.atleast_1d`, sin exogenas, rapido -- sin
     LSTM/Optuna/AR) para probar la integracion real dentro de `run()`, y un
     tercer bloque con `ar_model.py` (Family A, CON trials/config_usada) en
     2 regiones para confirmar que ese camino tambien reanuda sin recalcular.

Solo depende de pandas/numpy/statsmodels/sklearn -- corre en cualquier
entorno, sin tensorflow/optuna/lightgbm/xgboost.

Uso:
    python tests/test_checkpoint.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast import checkpoint as ckpt
from tesis_forecast.regions import REGIONS_ALL
from tesis_forecast.models import naive_trend_seasonal_model as ntsm
from tesis_forecast.models import ar_model as arm


# =========================================================
# PARTE 1: unit tests puros de checkpoint.py (CSV sinteticos)
# =========================================================

def _escribir_csvs(output_dir, regiones, forecast_horizon=48, n_modelos=1,
                    con_trials=False, trials_por_region=10, con_config=True,
                    region_parcial=None, pred_parcial=5):
    """
    Escribe series.csv/metricas.csv/[trials.csv]/[config_usada.csv]
    sinteticos, con una region completa por defecto para cada nombre en
    `regiones`, salvo `region_parcial` (si se da), a la que se le escriben
    menos filas de prediccion de las esperadas (`pred_parcial` en vez de
    `forecast_horizon`) para simular una region truncada/corrupta.
    """
    series_rows = []
    metrics_rows = []
    trials_rows = []
    config_rows = []

    fechas_base = pd.date_range("2024-01-01", periods=forecast_horizon, freq="h")

    for region in regiones:
        nombre_serie = f"{region}_DEMANDA"
        n_pred = pred_parcial if region == region_parcial else forecast_horizon

        for m_idx in range(n_modelos):
            modelo = f"Modelo{m_idx}" if n_modelos > 1 else "Modelo"

            for h in range(n_pred):
                series_rows.append({
                    "serie": nombre_serie, "fecha": fechas_base[h], "tipo": "prediccion",
                    "subset": "test", "modelo": modelo, "valor": 100.0 + h,
                })

            # Si la region es parcial, tambien se omite su fila de metricas
            # para ese modelo (asi se comporta un run() real: sin
            # `calcular_metricas` completo no se llama a `guardar_metricas`).
            if region == region_parcial:
                continue

            metrics_rows.append({
                "serie": nombre_serie, "modelo": modelo,
                "MAE": 1.0, "RMSE": 1.5, "MAPE": 2.0, "sMAPE": 2.1,
            })

            if con_trials:
                for lag in range(trials_por_region):
                    trials_rows.append({"serie": nombre_serie, "modelo": modelo, "lag": lag, "AIC": float(lag)})

            if con_config:
                config_rows.append({"serie": nombre_serie, "modelo": modelo, "parametros": "{}"})

    pd.DataFrame(series_rows).to_csv(os.path.join(output_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics_rows).to_csv(os.path.join(output_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    if con_trials:
        pd.DataFrame(trials_rows).to_csv(os.path.join(output_dir, "trials.csv"), index=False, encoding="utf-8-sig")
    if con_config:
        pd.DataFrame(config_rows).to_csv(os.path.join(output_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")


def test_region_es_completa_criterios():
    metricas_df = pd.DataFrame([
        {"serie": "BCA_DEMANDA", "modelo": "M", "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1},
        {"serie": "CEN_DEMANDA", "modelo": "M", "MAE": np.nan, "RMSE": 1, "MAPE": 1, "sMAPE": 1},
    ])
    series_df = pd.DataFrame([
        {"serie": "BCA_DEMANDA", "fecha": "2024-01-01", "tipo": "prediccion", "valor": 1},
        {"serie": "BCA_DEMANDA", "fecha": "2024-01-02", "tipo": "prediccion", "valor": 2},
        {"serie": "CEN_DEMANDA", "fecha": "2024-01-01", "tipo": "prediccion", "valor": 1},
    ])

    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_df, series_df, None, None, forecast_horizon=2) is True
    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_df, series_df, None, None, forecast_horizon=3) is False, \
        "conteo de predicciones incorrecto deberia fallar"
    assert ckpt.region_es_completa("CEN_DEMANDA", metricas_df, series_df, None, None, forecast_horizon=1) is False, \
        "metrica NaN deberia fallar"
    assert ckpt.region_es_completa("NES_DEMANDA", metricas_df, series_df, None, None, forecast_horizon=1) is False, \
        "region ausente deberia fallar"

    # forecast_horizon=None (split dinamico): basta con >=1 prediccion.
    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_df, series_df, None, None, forecast_horizon=None) is True

    # requiere_trials / requiere_config_usada
    trials_df = pd.DataFrame([{"serie": "BCA_DEMANDA", "lag": i} for i in range(5)])
    config_df = pd.DataFrame([{"serie": "BCA_DEMANDA", "parametros": "{}"}])

    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_df, series_df, trials_df, config_df,
                                    forecast_horizon=2, requiere_trials=True, requiere_config_usada=True) is True
    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_df, series_df, None, config_df,
                                    forecast_horizon=2, requiere_trials=True, requiere_config_usada=True) is False, \
        "sin trials.csv y requiere_trials=True deberia fallar"
    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_df, series_df, trials_df, config_df,
                                    forecast_horizon=2, requiere_trials=True, trials_esperados=999) is False, \
        "conteo exacto de trials incorrecto deberia fallar"

    # n_modelos_esperados > 1 (FCNN: 2 estrategias, siempre con forecast_horizon fijo)
    metricas_2m = pd.DataFrame([
        {"serie": "BCA_DEMANDA", "modelo": "A", "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1},
        {"serie": "BCA_DEMANDA", "modelo": "B", "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1},
    ])
    # series_df (definido arriba) solo tiene 2 filas de prediccion para BCA_DEMANDA
    # -- alcanza para 1 modelo (forecast_horizon=2) pero no para 2.
    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_2m, series_df, None, None,
                                    forecast_horizon=2, n_modelos_esperados=2) is False, \
        "con n_modelos_esperados=2 debe exigir forecast_horizon*2 filas de prediccion"

    series_df_2_modelos = pd.DataFrame([
        {"serie": "BCA_DEMANDA", "fecha": "2024-01-01", "tipo": "prediccion", "valor": 1},
        {"serie": "BCA_DEMANDA", "fecha": "2024-01-02", "tipo": "prediccion", "valor": 2},
        {"serie": "BCA_DEMANDA", "fecha": "2024-01-01", "tipo": "prediccion", "valor": 3},
        {"serie": "BCA_DEMANDA", "fecha": "2024-01-02", "tipo": "prediccion", "valor": 4},
    ])
    assert ckpt.region_es_completa("BCA_DEMANDA", metricas_2m, series_df_2_modelos, None, None,
                                    forecast_horizon=2, n_modelos_esperados=2) is True, \
        "con las 4 filas (2 modelos x horizon=2) deberia completar"

    print("OK test_region_es_completa_criterios")


def test_cargar_checkpoint_filas_3_de_8():
    """Escenario 2 del pedido: 3/8 regiones completas -> checkpoint detecta esas 3."""
    tmp = tempfile.mkdtemp(prefix="ckpt_unit_filas_")
    try:
        completas = REGIONS_ALL[:3]
        _escribir_csvs(tmp, completas, forecast_horizon=24, con_trials=False, con_config=False)

        regiones_completas, previos = ckpt.cargar_checkpoint_regiones(
            tmp, REGIONS_ALL, forecast_horizon=24, formato_series="filas",
        )

        assert regiones_completas == set(completas), regiones_completas
        assert len(previos["metrics"]) == 3
        assert len(previos["series"]) == 3 * 24
        assert previos["trials_df"] is None
        assert previos["config_usada"] == []

        series_de = {row["serie"] for row in previos["series"]}
        assert series_de == {f"{r}_DEMANDA" for r in completas}

        print("OK test_cargar_checkpoint_filas_3_de_8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cargar_checkpoint_bloques():
    """Family C (sarimax/ensemble_stl): series precargada como bloques con fecha/valor arreglo."""
    tmp = tempfile.mkdtemp(prefix="ckpt_unit_bloques_")
    try:
        completas = ["BCA", "CEN"]
        _escribir_csvs(tmp, completas, forecast_horizon=10, con_trials=False, con_config=True)

        regiones_completas, previos = ckpt.cargar_checkpoint_regiones(
            tmp, ["BCA", "CEN", "NES"], forecast_horizon=10,
            requiere_config_usada=True, formato_series="bloques",
        )

        assert regiones_completas == {"BCA", "CEN"}
        assert len(previos["series"]) == 2, "un bloque por (serie,tipo,subset,modelo)"
        for bloque in previos["series"]:
            assert len(bloque["fecha"]) == 10
            assert len(bloque["valor"]) == 10
        assert len(previos["config_usada"]) == 2

        print("OK test_cargar_checkpoint_bloques")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_region_parcial_se_excluye():
    """Escenario 3 del pedido: region parcial/corrupta -> excluida del checkpoint."""
    tmp = tempfile.mkdtemp(prefix="ckpt_unit_parcial_")
    try:
        completas_mas_parcial = ["BCA", "CEN", "NES"]
        _escribir_csvs(
            tmp, completas_mas_parcial, forecast_horizon=24,
            region_parcial="NES", pred_parcial=5,
        )

        regiones_completas, previos = ckpt.cargar_checkpoint_regiones(
            tmp, REGIONS_ALL, forecast_horizon=24,
        )

        assert regiones_completas == {"BCA", "CEN"}, regiones_completas
        assert "NES" not in regiones_completas

        series_de = {row["serie"] for row in previos["series"]}
        assert "NES_DEMANDA" not in series_de, "las filas de la region parcial no deben precargarse"

        print("OK test_region_parcial_se_excluye")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_8_de_8_completo():
    """Escenario 4 del pedido: 8/8 completas -> checkpoint las detecta todas."""
    tmp = tempfile.mkdtemp(prefix="ckpt_unit_8de8_")
    try:
        _escribir_csvs(tmp, REGIONS_ALL, forecast_horizon=12)

        regiones_completas, previos = ckpt.cargar_checkpoint_regiones(
            tmp, REGIONS_ALL, forecast_horizon=12,
        )

        assert regiones_completas == set(REGIONS_ALL)
        assert len(previos["metrics"]) == 8
        assert len(previos["series"]) == 8 * 12

        print("OK test_8_de_8_completo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_precargar_no_duplica():
    """Escenario 5 del pedido: precargar + agregar region nueva no duplica filas."""

    class _Acumulador:
        def __init__(self):
            self.series = []
            self.metrics = []
            self.trials = []
            self.config_usada = []

    tmp = tempfile.mkdtemp(prefix="ckpt_unit_noDup_")
    try:
        completas = ["BCA", "CEN"]
        _escribir_csvs(tmp, completas, forecast_horizon=6, con_trials=True, con_config=True)

        regiones_completas, previos = ckpt.cargar_checkpoint_regiones(
            tmp, ["BCA", "CEN", "NES"], forecast_horizon=6,
            requiere_trials=True, requiere_config_usada=True,
        )

        resultados = _Acumulador()
        ckpt.precargar_en_acumulador(resultados, previos)

        # Simula el trabajo fresco de la region pendiente (NES), igual a
        # como lo haria evaluar_serie()/evaluar_region() de un modelo real.
        for h in range(6):
            resultados.series.append({
                "serie": "NES_DEMANDA", "fecha": pd.Timestamp("2024-02-01") + pd.Timedelta(hours=h),
                "tipo": "prediccion", "subset": "test", "modelo": "Modelo", "valor": 50.0 + h,
            })
        resultados.metrics.append({"serie": "NES_DEMANDA", "modelo": "Modelo", "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1})
        resultados.trials.append(pd.DataFrame([{"serie": "NES_DEMANDA", "modelo": "Modelo", "lag": i} for i in range(10)]))
        resultados.config_usada.append({"serie": "NES_DEMANDA", "modelo": "Modelo", "parametros": "{}"})

        series_df = pd.DataFrame(resultados.series)
        metricas_df = pd.DataFrame(resultados.metrics)
        trials_df = pd.concat(resultados.trials, ignore_index=True)
        config_df = pd.DataFrame(resultados.config_usada)

        assert len(series_df) == 3 * 6, f"esperaba 18 filas de series (3 regiones x 6h), hubo {len(series_df)}"
        assert len(metricas_df) == 3, f"esperaba 3 filas de metricas, hubo {len(metricas_df)}"
        assert metricas_df["serie"].nunique() == 3, "no debe haber una serie duplicada"
        assert not metricas_df["serie"].duplicated().any()
        assert len(trials_df) == 3 * 10
        assert len(config_df) == 3

        print("OK test_precargar_no_duplica")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# PARTE 2: end-to-end con naive_trend_seasonal_model.py (Family B)
# =========================================================

def _construir_region_long_csv_ntsm(tmp_dir, region, n_horas, seed):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2024-01-01", periods=n_horas, freq="h")
    demanda = (
        100 + 0.01 * np.arange(n_horas)
        + 10 * np.sin(np.arange(n_horas) * 2 * np.pi / 24)
        + 3 * np.sin(np.arange(n_horas) * 2 * np.pi / 168)
        + rng.normal(0, 1, n_horas)
    )
    df = pd.DataFrame({
        "fecha": fechas.date, "Hora": (fechas.hour + 1), ntsm.COL_DEMANDA: demanda,
    })
    df.to_csv(os.path.join(tmp_dir, f"{region}_long.csv"), index=False)


def test_e2e_naive_trend_seasonal_resume():
    # Ventana reducida (override explicito, no el default 3600h/168h del
    # modulo) unicamente para que el test corra rapido -- STL(robust=True)
    # sobre una serie mas corta es mucho mas barato, y esto solo prueba la
    # mecanica de resume/checkpoint, no un resultado cientifico. 600h/48h
    # es la misma ventana reducida que ya usa tests/smoke_ensemble_stl.py.
    train_hours = 600
    forecast_horizon = 48
    n_horas = train_hours + forecast_horizon + 24

    tmp_data = tempfile.mkdtemp(prefix="ckpt_e2e_ntsm_data_")
    tmp_out = tempfile.mkdtemp(prefix="ckpt_e2e_ntsm_out_")

    try:
        primeras_3 = REGIONS_ALL[:3]
        restantes_5 = REGIONS_ALL[3:]

        # --- Escenario 1: run nuevo, solo datos de 3 regiones disponibles ---
        for i, region in enumerate(primeras_3):
            _construir_region_long_csv_ntsm(tmp_data, region, n_horas, seed=i)

        series_df, metricas_df, _, _ = ntsm.run(
            exogenas_globales={}, regions_all=primeras_3,
            train_hours=train_hours, forecast_horizon=forecast_horizon,
            data_dir=tmp_data, output_dir=tmp_out,
        )
        assert len(metricas_df) == 3, f"run nuevo: esperaba 3 filas, hubo {len(metricas_df)}"
        assert set(metricas_df["region"]) == set(primeras_3)
        print("OK e2e escenario 1 (run nuevo, 3 regiones)")

        # --- Escenario 2: aparecen datos de las 8, se reanuda con regions_all completo ---
        for i, region in enumerate(restantes_5):
            _construir_region_long_csv_ntsm(tmp_data, region, n_horas, seed=100 + i)

        metricas_antes = pd.read_csv(os.path.join(tmp_out, "metricas.csv"))
        valores_mape_originales = dict(zip(metricas_antes["region"], metricas_antes["MAPE"]))

        series_df2, metricas_df2, _, _ = ntsm.run(
            exogenas_globales={}, regions_all=REGIONS_ALL,
            train_hours=train_hours, forecast_horizon=forecast_horizon,
            data_dir=tmp_data, output_dir=tmp_out,
        )

        assert len(metricas_df2) == 8, f"reanudar: esperaba 8 filas, hubo {len(metricas_df2)}"
        assert set(metricas_df2["region"]) == set(REGIONS_ALL)
        assert not metricas_df2["serie"].duplicated().any(), "no debe haber series duplicadas tras reanudar"

        # Las 3 regiones originales NO deben haberse recalculado: mismo MAPE exacto.
        for region in primeras_3:
            mape_nuevo = metricas_df2.loc[metricas_df2["region"] == region, "MAPE"].iloc[0]
            # np.isclose (no ==): ambos valores pasaron por un CSV
            # (metricas.csv) via to_csv/read_csv, que no siempre preserva
            # el float64 bit-a-bit -- comparar exacto compararia precision
            # de serializacion, no si el modelo se recalculo de verdad.
            assert np.isclose(mape_nuevo, valores_mape_originales[region], rtol=1e-9), (
                f"{region} se recalculo al reanudar (MAPE cambio de "
                f"{valores_mape_originales[region]} a {mape_nuevo})"
            )
        print("OK e2e escenario 2 (reanuda desde 3/8, no recalcula las 3 ya completas)")

        n_pred_total = (series_df2["tipo"] == "prediccion").sum()
        assert n_pred_total == 8 * forecast_horizon, (
            f"esperaba {8 * forecast_horizon} predicciones totales sin duplicar, hubo {n_pred_total}"
        )
        print("OK e2e escenario 5 (no duplica filas de series.csv al reanudar)")

        # --- Escenario 4: 8/8 completas -> correr de nuevo debe ser un skip total ---
        series_df3, metricas_df3, _, _ = ntsm.run(
            exogenas_globales={}, regions_all=REGIONS_ALL,
            train_hours=train_hours, forecast_horizon=forecast_horizon,
            data_dir=tmp_data, output_dir=tmp_out,
        )
        assert len(metricas_df3) == 8
        for region in REGIONS_ALL:
            mape_3 = metricas_df3.loc[metricas_df3["region"] == region, "MAPE"].iloc[0]
            mape_2 = metricas_df2.loc[metricas_df2["region"] == region, "MAPE"].iloc[0]
            assert np.isclose(mape_3, mape_2, rtol=1e-9), (
                f"{region} se recalculo en una corrida 8/8 (deberia ser skip total): {mape_2} -> {mape_3}"
            )
        print("OK e2e escenario 4 (8/8 completas -> skip total, nada se recalcula)")

        # --- Escenario 3: se corrompe una region (menos predicciones de las esperadas) ---
        region_a_corromper = REGIONS_ALL[0]
        nombre_serie_corrupta = f"{region_a_corromper}_DEMANDA"

        series_csv_path = os.path.join(tmp_out, "series.csv")
        df_series_actual = pd.read_csv(series_csv_path)

        es_prediccion_de_la_region = (
            (df_series_actual["serie"] == nombre_serie_corrupta) & (df_series_actual["tipo"] == "prediccion")
        )
        idx_a_borrar = df_series_actual[es_prediccion_de_la_region].index[:15]  # deja solo 33/48
        df_series_corrupta = df_series_actual.drop(index=idx_a_borrar)
        df_series_corrupta.to_csv(series_csv_path, index=False, encoding="utf-8-sig")

        series_df4, metricas_df4, _, _ = ntsm.run(
            exogenas_globales={}, regions_all=REGIONS_ALL,
            train_hours=train_hours, forecast_horizon=forecast_horizon,
            data_dir=tmp_data, output_dir=tmp_out,
        )

        assert len(metricas_df4) == 8, f"esperaba 8 filas tras reparar la region corrupta, hubo {len(metricas_df4)}"
        n_pred_region_corrupta = (
            (series_df4["serie"] == nombre_serie_corrupta) & (series_df4["tipo"] == "prediccion")
        ).sum()
        assert n_pred_region_corrupta == forecast_horizon, (
            f"la region corrupta deberia haberse re-ejecutado completa ({forecast_horizon} predicciones), "
            f"tiene {n_pred_region_corrupta}"
        )
        assert not metricas_df4["serie"].duplicated().any()
        n_pred_total_4 = (series_df4["tipo"] == "prediccion").sum()
        assert n_pred_total_4 == 8 * forecast_horizon, (
            f"esperaba {8 * forecast_horizon} predicciones totales tras reparar, hubo {n_pred_total_4} "
            "(regiones sanas no deben duplicarse ni la reparada quedar con menos filas)"
        )
        print("OK e2e escenario 3 (region parcial/corrupta se reemplaza por completo, sin duplicar las demas)")

    finally:
        shutil.rmtree(tmp_data, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


# =========================================================
# PARTE 3: end-to-end con ar_model.py (Family A, CON trials/config_usada)
# =========================================================

def _construir_region_long_csv_ar(tmp_dir, region, n_horas, seed):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2024-01-01", periods=n_horas, freq="h")
    demanda = 100 + 10 * np.sin(np.arange(n_horas) * 2 * np.pi / 24) + rng.normal(0, 2, n_horas)
    df = pd.DataFrame({
        "fecha": fechas.date, "Hora": (fechas.hour + 1), arm.COL_DEMANDA: demanda,
    })
    df.to_csv(os.path.join(tmp_dir, f"{region}_long.csv"), index=False)


def test_e2e_ar_resume_con_trials_y_config():
    """
    Confirma que el camino CON trials.csv/config_usada.csv (xgboost/
    lightgbm/lstm_direct/ar/ar_resid_trend_seasonal/lstm_resid/fcnn/
    ensemble_stl comparten este patron) tambien reanuda sin recalcular y
    sin duplicar. Solo 2 regiones (el barrido de 168 lags por region no es
    instantaneo) para mantener el test rapido.
    """
    n_horas = 24 * 40
    regiones_2 = REGIONS_ALL[:2]

    tmp_data = tempfile.mkdtemp(prefix="ckpt_e2e_ar_data_")
    tmp_out = tempfile.mkdtemp(prefix="ckpt_e2e_ar_out_")

    try:
        _construir_region_long_csv_ar(tmp_data, regiones_2[0], n_horas, seed=1)

        _, metricas_df, trials_df, config_df = arm.run(
            exogenas_globales={}, regions_all=[regiones_2[0]],
            data_dir=tmp_data, output_dir=tmp_out,
        )
        assert len(metricas_df) == 1
        assert len(trials_df) == arm.MAX_LAG_AR
        assert len(config_df) == 1
        mape_original = metricas_df["MAPE"].iloc[0]

        # Aparece la segunda region; se reanuda con las 2.
        _construir_region_long_csv_ar(tmp_data, regiones_2[1], n_horas, seed=2)

        _, metricas_df2, trials_df2, config_df2 = arm.run(
            exogenas_globales={}, regions_all=regiones_2,
            data_dir=tmp_data, output_dir=tmp_out,
        )

        assert len(metricas_df2) == 2, f"esperaba 2 filas de metricas, hubo {len(metricas_df2)}"
        assert not metricas_df2["serie"].duplicated().any()
        assert len(trials_df2) == 2 * arm.MAX_LAG_AR, (
            f"esperaba {2 * arm.MAX_LAG_AR} filas de trials sin duplicar, hubo {len(trials_df2)}"
        )
        assert len(config_df2) == 2

        mape_region1_tras_reanudar = metricas_df2.loc[metricas_df2["region"] == regiones_2[0], "MAPE"].iloc[0]
        assert np.isclose(mape_region1_tras_reanudar, mape_original, rtol=1e-9), (
            f"{regiones_2[0]} se recalculo al reanudar (MAPE {mape_original} -> {mape_region1_tras_reanudar})"
        )

        print("OK test_e2e_ar_resume_con_trials_y_config")

    finally:
        shutil.rmtree(tmp_data, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


def main():
    test_region_es_completa_criterios()
    test_cargar_checkpoint_filas_3_de_8()
    test_cargar_checkpoint_bloques()
    test_region_parcial_se_excluye()
    test_8_de_8_completo()
    test_precargar_no_duplica()

    test_e2e_naive_trend_seasonal_resume()
    test_e2e_ar_resume_con_trials_y_config()

    print("\nTODOS LOS TESTS DE CHECKPOINT PASARON")


if __name__ == "__main__":
    main()

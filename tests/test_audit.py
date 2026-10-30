"""
Tests del motor de auditoria de solo lectura (src/tesis_forecast/audit.py)
contra la matriz de exogenas individuales.

Fixtures 100% sinteticas -- ninguna corrida real, ningun modelo se
entrena, y ningun archivo de una fixture se modifica una vez escrito
(el motor de auditoria es de solo lectura; estos tests ademas verifican
que las carpetas fixture sigan intactas despues de auditar). Solo
depende de pandas -- corre en cualquier entorno, sin tensorflow/optuna/
xgboost/lightgbm (el motor de auditoria -- a diferencia de
`construir_objetivos_desde_matriz_actual`, que no se testea aqui -- no
los necesita).

Uso:
    python tests/test_audit.py
"""

import json
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast import audit

REGIONES_TEST = ["BCA", "CEN"]


# =========================================================
# FIXTURES
# =========================================================

def _pred_fechas(forecast_horizon, offset_horas=0):
    base = pd.Timestamp("2024-06-01") + pd.Timedelta(hours=offset_horas)
    return pd.date_range(base, periods=forecast_horizon, freq="h")


def _escribir_run_moderno(pipeline_dir, run_name, modelo, exogenas, regiones,
                           train_hours=336, forecast_horizon=4, con_config_json=True,
                           mape_base=2.0, n_trials=3):
    """
    Estilo actual (post-runner.py): config.json + metricas.csv (1 fila/
    region, tuneado/horizonte_usado) + series.csv (solo prediccion, igual
    que xgboost/lightgbm/lstm_direct reales) + trials.csv/config_usada.csv.
    `con_config_json=False` simula una carpeta LEGACY con el mismo
    contenido pero sin config.json (para probar el fallback).
    """
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if con_config_json:
        cfg = {
            "modelo": modelo, "exogenas": exogenas, "train_hours": train_hours,
            "forecast_horizon": forecast_horizon, "optuna_n_trials": 10, "seed": 42,
            "notas": "", "run_name": run_name, "git_commit": "abc123", "generated_at": "2026-01-01T00:00:00",
        }
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    modelo_display = {
        "xgboost": "XGBoost_Tuned_2Semanas", "lightgbm": "LightGBM_Tuned", "lstm_direct": "LSTM_Directa",
        "lstm_resid": "LSTM_Resid_Trend_Seasonal",
    }.get(modelo, modelo)

    series_rows, metric_rows, trials_rows, config_rows = [], [], [], []
    for i, region in enumerate(regiones):
        nombre_serie = f"{region}_DEMANDA"
        for h, fecha in enumerate(_pred_fechas(forecast_horizon)):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": modelo_display, "valor": 100.0 + h, "region": region,
            })
        metric_rows.append({
            "serie": nombre_serie, "modelo": modelo_display, "tuneado": True,
            "horizonte_usado": f"{forecast_horizon}_horas",
            "MAPE": mape_base + i, "sMAPE": mape_base + i, "MAE": 1.0, "RMSE": 1.5, "region": region,
        })
        for t in range(n_trials):
            trials_rows.append({"serie": nombre_serie, "modelo": modelo, "number": t, "value": 1.0, "region": region})
        config_rows.append({
            "serie": nombre_serie, "modelo": modelo_display, "parametros": "{}",
            "horizonte_usado": f"{forecast_horizon}_horas", "train_horas": train_hours,
            "exogenas": str(exogenas), "region": region,
        })

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(trials_rows).to_csv(os.path.join(run_dir, "trials.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(config_rows).to_csv(os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_legacy_sarimax_sin_config(pipeline_dir, run_name, exogena_canonica, regiones,
                                             train_hours=1440, forecast_horizon=4, mape_base=1.5):
    """
    Carpeta VERDADERAMENTE legacy: sin config.json, con la columna
    'exogenas' de config_usada.csv usando el nombre INTERNO de SARIMAX
    (ej. 'Temperaturas' en vez de 'Temperatura') -- exactamente el
    formato real de sarimax_model.py, para probar la traduccion de
    vuelta a canonico.
    """
    exogena_interna = {"Temperatura": "Temperaturas", "IGAE": "IGAE", "Generacion": "Generacion",
                        "Importacion": "Importacion", "Exportacion": "Exportacion"}[exogena_canonica]

    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    series_rows, metric_rows, config_rows = [], [], []
    for i, region in enumerate(regiones):
        nombre_serie = f"{region}_DEMANDA"
        for h, fecha in enumerate(_pred_fechas(forecast_horizon)):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": "SARIMAX_1_1_1__1_0_1_168_EXOG_2M", "valor": 90.0 + h, "region": region,
            })
        metric_rows.append({
            "serie": nombre_serie, "modelo": "SARIMAX_1_1_1__1_0_1_168_EXOG_2M",
            "order": "(1, 1, 1)", "AIC": 100.0, "BIC": 110.0,
            "MAPE": mape_base + i, "sMAPE": mape_base + i, "MAE": 1.0, "RMSE": 1.2,
            "exogenas": str([exogena_interna]), "region": region,
        })
        config_rows.append({
            "serie": nombre_serie, "modelo": "SARIMAX_1_1_1__1_0_1_168_EXOG_2M", "parametros": "{}",
            "train_horas": train_hours, "exogenas": str([exogena_interna]), "region": region,
        })

    pd.DataFrame(series_rows).to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(config_rows).to_csv(os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _escribir_run_modelo_ambiguo(pipeline_dir, run_name, regiones):
    """Carpeta legacy cuya columna 'modelo' no se reconoce -- debe contarse como 'sin identidad', no crashear ni adivinar."""
    run_dir = os.path.join(pipeline_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    rows = [{"serie": f"{r}_DEMANDA", "modelo": "AlgoRaroDeUnExperimentoViejo", "MAPE": 1, "sMAPE": 1, "MAE": 1, "RMSE": 1, "region": r} for r in regiones]
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    return run_dir


def _objetivo(modelo, exogena, train_hours, forecast_horizon):
    return audit.ConfigObjetivo(
        modelo=modelo, exogena=exogena, train_hours=train_hours, forecast_horizon=forecast_horizon,
        run_name=f"{modelo}_{exogena}",
    )


# =========================================================
# TESTS
# =========================================================

def test_exact_match_complete_via_config_json():
    tmp = tempfile.mkdtemp(prefix="audit_test_complete_")
    try:
        _escribir_run_moderno(tmp, "XGBoost_train336h_fh4h_Temp", "xgboost", ["Temperatura"], REGIONES_TEST,
                               train_hours=336, forecast_horizon=4)

        objetivos = [_objetivo("xgboost", "Temperatura", 336, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        assert len(df) == 1
        fila = df.iloc[0]
        assert fila["estado"] == "exact_match_complete", fila["razon"]
        assert fila["regiones_validas"] == "2/2"
        assert fila["ruta_legacy"] == os.path.join(tmp, "XGBoost_train336h_fh4h_Temp")

        print("OK test_exact_match_complete_via_config_json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_exact_match_partial_falta_region():
    tmp = tempfile.mkdtemp(prefix="audit_test_partial_")
    try:
        _escribir_run_moderno(tmp, "XGBoost_train336h_fh4h_IGAE", "xgboost", ["IGAE"], REGIONES_TEST[:1],  # solo BCA
                               train_hours=336, forecast_horizon=4)

        objetivos = [_objetivo("xgboost", "IGAE", 336, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        assert len(df) == 1
        fila = df.iloc[0]
        assert fila["estado"] == "exact_match_partial", fila["razon"]
        assert fila["regiones_validas"] == "1/2 (BCA)"

        print("OK test_exact_match_partial_falta_region")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_not_equivalent_train_hours_distinto():
    tmp = tempfile.mkdtemp(prefix="audit_test_noeq_")
    try:
        # La carpeta existe con train_hours=200, pero el objetivo actual pide 336.
        _escribir_run_moderno(tmp, "XGBoost_train200h_fh4h_Generacion", "xgboost", ["Generacion"], REGIONES_TEST,
                               train_hours=200, forecast_horizon=4)

        objetivos = [_objetivo("xgboost", "Generacion", 336, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        assert len(df) == 1
        fila = df.iloc[0]
        assert fila["estado"] == "legacy_not_equivalent", fila["razon"]
        assert "train_hours difiere" in fila["razon"]

        print("OK test_legacy_not_equivalent_train_hours_distinto")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_sin_ninguna_carpeta():
    tmp = tempfile.mkdtemp(prefix="audit_test_missing_")
    try:
        _escribir_run_moderno(tmp, "XGBoost_train336h_fh4h_Temp", "xgboost", ["Temperatura"], REGIONES_TEST)

        objetivos = [
            _objetivo("xgboost", "Temperatura", 336, 4),
            _objetivo("xgboost", "Exportacion", 336, 4),  # esta no tiene carpeta
        ]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        assert len(df) == 2
        fila_faltante = df[df["exogena"] == "Exportacion"].iloc[0]
        assert fila_faltante["estado"] == "missing"
        assert fila_faltante["ruta_legacy"] is None
        assert fila_faltante["regiones_validas"] == "0/2"

        print("OK test_missing_sin_ninguna_carpeta")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_sin_config_json_xgboost_se_reconoce():
    tmp = tempfile.mkdtemp(prefix="audit_test_legacy_xgb_")
    try:
        _escribir_run_moderno(tmp, "xgboost_demanda_horario_2semanas_Temperatura_viejo",  # nombre pre-esquema actual
                               "xgboost", ["Temperatura"], REGIONES_TEST, train_hours=336, forecast_horizon=4,
                               con_config_json=False)

        objetivos = [_objetivo("xgboost", "Temperatura", 336, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        fila = df.iloc[0]
        assert fila["estado"] == "exact_match_complete", fila["razon"]
        assert "legacy (sin config.json)" in fila["razon"]

        print("OK test_legacy_sin_config_json_xgboost_se_reconoce")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_sarimax_traduce_exogena_interna_a_canonica():
    tmp = tempfile.mkdtemp(prefix="audit_test_legacy_sarimax_")
    try:
        _escribir_run_legacy_sarimax_sin_config(tmp, "sarimax_viejo_temperatura", "Temperatura", REGIONES_TEST,
                                                 train_hours=1440, forecast_horizon=4)

        objetivos = [_objetivo("sarimax", "Temperatura", 1440, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        fila = df.iloc[0]
        assert fila["estado"] == "exact_match_complete", fila["razon"]

        print("OK test_legacy_sarimax_traduce_exogena_interna_a_canonica")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_modelo_ambiguo_no_crashea_y_se_cuenta_sin_identidad():
    tmp = tempfile.mkdtemp(prefix="audit_test_ambiguo_")
    try:
        _escribir_run_modelo_ambiguo(tmp, "carpeta_rara", REGIONES_TEST)

        objetivos = [_objetivo("xgboost", "Temperatura", 336, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        assert len(df) == 1
        assert df.iloc[0]["estado"] == "missing"  # la carpeta ambigua no cuenta como candidata a nada

        print("OK test_modelo_ambiguo_no_crashea_y_se_cuenta_sin_identidad")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_solo_lectura_no_modifica_ni_borra_nada():
    tmp = tempfile.mkdtemp(prefix="audit_test_soloLectura_")
    try:
        run_dir = _escribir_run_moderno(tmp, "XGBoost_train336h_fh4h_Temp", "xgboost", ["Temperatura"], REGIONES_TEST)

        archivos_antes = {}
        for archivo in os.listdir(run_dir):
            ruta = os.path.join(run_dir, archivo)
            archivos_antes[archivo] = (os.path.getmtime(ruta), os.path.getsize(ruta))

        objetivos = [_objetivo("xgboost", "Temperatura", 336, 4), _objetivo("xgboost", "IGAE", 336, 4)]
        audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)
        audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)  # 2 veces, por si acaso

        archivos_despues = {}
        for archivo in os.listdir(run_dir):
            ruta = os.path.join(run_dir, archivo)
            archivos_despues[archivo] = (os.path.getmtime(ruta), os.path.getsize(ruta))

        assert archivos_antes == archivos_despues, "el agregador toco archivos existentes"
        assert set(os.listdir(tmp)) == {"XGBoost_train336h_fh4h_Temp"}, "el agregador creo/borro carpetas"

        print("OK test_solo_lectura_no_modifica_ni_borra_nada")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multiples_exogenas_no_es_candidata_a_individual():
    """Una carpeta baseline (5-8 exogenas) no debe matchear como 'individual' de ninguna exogena suelta."""
    tmp = tempfile.mkdtemp(prefix="audit_test_baseline_no_individual_")
    try:
        _escribir_run_moderno(tmp, "XGBoost_train336h_fh4h_baseline", "xgboost",
                               ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
                               REGIONES_TEST, train_hours=336, forecast_horizon=4)

        objetivos = [_objetivo("xgboost", "Temperatura", 336, 4)]
        df = audit.auditar_matriz_individual(objetivos, tmp, regiones_esperadas=REGIONES_TEST)

        assert df.iloc[0]["estado"] == "missing"

        print("OK test_multiples_exogenas_no_es_candidata_a_individual")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_exact_match_complete_via_config_json()
    test_exact_match_partial_falta_region()
    test_legacy_not_equivalent_train_hours_distinto()
    test_missing_sin_ninguna_carpeta()
    test_legacy_sin_config_json_xgboost_se_reconoce()
    test_legacy_sarimax_traduce_exogena_interna_a_canonica()
    test_modelo_ambiguo_no_crashea_y_se_cuenta_sin_identidad()
    test_solo_lectura_no_modifica_ni_borra_nada()
    test_multiples_exogenas_no_es_candidata_a_individual()

    print("\nTODOS LOS TESTS DE AUDITORIA PASARON")


if __name__ == "__main__":
    main()

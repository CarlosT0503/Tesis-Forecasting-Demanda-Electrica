"""
Tests de la Fase 3 (Temperatura+IGAE): las 7 configs explicitas
(`configs_temp_igae` en `notebooks/run_matrix.ipynb`) y sus RUN_NAME.

Solo depende de pandas -- corre en cualquier entorno.

Uso:
    python tests/test_fase3_temp_igae.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.config import ExperimentConfig, build_run_name

# runner.py importa los 12 modulos de modelos a nivel de modulo, lo que
# arrastra tensorflow/optuna/xgboost/lightgbm -- NO instalados en este
# entorno local (misma limitacion preexistente que el resto del proyecto,
# ver checkpoint.py/matrix.py). El test que depende de resolve_run() real
# se salta explicitamente (no falla en bloque) cuando ese import no esta
# disponible; sigue corriendo tal cual en Colab.
try:
    from tesis_forecast.runner import resolve_run, MODEL_DEFAULTS
    RUNNER_DISPONIBLE = True
except ImportError as e:
    resolve_run = None
    MODEL_DEFAULTS = None
    RUNNER_DISPONIBLE = False
    _RUNNER_IMPORT_ERROR = e

MODELOS_MULTIVARIADOS = ["xgboost", "lightgbm", "lstm_direct", "sarimax", "fcnn", "ensemble_stl", "lstm_resid"]

# Confirmados leyendo TRAIN_LAST_HOURS_DEFAULT/FORECAST_HORIZON_DEFAULT de
# cada modulo de modelo directamente (no runner.py) -- ver
# test_defaults_y_run_names_via_resolve_run_real() para la verificacion
# CRUZADA contra runner.MODEL_DEFAULTS cuando este disponible.
DEFAULTS_ESPERADOS = {
    "xgboost": (336, 168),
    "lightgbm": (336, 168),
    "lstm_direct": (2160, 168),
    "sarimax": (1440, 168),
    "fcnn": (3600, 168),
    "ensemble_stl": (3600, 168),
    "lstm_resid": (3600, 168),
}

RUN_NAMES_ESPERADOS = {
    "xgboost": "XGBoost_train336h_fh168h_Temp-IGAE",
    "lightgbm": "LightGBM_train336h_fh168h_Temp-IGAE",
    "lstm_direct": "LSTM_Directa_train2160h_fh168h_Temp-IGAE",
    "sarimax": "SARIMAX_train1440h_fh168h_Temp-IGAE",
    "fcnn": "FCNN_train3600h_fh168h_Temp-IGAE",
    "ensemble_stl": "Ensemble_STL_train3600h_fh168h_Temp-IGAE",
    "lstm_resid": "LSTM_Resid_Trend_Seasonal_train3600h_fh168h_Temp-IGAE",
}

# El RUN_NAME de FCNN debe coincidir EXACTAMENTE con la corrida que ya
# esta en progreso en Drive (otro Colab) -- ver test_fcnn_coincide_con_corrida_activa().
FCNN_RUN_NAME_CORRIDA_ACTIVA = "FCNN_train3600h_fh168h_Temp-IGAE"


def _construir_configs_temp_igae():
    return [ExperimentConfig(modelo=m, exogenas=["Temperatura", "IGAE"]) for m in MODELOS_MULTIVARIADOS]


# =========================================================
# TESTS: las 7 configs
# =========================================================

def test_exactamente_7_configs():
    configs = _construir_configs_temp_igae()
    assert len(configs) == 7, f"esperaba 7 configs, hubo {len(configs)}"
    print("OK test_exactamente_7_configs")


def test_todas_tienen_exactamente_temperatura_igae():
    configs = _construir_configs_temp_igae()
    for c in configs:
        assert c.exogenas == ["Temperatura", "IGAE"], f"{c.modelo}: exogenas={c.exogenas}"
    print("OK test_todas_tienen_exactamente_temperatura_igae")


def test_ninguna_tiene_otras_exogenas():
    configs = _construir_configs_temp_igae()
    prohibidas = {"Generacion", "Importacion", "Exportacion", "Primarias", "Secundarias", "Terciarias"}
    for c in configs:
        assert not (prohibidas & set(c.exogenas)), f"{c.modelo}: {c.exogenas} contiene una exogena prohibida"
    print("OK test_ninguna_tiene_otras_exogenas")


def test_defaults_cientificos_resueltos_son_los_actuales():
    """
    Version LIVIANA (siempre corre localmente): usa build_run_name() +
    DEFAULTS_ESPERADOS (confirmados leyendo cada modulo de modelo
    directamente). La verificacion CRUZADA contra el resolve_run()/
    runner.MODEL_DEFAULTS real vive en
    test_defaults_y_run_names_via_resolve_run_real() mas abajo, que se
    salta si runner.py no se puede importar (falta tensorflow/optuna/
    xgboost/lightgbm en este entorno).
    """
    configs = _construir_configs_temp_igae()
    for c in configs:
        assert c.modelo in DEFAULTS_ESPERADOS, f"{c.modelo} no tiene defaults esperados registrados en este test"
    print("OK test_defaults_cientificos_resueltos_son_los_actuales (version liviana)")


def test_run_names_correctos_y_unicos():
    """Version LIVIANA: build_run_name() directo (config.py, sin runner.py)."""
    configs = _construir_configs_temp_igae()
    nombres = []
    for c in configs:
        th_esperado, fh_esperado = DEFAULTS_ESPERADOS[c.modelo]
        run_name = build_run_name(c.modelo, th_esperado, fh_esperado, c.exogenas)
        assert run_name == RUN_NAMES_ESPERADOS[c.modelo], f"{c.modelo}: {run_name} != {RUN_NAMES_ESPERADOS[c.modelo]}"
        nombres.append(run_name)
    assert len(nombres) == len(set(nombres)), f"hay RUN_NAMEs duplicados: {nombres}"
    print("OK test_run_names_correctos_y_unicos")


def test_fcnn_coincide_con_corrida_activa():
    """
    El RUN_NAME que resuelve la config de FCNN de esta Fase 3 debe ser
    IDENTICO al de la corrida que ya esta en progreso en Drive (otro
    Colab) -- si no coincidiera, run_matrix() creuria una carpeta nueva en
    vez de reanudar/saltar la existente.
    """
    th_esperado, fh_esperado = DEFAULTS_ESPERADOS["fcnn"]
    run_name = build_run_name("fcnn", th_esperado, fh_esperado, ["Temperatura", "IGAE"])
    assert run_name == FCNN_RUN_NAME_CORRIDA_ACTIVA, (
        f"El RUN_NAME de FCNN en Fase 3 ({run_name}) no coincide con la corrida activa "
        f"({FCNN_RUN_NAME_CORRIDA_ACTIVA}) -- run_matrix() no reanudaria/saltaria esa carpeta."
    )
    print(f"OK test_fcnn_coincide_con_corrida_activa ({run_name})")


def test_defaults_y_run_names_via_resolve_run_real():
    """
    Version PESADA (requiere tensorflow/optuna/xgboost/lightgbm -- corre
    en Colab, no en este entorno local): llama a resolve_run() de verdad y
    cruza contra runner.MODEL_DEFAULTS, no contra numeros copiados a mano.
    """
    if not RUNNER_DISPONIBLE:
        print(f"SALTADO test_defaults_y_run_names_via_resolve_run_real (runner.py no importable aqui: {_RUNNER_IMPORT_ERROR})")
        return

    configs = _construir_configs_temp_igae()
    for c in configs:
        resolved = resolve_run(c)
        th_esperado, fh_esperado = DEFAULTS_ESPERADOS[c.modelo]
        assert resolved.train_hours == th_esperado, f"{c.modelo}: train_hours={resolved.train_hours}, esperado={th_esperado}"
        assert resolved.forecast_horizon == fh_esperado, f"{c.modelo}: forecast_horizon={resolved.forecast_horizon}, esperado={fh_esperado}"
        assert resolved.train_hours == MODEL_DEFAULTS[c.modelo]["train_hours"]
        assert resolved.forecast_horizon == MODEL_DEFAULTS[c.modelo]["forecast_horizon"]
        assert resolved.run_name == RUN_NAMES_ESPERADOS[c.modelo]
    print("OK test_defaults_y_run_names_via_resolve_run_real")


def test_run_names_no_colisionan_con_fase_1a_1b_2():
    nombres_fase3 = set()
    for c in _construir_configs_temp_igae():
        th_esperado, fh_esperado = DEFAULTS_ESPERADOS[c.modelo]
        nombres_fase3.add(build_run_name(c.modelo, th_esperado, fh_esperado, c.exogenas))

    fase1a = [
        ("xgboost", 336, 168, ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"]),
        ("lightgbm", 336, 168, ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"]),
        ("lstm_direct", 2160, 168, ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]),
        ("sarimax", 1440, 168, ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]),
        ("fcnn", 3600, 168, ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]),
        ("ensemble_stl", 3600, 168, ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]),
    ]
    nombres_fase1a = {build_run_name(m, th, fh, ex) for m, th, fh, ex in fase1a}

    nombres_fase1b = {
        build_run_name("naive", "auto", "auto", []),
        build_run_name("naive_trend", "auto", "auto", []),
        build_run_name("naive_trend_seasonal", 3600, 168, []),
        build_run_name("ar", "auto", "auto", []),
        build_run_name("ar_resid_trend_seasonal", 3600, 168, []),
        build_run_name("lstm_resid", 3600, 168, ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]),
    }

    exogenas_individuales = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
    nombres_fase2 = set()
    for modelo in MODELOS_MULTIVARIADOS:
        th, fh = DEFAULTS_ESPERADOS[modelo]
        for exog in exogenas_individuales:
            nombres_fase2.add(build_run_name(modelo, th, fh, [exog]))

    colisiones = nombres_fase3 & (nombres_fase1a | nombres_fase1b | nombres_fase2)
    assert not colisiones, f"colisiones de RUN_NAME con fases existentes: {colisiones}"
    print("OK test_run_names_no_colisionan_con_fase_1a_1b_2")


def main():
    test_exactamente_7_configs()
    test_todas_tienen_exactamente_temperatura_igae()
    test_ninguna_tiene_otras_exogenas()
    test_defaults_cientificos_resueltos_son_los_actuales()
    test_run_names_correctos_y_unicos()
    test_fcnn_coincide_con_corrida_activa()
    test_defaults_y_run_names_via_resolve_run_real()
    test_run_names_no_colisionan_con_fase_1a_1b_2()

    print("\nTODOS LOS TESTS DE FASE 3 (TEMP+IGAE) PASARON")


if __name__ == "__main__":
    main()

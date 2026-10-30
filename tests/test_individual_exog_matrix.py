"""
Tests de `matrix.build_individual_exog_matrix()` -- la matriz de exogenas
individuales (una exogena a la vez, sin acumular) para todos los modelos
multivariados.

`runner.py` importa los 12 modulos de modelos a nivel de modulo, y algunos
de ellos (xgboost/lightgbm/lstm_direct/fcnn/ensemble_stl/lstm_resid)
requieren `optuna`/`tensorflow`/`lightgbm`/`xgboost`, que no estan
instalados en este entorno de desarrollo. Como este archivo solo necesita
las CONSTANTES de nivel de modulo de cada modelo (TRAIN_LAST_HOURS_DEFAULT,
EXOG_COLS_DEFAULT, EXOG_CATALOGO, etc. -- literales de Python, no dependen
de que esas librerias hagan nada) y ninguna de sus funciones se llama
nunca aqui, se insertan stubs (`unittest.mock.MagicMock`) en `sys.modules`
para las librerias faltantes ANTES de importar `tesis_forecast.runner`/
`tesis_forecast.matrix`. Si una libreria SI esta instalada (como en Colab),
no se stubea -- se usa la real, sin ningun cambio de comportamiento.

Uso:
    python tests/test_individual_exog_matrix.py
"""

import importlib
import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _stub_si_falta(nombre):
    try:
        importlib.import_module(nombre)
    except ImportError:
        sys.modules[nombre] = MagicMock()


for _mod in [
    "optuna", "optuna.samplers",
    "xgboost", "lightgbm",
    "tensorflow", "tensorflow.keras", "tensorflow.keras.backend",
    "tensorflow.keras.models", "tensorflow.keras.layers",
    "tensorflow.keras.optimizers", "tensorflow.keras.callbacks",
]:
    _stub_si_falta(_mod)

from tesis_forecast import checkpoint as ckpt
from tesis_forecast.config import ExperimentConfig, EXOG_ABBR
from tesis_forecast.matrix import build_individual_exog_matrix
from tesis_forecast.regions import REGIONS_ALL
from tesis_forecast.runner import MODEL_DEFAULTS, resolve_run

EXOGENAS_INDIVIDUALES = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]

MODELOS_UNIVARIADOS_ESPERADOS = {
    "naive", "naive_trend", "ar", "naive_trend_seasonal", "ar_resid_trend_seasonal",
}
MODELOS_MULTIVARIADOS_ESPERADOS = {
    "xgboost", "lightgbm", "lstm_direct", "sarimax", "fcnn", "ensemble_stl", "lstm_resid",
}


def test_modelos_incluidos_y_excluidos():
    """
    Confirma en codigo (via MODEL_DEFAULTS["catalogo"], la misma fuente que
    usa resolve_run() para validar exogenas) cuales de los 12 modelos son
    multivariados vs. univariados -- no se asume la lista, se deriva.
    """
    assert set(MODEL_DEFAULTS.keys()) == MODELOS_UNIVARIADOS_ESPERADOS | MODELOS_MULTIVARIADOS_ESPERADOS, (
        "Hay un modelo nuevo en MODEL_DEFAULTS no contemplado en este test -- "
        "revisar si es univariado o multivariado y clasificarlo."
    )

    for modelo in MODELOS_UNIVARIADOS_ESPERADOS:
        assert MODEL_DEFAULTS[modelo]["catalogo"] == [], f"{modelo} deberia tener catalogo vacio (univariado)"

    for modelo in MODELOS_MULTIVARIADOS_ESPERADOS:
        assert len(MODEL_DEFAULTS[modelo]["catalogo"]) > 0, f"{modelo} deberia tener catalogo no vacio (multivariado)"
        for exogena in EXOGENAS_INDIVIDUALES:
            assert exogena in MODEL_DEFAULTS[modelo]["catalogo"], (
                f"{modelo} no reconoce '{exogena}' -- si esto es correcto, la matriz debe excluir "
                f"esa combinacion puntual (ver build_individual_exog_matrix)"
            )

    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)
    modelos_en_configs = {c.modelo for c in configs}

    assert modelos_en_configs == MODELOS_MULTIVARIADOS_ESPERADOS, (
        f"Modelos incluidos incorrectos. Esperado: {sorted(MODELOS_MULTIVARIADOS_ESPERADOS)}, "
        f"obtenido: {sorted(modelos_en_configs)}"
    )
    assert modelos_en_configs.isdisjoint(MODELOS_UNIVARIADOS_ESPERADOS), (
        "Un modelo univariado genero corridas en la matriz de exogenas individuales"
    )

    print(f"OK test_modelos_incluidos_y_excluidos -- incluidos: {sorted(modelos_en_configs)}")


def test_numero_exacto_de_configs():
    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)
    esperado = len(MODELOS_MULTIVARIADOS_ESPERADOS) * len(EXOGENAS_INDIVIDUALES)
    assert len(configs) == esperado, f"Esperaba {esperado} configs (7 modelos x 5 exogenas), hubo {len(configs)}"

    # Exactamente 5 configs por modelo multivariado (una por exogena), ni una mas ni una menos.
    for modelo in MODELOS_MULTIVARIADOS_ESPERADOS:
        n = sum(1 for c in configs if c.modelo == modelo)
        assert n == len(EXOGENAS_INDIVIDUALES), f"{modelo}: esperaba {len(EXOGENAS_INDIVIDUALES)} configs, hubo {n}"

    print(f"OK test_numero_exacto_de_configs -- {len(configs)} configs")


def test_una_exogena_por_config():
    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)
    for c in configs:
        assert isinstance(c.exogenas, list)
        assert len(c.exogenas) == 1, f"{c.modelo}: se esperaba exactamente 1 exogena, hubo {c.exogenas}"
        assert c.exogenas[0] in EXOGENAS_INDIVIDUALES

    # Cobertura: cada modelo multivariado tiene exactamente 1 config por cada una de las 5 exogenas.
    for modelo in MODELOS_MULTIVARIADOS_ESPERADOS:
        exogenas_del_modelo = sorted(c.exogenas[0] for c in configs if c.modelo == modelo)
        assert exogenas_del_modelo == sorted(EXOGENAS_INDIVIDUALES), (
            f"{modelo}: exogenas generadas {exogenas_del_modelo} != {sorted(EXOGENAS_INDIVIDUALES)}"
        )

    print("OK test_una_exogena_por_config")


def test_sin_primarias_secundarias_terciarias():
    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)
    prohibidas = {"Primarias", "Secundarias", "Terciarias"}
    for c in configs:
        assert not (set(c.exogenas) & prohibidas), f"{c.modelo}: exogena acumulativa/no pedida en {c.exogenas}"

    # Tambien si el llamador pidiera explicitamente Primarias/Secundarias/Terciarias
    # (que si estan en el catalogo de xgboost/lightgbm), solo se excluyen para los
    # modelos que NO las reconocen -- no se prueba aqui con la lista pedida por el
    # usuario, pero confirma que la funcion no las agrega por su cuenta.
    assert all("Primarias" not in c.exogenas and "Secundarias" not in c.exogenas and "Terciarias" not in c.exogenas
               for c in configs)

    print("OK test_sin_primarias_secundarias_terciarias")


def test_defaults_cientificos_preservados():
    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)

    for c in configs:
        # La config no debe forzar train_hours/forecast_horizon/optuna_n_trials:
        # deben quedar en None para que resolve_run() use el default del modelo.
        assert c.train_hours is None, f"{c.modelo}: train_hours no deberia estar fijado en la config generada"
        assert c.forecast_horizon is None, f"{c.modelo}: forecast_horizon no deberia estar fijado"
        assert c.optuna_n_trials is None, f"{c.modelo}: optuna_n_trials no deberia estar fijado"

        resolved = resolve_run(c)
        defaults = MODEL_DEFAULTS[c.modelo]

        assert resolved.train_hours == defaults["train_hours"], (
            f"{c.modelo}: train_hours resuelto {resolved.train_hours} != default {defaults['train_hours']}"
        )
        assert resolved.forecast_horizon == defaults["forecast_horizon"], (
            f"{c.modelo}: forecast_horizon resuelto {resolved.forecast_horizon} != default {defaults['forecast_horizon']}"
        )
        assert resolved.optuna_n_trials == defaults["optuna_n_trials"], (
            f"{c.modelo}: optuna_n_trials resuelto {resolved.optuna_n_trials} != default {defaults['optuna_n_trials']}"
        )
        assert resolved.exogenas == c.exogenas

    print("OK test_defaults_cientificos_preservados")


def test_run_name_sin_colision():
    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)
    run_names = [resolve_run(c).run_name for c in configs]

    assert len(run_names) == len(set(run_names)), "Hay RUN_NAME duplicados en la matriz de exogenas individuales"

    # Para un mismo modelo, dos exogenas distintas deben producir RUN_NAME distintos
    # (verificacion explicita del ejemplo que pide el usuario: Temperatura vs IGAE).
    run_names_xgboost = {
        exogena: resolve_run(ExperimentConfig(modelo="xgboost", exogenas=[exogena])).run_name
        for exogena in EXOGENAS_INDIVIDUALES
    }
    assert len(set(run_names_xgboost.values())) == len(EXOGENAS_INDIVIDUALES)
    assert run_names_xgboost["Temperatura"] != run_names_xgboost["IGAE"]
    assert "Temp" in run_names_xgboost["Temperatura"] and "IGAE" not in run_names_xgboost["Temperatura"]
    assert "IGAE" in run_names_xgboost["IGAE"] and "Temp" not in run_names_xgboost["IGAE"]

    # Y deben ser distintos del RUN_NAME de la matriz baseline (todas las exogenas
    # juntas) -- no debe haber riesgo de pisar esa carpeta.
    baseline_run_name = resolve_run(ExperimentConfig(modelo="xgboost")).run_name
    assert baseline_run_name not in run_names_xgboost.values()
    assert baseline_run_name != run_names_xgboost["Temperatura"]

    print("OK test_run_name_sin_colision")
    print(f"   ejemplo: {run_names_xgboost}")


def test_checkpoint_identifica_corrida_individual():
    """
    Una corrida con una sola exogena vive en su propia carpeta (RUN_NAME
    distinto), y el checkpoint por region (cargar_checkpoint_regiones) debe
    reconocerla como cualquier otra corrida -- no hay tratamiento especial
    por tener 1 exogena en vez de 5 u 8.
    """
    config = ExperimentConfig(modelo="sarimax", exogenas=["Temperatura"])
    resolved = resolve_run(config)

    tmp_out = tempfile.mkdtemp(prefix="ckpt_indiv_exog_")
    try:
        forecast_horizon = resolved.forecast_horizon

        series_rows = []
        metrics_rows = []
        config_rows = []
        fechas = pd.date_range("2024-01-01", periods=forecast_horizon, freq="h")

        for region in REGIONS_ALL:
            nombre_serie = f"{region}_DEMANDA"
            for h in range(forecast_horizon):
                series_rows.append({
                    "serie": nombre_serie, "fecha": fechas[h], "tipo": "prediccion",
                    "subset": "test", "modelo": "SARIMAX", "valor": 100.0 + h,
                })
            metrics_rows.append({"serie": nombre_serie, "modelo": "SARIMAX", "MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1})
            config_rows.append({"serie": nombre_serie, "modelo": "SARIMAX", "parametros": "{}"})

        pd.DataFrame(series_rows).to_csv(os.path.join(tmp_out, "series.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(metrics_rows).to_csv(os.path.join(tmp_out, "metricas.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(config_rows).to_csv(os.path.join(tmp_out, "config_usada.csv"), index=False, encoding="utf-8-sig")

        regiones_completas, previos = ckpt.cargar_checkpoint_regiones(
            tmp_out, REGIONS_ALL, forecast_horizon=forecast_horizon,
            requiere_config_usada=True, formato_series="bloques",
        )

        assert regiones_completas == set(REGIONS_ALL), (
            f"El checkpoint no reconocio la corrida individual (Temperatura) como completa: {regiones_completas}"
        )
        assert len(previos["metrics"]) == 8

        print(f"OK test_checkpoint_identifica_corrida_individual (run_name={resolved.run_name})")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_univariados_no_duplican():
    """
    Modelos univariados: ni una corrida, ni 5 corridas redundantes. Se
    prueba tanto con el universo completo (default) como pasando
    explicitamente un modelo univariado en `modelos=`.
    """
    configs = build_individual_exog_matrix(exogenas=EXOGENAS_INDIVIDUALES)
    for modelo in MODELOS_UNIVARIADOS_ESPERADOS:
        assert modelo not in {c.modelo for c in configs}

    configs_mixto = build_individual_exog_matrix(
        exogenas=EXOGENAS_INDIVIDUALES,
        modelos=["naive", "ar", "xgboost"],
    )
    modelos_mixto = [c.modelo for c in configs_mixto]
    assert modelos_mixto.count("naive") == 0, "Naive (univariado) no deberia generar ninguna corrida"
    assert modelos_mixto.count("ar") == 0, "AR (univariado) no deberia generar ninguna corrida"
    assert modelos_mixto.count("xgboost") == len(EXOGENAS_INDIVIDUALES)

    print("OK test_univariados_no_duplican")


def test_exogena_no_reconocida_excluye_solo_esa_combinacion():
    """
    Si se pide una exogena que un modelo multivariado no reconoce, se
    excluye SOLO esa combinacion puntual (el resto de exogenas para ese
    modelo, y el resto de modelos, siguen generandose).
    """
    exogenas_con_una_invalida = ["Temperatura", "NoExisteEnNingunCatalogo"]
    configs = build_individual_exog_matrix(exogenas=exogenas_con_una_invalida, modelos=["xgboost"])

    exogenas_generadas = sorted(c.exogenas[0] for c in configs)
    assert exogenas_generadas == ["Temperatura"], (
        f"Se esperaba que solo 'Temperatura' generara config (la otra no existe en ningun catalogo), "
        f"hubo {exogenas_generadas}"
    )

    print("OK test_exogena_no_reconocida_excluye_solo_esa_combinacion")


def main():
    test_modelos_incluidos_y_excluidos()
    test_numero_exacto_de_configs()
    test_una_exogena_por_config()
    test_sin_primarias_secundarias_terciarias()
    test_defaults_cientificos_preservados()
    test_run_name_sin_colision()
    test_checkpoint_identifica_corrida_individual()
    test_univariados_no_duplican()
    test_exogena_no_reconocida_excluye_solo_esa_combinacion()

    print("\nTODOS LOS TESTS DE LA MATRIZ DE EXOGENAS INDIVIDUALES PASARON")


if __name__ == "__main__":
    main()

"""
Smoke test del pipeline LightGBM (models/lightgbm_model.py).

Objetivo: correr el pipeline COMPLETO (carga de datos, construccion de
exogenas, features, tuning de Optuna, forecast recursivo, guardado) contra
datos sinteticos minimos, para detectar errores de forma/tipos/dependencias
ANTES de lanzar una corrida real de 8 regiones en Colab. No usa datos reales
ni Google Drive; corre en segundos con 1 sola region y 1 trial de Optuna.

Requiere `lightgbm` y `optuna` instalados (no corre en un entorno sin esas
dependencias) -- pensado para ejecutarse en Colab o cualquier entorno con
los mismos requirements que run_experiments.ipynb.

Uso:
    python tests/smoke_lightgbm.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import lightgbm_model as m

REGION = "BCA"

# TRAIN_HOURS/FORECAST_HORIZON elegidos para que _resolver_window() calcule
# un window ADAPTATIVO menor al techo de 168h (336-168 -> window=84 en el
# default vigente; aqui 200-48 -> window=76), en vez de caer justo en el
# caso trivial donde el resultado coincidiria con el techo sin probar nada.
# La version anterior de este test (train_hours=48) exponia el bug real:
# con window=168 fijo, cualquier train_hours <= ~170h dejaba el DataFrame
# de features vacio tras el dropna() -- ver docstring de lightgbm_model.py.
TRAIN_HOURS = 200
FORECAST_HORIZON = 48

# Suficientemente largo para cubrir train + forecast + el lag de 336h
# (2 semanas) que construir_future_exog necesita para estimar Gen/Imp/Exp.
N_HORAS = TRAIN_HOURS + FORECAST_HORIZON + 336 + 24  # margen extra


def _serie_horaria_sintetica(n_horas, seed):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2024-01-01", periods=n_horas, freq="h")
    valores = 100 + 10 * np.sin(np.arange(n_horas) * 2 * np.pi / 24) + rng.normal(0, 2, n_horas)
    return fechas, valores


def _construir_region_long_csv(tmp_dir, region, n_horas):
    fechas, demanda = _serie_horaria_sintetica(n_horas, seed=1)
    df = pd.DataFrame({
        "fecha": fechas.date,
        "Hora": (fechas.hour + 1),  # convencion 1-24 del pipeline
        m.COL_DEMANDA: demanda,
    })
    df.to_csv(os.path.join(tmp_dir, f"{region}_long.csv"), index=False)


def _construir_region_exog_csv(tmp_dir, region, sufijo, n_horas, seed):
    fechas, valores = _serie_horaria_sintetica(n_horas, seed=seed)
    df = pd.DataFrame({
        "fecha": fechas.date,
        "hora": (fechas.hour + 1),
        "valor": np.abs(valores),
    })
    df.to_csv(os.path.join(tmp_dir, f"{region}_{sufijo}.csv"), index=False)


def _construir_exogena_global_h(n_horas, seed):
    fechas, valores = _serie_horaria_sintetica(n_horas, seed=seed)
    return pd.DataFrame({
        "fecha": fechas.date,
        "hora": (fechas.hour + 1),
        "valor": valores,
    })


def main():
    tmp_data_dir = tempfile.mkdtemp(prefix="smoke_lightgbm_data_")
    tmp_out_dir = tempfile.mkdtemp(prefix="smoke_lightgbm_out_")

    try:
        _construir_region_long_csv(tmp_data_dir, REGION, N_HORAS)
        _construir_region_exog_csv(tmp_data_dir, REGION, "GEN", N_HORAS, seed=2)
        _construir_region_exog_csv(tmp_data_dir, REGION, "IMP", N_HORAS, seed=3)
        _construir_region_exog_csv(tmp_data_dir, REGION, "EXP", N_HORAS, seed=4)

        exogenas_globales = {
            "Temperaturas_H": _construir_exogena_global_h(N_HORAS, seed=5),
            "Primarias_H": _construir_exogena_global_h(N_HORAS, seed=6),
            "Secundarias_H": _construir_exogena_global_h(N_HORAS, seed=7),
            "Terciarias_H": _construir_exogena_global_h(N_HORAS, seed=8),
            "IGAE_H": _construir_exogena_global_h(N_HORAS, seed=9),
        }

        print(f"Datos sinteticos en: {tmp_data_dir}")
        print(f"Salida en: {tmp_out_dir}")
        print(f"N_HORAS={N_HORAS}, TRAIN_HOURS={TRAIN_HOURS}, FORECAST_HORIZON={FORECAST_HORIZON}")

        series_df, metricas_df, trials_df, config_usada_df = m.run(
            exogenas_globales=exogenas_globales,
            regions_all=[REGION],
            train_hours=TRAIN_HOURS,
            forecast_horizon=FORECAST_HORIZON,
            exog_cols=list(m.EXOG_COLS_DEFAULT),
            optuna_n_trials=1,
            data_dir=tmp_data_dir,
            output_dir=tmp_out_dir,
        )

        # ---------------------------------------------------------
        # VALIDACIONES
        # ---------------------------------------------------------

        window_resuelto = m._resolver_window(TRAIN_HOURS, FORECAST_HORIZON)
        print(f"Window adaptativo resuelto: {window_resuelto}h (techo={m.WINDOW_DEFAULT}h)")
        assert window_resuelto < m.WINDOW_DEFAULT, (
            "Este test deberia ejercitar el caso adaptativo (window < techo); "
            "ajusta TRAIN_HOURS/FORECAST_HORIZON si esto ya no se cumple."
        )

        assert len(metricas_df) == 1, f"Se esperaba 1 fila de metricas, hubo {len(metricas_df)}"
        assert metricas_df["region"].iloc[0] == REGION

        for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
            valor = metricas_df[col].iloc[0]
            assert np.isfinite(valor), f"{col} no es finito: {valor}"

        assert len(series_df) == FORECAST_HORIZON, (
            f"Se esperaban {FORECAST_HORIZON} predicciones, hubo {len(series_df)}"
        )

        assert len(trials_df) >= 1, "No se registro ningun trial de Optuna"
        assert len(config_usada_df) == 1

        for archivo in ["series.csv", "metricas.csv", "trials.csv", "config_usada.csv"]:
            ruta = os.path.join(tmp_out_dir, archivo)
            assert os.path.exists(ruta), f"No se genero {archivo}"

        print("\nSMOKE TEST LIGHTGBM: OK")
        print(metricas_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

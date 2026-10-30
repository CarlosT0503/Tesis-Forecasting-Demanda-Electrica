"""
Smoke test del pipeline FCNN multivariada (models/fcnn_model.py).

Corre el pipeline completo (ambas estrategias: directa y STL-residuos)
contra datos sinteticos minimos, 1 region, 1 trial de Optuna por
estrategia. Requiere `tensorflow`, `optuna` y `statsmodels` (STL) -- no se
pudo ejecutar en el entorno de desarrollo local (no tiene tensorflow/optuna
instalados); pensado para correr en Colab antes de lanzar el pipeline
completo de 8 regiones.

Uso:
    python tests/smoke_fcnn.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import fcnn_model as m

REGION = "BCA"

# WINDOW=168 es fijo (no configurable) dentro de fcnn_model -- el train
# debe ser sustancialmente mayor a 168 para tener ventanas suficientes
# tanto para el modelo directo como para el de residuos STL.
TRAIN_HOURS = 600
FORECAST_HORIZON = 48
N_HORAS = TRAIN_HOURS + FORECAST_HORIZON + 200  # margen para lag168 + STL


def _serie_horaria_sintetica(n_horas, seed):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2024-01-01", periods=n_horas, freq="h")
    valores = 100 + 10 * np.sin(np.arange(n_horas) * 2 * np.pi / 24) + rng.normal(0, 2, n_horas)
    return fechas, valores


def _construir_region_long_csv(tmp_dir, region, n_horas):
    fechas, demanda = _serie_horaria_sintetica(n_horas, seed=1)
    df = pd.DataFrame({
        "fecha": fechas.date,
        "Hora": (fechas.hour + 1),
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
    tmp_data_dir = tempfile.mkdtemp(prefix="smoke_fcnn_data_")
    tmp_out_dir = tempfile.mkdtemp(prefix="smoke_fcnn_out_")

    try:
        _construir_region_long_csv(tmp_data_dir, REGION, N_HORAS)
        _construir_region_exog_csv(tmp_data_dir, REGION, "GEN", N_HORAS, seed=2)
        _construir_region_exog_csv(tmp_data_dir, REGION, "IMP", N_HORAS, seed=3)
        _construir_region_exog_csv(tmp_data_dir, REGION, "EXP", N_HORAS, seed=4)

        exogenas_globales = {
            "Temperaturas_H": _construir_exogena_global_h(N_HORAS, seed=5),
            "IGAE_H": _construir_exogena_global_h(N_HORAS, seed=6),
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

        # 2 modelos por region (directa + STL-residuos)
        assert len(metricas_df) == 2, f"Se esperaban 2 filas de metricas (2 estrategias), hubo {len(metricas_df)}"
        assert set(metricas_df["modelo"]) == {m.MODELO_DIRECTA, m.MODELO_STL_RESIDUOS}
        assert (metricas_df["region"] == REGION).all()

        for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
            assert metricas_df[col].notna().all(), f"{col} tiene NaN inesperado"
            assert np.isfinite(metricas_df[col]).all(), f"{col} no es finito"

        n_pred = (series_df["tipo"] == "prediccion").sum()
        assert n_pred == 2 * FORECAST_HORIZON, f"Se esperaban {2*FORECAST_HORIZON} predicciones, hubo {n_pred}"

        assert len(config_usada_df) == 2
        assert len(trials_df) >= 2, "Se esperaban trials de ambas estrategias"

        for archivo in ["series.csv", "metricas.csv", "trials.csv", "config_usada.csv"]:
            ruta = os.path.join(tmp_out_dir, archivo)
            assert os.path.exists(ruta), f"No se genero {archivo}"

        print("\nSMOKE TEST FCNN: OK")
        print(metricas_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

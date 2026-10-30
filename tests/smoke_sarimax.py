"""
Smoke test del pipeline SARIMAX (models/sarimax_model.py).

Corre el pipeline completo (carga, exogenas, SARIMAX, guardado) contra
datos sinteticos minimos: 1 region, ventana chica. No requiere Optuna
(SARIMAX no tunea) -- solo pandas/numpy/scikit-learn/statsmodels, por lo
que puede correr en cualquier entorno con esas dependencias, incluyendo
fuera de Colab.

Uso:
    python tests/smoke_sarimax.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import sarimax_model as m

REGION = "BCA"

TRAIN_HOURS = 48
FORECAST_HORIZON = 24
N_HORAS = TRAIN_HOURS + FORECAST_HORIZON + 336 + 24  # margen para el lag de 336h


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
    tmp_data_dir = tempfile.mkdtemp(prefix="smoke_sarimax_data_")
    tmp_out_dir = tempfile.mkdtemp(prefix="smoke_sarimax_out_")

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
            data_dir=tmp_data_dir,
            output_dir=tmp_out_dir,
        )

        assert len(metricas_df) == 1, f"Se esperaba 1 fila de metricas, hubo {len(metricas_df)}"
        assert metricas_df["region"].iloc[0] == REGION

        for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
            valor = metricas_df[col].iloc[0]
            assert np.isfinite(valor), f"{col} no es finito: {valor}"

        # series.csv: 1 fila "real" completa + FORECAST_HORIZON predicciones
        n_pred = (series_df["tipo"] == "prediccion").sum()
        assert n_pred == FORECAST_HORIZON, f"Se esperaban {FORECAST_HORIZON} predicciones, hubo {n_pred}"

        assert len(config_usada_df) == 1
        assert trials_df is not None and len(trials_df) == 0, "SARIMAX no deberia generar trials (no tunea)"

        for archivo in ["series.csv", "metricas.csv", "config_usada.csv"]:
            ruta = os.path.join(tmp_out_dir, archivo)
            assert os.path.exists(ruta), f"No se genero {archivo}"

        assert not os.path.exists(os.path.join(tmp_out_dir, "trials.csv")), (
            "No deberia existir trials.csv para SARIMAX (no tunea)"
        )

        print("\nSMOKE TEST SARIMAX: OK")
        print(metricas_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

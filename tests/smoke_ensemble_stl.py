"""
Smoke test del pipeline Ensemble STL (models/ensemble_stl.py).

Corre el pipeline completo (STL + LSTM tendencia + FCNN estacionalidad +
AR residuo) contra datos sinteticos minimos, 1 region, 1 trial de Optuna
por submodelo, barrido de AR acotado. Requiere `tensorflow`, `optuna` y
`statsmodels` -- no se pudo ejecutar en el entorno de desarrollo local (no
tiene tensorflow/optuna instalados); pensado para correr en Colab antes de
lanzar el pipeline completo de 8 regiones. Es el smoke test mas pesado de
los cuatro (entrena 2 redes + STL + barrido AR de hasta 168 lags), pero
sigue siendo minutos, no horas, gracias al train chico y 1 solo trial.

Uso:
    python tests/smoke_ensemble_stl.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import ensemble_stl as m

REGION = "BCA"

# WINDOW=168 y STL_PERIOD=168 son fijos -- el train debe ser sustancialmente
# mayor a 168 para que la descomposicion STL y las ventanas tengan sentido.
TRAIN_HOURS = 600
FORECAST_HORIZON = 48
N_HORAS = TRAIN_HOURS + FORECAST_HORIZON + 200


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
    tmp_data_dir = tempfile.mkdtemp(prefix="smoke_ensemble_data_")
    tmp_out_dir = tempfile.mkdtemp(prefix="smoke_ensemble_out_")

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

        assert len(metricas_df) == 1, f"Se esperaba 1 fila de metricas (modelo final combinado), hubo {len(metricas_df)}"
        assert metricas_df["modelo"].iloc[0] == m.NOMBRE_MODELO_FINAL
        assert metricas_df["region"].iloc[0] == REGION

        for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
            valor = metricas_df[col].iloc[0]
            assert np.isfinite(valor), f"{col} no es finito: {valor}"

        # series.csv: real + prediccion final + 3 componentes, cada uno FORECAST_HORIZON filas
        tipos = series_df["tipo"].value_counts()
        assert tipos.get("prediccion", 0) == FORECAST_HORIZON
        assert tipos.get("componente_pred", 0) == 3 * FORECAST_HORIZON
        assert set(series_df.loc[series_df["tipo"] == "componente_pred", "modelo"]) == {
            "LSTM_trend_EXOG_ALL_Lag168", "FCNN_seasonal_EXOG_ALL_Lag168", "AR_resid",
        }

        assert len(config_usada_df) == 1
        # trials.csv: LSTM trend + FCNN seasonal + barrido de lags del AR
        assert set(trials_df["modelo"]) == {"LSTM_trend", "FCNN_seasonal", "AR_resid"}

        for archivo in ["series.csv", "metricas.csv", "trials.csv", "config_usada.csv"]:
            ruta = os.path.join(tmp_out_dir, archivo)
            assert os.path.exists(ruta), f"No se genero {archivo}"

        print("\nSMOKE TEST ENSEMBLE STL: OK")
        print(metricas_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

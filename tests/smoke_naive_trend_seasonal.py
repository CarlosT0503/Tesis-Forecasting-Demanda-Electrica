"""
Smoke test del pipeline Naive + Tendencia + Estacionalidad
(models/naive_trend_seasonal_model.py).

Corre el pipeline completo (STL, tendencia lineal, estacionalidad repetida,
guardado) contra datos sinteticos, 1 region. Solo depende de pandas/numpy/
scikit-learn/statsmodels -- corre en cualquier entorno.

Uso:
    python tests/smoke_naive_trend_seasonal.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import naive_trend_seasonal_model as m

REGION = "BCA"

TRAIN_HOURS = m.TRAIN_LAST_HOURS_DEFAULT   # 3600h -- STL con period=168 necesita una ventana asi
FORECAST_HORIZON = m.FORECAST_HORIZON_DEFAULT  # 168h
N_HORAS = TRAIN_HOURS + FORECAST_HORIZON + 24  # margen chico


def _construir_region_long_csv(tmp_dir, region, n_horas):
    rng = np.random.default_rng(1)
    fechas = pd.date_range("2024-01-01", periods=n_horas, freq="h")
    demanda = (
        100
        + 0.005 * np.arange(n_horas)
        + 10 * np.sin(np.arange(n_horas) * 2 * np.pi / 24)
        + 3 * np.sin(np.arange(n_horas) * 2 * np.pi / 168)
        + rng.normal(0, 1, n_horas)
    )

    df = pd.DataFrame({
        "fecha": fechas.date,
        "Hora": (fechas.hour + 1),
        m.COL_DEMANDA: demanda,
    })
    df.to_csv(os.path.join(tmp_dir, f"{region}_long.csv"), index=False)


def main():
    tmp_data_dir = tempfile.mkdtemp(prefix="smoke_ntss_data_")
    tmp_out_dir = tempfile.mkdtemp(prefix="smoke_ntss_out_")

    try:
        _construir_region_long_csv(tmp_data_dir, REGION, N_HORAS)

        print(f"Datos sinteticos en: {tmp_data_dir}")
        print(f"Salida en: {tmp_out_dir}")
        print(f"N_HORAS={N_HORAS}, TRAIN_HOURS={TRAIN_HOURS}, FORECAST_HORIZON={FORECAST_HORIZON}")

        series_df, metricas_df, trials_df, config_usada_df = m.run(
            exogenas_globales={},
            regions_all=[REGION],
            data_dir=tmp_data_dir,
            output_dir=tmp_out_dir,
        )

        assert len(metricas_df) == 1, f"Se esperaba 1 fila de metricas, hubo {len(metricas_df)}"
        assert metricas_df["modelo"].iloc[0] == m.NOMBRE_MODELO
        assert metricas_df["region"].iloc[0] == REGION

        for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
            valor = metricas_df[col].iloc[0]
            assert np.isfinite(valor), f"{col} no es finito: {valor}"

        n_pred = (series_df["tipo"] == "prediccion").sum()
        assert n_pred == FORECAST_HORIZON, f"Se esperaban {FORECAST_HORIZON} predicciones, hubo {n_pred}"

        # La fila "real" usa la serie COMPLETA sin recortar (igual que la
        # celda 64 original: guardar_resultados recibe `serie`/`fechas` sin
        # slicear a la ventana reciente de train+test).
        n_real = (series_df["tipo"] == "real").sum()
        assert n_real == N_HORAS, f"Filas 'real' inesperadas: {n_real} (se esperaban {N_HORAS})"

        try:
            m.run(exogenas_globales={}, regions_all=[REGION], exog_cols=["Temperatura"],
                  data_dir=tmp_data_dir, output_dir=tmp_out_dir)
            raise SystemExit("Se esperaba ValueError al pasar exogenas a un modelo univariado")
        except ValueError as e:
            print(f"Rechazo esperado de exogenas: {e}")

        assert len(trials_df) == 0
        assert len(config_usada_df) == 0

        for archivo in ["series.csv", "metricas.csv"]:
            ruta = os.path.join(tmp_out_dir, archivo)
            assert os.path.exists(ruta), f"No se genero {archivo}"

        print("\nSMOKE TEST NAIVE_TREND_SEASONAL: OK")
        print(metricas_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

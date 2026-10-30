"""
Smoke test del pipeline Naive (models/naive_model.py).

Corre el pipeline completo (carga, split dinamico, forecast, guardado)
contra datos sinteticos minimos, 1 region. Solo depende de pandas/numpy --
corre en cualquier entorno, incluyendo local sin Colab.

Uso:
    python tests/smoke_naive.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import naive_model as m

REGION = "BCA"
N_HORAS = 24 * 40  # 40 dias: suficiente para MIN_OBS (30 dias) + un test_size razonable


def _construir_region_long_csv(tmp_dir, region, n_horas):
    rng = np.random.default_rng(1)
    fechas = pd.date_range("2024-01-01", periods=n_horas, freq="h")
    demanda = 100 + 10 * np.sin(np.arange(n_horas) * 2 * np.pi / 24) + rng.normal(0, 2, n_horas)

    df = pd.DataFrame({
        "fecha": fechas.date,
        "Hora": (fechas.hour + 1),
        m.COL_DEMANDA: demanda,
    })
    df.to_csv(os.path.join(tmp_dir, f"{region}_long.csv"), index=False)


def main():
    tmp_data_dir = tempfile.mkdtemp(prefix="smoke_naive_data_")
    tmp_out_dir = tempfile.mkdtemp(prefix="smoke_naive_out_")

    try:
        _construir_region_long_csv(tmp_data_dir, REGION, N_HORAS)

        print(f"Datos sinteticos en: {tmp_data_dir}")
        print(f"Salida en: {tmp_out_dir}")
        print(f"N_HORAS={N_HORAS}")

        series_df, metricas_df, trials_df, config_usada_df = m.run(
            exogenas_globales={},
            regions_all=[REGION],
            data_dir=tmp_data_dir,
            output_dir=tmp_out_dir,
        )

        # ---------------------------------------------------------
        # VALIDACIONES
        # ---------------------------------------------------------

        assert len(metricas_df) == 1, f"Se esperaba 1 fila de metricas, hubo {len(metricas_df)}"
        assert metricas_df["modelo"].iloc[0] == "Naive"
        assert metricas_df["region"].iloc[0] == REGION
        assert metricas_df["tuneado"].iloc[0] == False  # noqa: E712 (sin Optuna)

        for col in ["MAPE", "sMAPE", "MAE", "RMSE"]:
            valor = metricas_df[col].iloc[0]
            assert np.isfinite(valor), f"{col} no es finito: {valor}"

        # Split dinamico esperado: max(720, 10% de N_HORAS), acotado a N_HORAS//3
        test_size_esperado = min(max(24 * 30, int(N_HORAS * 0.10)), N_HORAS // 3)
        n_pred = (series_df["tipo"] == "prediccion").sum()
        assert n_pred == test_size_esperado, f"Se esperaban {test_size_esperado} predicciones, hubo {n_pred}"

        n_real = (series_df["tipo"] == "real").sum()
        assert n_real == N_HORAS, f"Se esperaban {N_HORAS} filas 'real', hubo {n_real}"

        # Sin exogenas: exog_cols debe ser rechazado
        try:
            m.run(exogenas_globales={}, regions_all=[REGION], exog_cols=["Temperatura"],
                  data_dir=tmp_data_dir, output_dir=tmp_out_dir)
            raise SystemExit("Se esperaba ValueError al pasar exogenas a un modelo univariado")
        except ValueError as e:
            print(f"Rechazo esperado de exogenas: {e}")

        # Sin tuning: no deberian existir trials.csv/config_usada.csv
        assert len(trials_df) == 0
        assert len(config_usada_df) == 0

        for archivo in ["series.csv", "metricas.csv"]:
            ruta = os.path.join(tmp_out_dir, archivo)
            assert os.path.exists(ruta), f"No se genero {archivo}"

        assert not os.path.exists(os.path.join(tmp_out_dir, "trials.csv"))
        assert not os.path.exists(os.path.join(tmp_out_dir, "config_usada.csv"))

        print("\nSMOKE TEST NAIVE: OK")
        print(metricas_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

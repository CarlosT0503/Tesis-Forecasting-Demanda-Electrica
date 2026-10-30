"""
Corre localmente (sin TensorFlow/Optuna) los modelos estadisticos de la
extension a cuatro semanas: naive, naive_trend, naive_trend_seasonal, ar,
ar_resid_trend_seasonal y sarimax.

    python scripts/run_local_estadisticos.py --data-dir datos_locales \
        --out resultados_locales/Pipeline_Resultados_Ventanas --ventanas S4 S1 S2 S3

Los resultados se escriben en --out/<S#>/<RUN_NAME>/ (misma estructura que en
Colab) y se pueden copiar a Pipeline_Resultados_Ventanas/ de Drive. Los
modelos con redes/arboles (xgboost, lightgbm, fcnn, lstm_*, ensemble_stl) NO
se pueden correr aqui: se corren en Colab con notebooks/run_ventanas.ipynb.
"""

import argparse
import os
import sys
from unittest import mock

# Un hilo de BLAS por proceso: con varios procesos en paralelo, cada uno con un
# hilo por nucleo satura la CPU y los hilos se estorban (mide mas lento).
# Debe fijarse ANTES de importar numpy.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# runner.py importa todos los modelos; los que necesitan librerias pesadas se
# sustituyen por stubs (aqui solo se ejecutan modelos que no las usan).
for nombre in [
    "optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras",
    "tensorflow.keras.models", "tensorflow.keras.layers", "tensorflow.keras.callbacks",
    "tensorflow.keras.optimizers", "tensorflow.keras.backend",
]:
    try:
        __import__(nombre)
    except ImportError:
        sys.modules[nombre] = mock.MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tesis_forecast import io_drive, ventanas_tools as vt  # noqa: E402
from tesis_forecast.matrix import run_matrix  # noqa: E402

MODELOS_LOCALES = ["naive", "naive_trend", "naive_trend_seasonal", "ar_resid_trend_seasonal", "ar", "sarimax"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datos_locales")
    ap.add_argument("--out", default="resultados_locales/Pipeline_Resultados_Ventanas")
    ap.add_argument("--ventanas", nargs="+", default=["S4", "S1", "S2", "S3"])
    ap.add_argument("--modelos", nargs="+", default=MODELOS_LOCALES)
    ap.add_argument("--niveles", nargs="+", default=["T1"], choices=["T1", "T2", "T3"])
    args = ap.parse_args()

    io_drive.BASE_DIR_VENTANAS = os.path.abspath(args.out)
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(io_drive.BASE_DIR_VENTANAS, exist_ok=True)
    tiempos = os.path.join(
        io_drive.BASE_DIR_VENTANAS, "_auditoria",
        "tiempos_local_" + "-".join(args.ventanas) + "_" + "-".join(args.niveles) + "_" + "-".join(sorted(args.modelos))[:40] + ".csv",
    )

    for v in args.ventanas:
        configs = []
        for nivel in args.niveles:
            orden = {m: i for i, m in enumerate(args.modelos)}
            parte = [c for c in vt.construir_configs(nivel, v) if c.modelo in args.modelos]
            parte.sort(key=lambda c: orden[c.modelo])
            configs += parte
        print(f"\n##### Ventana {v}: {len(configs)} configuraciones -> {io_drive.BASE_DIR_VENTANAS}")
        registros = run_matrix(configs, data_dir=data_dir, mount_drive=False)
        vt.guardar_tiempos(registros, tiempos)


if __name__ == "__main__":
    main()

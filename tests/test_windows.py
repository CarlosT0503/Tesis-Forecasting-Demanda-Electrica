"""
Pruebas de la extension a cuatro semanas (windows.py + runner/config con
`ventana`). Solo usa datos sinteticos y corre localmente sin tensorflow,
optuna, xgboost ni lightgbm (se sustituyen por stubs solo para poder
importar runner.py; el modelo que se ejecuta de verdad es
naive_trend_seasonal, que solo necesita pandas/numpy/statsmodels/sklearn).

Cubre:
  1. Las 4 ventanas: 168 h, consecutivas, S4 = semana original.
  2. Materializacion: ningun dato posterior al fin de la ventana, region con
     historia insuficiente omitida, archivos originales intactos.
  3. runner: resultados en Pipeline_Resultados_Ventanas/<S#>/, config.json
     con la ventana, y la semana de prueba evaluada es exactamente la pedida.
  4. PRUEBA DE FUGA: dos datasets identicos hasta el fin de S1 y MUY
     distintos despues producen metricas y predicciones identicas en S1.
  5. Sin `ventana` el comportamiento es el de siempre.

Uso:
    python tests/test_windows.py
"""

import json
import os
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

for nombre in [
    "optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras",
    "tensorflow.keras.models", "tensorflow.keras.layers", "tensorflow.keras.callbacks",
    "tensorflow.keras.optimizers", "tensorflow.keras.backend",
]:
    sys.modules.setdefault(nombre, mock.MagicMock())

from tesis_forecast import io_drive, runner, windows  # noqa: E402
from tesis_forecast.config import ExperimentConfig, resolved_config_dict  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def construir_datos(dir_, futuro_factor=1.0):
    """Datos horarios 2025-10-01 .. 2026-05-23 23:00 (convencion Hora 1..24).
    `futuro_factor` multiplica todo lo posterior al 2 de mayo (fin de S1)."""
    rng = np.random.default_rng(7)
    fin_s1 = pd.Timestamp("2026-05-02 23:00")

    def serie(fechas, base):
        n = len(fechas)
        v = (base + 0.004 * np.arange(n) + 12 * np.sin(np.arange(n) * 2 * np.pi / 24)
             + 5 * np.sin(np.arange(n) * 2 * np.pi / 168) + rng.normal(0, 1, n))
        return np.where(fechas > fin_s1, v * futuro_factor, v)

    def escribir(region, desde):
        f = pd.date_range(desde, "2026-05-23 23:00", freq="h")
        pd.DataFrame({
            "fecha": f.strftime("%Y-%m-%d"),
            "Hora": f.hour + 1,
            windows.COL_DEMANDA: serie(f, 1000.0),
        }).to_csv(os.path.join(dir_, f"{region}_long.csv"), index=False)
        for suf in windows.SUFIJOS_EXOG_REGIONAL:
            pd.DataFrame({
                "fecha": f.strftime("%Y-%m-%d"), "hora": f.hour + 1, "valor": serie(f, 50.0),
            }).to_csv(os.path.join(dir_, f"{region}_{suf}.csv"), index=False)

    escribir("CEN", "2025-10-01")
    escribir("NES", "2025-10-01")
    escribir("BCA", "2026-04-20")  # historia corta: debe omitirse con train_hours grande
    open(os.path.join(dir_, "IGAE_2.xlsx"), "wb").write(b"placeholder")
    open(os.path.join(dir_, "Temperaturas promedio.csv"), "w").write("placeholder\n")


def main():
    print("1. Definicion de ventanas")
    ids = ["S1", "S2", "S3", "S4"]
    for v_id in ids:
        v = windows.get_ventana(v_id)
        horas = int((v.test_end - v.test_start) / pd.Timedelta(hours=1)) + 1
        check(horas == 168, f"{v_id}: {v.test_start} .. {v.test_end} = {horas} h")
    check(windows.VENTANAS["S4"].inicio == "2026-05-17" and windows.VENTANAS["S4"].fin == "2026-05-23",
          "S4 es la semana original 17-23 mayo 2026")
    for a, b in zip(ids, ids[1:]):
        check(windows.VENTANAS[b].test_start == windows.VENTANAS[a].test_end + pd.Timedelta(hours=1),
              f"{a} y {b} son consecutivas")

    tmp = tempfile.mkdtemp(prefix="test_windows_")
    try:
        datos = os.path.join(tmp, "datos")
        os.makedirs(datos)
        construir_datos(datos)
        antes = {f: os.path.getsize(os.path.join(datos, f)) for f in os.listdir(datos)}

        print("\n2. Materializacion")
        for v_id in ids:
            out = os.path.join(tmp, "vista", v_id)
            m = windows.materializar_datos_ventana(datos, v_id, out, train_hours=720)
            fin = pd.Timestamp(windows.VENTANAS[v_id].fin)
            for region in ["CEN", "NES"]:
                df = pd.read_csv(os.path.join(out, f"{region}_long.csv"))
                check(pd.to_datetime(df["fecha"]).max() == fin, f"{v_id}/{region}: la demanda termina el {fin.date()}")
                for suf in windows.SUFIJOS_EXOG_REGIONAL:
                    e = pd.read_csv(os.path.join(out, f"{region}_{suf}.csv"))
                    check(pd.to_datetime(e["fecha"]).max() == fin, f"{v_id}/{region}_{suf}: sin filas posteriores al fin")
            check("BCA" in m["regiones_omitidas"], f"{v_id}: BCA (history corta, train=720) omitida -> {m['regiones_omitidas'].get('BCA')}")
            check(not os.path.exists(os.path.join(out, "BCA_long.csv")), f"{v_id}: no se escribio BCA_long.csv")

        orig = pd.read_csv(os.path.join(datos, "CEN_long.csv"), dtype=str)
        s4 = pd.read_csv(os.path.join(tmp, "vista", "S4", "CEN_long.csv"), dtype=str)
        check(orig.equals(s4), "S4 (datos hasta el 23-may) es identica a los datos originales")

        m_bca = windows.materializar_datos_ventana(datos, "S4", os.path.join(tmp, "vista", "bca_ok"), train_hours=100)
        check("BCA" not in m_bca["regiones_omitidas"], "BCA entra si train_hours=100 cabe en su historia")

        despues = {f: os.path.getsize(os.path.join(datos, f)) for f in os.listdir(datos)}
        check(antes == despues, "los archivos originales no se modificaron")

        try:
            windows.materializar_datos_ventana(datos, "S1", datos)
            check(False, "debe negarse a escribir sobre data_dir")
        except ValueError:
            check(True, "se niega a escribir sobre data_dir")

        print("\n3. Runner con ventana")
        base_ventanas = os.path.join(tmp, "Pipeline_Resultados_Ventanas")
        resultados = {}
        with mock.patch.object(io_drive, "BASE_DIR_VENTANAS", base_ventanas), \
             mock.patch.object(windows, "carpeta_datos_ventana", lambda v: os.path.join(tmp, "scratch", v)), \
             mock.patch.object(runner, "cargar_exogenas_horarias", lambda data_dir: {}):
            for v_id in ["S1", "S4"]:
                cfg = ExperimentConfig(modelo="naive_trend_seasonal", train_hours=720, ventana=v_id)
                res = runner.run_experiment(cfg, data_dir=datos, mount_drive=False)
                resultados[v_id] = res
                check(res.run_dir == os.path.join(base_ventanas, v_id, res.run_name),
                      f"{v_id}: resultados en Pipeline_Resultados_Ventanas/{v_id}/")
                cj = json.load(open(os.path.join(res.run_dir, "config.json"), encoding="utf-8"))
                check(cj["ventana"] == v_id and cj["test_end"].startswith(windows.VENTANAS[v_id].fin),
                      f"{v_id}: config.json registra la ventana ({cj['test_start']} .. {cj['test_end']})")
                check(os.path.exists(os.path.join(res.run_dir, windows.MANIFEST_NOMBRE)), f"{v_id}: manifiesto de datos guardado")
                ser = pd.read_csv(os.path.join(res.run_dir, "series.csv"), encoding="utf-8-sig")
                pred = ser[ser["tipo"] == "prediccion"]
                f = pd.to_datetime(pred["fecha"])
                check(f.min() == windows.VENTANAS[v_id].test_start and f.max() == windows.VENTANAS[v_id].test_end,
                      f"{v_id}: la semana evaluada es {f.min()} .. {f.max()}")
                check(set(pd.read_csv(os.path.join(res.run_dir, "metricas.csv"))["region"]) == {"CEN", "NES"},
                      f"{v_id}: metricas solo de CEN y NES (BCA omitida)")

            print("\n4. Prueba de fuga (datos posteriores al fin de S1 completamente distintos)")
            datos_b = os.path.join(tmp, "datos_b")
            os.makedirs(datos_b)
            construir_datos(datos_b, futuro_factor=10.0)
            base_b = os.path.join(tmp, "Pipeline_Resultados_Ventanas_B")
            with mock.patch.object(io_drive, "BASE_DIR_VENTANAS", base_b), \
                 mock.patch.object(windows, "carpeta_datos_ventana", lambda v: os.path.join(tmp, "scratch_b", v)):
                cfg = ExperimentConfig(modelo="naive_trend_seasonal", train_hours=720, ventana="S1")
                res_b = runner.run_experiment(cfg, data_dir=datos_b, mount_drive=False)

            a = pd.read_csv(os.path.join(resultados["S1"].run_dir, "metricas.csv"), encoding="utf-8-sig")
            b = pd.read_csv(os.path.join(res_b.run_dir, "metricas.csv"), encoding="utf-8-sig")
            cols = ["MAE", "RMSE", "MAPE", "sMAPE"]
            check(np.allclose(a[cols].values, b[cols].values, rtol=0, atol=0), "S1: metricas identicas con futuro distinto")
            sa = pd.read_csv(os.path.join(resultados["S1"].run_dir, "series.csv"), encoding="utf-8-sig")
            sb = pd.read_csv(os.path.join(res_b.run_dir, "series.csv"), encoding="utf-8-sig")
            check(sa["valor"].equals(sb["valor"]), "S1: series (reales y predicciones) identicas con futuro distinto")

            m_s1 = pd.read_csv(os.path.join(resultados["S1"].run_dir, "metricas.csv"), encoding="utf-8-sig")
            m_s4 = pd.read_csv(os.path.join(resultados["S4"].run_dir, "metricas.csv"), encoding="utf-8-sig")
            check(not np.allclose(m_s1[cols].values, m_s4[cols].values), "sanity: S1 y S4 dan metricas distintas")

        print("\n4b. Baselines univariados con ventana fija (Naive, Naive_Trend, AR)")
        with mock.patch.object(io_drive, "BASE_DIR_VENTANAS", base_ventanas), \
             mock.patch.object(windows, "carpeta_datos_ventana", lambda v: os.path.join(tmp, "scratch", v)), \
             mock.patch.object(runner, "cargar_exogenas_horarias", lambda data_dir: {}):
            for modelo in ["naive", "naive_trend", "ar"]:
                cfg = ExperimentConfig(modelo=modelo, train_hours=720, forecast_horizon=168, ventana="S2")
                res = runner.run_experiment(cfg, data_dir=datos, mount_drive=False)
                ser = pd.read_csv(os.path.join(res.run_dir, "series.csv"), encoding="utf-8-sig")
                pred = ser[ser["tipo"] == "prediccion"]
                f = pd.to_datetime(pred["fecha"])
                check(f.min() == windows.VENTANAS["S2"].test_start and f.max() == windows.VENTANAS["S2"].test_end
                      and pred["serie"].nunique() == 2 and len(pred) == 2 * 168,
                      f"{modelo}: S2 evaluada exactamente ({f.min()} .. {f.max()}), 168 h x 2 regiones")

        print("\n5. Sin ventana todo sigue igual")
        cfg = ExperimentConfig(modelo="naive_trend_seasonal")
        r = runner.resolve_run(cfg)
        check(r.run_dir.startswith(io_drive.BASE_DIR + "/") or r.run_dir.startswith(io_drive.BASE_DIR + os.sep),
              "sin ventana: carpeta en Pipeline_Resultados/")
        check("ventana" not in resolved_config_dict(cfg, r.run_name, 3600, 168, None, []),
              "sin ventana: config.json no incorpora la clave 'ventana'")
        r1 = runner.resolve_run(ExperimentConfig(modelo="naive_trend_seasonal", ventana="S2"))
        check(r1.run_name == r.run_name and r1.run_dir != r.run_dir, "con ventana: mismo RUN_NAME, carpeta distinta")
        try:
            runner.resolve_run(ExperimentConfig(modelo="naive_trend_seasonal", ventana="S2", forecast_horizon=24))
            check(False, "con ventana, horizonte != 168 debe rechazarse")
        except ValueError:
            check(True, "con ventana, horizonte != 168 se rechaza")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODO OK" if not FALLOS else f"{len(FALLOS)} FALLO(S): {FALLOS}"))
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())

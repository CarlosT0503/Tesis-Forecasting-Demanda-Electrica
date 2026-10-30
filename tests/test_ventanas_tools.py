"""
Pruebas de ventanas_tools.py con datos sinteticos (sin tensorflow/optuna/
xgboost/lightgbm: se stubbean solo para poder importar runner.py).

    python tests/test_ventanas_tools.py
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

from tesis_forecast import ventanas_tools as vt, windows  # noqa: E402
from tesis_forecast.runner import resolve_run  # noqa: E402
from tesis_forecast.regions import REGIONS_ALL  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def escribir_corrida(base, ventana, run_name, modelo, exog, mape_por_region, estrategias=("M",)):
    d = os.path.join(base, ventana, run_name)
    os.makedirs(d, exist_ok=True)
    json.dump({"modelo": modelo, "exogenas": exog, "train_hours": 336, "ventana": ventana},
              open(os.path.join(d, "config.json"), "w"))
    filas = []
    for reg, mape in mape_por_region.items():
        for est in estrategias:
            filas.append({"serie": f"{reg}_DEMANDA", "modelo": est, "MAE": mape * 10, "RMSE": mape * 12,
                          "MAPE": mape, "sMAPE": mape * 1.01, "region": reg})
    pd.DataFrame(filas).to_csv(os.path.join(d, "metricas.csv"), index=False, encoding="utf-8-sig")
    return d


def main():
    print("1. Plan de experimentos")
    for ventana in ["S1", "S4"]:
        t1, t2, t3 = (vt.construir_configs(n, ventana) for n in ("T1", "T2", "T3"))
        todos = [resolve_run(c) for c in t1 + t2 + t3]
        nombres = [r.run_name for r in todos]
        check(len(t1) == 8 and len(t2) == 4, f"{ventana}: T1=8 y T2=4 configuraciones")
        check(len(nombres) == len(set(nombres)), f"{ventana}: sin RUN_NAME repetidos entre T1+T2+T3 ({len(nombres)} corridas)")
        check(all(f"/{ventana}" in r.run_dir.replace(os.sep, "/") for r in todos), f"{ventana}: todas escriben en Pipeline_Resultados_Ventanas/{ventana}/")
        check(all(r.forecast_horizon == 168 for r in todos), f"{ventana}: horizonte 168 h en todas")
    t1 = vt.construir_configs("T1", "S1")
    r = {c.modelo: resolve_run(c) for c in t1}
    check(r["xgboost"].train_hours == 336 and r["sarimax"].train_hours == 1440, "T1 conserva train_hours originales (336/1440)")
    check(resolve_run(vt.construir_configs("T2", "S1")[2]).train_hours == 2160, "T2 lstm_direct conserva 2160 h")
    modelos_t3 = {c.modelo for c in vt.construir_configs("T3", "S1")}
    check("naive" not in modelos_t3 and "ar" not in modelos_t3, "T3 no repite los baselines univariados")
    base = {c.modelo: resolve_run(c) for c in vt.construir_configs("T1", "S3")}
    check(all(base[m].train_hours == 3600 and base[m].forecast_horizon == 168 for m in ("naive", "naive_trend", "ar")),
          "Naive/Naive_Trend/AR en T1 con ventana fija 3600 h / 168 h")
    plan = vt.plan_dataframe()
    por_entorno = plan.groupby(["ventana", "ejecuta_en"]).size().unstack()
    check((por_entorno["local"] == 26).all() and (por_entorno["colab"] == 28).all(),
          f"reparto por semana: 26 corridas en la PC + 28 en Colab ({por_entorno.iloc[0].to_dict()})")
    cfgs = vt.construir_configs("T1", "S1") + vt.construir_configs("T2", "S1") + vt.construir_configs("T3", "S1")
    loc, col = vt.filtrar_por_entorno(cfgs, "local"), vt.filtrar_por_entorno(cfgs, "colab")
    nl = {resolve_run(c).run_name for c in loc}; nc = {resolve_run(c).run_name for c in col}
    check(not (nl & nc) and len(loc) + len(col) == 54, "ninguna corrida esta asignada a la PC y a Colab a la vez")
    check({c.modelo for c in col}.isdisjoint(vt.MODELOS_LOCALES), "Colab no recibe modelos locales")
    check(len(plan) == 216 and plan["run_name"].nunique() * 4 == 216 and plan.groupby("ventana").size().eq(54).all(),
          f"plan total = {len(plan)} corridas (54 por semana x 4)")

    print("\n2. Consolidado")
    tmp = tempfile.mkdtemp(prefix="test_vt_")
    try:
        base = os.path.join(tmp, "Ventanas")
        valores = {"S1": 4.0, "S2": 6.0, "S3": 8.0, "S4": 10.0}
        for v, m in valores.items():
            escribir_corrida(base, v, "Mod_A", "xgboost", ["Temperatura"], {r: m for r in REGIONS_ALL})
        escribir_corrida(base, "S1", "Mod_B", "fcnn", ["IGAE"], {r: 3.0 for r in REGIONS_ALL[:5]}, estrategias=("dir", "stl"))
        out = os.path.join(tmp, "Consolidado")
        res = vt.consolidar_ventanas(base, out)
        largo = res["largo"]
        check(len(largo) == 4 * 8 + 5 * 2, f"largo: {len(largo)} filas (32 + 10)")
        check(all(os.path.exists(os.path.join(out, f)) for f in [
            "metricas_ventanas_largo.csv", "resumen_modelo_region_entre_semanas.csv",
            "resumen_modelo_semana.csv", "resumen_modelo_global_entre_semanas.csv", "inventario_corridas_ventanas.csv"]),
            "se escriben los 5 CSV")
        g = res["global_"]
        a = g[g["modelo_config"] == "xgboost"].iloc[0]
        check(abs(a["MAPE_media"] - 7.0) < 1e-9, f"media entre semanas = {a['MAPE_media']} (esperado 7.0)")
        check(abs(a["MAPE_std"] - np.std([4, 6, 8, 10], ddof=1)) < 1e-9, f"std (ddof=1) = {a['MAPE_std']:.4f}")
        check(bool(a["completo_4_semanas_8_regiones"]), "Mod_A marcado completo (4 semanas x 8 regiones)")
        b = g[g["modelo_config"] == "fcnn"]
        check(len(b) == 2 and not b["completo_4_semanas_8_regiones"].any(), "Mod_B (1 semana, 5 regiones) marcado incompleto")
        pr = res["por_region"]
        x = pr[(pr["modelo_config"] == "xgboost") & (pr["region"] == "CEN")].iloc[0]
        check(x["MAPE_count"] == 4 and abs(x["MAPE_mean"] - 7.0) < 1e-9, "por region: 4 semanas, media 7.0")

        print("\n3. Comparacion S4 vs original")
        orig = os.path.join(tmp, "Pipeline_Resultados")
        cfg_x = [c for c in vt.construir_configs("T1", "S4") if c.modelo == "xgboost"][0]
        cfg_s = [c for c in vt.construir_configs("T1", "S4") if c.modelo == "sarimax"][0]
        for c, mape_nuevo_delta in [(cfg_x, 0.0), (cfg_s, 0.5)]:
            rn = resolve_run(c).run_name
            escribir_corrida(orig, "", rn, c.modelo, c.exogenas, {r: 5.0 for r in REGIONS_ALL})
            escribir_corrida(os.path.join(tmp, "V2"), "S4", rn, c.modelo, c.exogenas, {r: 5.0 + mape_nuevo_delta for r in REGIONS_ALL})
        # orig se escribio en orig//<rn> (ventana ""): equivale a orig/<rn>
        det, resu = vt.comparar_s4_con_original([cfg_x, cfg_s], orig, os.path.join(tmp, "V2"))
        sx = resu[resu["modelo"] == "xgboost"].iloc[0]
        ss = resu[resu["modelo"] == "sarimax"].iloc[0]
        check(bool(sx["reproduccion_satisfactoria"]), "xgboost identico -> reproduccion satisfactoria")
        check(not bool(ss["reproduccion_satisfactoria"]) and ss["difiere"] == 8, "sarimax con +0.5 pp -> DIFIERE en las 8 regiones")
        cfg_f = vt.construir_configs("T2", "S4")[0]
        check(vt.nombre_corrida_original("X", cfg_f.modelo) == "X_timefix", "fcnn se compara contra la corrida _timefix")
        check(vt.nombre_corrida_original("X", "sarimax") == "X", "sarimax se compara contra la corrida sin sufijo")
        cfg_n = [c for c in vt.construir_configs("T1", "S4") if c.modelo == "naive"][0]
        det_n, res_n = vt.comparar_s4_con_original([cfg_n], orig, os.path.join(tmp, "V2"))
        check((det_n["estado"] == "NO_COMPARABLE").all(), "naive: reproduccion marcada NO_COMPARABLE (original con ventana dinamica)")

        print("\n4. Auditoria de datos")
        datos = os.path.join(tmp, "datos")
        os.makedirs(datos)
        f = pd.date_range("2026-01-01", "2026-05-23 23:00", freq="h")
        # CEN completa; NES con un hueco de 5 h dentro de S2
        pd.DataFrame({"fecha": f.strftime("%Y-%m-%d"), "Hora": f.hour + 1, windows.COL_DEMANDA: 100.0}).to_csv(
            os.path.join(datos, "CEN_long.csv"), index=False)
        g2 = f[~((f >= "2026-05-05 10:00") & (f <= "2026-05-05 14:00"))]
        pd.DataFrame({"fecha": g2.strftime("%Y-%m-%d"), "Hora": g2.hour + 1, windows.COL_DEMANDA: 100.0}).to_csv(
            os.path.join(datos, "NES_long.csv"), index=False)
        dem, ex, gl = vt.auditar_datos_ventanas(datos, regiones=["CEN", "NES", "BCA"])
        c = dem[dem["region"] == "CEN"].set_index("ventana")
        n = dem[dem["region"] == "NES"].set_index("ventana")
        check(c["apta_para_ventana"].all(), "CEN apta en las 4 semanas")
        check(not n.loc["S2", "apta_para_ventana"] and n.loc["S1", "apta_para_ventana"], "NES: S2 no apta (hueco de 5 h), S1 si")
        check(int(n.loc["S1", "n_huecos"]) == 1 and int(n.loc["S1", "hueco_max_h"]) == 5, "NES: detecta 1 hueco de 5 h")
        check(dem[dem["region"] == "BCA"]["problema"].notna().all(), "BCA sin archivo: reportada")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODO OK" if not FALLOS else f"{len(FALLOS)} FALLO(S): {FALLOS}"))
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())

"""Deriva los archivos publicables de `results/` a partir de las carpetas de resultados de las 216 corridas (solo lectura).

No entrena ni modifica ninguna carpeta de origen. Lee, para cada una de las 216 corridas listadas en
`consolidado/metricas_largas.csv`, sus archivos `config.json`, `config_usada.csv` y `series.csv`, y escribe:

  results/configuracion_corridas.csv        configuracion de cada corrida (modelo, variables, entrenamiento, semilla, commit, fecha)
  results/hiperparametros_por_corrida.csv   hiperparametros registrados por la corrida (`config_usada.csv`), una fila por region y modelo
  results/predicciones_prueba.csv.gz        pronosticos de las semanas de prueba (168 h por region, modelo y corrida)
  results/valores_reales_prueba.csv.gz      (NO se publica en este repositorio; ver docs/DATOS.md)

Uso (desde la raiz del proyecto original):
  python preparar_resultados_publicos.py --origen resultados_locales --salida results
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def carpeta_corrida(origen, fila):
    if fila["origen"] == "PC":
        return os.path.join(origen, "Pipeline_Resultados_Ventanas", fila["ventana"], fila["run_name"])
    return os.path.join(origen, "distribuido", "staging", fila["origen"], fila["ventana"], fila["run_name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    larg = pd.read_csv(os.path.join(a.origen, "consolidado", "metricas_largas.csv"), encoding="utf-8-sig")
    corridas = larg[["ventana", "run_name", "origen"]].drop_duplicates().sort_values(["ventana", "run_name"]).reset_index(drop=True)
    print("corridas:", len(corridas))
    os.makedirs(a.salida, exist_ok=True)

    cfg_filas, hip_frames, pred_frames, sin_config = [], [], [], []
    reales = {}      # (ventana, region, fecha) -> valor
    reales_hash = {}  # comprobacion entre corridas
    discrepancias = 0
    for k, f in corridas.iterrows():
        d = carpeta_corrida(a.origen, f)
        cj = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        cj = {kk: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for kk, v in cj.items()}
        cj["origen"] = f["origen"]
        cfg_filas.append(cj)
        ruta_cu = os.path.join(d, "config_usada.csv")
        if os.path.exists(ruta_cu):
            cu = pd.read_csv(ruta_cu, encoding="utf-8-sig")
            cu.insert(0, "run_name", f["run_name"])
            cu.insert(0, "ventana", f["ventana"])
            hip_frames.append(cu)
        else:
            sin_config.append(f["run_name"])
        s = pd.read_csv(os.path.join(d, "series.csv"), encoding="utf-8-sig", usecols=["fecha", "tipo", "subset", "modelo", "valor", "region"])
        t0, t1 = pd.Timestamp(cj["test_start"]), pd.Timestamp(cj["test_end"])
        s["fecha"] = pd.to_datetime(s["fecha"])
        p = s[(s.tipo == "prediccion") & (s.subset == "test")][["region", "modelo", "fecha", "valor"]].copy()
        p.insert(0, "run_name", f["run_name"])
        p.insert(0, "ventana", f["ventana"])
        p = p.rename(columns={"modelo": "modelo_metrica", "valor": "prediccion"})
        pred_frames.append(p)
        r = s[(s.tipo == "real") & (s.fecha >= t0) & (s.fecha <= t1)]
        for reg, fe, v in zip(r.region, r.fecha, r.valor):
            clave = (f["ventana"], reg, fe)
            if clave in reales:
                if abs(reales[clave] - v) > 1e-9:
                    discrepancias += 1
            else:
                reales[clave] = v
        if (k + 1) % 20 == 0:
            print("  procesadas", k + 1, flush=True)

    pd.DataFrame(cfg_filas).to_csv(os.path.join(a.salida, "configuracion_corridas.csv"), index=False, encoding="utf-8")
    hip = pd.concat(hip_frames, ignore_index=True)
    hip.to_csv(os.path.join(a.salida, "hiperparametros_por_corrida.csv"), index=False, encoding="utf-8")
    pred = pd.concat(pred_frames, ignore_index=True)
    pred["fecha"] = pred["fecha"].dt.strftime("%Y-%m-%d %H:%M:%S")
    pred.to_csv(os.path.join(a.salida, "predicciones_prueba.csv.gz"), index=False, encoding="utf-8", compression="gzip")
    rl = pd.DataFrame([(v, reg, fe.strftime("%Y-%m-%d %H:%M:%S"), val) for (v, reg, fe), val in sorted(reales.items())], columns=["ventana", "region", "fecha", "real"])
    rl.to_csv(os.path.join(a.salida, "valores_reales_prueba.csv.gz"), index=False, encoding="utf-8", compression="gzip")
    print("corridas sin config_usada.csv (modelos sin hiperparametros):", len(set(sin_config)))
    print("predicciones:", len(pred), "| reales:", len(rl), "| discrepancias de valores reales entre corridas:", discrepancias)


if __name__ == "__main__":
    main()

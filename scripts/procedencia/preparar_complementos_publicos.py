"""Complementos publicables de `results/` (solo lectura; no entrena ni modifica carpetas de origen).

  results/trials_optuna.csv.gz                     intentos de la busqueda de hiperparametros de cada corrida (`trials.csv`)
  results/variacion_reentrenamiento_FCNN_S3_Temp.csv   comparacion de dos ejecuciones de la misma configuracion de la FCNN
                                                    (semana S3, variable temperatura): metricas antes y despues de recalcularla

Uso: python preparar_complementos_publicos.py --origen resultados_locales --salida results
"""
import argparse
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
    corridas = larg[["ventana", "run_name", "origen"]].drop_duplicates().sort_values(["ventana", "run_name"])
    marcos, sin = [], 0
    for _, f in corridas.iterrows():
        ruta = os.path.join(carpeta_corrida(a.origen, f), "trials.csv")
        if not os.path.exists(ruta):
            sin += 1
            continue
        t = pd.read_csv(ruta, encoding="utf-8-sig")
        if len(t) == 0:
            sin += 1
            continue
        t.insert(0, "run_name", f["run_name"])
        t.insert(0, "ventana", f["ventana"])
        marcos.append(t)
    tr = pd.concat(marcos, ignore_index=True)
    tr.to_csv(os.path.join(a.salida, "trials_optuna.csv.gz"), index=False, encoding="utf-8", compression="gzip")
    print("trials: corridas con intentos", len(marcos), "| sin intentos (modelos sin busqueda):", sin, "| filas", len(tr))

    d = os.path.join(a.origen, "distribuido")
    mn = pd.read_csv(os.path.join(d, "staging", "PRINCIPAL", "S3", "FCNN_train3600h_fh168h_Temp", "metricas.csv"), encoding="utf-8-sig")
    mo = pd.read_csv(os.path.join(d, "snapshots_previos", "S3_FCNN_Temp_antes_de_recalcular", "metricas.csv"), encoding="utf-8-sig")
    j = mn.merge(mo, on=["region", "modelo"], suffixes=("_recalculada", "_anterior"))
    j = j[j.region != "BCA"].copy()
    j["dif_abs_MAPE"] = (j["MAPE_recalculada"] - j["MAPE_anterior"]).abs()
    j = j[["region", "modelo", "MAPE_anterior", "MAPE_recalculada", "dif_abs_MAPE", "MAE_anterior", "MAE_recalculada"]]
    j.to_csv(os.path.join(a.salida, "variacion_reentrenamiento_FCNN_S3_Temp.csv"), index=False, encoding="utf-8")
    print("variacion: n=%d | mediana=%.2f | max=%.2f | >2pp: %d" % (len(j), j.dif_abs_MAPE.median(), j.dif_abs_MAPE.max(), int((j.dif_abs_MAPE > 2).sum())))


if __name__ == "__main__":
    main()

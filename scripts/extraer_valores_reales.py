"""Extrae de los archivos procesados por region (`{REGION}_long.csv`, ver docs/DATOS.md) los valores reales de demanda de las cuatro semanas de
prueba y los escribe en el formato que espera `verificar_resultados.py --reales`.

Los valores de CENACE no se distribuyen con este repositorio: quien los obtenga de la fuente y los organice como describe docs/DATOS.md puede usar este
script. La verificacion posterior compara el resultado con las huellas publicadas en docs/huellas_valores_reales.csv.

Regla: la demanda es la columna «Estimacion de Demanda por Balance (MWh)»; la hora `Hora` (1 a 24) del dia `fecha` corresponde a la marca
fecha + (Hora - 1) horas. Cada semana tiene 168 horas. Ademas se extrae, con la ventana «PREVIA», la ultima hora anterior a S1 (25 de abril de 2026, 23:00),
necesaria para reconstruir el pronostico del modelo Naive de S1 (el Naive repite la ultima hora observada).

Uso:  python scripts/extraer_valores_reales.py --data-dir datos_locales --salida valores_reales_prueba.csv.gz
"""
import argparse
import os
import sys

import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from tesis_forecast.windows import VENTANAS  # noqa: E402

REGIONES = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="datos_locales")
    ap.add_argument("--salida", default="valores_reales_prueba.csv.gz")
    a = ap.parse_args()
    filas = []
    for reg in REGIONES:
        df = pd.read_csv(os.path.join(a.data_dir, "%s_long.csv" % reg), encoding="utf-8-sig", usecols=["fecha", "Hora", COL_DEMANDA])
        df["fecha_hora"] = pd.to_datetime(df["fecha"]) + pd.to_timedelta(df["Hora"].astype(int) - 1, unit="h")
        for v, w in VENTANAS.items():
            g = df[(df["fecha_hora"] >= w.test_start) & (df["fecha_hora"] <= w.test_end)].sort_values("fecha_hora")
            if len(g) != 168 or g["fecha_hora"].duplicated().any():
                raise SystemExit("%s %s: se esperaban 168 horas distintas y hay %d" % (reg, v, len(g)))
            filas.append(pd.DataFrame({"ventana": v, "region": reg, "fecha": g["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S"), "real": g[COL_DEMANDA].astype(float).to_numpy()}))
        previa = df[df["fecha_hora"] == VENTANAS["S1"].test_start - pd.Timedelta(hours=1)]
        if len(previa) != 1:
            raise SystemExit("%s: no se encontro la hora previa a S1" % reg)
        filas.append(pd.DataFrame({"ventana": "PREVIA", "region": reg, "fecha": previa["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S"), "real": previa[COL_DEMANDA].astype(float).to_numpy()}))
    out = pd.concat(filas, ignore_index=True)
    out.to_csv(a.salida, index=False, compression="gzip")
    print("escrito %s (%d valores)" % (a.salida, len(out)))


if __name__ == "__main__":
    main()

"""Comportamiento de los errores a partir de las predicciones guardadas (solo lectura) de las corridas de referencia (D1).

Para cada modelo, region y semana recalcula, con `series.csv`:
  - MAPE por dia del horizonte (dia 1 = primeras 24 h del pronostico ... dia 7); las semanas de prueba empiezan en domingo,
    asi que el dia del horizonte coincide con el dia de la semana (dom..sab);
  - sesgo porcentual = 100 * mean(pronostico - real) / mean(real) (positivo = sobreestima);
  - MAPE por hora del dia.
Escribe resultados_locales/analisis_cap6/tablas/error_por_dia.csv, error_sesgo.csv y error_por_hora.csv.
No modifica ninguna carpeta de origen.
"""
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import tablas_cap6 as T  # noqa: E402

PC = os.path.join(RAIZ, "resultados_locales", "Pipeline_Resultados_Ventanas")
STAGING = os.path.join(RAIZ, "resultados_locales", "distribuido", "staging")
SAL = os.path.join(RAIZ, "resultados_locales", "analisis_cap6", "tablas")


def carpeta(origen, ventana, run):
    return os.path.join(PC if origen == "PC" else os.path.join(STAGING, origen), ventana, run)


def main():
    ref = pd.read_csv(os.path.join(SAL, "referencia_D1_con_posiciones.csv"))
    corridas = ref[["ventana", "run_name", "origen"]].drop_duplicates().sort_values(["ventana", "run_name"]).values.tolist()
    # Paso 1: leer predicciones de cada corrida y los valores reales que traiga (no todas las corridas guardan filas 'real').
    preds, reales, conflictos = [], {}, 0
    for n, (ventana, run, origen) in enumerate(corridas, 1):
        ruta = os.path.join(carpeta(origen, ventana, run), "series.csv")
        s = pd.read_csv(ruta, encoding="utf-8-sig", usecols=["fecha", "tipo", "modelo", "valor", "region"])
        modelos_ref = set(ref[(ref["ventana"] == ventana) & (ref["run_name"] == run)]["modelo_metrica"])
        pred = s[(s["tipo"] == "prediccion") & (s["modelo"].isin(modelos_ref))].copy()
        pred["ventana"] = ventana
        preds.append(pred)
        r = s[s["tipo"] == "real"].drop_duplicates(["region", "fecha"])
        if len(r):
            ventana_fechas = set(pred["fecha"])
            for reg, fec, val in zip(r["region"], r["fecha"], r["valor"]):
                if fec in ventana_fechas:
                    k = (ventana, reg, fec)
                    if k in reales and not np.isclose(reales[k], val, rtol=0, atol=1e-6):
                        conflictos += 1
                    reales[k] = float(val)
        print("  [%d/%d] %s %s" % (n, len(corridas), ventana, run), flush=True)
    print("valores reales reunidos:", len(reales), "| conflictos entre corridas (mismo dato, distinto valor):", conflictos)
    assert conflictos == 0
    pred = pd.concat(preds, ignore_index=True)
    por_dia, sesgo, por_hora = [], [], []
    for (ventana, reg, mm), g in pred.groupby(["ventana", "region", "modelo"]):
        g = g.sort_values("fecha")
        f = pd.to_datetime(g["fecha"])
        y = np.array([reales.get((ventana, reg, x), np.nan) for x in g["fecha"]], dtype=float)
        yh = g["valor"].to_numpy(dtype=float)
        assert len(y) == 168 and np.isfinite(y).all(), (ventana, reg, mm, len(y), int(np.isnan(y).sum()))
        h = ((f - f.min()).dt.total_seconds() / 3600).astype(int).to_numpy()
        ape = np.where(y != 0, 100 * np.abs(y - yh) / np.where(y != 0, np.abs(y), 1), np.nan)
        nombre = T.NOMBRE[mm]
        for d in range(7):
            m = (h // 24) == d
            por_dia.append({"modelo": nombre, "region": reg, "semana": ventana, "dia_horizonte": d + 1, "MAPE": np.nanmean(ape[m]),
                            "MAE": np.mean(np.abs(y[m] - yh[m]))})
        hora = f.dt.hour.to_numpy()
        for hh in range(24):
            m = hora == hh
            por_hora.append({"modelo": nombre, "region": reg, "semana": ventana, "hora": hh, "MAPE": np.nanmean(ape[m])})
        sesgo.append({"modelo": nombre, "region": reg, "semana": ventana, "sesgo_pct": 100 * np.mean(yh - y) / np.mean(y),
                      "MAPE_recalculado": np.nanmean(ape), "MAE_recalculado": np.mean(np.abs(y - yh))})
    # correspondencia con las metricas guardadas
    sb = pd.DataFrame(sesgo)
    j = sb.merge(ref[["modelo_nombre", "region", "ventana", "MAPE", "MAE"]], left_on=["modelo", "region", "semana"],
                 right_on=["modelo_nombre", "region", "ventana"])
    dif = np.maximum(np.abs(j["MAPE_recalculado"] - j["MAPE"]) / j["MAPE"], np.abs(j["MAE_recalculado"] - j["MAE"]) / j["MAE"])
    print("correspondencia metricas guardadas <-> recalculadas desde predicciones: %d de %d con diferencia relativa < 1e-3 | max %.2e" % (
        int((dif < 1e-3).sum()), len(j), dif.max()))
    pd.DataFrame(por_dia).to_csv(os.path.join(SAL, "error_por_dia.csv"), index=False)
    sb.to_csv(os.path.join(SAL, "error_sesgo.csv"), index=False)
    pd.DataFrame(por_hora).to_csv(os.path.join(SAL, "error_por_hora.csv"), index=False)
    print("listo:", len(sb), "combinaciones modelo-region-semana (se esperan 416)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

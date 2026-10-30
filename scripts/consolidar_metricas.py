"""Reune, en una tabla larga, las metricas reales de las 216 corridas (PC + copias de los Drive). Solo lectura.

No reentrena, no modifica carpetas de origen y no calcula rankings: solo junta lo que cada corrida ya escribio
en metricas.csv y dice cuales corridas estan completas (8 regiones, valores finitos) para poder armar las tablas
del Capitulo 6 cuando termine la sincronizacion.

Fuentes:
  - PC:     resultados_locales/Pipeline_Resultados_Ventanas/<ventana>/<run_name>/
  - Colab:  resultados_locales/distribuido/staging/<TRABAJADOR>/<ventana>/<run_name>/ (solo si su validacion JSON dice ok)
Salida:   resultados_locales/consolidado/metricas_largas.csv y cobertura.csv
Uso:      python scripts/consolidar_metricas.py
"""
import csv
import json
import math
import os
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLAN = os.path.join(RAIZ, "docs", "plan_corridas_ventanas.csv")
PC_DIR = os.path.join(RAIZ, "resultados_locales", "Pipeline_Resultados_Ventanas")
DIST = os.path.join(RAIZ, "resultados_locales", "distribuido")
SALIDA = os.path.join(RAIZ, "resultados_locales", "consolidado")
REGIONES = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]
METRICAS = ("MAE", "RMSE", "MAPE", "sMAPE")


def leer(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validada_colab(trab, ventana, run):
    ruta = os.path.join(DIST, "validacion", "%s__%s__%s.json" % (trab, ventana, run))
    if not os.path.exists(ruta):
        return False
    return bool(json.load(open(ruta, encoding="utf-8")).get("ok"))


def buscar(ventana, run, ejecuta_en):
    """Devuelve (origen, carpeta) de la fuente valida, o (None, motivo)."""
    if ejecuta_en.strip().lower() == "local":
        carpeta = os.path.join(PC_DIR, ventana, run)
        return ("PC", carpeta) if os.path.isdir(carpeta) else (None, "falta carpeta en la PC")
    candidatas = []
    for t in ("PRINCIPAL", "W2", "W3"):
        carpeta = os.path.join(DIST, "staging", t, ventana, run)
        if os.path.isdir(carpeta):
            candidatas.append((t, carpeta, validada_colab(t, ventana, run)))
    buenas = [c for c in candidatas if c[2]]
    if len(buenas) == 1:
        return buenas[0][0], buenas[0][1]
    if len(buenas) > 1:
        return None, "validada en mas de un trabajador: " + "+".join(c[0] for c in buenas)
    if candidatas:
        return None, "presente pero no validada en " + "+".join(c[0] for c in candidatas)
    return None, "no sincronizada"


def main():
    plan = leer(PLAN)
    largas, cobertura = [], []
    for r in plan:
        origen, carpeta = buscar(r["ventana"], r["run_name"], r["ejecuta_en"])
        fila = {"ventana": r["ventana"], "nivel": r["nivel"], "modelo": r["modelo"], "run_name": r["run_name"],
                "exogenas": r["exogenas"], "ejecuta_en": r["ejecuta_en"], "origen": origen or "", "estado": "", "motivo": ""}
        if origen is None:
            fila["estado"], fila["motivo"] = "sin_datos", carpeta
            cobertura.append(fila)
            continue
        ruta = os.path.join(carpeta, "metricas.csv")
        if not os.path.exists(ruta):
            fila["estado"], fila["motivo"] = "sin_datos", "sin metricas.csv"
            cobertura.append(fila)
            continue
        # Una corrida puede traer varios modelos por region (p. ej. la FCNN: directa y sobre residuo STL).
        regs = {}
        for m in leer(ruta):
            reg = (m.get("region") or m.get("serie", "")[:3]).strip()
            vals = {k: float(m[k]) for k in METRICAS if m.get(k) not in (None, "")}
            if reg and len(vals) == 4 and all(math.isfinite(v) for v in vals.values()):
                regs[(reg, (m.get("modelo") or "").strip())] = vals
        modelos_run = sorted({k[1] for k in regs})
        faltan = [x for x in REGIONES for mm in modelos_run if (x, mm) not in regs]
        fila["estado"] = "completa" if not faltan else "incompleta"
        fila["motivo"] = "" if not faltan else "faltan: " + ",".join(faltan)
        cobertura.append(fila)
        if not faltan:
            for reg in REGIONES:
                for mm in modelos_run:
                    largas.append({"ventana": r["ventana"], "nivel": r["nivel"], "modelo": r["modelo"], "modelo_metrica": mm,
                                   "run_name": r["run_name"], "exogenas": r["exogenas"], "origen": origen, "region": reg, **regs[(reg, mm)]})
    os.makedirs(SALIDA, exist_ok=True)
    with open(os.path.join(SALIDA, "metricas_largas.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ventana", "nivel", "modelo", "modelo_metrica", "run_name", "exogenas", "origen", "region", *METRICAS])
        w.writeheader()
        w.writerows(largas)
    with open(os.path.join(SALIDA, "cobertura.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cobertura[0].keys()))
        w.writeheader()
        w.writerows(cobertura)
    completas = [c for c in cobertura if c["estado"] == "completa"]
    print("Corridas del plan:", len(plan), "| completas con 8 regiones:", len(completas), "| filas de metricas:", len(largas))
    por_ventana = {}
    for c in cobertura:
        por_ventana.setdefault(c["ventana"], [0, 0])
        por_ventana[c["ventana"]][1] += 1
        por_ventana[c["ventana"]][0] += c["estado"] == "completa"
    for v, (a, b) in sorted(por_ventana.items()):
        print("  %s: %d de %d" % (v, a, b))
    motivos = {}
    for c in cobertura:
        if c["estado"] != "completa":
            motivos[c["motivo"]] = motivos.get(c["motivo"], 0) + 1
    for m, n in sorted(motivos.items(), key=lambda x: -x[1]):
        print("  sin datos (%d): %s" % (n, m))
    return 0


if __name__ == "__main__":
    sys.exit(main())

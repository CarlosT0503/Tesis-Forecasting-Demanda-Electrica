"""Auditoria de solo lectura: cruza el plan de 216 corridas con lo sincronizado de cada trabajador.

No entrena, no copia y no modifica nada. Lee:
  - docs/plan_corridas_ventanas.csv        (las 112 configuraciones de Colab = ejecuta_en distinto de 'local')
  - docs/asignacion/<TRABAJADOR>.csv       (quien debia correr cada una)
  - resultados_locales/distribuido/staging/<TRABAJADOR>/<ventana>/<run_name>/   (copia de cada Drive)
  - resultados_locales/distribuido/validacion/<TRABAJADOR>__<ventana>__<run_name>.json

Escribe resultados_locales/distribuido/auditoria/auditoria_112.csv y un resumen en pantalla.
Uso: python scripts/auditar_consolidacion.py
"""
import csv
import glob
import json
import os
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLAN = os.path.join(RAIZ, "docs", "plan_corridas_ventanas.csv")
ASIG = os.path.join(RAIZ, "docs", "asignacion")
DIST = os.path.join(RAIZ, "resultados_locales", "distribuido")
TRABAJADORES = ("PRINCIPAL", "W2", "W3")
REGIONES = ("BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN")


def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def asignaciones():
    quien = {}
    for t in TRABAJADORES:
        ruta = os.path.join(ASIG, t + ".csv")
        if os.path.exists(ruta):
            for r in leer_csv(ruta):
                quien[(r["ventana"], r["run_name"])] = t
    return quien


def estado_en(trab, ventana, run):
    """('ausente'|'validada'|'con_errores', n_errores, primer_error) para un trabajador."""
    carpeta = os.path.join(DIST, "staging", trab, ventana, run)
    if not os.path.isdir(carpeta):
        return "ausente", 0, "", False
    ruta = os.path.join(DIST, "validacion", "%s__%s__%s.json" % (trab, ventana, run))
    if not os.path.exists(ruta):
        return "sin_validar", 0, "", False
    v = json.load(open(ruta, encoding="utf-8"))
    errores = [h["mensaje"] for h in v.get("hallazgos", []) if h.get("nivel") == "ERROR"]
    solo_predicciones = bool(errores) and all(h.get("codigo") == "predicciones" for h in v.get("hallazgos", []) if h.get("nivel") == "ERROR")
    return ("validada" if v.get("ok") else "con_errores"), len(errores), (errores[0] if errores else ""), solo_predicciones


def metricas_con_8_regiones(trab, ventana, run):
    ruta = os.path.join(DIST, "staging", trab, ventana, run, "metricas.csv")
    if not os.path.exists(ruta):
        return False
    try:
        return len({m.get("region", "").strip() for m in leer_csv(ruta)} - {""}) == len(REGIONES)
    except (OSError, ValueError):
        return False


def main():
    plan = [r for r in leer_csv(PLAN) if r["ejecuta_en"].strip().lower() != "local"]
    quien = asignaciones()
    filas, resumen = [], {}
    for r in plan:
        clave = (r["ventana"], r["run_name"])
        enc = {t: estado_en(t, *clave) for t in TRABAJADORES}
        validadas = [t for t, e in enc.items() if e[0] == "validada"]
        presentes = [t for t, e in enc.items() if e[0] != "ausente"]
        # Comprobacion pedida por el autor: metricas de 8 regiones pero predicciones faltantes
        sin_pred = [t for t, e in enc.items() if e[0] == "con_errores" and e[3] and metricas_con_8_regiones(t, *clave)]
        if len(validadas) == 1:
            estado = "completa_validada"
        elif len(validadas) > 1:
            estado = "duplicada_validada"
        elif presentes:
            estado = "parcial_o_con_errores"
        else:
            estado = "no_encontrada"
        if sin_pred:
            estado = "metricas_completas_sin_predicciones"
        resumen[estado] = resumen.get(estado, 0) + 1
        err = next((e[2] for e in enc.values() if e[2]), "")
        filas.append({"ventana": r["ventana"], "nivel": r["nivel"], "modelo": r["modelo"], "run_name": r["run_name"],
                      "asignada_a": quien.get(clave, "(ya completa en el principal antes del reparto)"),
                      "validada_en": "+".join(validadas), "presente_en": "+".join(presentes), "estado": estado,
                      "primer_error": err})
    os.makedirs(os.path.join(DIST, "auditoria"), exist_ok=True)
    salida = os.path.join(DIST, "auditoria", "auditoria_112.csv")
    with open(salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print("Configuraciones de Colab en el plan:", len(plan))
    for k, v in sorted(resumen.items()):
        print("  %-24s %d" % (k, v))
    print("Detalle:", salida)
    for f in filas:
        if f["estado"] != "completa_validada":
            print("  !", f["estado"], f["ventana"], f["run_name"], "| asignada a:", f["asignada_a"], "|", f["primer_error"])
    # corridas de staging que no estan en el plan (por si hay basura o duplicados)
    plan_claves = {(r["ventana"], r["run_name"]) for r in plan}
    extra = []
    for t in TRABAJADORES:
        for d in glob.glob(os.path.join(DIST, "staging", t, "S*", "*")):
            if os.path.isdir(d):
                clave = (os.path.basename(os.path.dirname(d)), os.path.basename(d))
                if clave not in plan_claves:
                    extra.append((t,) + clave)
    print("Carpetas sincronizadas que no son de las 112:", len(extra))
    for e in extra[:15]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())

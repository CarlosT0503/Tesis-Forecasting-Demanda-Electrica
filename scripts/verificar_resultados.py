"""Ruta A: verifica, SIN entrenar modelos, que los resultados guardados en `results/` reproducen las metricas, las tablas y las figuras.

Script escrito para este repositorio. Reutiliza `tesis_forecast.metrics` (la misma funcion que calculo las metricas) y los scripts
originales `tablas_cap6.py`, `errores_cap6.py`, `extras_cap6.py` y `figuras_cap6.py`, que se ejecutan sin modificarlos en un
directorio temporal con la estructura que esperan.

Comprobaciones:
  1. Cobertura: 216 corridas y 1 952 conjuntos de metricas (region x semana x configuracion x modelo).
  2. Recalculo de las 1 952 metricas (MAE, RMSE, MAPE, sMAPE) desde `predicciones_prueba.csv.gz` y los valores reales de las semanas de prueba.
  3. Regeneracion de las tablas del Capitulo 6 desde las metricas y comparacion con `results/referencia_tesis/tablas/`.
  4. Regeneracion de las figuras y comparacion con `results/referencia_tesis/figuras/` (pixeles).

Los valores reales de demanda (CENACE) NO se distribuyen con este repositorio, y tampoco las predicciones del modelo Naive (son una constante: la
ultima hora observada, es decir, un dato de CENACE). Hay dos modos:
  - Verificacion PARCIAL (por omision, sin valores reales): hace 1, 3 y 4. No puede hacer 2. Las tres tablas de error que se calculan con los
    valores reales (error_por_dia, error_por_hora, error_sesgo) se toman de la referencia y no se regeneran; las tablas y figuras que dependen
    de ellas se regeneran a partir de esa referencia.
  - Verificacion COMPLETA (con `--reales archivo.csv.gz`): el archivo debe tener las columnas ventana, region, fecha y real, y sus valores deben
    coincidir con las huellas SHA-256 de `docs/huellas_valores_reales.csv`; si no coinciden, el script se detiene. Hace 1, 2, 3 y 4 completas; las predicciones del Naive se reconstruyen exactamente desde esos valores.

Uso:  python scripts/verificar_resultados.py [--results results] [--reales archivo] [--tolerancia 1e-6] [--sin-figuras]
Codigo de salida: 0 si todo lo comprobado coincide; 1 si alguna comprobacion falla. La verificacion parcial lo declara en la ultima linea.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from tesis_forecast.metrics import calcular_metricas  # noqa: E402

COLS_METRICAS = ["MAE", "RMSE", "MAPE", "sMAPE"]
SCRIPTS_CAP6 = ["tablas_cap6.py", "errores_cap6.py", "extras_cap6.py", "figuras_cap6.py"]


def leer(res, nombre, **kw):
    return pd.read_csv(os.path.join(res, nombre), encoding="utf-8-sig", **kw)


def comprobar_cobertura(larg):
    n_corridas = larg[["ventana", "run_name"]].drop_duplicates().shape[0]
    ok = n_corridas == 216 and len(larg) == 1952
    print("[1] Cobertura: %d corridas, %d filas de metricas (se esperan 216 y 1952) -> %s" % (n_corridas, len(larg), "OK" if ok else "FALLA"))
    return ok


TABLAS_CON_REALES = ["error_por_dia.csv", "error_sesgo.csv", "error_por_hora.csv"]


def huella_serie(fechas, valores):
    """SHA-256 de la serie en forma canonica: una linea 'fecha,valor' (5 decimales) por hora, en orden cronologico."""
    orden = sorted(range(len(fechas)), key=lambda i: fechas[i])
    texto = "\n".join("%s,%.5f" % (fechas[i], valores[i]) for i in orden)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def comprobar_huellas(real):
    """Compara los valores reales aportados con las huellas publicadas. Devuelve la lista de (ventana, region) que no coinciden."""
    ref = pd.read_csv(os.path.join(RAIZ, "docs", "huellas_valores_reales.csv"))
    malas = []
    for _, f in ref.iterrows():
        g = real[(real["ventana"] == f["ventana"]) & (real["region"] == f["region"])]
        fechas, vals = g["fecha"].astype(str).tolist(), g["real"].astype(float).tolist()
        if len(g) != f["n"] or huella_serie(fechas, vals) != f["sha256"]:
            malas.append((f["ventana"], f["region"]))
    print("[0] Valores reales aportados: %d de %d series (semana x region) coinciden con las huellas publicadas -> %s" % (
        len(ref) - len(malas), len(ref), "OK" if not malas else "FALLA"))
    return malas


SEMANA_ANTERIOR = {"S1": "PREVIA", "S2": "S1", "S3": "S2", "S4": "S3"}


def reconstruir_naive(larg, real):
    """El pronostico del modelo Naive es una constante: la ultima hora observada antes de la semana de prueba. Como repite datos de CENACE,
    no se publica; se reconstruye exactamente desde los valores reales aportados (la semana anterior, o la hora previa a S1)."""
    filas = []
    for v, run, reg in larg[larg["modelo_metrica"] == "Naive"][["ventana", "run_name", "region"]].drop_duplicates().itertuples(index=False):
        ultimo = float(real[(real["ventana"] == SEMANA_ANTERIOR[v]) & (real["region"] == reg)].sort_values("fecha")["real"].iloc[-1])
        for f in real[(real["ventana"] == v) & (real["region"] == reg)]["fecha"]:
            filas.append((v, run, reg, "Naive", f, ultimo))
    return pd.DataFrame(filas, columns=["ventana", "run_name", "region", "modelo_metrica", "fecha", "prediccion"])


def recalcular_metricas(res, larg, real, tol):
    pred = leer(res, "predicciones_prueba.csv.gz")
    pred = pd.concat([pred, reconstruir_naive(larg, real)], ignore_index=True)
    real_d = {(v, r, f): y for v, r, f, y in zip(real["ventana"], real["region"], real["fecha"], real["real"])}
    guardadas = larg.set_index(["ventana", "run_name", "modelo_metrica", "region"])
    peor, n, malas = 0.0, 0, []
    for (v, run, mm, reg), g in pred.groupby(["ventana", "run_name", "modelo_metrica", "region"], sort=False):
        g = g.sort_values("fecha")
        y = np.array([real_d[(v, reg, f)] for f in g["fecha"]], dtype=float)
        m = calcular_metricas(y, g["prediccion"].to_numpy(dtype=float))
        fila = guardadas.loc[(v, run, mm, reg)]
        for c in COLS_METRICAS:
            dif = abs(m[c] - fila[c]) / max(abs(fila[c]), 1e-12)
            peor = max(peor, dif)
            if dif > tol:
                malas.append((v, run, mm, reg, c, m[c], fila[c]))
        n += 1
    ok = n == len(larg) and not malas
    print("[2] Metricas recalculadas desde las predicciones: %d conjuntos | diferencia relativa maxima %.2e | tolerancia %.0e -> %s" % (n, peor, tol, "OK" if ok else "FALLA"))
    for x in malas[:5]:
        print("    difiere:", x)
    return ok, pred


def montar_espacio(tmp, res, larg, pred, real):
    """Reconstruye, en un directorio temporal, la estructura de carpetas que esperan los scripts originales."""
    os.makedirs(os.path.join(tmp, "scripts"))
    for s in SCRIPTS_CAP6:
        shutil.copy(os.path.join(RAIZ, "scripts", s), os.path.join(tmp, "scripts", s))
    cons = os.path.join(tmp, "resultados_locales", "consolidado")
    os.makedirs(cons)
    larg.to_csv(os.path.join(cons, "metricas_largas.csv"), index=False, encoding="utf-8")
    os.makedirs(os.path.join(tmp, "resultados_locales", "analisis_cap6", "tablas"))
    os.makedirs(os.path.join(tmp, "resultados_locales", "analisis_cap6", "figuras"))
    origen = larg[["ventana", "run_name", "origen"]].drop_duplicates()
    if real is None:   # modo parcial: las tablas que necesitan los valores reales se toman de la referencia
        for t in TABLAS_CON_REALES:
            shutil.copy(os.path.join(res, "referencia_tesis", "tablas", t), os.path.join(tmp, "resultados_locales", "analisis_cap6", "tablas", t))
        real_df = pd.DataFrame(columns=["ventana", "region", "fecha", "valor", "tipo", "modelo"])
    else:
        real_df = real.rename(columns={"real": "valor"})
        real_df["tipo"] = "real"
        real_df["modelo"] = "real"
    for _, f in origen.iterrows():
        if f["origen"] == "PC":
            ruta = os.path.join(tmp, "resultados_locales", "Pipeline_Resultados_Ventanas", f["ventana"], f["run_name"])
        else:
            ruta = os.path.join(tmp, "resultados_locales", "distribuido", "staging", f["origen"], f["ventana"], f["run_name"])
        os.makedirs(ruta)
        p = pred[(pred.ventana == f["ventana"]) & (pred.run_name == f["run_name"])].rename(columns={"prediccion": "valor", "modelo_metrica": "modelo"})
        p = p.assign(tipo="prediccion")[["fecha", "tipo", "modelo", "valor", "region"]]
        r = real_df[real_df.ventana == f["ventana"]][["fecha", "tipo", "modelo", "valor", "region"]]
        partes = [x for x in (p, r) if len(x)]
        (pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["fecha", "tipo", "modelo", "valor", "region"])).to_csv(
            os.path.join(ruta, "series.csv"), index=False, encoding="utf-8-sig")


def ejecutar(tmp, script):
    r = subprocess.run([sys.executable, os.path.join("scripts", script)], cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg"))
    if r.returncode != 0:
        print("    %s fallo (codigo %d):\n%s" % (script, r.returncode, (r.stderr or r.stdout)[-800:]))
    return r.returncode == 0


def comparar_tablas(tmp, res, tol):
    gen = os.path.join(tmp, "resultados_locales", "analisis_cap6", "tablas")
    ref = os.path.join(res, "referencia_tesis", "tablas")
    difieren, faltan = [], []
    for nombre in sorted(os.listdir(ref)):
        if not os.path.exists(os.path.join(gen, nombre)):
            faltan.append(nombre)
            continue
        a = pd.read_csv(os.path.join(gen, nombre))
        b = pd.read_csv(os.path.join(ref, nombre))
        ok = a.shape == b.shape and list(a.columns) == list(b.columns)
        if ok:
            for c in a.columns:
                if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                    ok = ok and np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=tol, atol=tol, equal_nan=True)
                else:
                    ok = ok and a[c].fillna("").astype(str).tolist() == b[c].fillna("").astype(str).tolist()
        if not ok:
            difieren.append(nombre)
    n_ok = len(os.listdir(ref)) - len(difieren) - len(faltan)
    print("[3] Tablas del Capitulo 6: %d de %d coinciden con la referencia%s%s" % (n_ok, len(os.listdir(ref)), " | difieren: %s" % difieren if difieren else "",
                                                                          " | no generadas: %s" % faltan if faltan else ""))
    return not difieren and not faltan


def version_matplotlib_png(ruta):
    """Version de matplotlib registrada en los metadatos (campo Software) de un PNG; None si no consta."""
    import re
    m = re.search(rb"Matplotlib version([0-9][0-9.]*)", open(ruta, "rb").read())
    return m.group(1).decode() if m else None


def comparar_figuras(tmp, res):
    import matplotlib.image as mpimg
    gen = os.path.join(tmp, "resultados_locales", "analisis_cap6", "figuras")
    ref = os.path.join(res, "referencia_tesis", "figuras")
    ok_n, malas, omitidas = 0, [], []
    for nombre in sorted(os.listdir(ref)):
        ruta = os.path.join(gen, nombre)
        if not os.path.exists(ruta):
            malas.append(nombre + " (no generada)")
            continue
        v_ref, v_gen = version_matplotlib_png(os.path.join(ref, nombre)), version_matplotlib_png(ruta)
        if v_ref != v_gen:   # el aspecto de una figura depende de la version de matplotlib: no es comparable con otra version
            omitidas.append("%s (referencia con matplotlib %s; aqui %s)" % (nombre, v_ref, v_gen))
            continue
        a, b = mpimg.imread(ruta), mpimg.imread(os.path.join(ref, nombre))
        if a.shape == b.shape and np.array_equal(a, b):
            ok_n += 1
        else:
            malas.append(nombre)
    print("[4] Figuras: %d identicas pixel a pixel | %d no comparables por la version de matplotlib | %d con diferencias" % (ok_n, len(omitidas), len(malas)))
    for o in omitidas:
        print("    no comparable:", o)
    for m in malas:
        print("    difiere:", m)
    return not malas and ok_n > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(RAIZ, "results"))
    ap.add_argument("--reales", default=None, help="archivo con los valores reales de las semanas de prueba (ventana, region, fecha, real); "
                    "por omision se busca results/valores_reales_prueba.csv.gz y, si no existe, la verificacion es parcial")
    ap.add_argument("--tolerancia", type=float, default=1e-6)
    ap.add_argument("--sin-figuras", action="store_true", help="omite la regeneracion y comparacion de figuras")
    a = ap.parse_args()
    ruta_real = a.reales or os.path.join(a.results, "valores_reales_prueba.csv.gz")
    if a.reales and not os.path.isfile(ruta_real):
        raise SystemExit("No existe el archivo indicado en --reales: %s" % ruta_real)
    real = pd.read_csv(ruta_real, encoding="utf-8-sig") if os.path.isfile(ruta_real) else None
    larg = leer(os.path.join(a.results, "consolidado"), "metricas_largas.csv")
    resultados = [comprobar_cobertura(larg)]
    pred = leer(a.results, "predicciones_prueba.csv.gz")
    if real is None:
        print("[0] Sin valores reales (no se distribuyen con el repositorio): VERIFICACION PARCIAL. Ver docs/DATOS.md para obtenerlos y usar --reales.")
        print("[2] Recalculo de las metricas desde las predicciones: NO REALIZADO (necesita los valores reales)")
    else:
        if comprobar_huellas(real):
            print("\nRESULTADO: LOS VALORES REALES APORTADOS NO SON LOS DEL ESTUDIO; no se continua.")
            return 1
        ok2, pred = recalcular_metricas(a.results, larg, real, a.tolerancia)
        resultados.append(ok2)
    tmp = tempfile.mkdtemp(prefix="verificacion_")
    try:
        montar_espacio(tmp, a.results, larg, pred, real)
        pasos = ["tablas_cap6.py"] + (["errores_cap6.py"] if real is not None else []) + ["extras_cap6.py"] + ([] if a.sin_figuras else ["figuras_cap6.py"])
        ok_pasos = all(ejecutar(tmp, s) for s in pasos)
        resultados.append(ok_pasos)
        if ok_pasos:
            resultados.append(comparar_tablas(tmp, a.results, a.tolerancia))
            if real is None:
                print("    (de esas tablas, %s se tomaron de la referencia, no se regeneraron: su coincidencia es trivial)" % ", ".join(TABLAS_CON_REALES))
            if not a.sin_figuras:
                resultados.append(comparar_figuras(tmp, a.results))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(resultados)
    if not ok:
        print("\nRESULTADO: HAY DIFERENCIAS (ver arriba)")
    elif real is None:
        print("\nRESULTADO: LO COMPROBADO COINCIDE. Verificacion PARCIAL: sin valores reales no se recalcularon las metricas ni se regeneraron %s." % ", ".join(TABLAS_CON_REALES))
    else:
        print("\nRESULTADO: TODO COINCIDE (verificacion completa)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

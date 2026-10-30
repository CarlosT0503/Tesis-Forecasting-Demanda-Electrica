"""Cifras adicionales para la redaccion de los Caps. 6 y 7 (solo estadistica descriptiva, sin pruebas de significancia).

Lee resultados_locales/consolidado/metricas_largas.csv y resultados_locales/analisis_cap6/tablas/ y escribe en la
misma carpeta de tablas: cobertura_T6_1.csv, pares_modelos.csv, mejor_por_region.csv, sesgo_por_modelo.csv.
"""
import csv
import os
import statistics as st
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
TABLAS = os.path.join(RAIZ, "resultados_locales", "analisis_cap6", "tablas")
LARGAS = os.path.join(RAIZ, "resultados_locales", "consolidado", "metricas_largas.csv")
REF = os.path.join(TABLAS, "referencia_D1_con_posiciones.csv")
SESGO = os.path.join(TABLAS, "error_sesgo.csv")

SEMANAS = ["S1", "S2", "S3", "S4"]
REGIONES = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]


def leer(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def escribir(nombre, filas):
    with open(os.path.join(TABLAS, nombre), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def main():
    largas = leer(LARGAS)

    # Cobertura: por modelo, configuraciones distintas y filas (configuracion x region) por semana
    cfg = defaultdict(set)
    cuenta = defaultdict(int)
    for r in largas:
        cfg[r["modelo_metrica"]].add(r["exogenas"])
        cuenta[(r["modelo_metrica"], r["ventana"])] += 1
    filas = []
    for m in sorted(cfg):
        fila = {"modelo": m, "configuraciones": len(cfg[m])}
        for s in SEMANAS:
            fila[s] = cuenta[(m, s)]
        fila["total"] = sum(cuenta[(m, s)] for s in SEMANAS)
        filas.append(fila)
    escribir("cobertura_T6_1.csv", filas)

    # Referencia (D1): 416 observaciones
    ref = leer(REF)
    obs = {(r["modelo_nombre"], r["region"], r["ventana"]): float(r["MAPE"]) for r in ref}

    def pares(a, b):
        d = [obs[(a, rg, s)] - obs[(b, rg, s)] for rg in REGIONES for s in SEMANAS]
        return {"A": a, "B": b, "n": len(d), "A_mejor": sum(1 for x in d if x < 0),
                "A_peor": sum(1 for x in d if x > 0), "mediana_dif_pp": round(st.median(d), 2), "media_dif_pp": round(st.mean(d), 2)}

    comparaciones = [
        ("Naive + tendencia + estacionalidad", "SARIMAX"),
        ("Naive + tendencia + estacionalidad", "AR"),
        ("Naive + tendencia + estacionalidad", "Naive + tendencia"),
        ("Naive + tendencia + estacionalidad", "Naive"),
        ("Naive + tendencia + estacionalidad", "AR sobre residuo STL"),
        ("Naive + tendencia + estacionalidad", "XGBoost"),
        ("Naive + tendencia + estacionalidad", "LightGBM"),
        ("SARIMAX", "AR"),
        ("LSTM directa", "LSTM sobre residuo STL"),
        ("LSTM directa", "FCNN directa"),
        ("LSTM directa", "Naive + tendencia + estacionalidad"),
        ("LSTM directa", "Naive"),
        ("LSTM sobre residuo STL", "Ensemble STL"),
    ]
    salida = [pares(a, b) for a, b in comparaciones]
    escribir("pares_modelos.csv", salida)

    # Mejor modelo por region segun la media de las 4 semanas, y cuantos ganadores distintos hay por region
    modelos = sorted({r["modelo_nombre"] for r in ref})
    mejor = []
    for rg in REGIONES:
        medias = {m: st.mean(obs[(m, rg, s)] for s in SEMANAS) for m in modelos}
        orden = sorted(medias, key=medias.get)
        ganadores = defaultdict(int)
        for s in SEMANAS:
            g = min(modelos, key=lambda m: obs[(m, rg, s)])
            ganadores[g] += 1
        mejor.append({"region": rg, "primero_4_semanas": orden[0], "MAPE_1": round(medias[orden[0]], 2),
                      "segundo_4_semanas": orden[1], "MAPE_2": round(medias[orden[1]], 2),
                      "ganadores_distintos_en_4_semanas": len(ganadores),
                      "peor": orden[-1], "MAPE_peor": round(medias[orden[-1]], 2)})
    escribir("mejor_por_region.csv", mejor)

    # Sesgo: mediana por modelo
    sesgo = defaultdict(list)
    for r in leer(SESGO):
        sesgo[r["modelo"]].append(float(r["sesgo_pct"]))
    filas = [{"modelo": m, "sesgo_mediano_pct": round(st.median(v), 2), "sesgo_medio_pct": round(st.mean(v), 2),
              "sub_estima_en": sum(1 for x in v if x < 0), "de": len(v)} for m, v in sorted(sesgo.items())]
    escribir("sesgo_por_modelo.csv", filas)
    # Mediana del MAPE diario entre los 13 modelos y las 8 regiones, por semana y dia del horizonte
    por_dia = defaultdict(list)
    for r in leer(os.path.join(TABLAS, "error_por_dia.csv")):
        por_dia[(r["semana"], int(r["dia_horizonte"]))].append(float(r["MAPE"]))
    escribir("error_por_dia_mediana.csv",
             [dict([("semana", s)] + [("dia%d" % d, round(st.median(por_dia[(s, d)]), 2)) for d in range(1, 8)]) for s in SEMANAS])

    figura_referencia_estadisticos()
    print("OK: cobertura_T6_1.csv, pares_modelos.csv, mejor_por_region.csv, sesgo_por_modelo.csv, "
          "error_por_dia_mediana.csv, F6_11_referencia_y_estadisticos.png")


def figura_referencia_estadisticos():
    """F6.11: MAPE medio regional por semana de los modelos de referencia y estadisticos (apoya R20 y R21)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = leer(REF)
    modelos = ["Naive + tendencia", "Naive + tendencia + estacionalidad", "AR", "SARIMAX"]
    colores = ["#7f7f7f", "#4d4d4d", "#9ecae1", "#1f77b4"]
    ancho = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (m, c) in enumerate(zip(modelos, colores)):
        valores = []
        for s in SEMANAS:
            v = [float(r["MAPE"]) for r in ref if r["modelo_nombre"] == m and r["ventana"] == s]
            valores.append(st.mean(v))
        pos = [k + (i - 1.5) * ancho for k in range(4)]
        ax.bar(pos, valores, ancho, label=m, color=c)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SEMANAS)
    ax.set_ylabel("MAPE medio entre las 8 regiones (%)")
    ax.set_title("Modelos de referencia y estadísticos por semana de prueba")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RAIZ, "resultados_locales", "analisis_cap6", "figuras", "F6_11_referencia_y_estadisticos.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

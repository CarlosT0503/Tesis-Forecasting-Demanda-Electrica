"""Figuras descriptivas del analisis de resultados (Capitulo 6). Solo lee los CSV de resultados_locales/analisis_cap6/tablas.

Salida: resultados_locales/analisis_cap6/figuras/*.png. No entrena ni modifica resultados.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import tablas_cap6 as T  # noqa: E402

TAB = os.path.join(RAIZ, "resultados_locales", "analisis_cap6", "tablas")
FIG = os.path.join(RAIZ, "resultados_locales", "analisis_cap6", "figuras")
COL = {"Referencia": "#7f7f7f", "Estadístico": "#1f77b4", "Aprendizaje automático": "#2ca02c",
       "Aprendizaje profundo": "#d62728", "Híbrido con STL": "#9467bd"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130, "savefig.bbox": "tight"})


def leer(n):
    return pd.read_csv(os.path.join(TAB, n + ".csv"))


def guardar(fig, nombre):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, nombre + ".png"))
    plt.close(fig)


def leyenda_familias(ax, loc="best"):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[f], markersize=8, label=f) for f in T.ORDEN_FAMILIAS],
              loc=loc, fontsize=8, title="Familia", title_fontsize=8)


def f1_global():
    g = leer("global_T6_2").sort_values("MAPE_media", ascending=False)
    ref = leer("referencia_D1_con_posiciones")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(g.itertuples()):
        ax.hlines(i, r.semana_min, r.semana_max, color=COL[r.familia], lw=3, alpha=0.5)
        vals = ref[ref.modelo_nombre == r.modelo]["MAPE"]
        ax.scatter(vals, [i] * len(vals), s=6, color=COL[r.familia], alpha=0.35, zorder=2)
        ax.scatter([r.MAPE_media], [i], marker="D", s=40, color=COL[r.familia], edgecolor="k", zorder=3)
        ax.scatter([r.MAPE_mediana], [i], marker="|", s=120, color="k", zorder=4)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["modelo"])
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("MAPE (%), escala logarítmica")
    ax.set_title("MAPE por modelo (configuración de referencia): 32 observaciones región–semana\n◆ media · | mediana · barra = rango de la media regional entre semanas")
    leyenda_familias(ax, "lower right")
    guardar(fig, "F6_1_MAPE_por_modelo")


def f2_mapa_calor():
    m = leer("matriz_region_modelo_media").set_index("region")
    orden = leer("global_T6_2")["modelo"].tolist()
    m = m[orden]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(m.to_numpy(), cmap="YlOrRd", aspect="auto", vmax=20)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels(orden, rotation=40, ha="right")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, "%.1f" % m.iat[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="MAPE medio de las 4 semanas (%) (escala truncada en 20)")
    ax.set_title("MAPE medio por región y modelo (4 semanas), modelos ordenados por MAPE medio global")
    guardar(fig, "F6_2_mapa_calor_region_modelo")


def f3_semanas():
    t = leer("por_semana_T6_4")
    g = leer("global_T6_2")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r in g.itertuples():
        s = t[t.modelo == r.modelo].sort_values("semana")
        ax.plot(s["semana"], s["MAPE_media_regional"], marker="o", color=COL[r.familia], alpha=0.8, lw=1.4)
        ax.annotate(r.modelo, (3, s["MAPE_media_regional"].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, va="center")
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylabel("MAPE medio entre las 8 regiones (%)")
    ax.set_title("MAPE medio regional por semana de prueba")
    leyenda_familias(ax, "upper left")
    guardar(fig, "F6_3_MAPE_por_semana")


def f4_posiciones():
    e = leer("estabilidad_T6_6")
    cols = ["pos_S1", "pos_S2", "pos_S3", "pos_S4"]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(e[cols].to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["S1", "S2", "S3", "S4"])
    ax.set_yticks(range(len(e)))
    ax.set_yticklabels(e["modelo"])
    ax.grid(False)
    for i in range(len(e)):
        for j in range(4):
            ax.text(j, i, int(e[cols].iat[i, j]), ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Posición del modelo en cada semana\n(por su MAPE medio regional; modelos ordenados por MAPE medio de las 4 semanas)", fontsize=8.5)
    guardar(fig, "F6_4_posiciones_por_semana")


def f5_exogenas():
    e = leer("exogenas_T6_5")
    modelos = [m for m in leer("global_T6_2")["modelo"] if m in set(e["modelo"])]
    configs = ["Temperatura", "IGAE", "Temperatura+IGAE", "Generacion", "Importacion", "Exportacion"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ancho = 0.12
    for k, c in enumerate(configs):
        v = [e[(e.modelo == m) & (e.configuracion == c)]["dif_MAPE_mediana"].iloc[0] for m in modelos]
        ax.bar(np.arange(len(modelos)) + (k - 2.5) * ancho, v, ancho, label=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(modelos)))
    ax.set_xticklabels(modelos, rotation=25, ha="right")
    ax.set_ylabel("Mediana de la diferencia de MAPE\n(configuración − referencia), pp")
    ax.set_ylim(-1.0, 2.0)
    ax.set_title("Efecto de usar menos variables exógenas que la configuración de referencia\n(mediana de 32 diferencias; positivo = peor que la referencia)")
    ax.legend(ncol=3, fontsize=7)
    guardar(fig, "F6_5_efecto_exogenas")


def f6_stl():
    ref = pd.read_csv(os.path.join(TAB, "referencia_D1_con_posiciones.csv"))
    piv = ref.pivot_table(index=["region", "ventana"], columns="modelo_nombre", values="MAPE")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rng = np.random.default_rng(0)
    for i, (et, d, s) in enumerate(T.PARES_STL):
        dif = (piv[s] - piv[d]).to_numpy()
        ax.scatter(dif, i + rng.uniform(-0.15, 0.15, len(dif)), s=14, alpha=0.6, color="#9467bd")
        ax.scatter([np.median(dif)], [i], marker="|", s=300, color="k", zorder=4)
        ax.scatter([np.mean(dif)], [i], marker="D", s=40, color="orange", edgecolor="k", zorder=5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(T.PARES_STL)))
    ax.set_yticklabels([p[0].replace(": ", ":\n") for p in T.PARES_STL], fontsize=7.5)
    ax.set_xlim(-25, 25)
    ax.set_xlabel("MAPE(variante con STL) − MAPE(variante directa), pp  (negativo = STL mejor)")
    ax.set_title("Estrategias con STL frente al modelado directo: 32 pares región–semana\n◆ media · | mediana (eje truncado en ±25 pp)")
    guardar(fig, "F6_6_STL_frente_a_directo")


def f7_dia():
    d = leer("error_por_dia")
    g = leer("global_T6_2")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r in g.itertuples():
        s = d[d.modelo == r.modelo].groupby("dia_horizonte")["MAPE"].median()
        ax.plot(s.index, s.values, marker="o", color=COL[r.familia], lw=1.3, alpha=0.85)
        ax.annotate(r.modelo, (7, s.values[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, va="center")
    ax.set_xlim(0.8, 9.2)
    ax.set_xlabel("Día del horizonte (1 = domingo … 7 = sábado)")
    ax.set_ylabel("Mediana del MAPE diario (%) entre 32 región–semana")
    ax.set_title("Error según el día del horizonte de pronóstico")
    leyenda_familias(ax, "upper left")
    guardar(fig, "F6_7_error_por_dia_del_horizonte")


def f8_sesgo():
    s = leer("error_sesgo")
    orden = leer("global_T6_2")["modelo"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(orden):
        v = s[s.modelo == m]["sesgo_pct"].to_numpy()
        ax.boxplot(v, positions=[i], widths=0.55, showfliers=True, flierprops={"markersize": 3},
                   boxprops={"color": COL[T.FAMILIA[m]]}, medianprops={"color": "k"})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels(orden, rotation=40, ha="right")
    ax.set_ylabel("Sesgo (%) = 100·media(pronóstico − real)/media(real)")
    ax.set_ylim(-60, 60)
    ax.set_title("Sesgo por modelo en las 32 combinaciones región–semana (positivo = sobreestima; eje truncado en ±60 %)")
    guardar(fig, "F6_8_sesgo")


def f9_regiones():
    d = leer("dificultad_region")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, s in enumerate(T.SEMANAS):
        v = d[d.ventana == s].set_index("region").loc[T.REGIONES]
        ax.bar(np.arange(8) + (k - 1.5) * 0.2, v["mediana_entre_modelos"], 0.2, label=s)
    ax.set_xticks(range(8))
    ax.set_xticklabels(T.REGIONES)
    ax.set_ylabel("Mediana del MAPE entre los 13 modelos (%)")
    ax.set_title("Dificultad de cada región por semana (mediana entre modelos)")
    ax.legend(ncol=4, fontsize=8)
    guardar(fig, "F6_9_dificultad_por_region")


def f10_colas():
    ref = pd.read_csv(os.path.join(TAB, "referencia_D1_con_posiciones.csv"))
    orden = leer("global_T6_2")["modelo"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(orden):
        v = ref[ref.modelo_nombre == m]["MAPE"].to_numpy()
        ax.boxplot(v, positions=[i], widths=0.55, boxprops={"color": COL[T.FAMILIA[m]]}, medianprops={"color": "k"}, flierprops={"markersize": 3})
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 50, 100])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels(orden, rotation=40, ha="right")
    ax.set_ylabel("MAPE (%), escala logarítmica")
    ax.set_title("Distribución del MAPE en las 32 combinaciones región–semana (colas)")
    guardar(fig, "F6_10_colas_MAPE")


def main():
    for f in (f1_global, f2_mapa_calor, f3_semanas, f4_posiciones, f5_exogenas, f6_stl, f7_dia, f8_sesgo, f9_regiones, f10_colas):
        f()
        print("ok", f.__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())

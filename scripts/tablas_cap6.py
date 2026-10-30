"""Tablas descriptivas del analisis de resultados (Capitulo 6) segun las decisiones D1-D6 del autor.

Solo lectura de `resultados_locales/consolidado/metricas_largas.csv` (216 corridas validadas); no entrena nada y no
modifica ningun resultado. Decisiones aplicadas:
  D1  ranking principal con la configuracion comun de cinco variables exogenas (ocho en XGBoost y LightGBM; sin exogenas en
      los cinco modelos de referencia); las demas configuraciones solo para el efecto de las exogenas y anexos.
  D2  empate si el MAPE coincide al redondear a dos decimales; desempata el MAE (tambien a dos decimales); si tambien coincide,
      posicion compartida (ranking de competencia: 1, 2, 2, 4).
  D3  agregacion entre regiones con media aritmetica simple Y mediana; nunca una sola medida sin explicar que representa.
  D4  configuraciones con resultados identicos = equivalentes (no se interpretan como ausencia de efecto).
  D6  ranking principal de 13 modelos con lectura por familias.
No se hacen pruebas de significancia: solo estadisticos descriptivos (el autor lo definio asi).
"""
import json
import math
import os
import statistics
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LARGAS = os.path.join(RAIZ, "resultados_locales", "consolidado", "metricas_largas.csv")
SALIDA = os.path.join(RAIZ, "resultados_locales", "analisis_cap6", "tablas")
REGIONES = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]
SEMANAS = ["S1", "S2", "S3", "S4"]
DEC = 2  # D2: precision con la que se decide el empate

CINCO = "Temperatura+IGAE+Generacion+Importacion+Exportacion"
OCHO = "Temperatura+Primarias+Secundarias+Terciarias+IGAE+Generacion+Importacion+Exportacion"
NINGUNA = "(ninguna)"

# modelo_metrica -> (nombre, familia, configuracion de referencia segun D1)
MODELOS = {
    "Naive": ("Naive", "Referencia", NINGUNA),
    "Naive_Trend": ("Naive + tendencia", "Referencia", NINGUNA),
    "Naive_Trend_Seasonal": ("Naive + tendencia + estacionalidad", "Referencia", NINGUNA),
    "AR": ("AR", "Estadístico", NINGUNA),
    "SARIMAX_1_1_1__1_0_1_168_EXOG_2M": ("SARIMAX", "Estadístico", CINCO),
    "XGBoost_Tuned_2Semanas": ("XGBoost", "Aprendizaje automático", OCHO),
    "LightGBM_Tuned": ("LightGBM", "Aprendizaje automático", OCHO),
    "FCNN_Multivariada_EXOG_Lag168": ("FCNN directa", "Aprendizaje profundo", CINCO),
    "LSTM_Directa": ("LSTM directa", "Aprendizaje profundo", CINCO),
    "AR_Resid_Trend_Seasonal": ("AR sobre residuo STL", "Híbrido con STL", NINGUNA),
    "STL_FCNN_Multivariada_Residuos_EXOG_Lag168": ("FCNN sobre residuo STL", "Híbrido con STL", CINCO),
    "LSTM_Resid_Trend_Seasonal": ("LSTM sobre residuo STL", "Híbrido con STL", CINCO),
    "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid_EXOG_ALL_Lag168": ("Ensemble STL", "Híbrido con STL", CINCO),
}
NOMBRE = {k: v[0] for k, v in MODELOS.items()}
FAMILIA = {v[0]: v[1] for v in MODELOS.values()}
ORDEN_FAMILIAS = ["Referencia", "Estadístico", "Aprendizaje automático", "Aprendizaje profundo", "Híbrido con STL"]

# pares STL frente a directo (misma configuracion de exogenas): (nombre, variante directa, variante con STL)
PARES_STL = [
    ("FCNN: sobre residuo STL frente a directa", "FCNN directa", "FCNN sobre residuo STL"),
    ("LSTM: sobre residuo STL frente a directa", "LSTM directa", "LSTM sobre residuo STL"),
    ("LSTM directa frente a Ensemble STL", "LSTM directa", "Ensemble STL"),
    ("AR: sobre residuo STL frente a directo", "AR", "AR sobre residuo STL"),
    ("Naive + tendencia: con estacionalidad (STL) frente a sin ella", "Naive + tendencia", "Naive + tendencia + estacionalidad"),
]


# ------------------------------------------------------------------------------------------------ carga y seleccion
def cargar(ruta=LARGAS):
    df = pd.read_csv(ruta, encoding="utf-8")
    df["modelo_nombre"] = df["modelo_metrica"].map(NOMBRE)
    assert df["modelo_nombre"].notna().all(), "modelo no reconocido: %s" % sorted(set(df.loc[df["modelo_nombre"].isna(), "modelo_metrica"]))
    df["familia"] = df["modelo_nombre"].map(FAMILIA)
    return df


def seleccionar_referencia(df):
    """D1: una sola configuracion por modelo. Devuelve (df_referencia, informe) y verifica el diseno."""
    partes, informe = [], []
    for mm, (nombre, fam, exog) in MODELOS.items():
        sub = df[(df["modelo_metrica"] == mm) & (df["exogenas"] == exog)]
        corridas = sub[["ventana", "run_name"]].drop_duplicates()  # el run_name no incluye la semana: una corrida = (semana, run_name)
        informe.append({"modelo": nombre, "familia": fam, "exogenas_referencia": exog, "corridas": len(corridas),
                        "filas": len(sub), "esperadas": len(SEMANAS) * len(REGIONES)})
        partes.append(sub)
    ref = pd.concat(partes, ignore_index=True)
    inf = pd.DataFrame(informe)
    assert (inf["corridas"] == len(SEMANAS)).all(), "cada modelo debe tener 1 corrida de referencia por semana:\n%s" % inf
    assert (inf["filas"] == inf["esperadas"]).all(), "cada modelo debe tener 32 filas de referencia:\n%s" % inf
    return ref, inf


# ------------------------------------------------------------------------------------------------ D2: posiciones
def posiciones(mape, mae, dec=DEC):
    """Posiciones de competencia (1, 2, 2, 4...). Empate de MAPE a `dec` decimales; desempata MAE a `dec` decimales."""
    claves = [(round(float(m), dec), round(float(a), dec)) for m, a in zip(mape, mae)]
    return [1 + sum(1 for otra in claves if otra < c) for c in claves]


def con_posiciones(ref):
    """Agrega columna `posicion` (entre los 13 modelos de referencia) a cada par region-semana."""
    ref = ref.copy()
    ref["posicion"] = 0
    for (reg, sem), idx in ref.groupby(["region", "ventana"]).groups.items():
        g = ref.loc[idx]
        assert len(g) == len(MODELOS), (reg, sem, len(g))
        ref.loc[idx, "posicion"] = posiciones(g["MAPE"].to_numpy(), g["MAE"].to_numpy())
    return ref


# ------------------------------------------------------------------------------------------------ tablas
def tabla_global(ref):
    """T6.2: por modelo, media y mediana de las 32 observaciones region-semana (D3) + dispersion + posiciones."""
    filas = []
    for nombre, g in ref.groupby("modelo_nombre"):
        por_sem = g.groupby("ventana")["MAPE"].mean()   # media regional por semana
        por_reg = g.groupby("region")["MAPE"].mean()    # media semanal por region
        filas.append({
            "modelo": nombre, "familia": FAMILIA[nombre],
            "MAPE_media": g["MAPE"].mean(), "MAPE_mediana": g["MAPE"].median(),
            "sd_entre_semanas": por_sem.std(ddof=1), "semana_min": por_sem.min(), "semana_max": por_sem.max(),
            "sd_entre_regiones": por_reg.std(ddof=1),
            "MAE_media": g["MAE"].mean(), "MAE_mediana": g["MAE"].median(),
            "primeros_lugares": int((g["posicion"] == 1).sum()), "entre_los_3_primeros": int((g["posicion"] <= 3).sum()),
            "posicion_media": g["posicion"].mean(), "posicion_mediana": g["posicion"].median(),
            "peor_MAPE": g["MAPE"].max(),
        })
    t = pd.DataFrame(filas).sort_values(["MAPE_media", "MAE_media"]).reset_index(drop=True)
    t.insert(0, "orden_por_MAPE_media", range(1, len(t) + 1))
    return t


def tabla_por_semana(ref):
    """T6.4: MAPE (media y mediana entre regiones) por modelo y semana."""
    out = []
    for (nombre, sem), g in ref.groupby(["modelo_nombre", "ventana"]):
        out.append({"modelo": nombre, "familia": FAMILIA[nombre], "semana": sem, "MAPE_media_regional": g["MAPE"].mean(),
                    "MAPE_mediana_regional": g["MAPE"].median(), "MAE_media_regional": g["MAE"].mean()})
    return pd.DataFrame(out)


def tabla_estabilidad(ref):
    """T6.6: posicion de cada modelo en cada semana (por su media regional, regla D2) y estabilidad entre semanas."""
    sem = tabla_por_semana(ref)
    filas = []
    pos = {}
    for s in SEMANAS:
        g = sem[sem["semana"] == s].reset_index(drop=True)
        g["pos"] = posiciones(g["MAPE_media_regional"].to_numpy(), g["MAE_media_regional"].to_numpy())
        pos[s] = dict(zip(g["modelo"], g["pos"]))
    for nombre in sorted(FAMILIA):
        v = [pos[s][nombre] for s in SEMANAS]
        m = sem[sem["modelo"] == nombre]["MAPE_media_regional"]
        filas.append({"modelo": nombre, "familia": FAMILIA[nombre], **{"pos_" + s: pos[s][nombre] for s in SEMANAS},
                      "rango_de_posiciones": max(v) - min(v), "MAPE_media_4_semanas": m.mean(),
                      "coef_variacion_entre_semanas": m.std(ddof=1) / m.mean()})
    t = pd.DataFrame(filas).sort_values("MAPE_media_4_semanas").reset_index(drop=True)
    rho = pd.DataFrame(pos).rank(axis=0)  # rangos por semana
    spearman = rho.corr(method="pearson")  # Spearman = Pearson sobre rangos
    return t, spearman


def tabla_ganadores(ref):
    """T6.3: primer y segundo lugar por region y semana (entre los 13 de referencia)."""
    filas = []
    for (reg, sem), g in ref.groupby(["region", "ventana"]):
        g = g.sort_values(["posicion", "MAPE", "MAE"]).reset_index(drop=True)
        primeros = g[g["posicion"] == 1]
        segundo = g[g["posicion"] > 1].head(1)
        filas.append({"region": reg, "semana": sem, "primer_lugar": " | ".join(primeros["modelo_nombre"]),
                      "MAPE_1": primeros["MAPE"].iloc[0], "empate_en_primer_lugar": len(primeros) > 1,
                      "segundo_lugar": segundo["modelo_nombre"].iloc[0] if len(segundo) else "",
                      "MAPE_2": segundo["MAPE"].iloc[0] if len(segundo) else np.nan,
                      "brecha_pp": (segundo["MAPE"].iloc[0] - primeros["MAPE"].iloc[0]) if len(segundo) else np.nan})
    return pd.DataFrame(filas)


def matriz_region_modelo(ref):
    """MAPE medio sobre las 4 semanas, region x modelo (y mediana)."""
    media = ref.pivot_table(index="region", columns="modelo_nombre", values="MAPE", aggfunc="mean")
    mediana = ref.pivot_table(index="region", columns="modelo_nombre", values="MAPE", aggfunc="median")
    return media, mediana


def dificultad_region(ref):
    """Por region y semana: mediana y minimo del MAPE entre los 13 modelos (que tan dificil es la region)."""
    g = ref.groupby(["region", "ventana"])["MAPE"].agg(mediana_entre_modelos="median", minimo="min", maximo="max").reset_index()
    return g


def efecto_exogenas(df):
    """T6.5: configuraciones alternativas frente a la de referencia, mismo modelo, region y semana (D4 para equivalentes)."""
    filas = []
    for mm, (nombre, fam, exog_ref) in MODELOS.items():
        if exog_ref == NINGUNA:
            continue
        base = df[(df["modelo_metrica"] == mm) & (df["exogenas"] == exog_ref)].set_index(["region", "ventana"])
        for exog in sorted(df[(df["modelo_metrica"] == mm)]["exogenas"].unique()):
            if exog == exog_ref:
                continue
            alt = df[(df["modelo_metrica"] == mm) & (df["exogenas"] == exog)].set_index(["region", "ventana"])
            j = alt[["MAPE", "MAE"]].join(base[["MAPE", "MAE"]], lsuffix="_alt", rsuffix="_ref")
            assert len(j) == len(REGIONES) * len(SEMANAS)
            d = j["MAPE_alt"] - j["MAPE_ref"]
            r_alt, r_ref = j["MAPE_alt"].round(DEC), j["MAPE_ref"].round(DEC)
            idem = (j["MAPE_alt"] == j["MAPE_ref"]) & (j["MAE_alt"] == j["MAE_ref"])
            filas.append({"modelo": nombre, "familia": fam, "configuracion": exog, "n_pares": len(j),
                          "dif_MAPE_media": d.mean(), "dif_MAPE_mediana": d.median(),
                          "mejora": int((r_alt < r_ref).sum()), "empeora": int((r_alt > r_ref).sum()),
                          "empate_a_2_decimales": int((r_alt == r_ref).sum()), "equivalentes_identicas": int(idem.sum()),
                          "MAPE_media_config": j["MAPE_alt"].mean(), "MAPE_media_ref": j["MAPE_ref"].mean()})
    return pd.DataFrame(filas)


def comparacion_stl(ref):
    """T6.7: pares (STL frente a directo) con la misma configuracion; diferencia = variante STL - variante directa."""
    filas = []
    piv = ref.pivot_table(index=["region", "ventana"], columns="modelo_nombre", values=["MAPE", "MAE"])
    for etiqueta, directa, stl in PARES_STL:
        d = piv["MAPE"][stl] - piv["MAPE"][directa]
        rd, rs = piv["MAPE"][directa].round(DEC), piv["MAPE"][stl].round(DEC)
        filas.append({"comparacion": etiqueta, "directa": directa, "variante_STL": stl, "n_pares": len(d),
                      "dif_MAPE_media": d.mean(), "dif_MAPE_mediana": d.median(),
                      "STL_mejor": int((rs < rd).sum()), "STL_peor": int((rs > rd).sum()), "empate_a_2_decimales": int((rs == rd).sum()),
                      "MAPE_media_directa": piv["MAPE"][directa].mean(), "MAPE_media_STL": piv["MAPE"][stl].mean()})
    return pd.DataFrame(filas)


def equivalencias(df):
    """D4: para cada modelo, pares de configuraciones de exogenas con metricas identicas (MAPE y MAE exactos) en cada
    par region-semana. `identicas_en` cuenta en cuantos de los 32 pares coinciden; 32 = equivalentes en todo el estudio."""
    filas = []
    for mm, (nombre, fam, exog_ref) in MODELOS.items():
        sub = df[df["modelo_metrica"] == mm]
        configs = sorted(sub["exogenas"].unique())
        if len(configs) < 2:
            continue
        piv = sub.set_index(["region", "ventana", "exogenas"])[["MAPE", "MAE"]]
        for i, c1 in enumerate(configs):
            for c2 in configs[i + 1:]:
                a_ = piv.xs(c1, level="exogenas")
                b_ = piv.xs(c2, level="exogenas")
                iguales = int(((a_["MAPE"] == b_["MAPE"]) & (a_["MAE"] == b_["MAE"])).sum())
                filas.append({"modelo": nombre, "configuracion_1": c1, "configuracion_2": c2, "identicas_en": iguales, "de": len(a_),
                              "involucra_referencia": exog_ref in (c1, c2)})
    t = pd.DataFrame(filas)
    t["equivalentes_en_todo_el_estudio"] = t["identicas_en"] == t["de"]
    return t


def ranking_con_todas(df):
    """Anexo: ranking con las 61 configuraciones (ilustra el sesgo de elegir la mejor configuracion de cada modelo)."""
    df = df.copy()
    df["pos61"] = 0
    for (reg, sem), idx in df.groupby(["region", "ventana"]).groups.items():
        g = df.loc[idx]
        df.loc[idx, "pos61"] = posiciones(g["MAPE"].to_numpy(), g["MAE"].to_numpy())
    return df


def todas_las_tablas(df=None):
    df = cargar() if df is None else df
    ref, sel = seleccionar_referencia(df)
    ref = con_posiciones(ref)
    t = {"seleccion_D1": sel, "global_T6_2": tabla_global(ref), "ganadores_T6_3": tabla_ganadores(ref),
         "por_semana_T6_4": tabla_por_semana(ref), "exogenas_T6_5": efecto_exogenas(df),
         "dificultad_region": dificultad_region(ref), "comparacion_STL_T6_7": comparacion_stl(ref), "equivalencias_D4": equivalencias(df)}
    est, spearman = tabla_estabilidad(ref)
    t["estabilidad_T6_6"] = est
    t["spearman_semanas"] = spearman.reset_index()
    # tabla completa pedida por el autor: modelo x combinacion de exogenas x semana, con el MAPE de cada region
    comp = df.pivot_table(index=["modelo_nombre", "familia", "exogenas", "ventana"], columns="region", values="MAPE").reset_index()
    comp["media_regiones"] = comp[REGIONES].mean(axis=1)
    comp["mediana_regiones"] = comp[REGIONES].median(axis=1)
    t["completa_modelo_exogenas_semana_region_MAPE"] = comp
    media, mediana = matriz_region_modelo(ref)
    t["matriz_region_modelo_media"] = media.reset_index()
    t["matriz_region_modelo_mediana"] = mediana.reset_index()
    return df, ref, t


def main():
    df, ref, t = todas_las_tablas()
    os.makedirs(SALIDA, exist_ok=True)
    for nombre, tabla in t.items():
        tabla.to_csv(os.path.join(SALIDA, nombre + ".csv"), index=False, encoding="utf-8")
    ref.to_csv(os.path.join(SALIDA, "referencia_D1_con_posiciones.csv"), index=False, encoding="utf-8")
    compartidas = int((ref.groupby(["region", "ventana", "posicion"]).size() > 1).sum())
    print("tablas escritas en", SALIDA, "|", len(t), "tablas | filas de referencia:", len(ref), "| grupos con posicion compartida:", compartidas)
    return 0


if __name__ == "__main__":
    sys.exit(main())

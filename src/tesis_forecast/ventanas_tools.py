"""
Herramientas de la extension a cuatro semanas de prueba (S1..S4).

  1. `auditar_datos_ventanas()`  -- cobertura real de datos por region y semana
                                    (para correr en Colab ANTES de entrenar).
  2. `construir_configs()`       -- lista de ExperimentConfig por nivel de costo
                                    (T1 estadisticos/arboles, T2 redes, T3 exogenas).
  3. `comparar_s4_con_original()`-- compara la reproduccion de S4 contra los
                                    resultados originales ya guardados.
  4. `consolidar_ventanas()`     -- tabla larga modelo x region x semana mas
                                    medias y desviaciones estandar.

Solo lectura sobre Pipeline_Resultados/ (originales): nada de aqui los toca.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from . import windows
from .regions import REGIONS_ALL

METRICAS = ["MAE", "RMSE", "MAPE", "sMAPE"]

# Familias cuyas corridas originales invalidas por el bug +1h fueron
# reemplazadas por una corrida `_timefix` (Fase 4 de run_matrix.ipynb).
FAMILIAS_TIMEFIX = {"fcnn", "ensemble_stl", "lstm_resid"}

# Modelos con division original dinamica ("auto"): no son comparables con la
# corrida original; en las ventanas semanales se ejecutan con ventana fija.
BASELINES_AUTO = {"naive", "naive_trend", "ar"}

# Reparto de ejecucion para no duplicar corridas entre la PC y Colab: estos modelos
# no necesitan TensorFlow (pandas/statsmodels, XGBoost, LightGBM) y se corren en la PC
# (scripts/run_local_estadisticos.py); las redes neuronales se corren en Colab (GPU).
MODELOS_LOCALES = {
    "naive", "naive_trend", "naive_trend_seasonal", "ar_resid_trend_seasonal", "ar", "sarimax",
    "xgboost", "lightgbm",
}


def ejecuta_en(modelo: str) -> str:
    return "local" if modelo in MODELOS_LOCALES else "colab"


def filtrar_por_entorno(configs: list, entorno: str) -> list:
    """Deja solo las configuraciones que corresponden a 'local' o 'colab'."""
    if entorno not in ("local", "colab"):
        raise ValueError("entorno debe ser 'local' o 'colab'")
    return [c for c in configs if ejecuta_en(c.modelo) == entorno]
BASELINES_TRAIN_HOURS = 3600

EXOG_5 = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
EXOG_8 = ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"]


# =========================================================
# 1. AUDITORIA DE DATOS (Colab)
# =========================================================

def _leer_long(data_dir: str, region: str) -> Optional[pd.DataFrame]:
    ruta = os.path.join(data_dir, f"{region}_long.csv")
    if not os.path.exists(ruta):
        return None
    df, _ = windows._leer_csv_texto(ruta)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _huecos_horarios(df: pd.DataFrame) -> dict:
    fecha = pd.to_datetime(df[windows.COL_FECHA], errors="coerce")
    hora = pd.to_numeric(df[windows.COL_HORA], errors="coerce")
    ok = fecha.notna() & hora.notna()
    ts = (fecha[ok] + pd.to_timedelta(hora[ok].astype(int) - 1, unit="h")).drop_duplicates().sort_values()
    if len(ts) < 2:
        return {"horas_totales": int(len(ts)), "n_huecos": 0, "hueco_max_h": 0}
    saltos = ts.diff().dropna() / pd.Timedelta(hours=1)
    huecos = saltos[saltos > 1]
    return {
        "horas_totales": int(len(ts)),
        "n_huecos": int(len(huecos)),
        "hueco_max_h": int(huecos.max() - 1) if len(huecos) else 0,
    }


def auditar_datos_ventanas(
    data_dir: str,
    train_hours_por_familia: Optional[dict] = None,
    regiones: Optional[list] = None,
    ventanas: Optional[list] = None,
):
    """
    Devuelve (df_demanda, df_exog_regional, df_globales).

    - df_demanda: una fila por (ventana, region) con rango total del archivo,
      huecos, cobertura de la semana de prueba y, por cada familia de
      `train_hours_por_familia`, si hay historia suficiente.
    - df_exog_regional: ultima fecha y NaN de GEN/IMP/EXP hasta el fin de la
      ventana y en las 336 h previas a su inicio (rezagos t-168/t-336).
    - df_globales: ultima fecha con dato REAL de Temperatura/IGAE/Primarias/
      Secundarias/Terciarias frente al inicio de cada ventana.
    """
    train_hours_por_familia = train_hours_por_familia or {
        "xgboost_lightgbm": 336, "sarimax": 1440, "lstm_direct": 2160, "stl_redes": 3600,
    }
    regiones = list(regiones) if regiones is not None else list(REGIONS_ALL)
    ventanas = list(ventanas) if ventanas is not None else sorted(windows.VENTANAS)

    filas, filas_ex = [], []
    for region in regiones:
        df = _leer_long(data_dir, region)
        if df is None:
            for v in ventanas:
                filas.append({"ventana": v, "region": region, "problema": "no existe el archivo _long.csv"})
            continue

        fecha = pd.to_datetime(df[windows.COL_FECHA], errors="coerce")
        base = {
            "region": region,
            "archivo_fecha_min": str(fecha.min().date()),
            "archivo_fecha_max": str(fecha.max().date()),
            **_huecos_horarios(df),
        }

        for v in ventanas:
            vent = windows.get_ventana(v)
            trunc = df[fecha <= pd.Timestamp(vent.fin)]
            diag = windows.diagnosticar_demanda(trunc, vent)
            fila = {"ventana": v, **base, **diag}
            for fam, h in train_hours_por_familia.items():
                fila[f"historia_ok_{fam}_{h}h"] = diag["horas_previas_validas"] >= h
            fila["apta_para_ventana"] = bool(diag["termina_en_fin_de_ventana"] and diag["ventana_completa"])
            filas.append(fila)

            for suf in windows.SUFIJOS_EXOG_REGIONAL:
                ruta = os.path.join(data_dir, f"{region}_{suf}.csv")
                if not os.path.exists(ruta):
                    filas_ex.append({"ventana": v, "region": region, "exog": suf, "problema": "no existe el archivo"})
                    continue
                ex, _ = windows._leer_csv_texto(ruta)
                cols = {str(c).strip().lower(): c for c in ex.columns}
                f = pd.to_datetime(ex[cols["fecha"]], errors="coerce")
                val = pd.to_numeric(ex[cols["valor"]], errors="coerce")
                hasta_fin = f <= pd.Timestamp(vent.fin)
                previos = (f >= vent.test_start.normalize() - pd.Timedelta(days=14)) & (f < vent.test_start.normalize())
                con_valor = hasta_fin & val.notna()
                filas_ex.append({
                    "ventana": v, "region": region, "exog": suf,
                    "ultima_fecha_hasta_fin": None if not hasta_fin.any() else str(f[hasta_fin].max().date()),
                    "primera_fecha_con_valor": None if not con_valor.any() else str(f[con_valor].min().date()),
                    "ultima_fecha_con_valor": None if not con_valor.any() else str(f[con_valor].max().date()),
                    "pct_filas_con_valor": round(100.0 * float(con_valor.sum()) / max(int(hasta_fin.sum()), 1), 2),
                    "filas_previas_14d": int(previos.sum()),
                    "nan_previas_14d": int(val[previos].isna().sum()),
                    "cubre_fin_de_ventana": bool(hasta_fin.any() and f[hasta_fin].max().normalize() == pd.Timestamp(vent.fin)),
                })

    filas_gl = []
    try:
        from .data.exogenas import cargar_exogenas_horarias

        globales = cargar_exogenas_horarias(data_dir=data_dir)
        for nombre, d in globales.items():
            ultima = pd.to_datetime(d["fecha"]).max()
            for v in ventanas:
                vent = windows.get_ventana(v)
                filas_gl.append({
                    "serie": nombre, "ventana": v,
                    "ultima_fecha_con_dato_real": str(ultima.date()),
                    "dias_sin_dato_real_antes_de_la_prueba": int((vent.test_start.normalize() - ultima.normalize()).days),
                    "sin_dato_real_en_la_prueba": bool(ultima.normalize() < vent.test_start.normalize()),
                })
    except Exception as e:  # pragma: no cover - depende de los archivos reales
        filas_gl.append({"serie": "(error)", "ventana": "", "ultima_fecha_con_dato_real": f"{type(e).__name__}: {e}"})

    return pd.DataFrame(filas), pd.DataFrame(filas_ex), pd.DataFrame(filas_gl)


# =========================================================
# 2. PLAN DE EXPERIMENTOS POR NIVEL DE COSTO
# =========================================================

def _cfg(modelo, exogenas, ventana, train_hours=None, notas=""):
    from .config import ExperimentConfig

    return ExperimentConfig(
        modelo=modelo, exogenas=exogenas, train_hours=train_hours,
        forecast_horizon=windows.HORAS_VENTANA, ventana=ventana, notas=notas,
    )


def construir_configs(nivel: str, ventana: str) -> list:
    """
    Niveles (de menor a mayor costo). Las configuraciones replican EXACTAMENTE
    las de Fases 1A/1B/2/3/4 de notebooks/run_matrix.ipynb (mismas exogenas y
    mismo train_hours); solo cambia la ventana.

      T1: sarimax, xgboost, lightgbm, naive_trend_seasonal, ar_resid_trend_seasonal,
          naive, naive_trend, ar (estos tres con ventana fija 3600 h / 168 h)
      T2: fcnn, lstm_resid, lstm_direct, ensemble_stl (configuracion base, 5 exogenas)
      T3: variantes de exogenas (individuales, Temperatura+IGAE y, para fcnn/
          ensemble_stl/lstm_resid, las 7 combinaciones de la Fase 4)

    Naive, Naive_Trend y AR: su division por defecto es dinamica ("auto":
    prueba = max(720 h, 10 % de la serie)), no una semana de 168 h. Esos
    modulos ya aceptan enteros, asi que aqui se ejecutan con
    `train_hours=3600` y `forecast_horizon=168`. Sus resultados NO son
    comparables con las corridas originales (ventana dinamica): la
    comparacion de S4 contra el original los marca NO_COMPARABLE.
    """
    windows.get_ventana(ventana)
    from .matrix import build_individual_exog_matrix

    t1 = [
        # De menor a mayor costo (SARIMAX con estacionalidad de 168 h es el mas lento).
        _cfg("naive", [], ventana, BASELINES_TRAIN_HOURS, "S-ext T1 baseline ventana fija"),
        _cfg("naive_trend", [], ventana, BASELINES_TRAIN_HOURS, "S-ext T1 baseline ventana fija"),
        _cfg("naive_trend_seasonal", [], ventana, None, "S-ext T1"),
        _cfg("ar_resid_trend_seasonal", [], ventana, None, "S-ext T1"),
        _cfg("ar", [], ventana, BASELINES_TRAIN_HOURS, "S-ext T1 baseline ventana fija"),
        _cfg("xgboost", EXOG_8, ventana, 336, "S-ext T1"),
        _cfg("lightgbm", EXOG_8, ventana, 336, "S-ext T1"),
        _cfg("sarimax", EXOG_5, ventana, 1440, "S-ext T1"),
    ]
    # Naive/Naive_Trend/AR: su division original es dinamica ("auto"); aqui entrenan
    # con las 3600 h previas (igual que la familia STL) y pronostican 168 h. Esos
    # modulos ya aceptan enteros (capacidad existente).
    t2 = [
        _cfg("fcnn", EXOG_5, ventana, 3600, "S-ext T2"),
        _cfg("lstm_resid", EXOG_5, ventana, 3600, "S-ext T2"),
        _cfg("lstm_direct", EXOG_5, ventana, 2160, "S-ext T2"),
        _cfg("ensemble_stl", EXOG_5, ventana, 3600, "S-ext T2"),
    ]

    t3 = []
    for c in build_individual_exog_matrix(exogenas=["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]):
        t3.append(_cfg(c.modelo, c.exogenas, ventana, None, "S-ext T3 individual"))
    for modelo in ["xgboost", "lightgbm", "lstm_direct", "sarimax", "fcnn", "ensemble_stl", "lstm_resid"]:
        t3.append(_cfg(modelo, ["Temperatura", "IGAE"], ventana, None, "S-ext T3 Temp+IGAE"))
    for modelo in sorted(FAMILIAS_TIMEFIX):
        t3.append(_cfg(modelo, EXOG_5, ventana, 3600, "S-ext T3 combinacion completa"))

    def _quitar_repetidas(lista, ya):
        from .runner import resolve_run

        out = []
        for c in lista:
            nombre = resolve_run(c).run_name
            if nombre in ya:
                continue
            ya.add(nombre)
            out.append(c)
        return out

    from .runner import resolve_run

    nombres_previos = {resolve_run(c).run_name for c in t1 + t2}
    if nivel == "T1":
        return t1
    if nivel == "T2":
        return t2
    if nivel == "T3":
        return _quitar_repetidas(t3, set(nombres_previos))
    raise ValueError("nivel debe ser 'T1', 'T2' o 'T3'")


def plan_dataframe(ventanas: Optional[list] = None, niveles: tuple = ("T1", "T2", "T3")) -> pd.DataFrame:
    """Una fila por corrida planeada (ventana x nivel x configuracion)."""
    from .runner import resolve_run

    ventanas = list(ventanas) if ventanas is not None else sorted(windows.VENTANAS)
    filas = []
    for v in ventanas:
        for nivel in niveles:
            for c in construir_configs(nivel, v):
                r = resolve_run(c)
                filas.append({
                    "ventana": v, "nivel": nivel, "ejecuta_en": ejecuta_en(c.modelo),
                    "modelo": c.modelo, "run_name": r.run_name,
                    "exogenas": "+".join(r.exogenas) or "(ninguna)", "train_hours": r.train_hours,
                    "forecast_horizon": r.forecast_horizon, "optuna_n_trials": r.optuna_n_trials,
                    "run_dir_relativo": os.path.join("Pipeline_Resultados_Ventanas", v, r.run_name).replace(os.sep, "/"),
                })
    return pd.DataFrame(filas)


# =========================================================
# 3. REPRODUCCION DE S4 VS ORIGINAL
# =========================================================

# Tolerancias propuestas (a confirmar con quien decide): los modelos sin
# aleatoriedad deben coincidir casi exactamente; las redes dependen de TF.
TOLERANCIA_REL = {"deterministico": 1e-6, "arboles": 1e-4}
TOLERANCIA_MAPE_PP_REDES = 0.5  # puntos porcentuales de MAPE por region

_DETERMINISTICOS = {"sarimax", "naive_trend_seasonal", "ar_resid_trend_seasonal"}
_ARBOLES = {"xgboost", "lightgbm"}


def nombre_corrida_original(run_name: str, modelo: str) -> str:
    """Las corridas de fcnn/ensemble_stl/lstm_resid vigentes en el consolidado
    son las `_timefix`; las sin sufijo quedaron invalidadas por el bug +1h."""
    return f"{run_name}_timefix" if modelo in FAMILIAS_TIMEFIX else run_name


def comparar_s4_con_original(
    configs: list,
    pipeline_resultados_dir: str,
    ventanas_base_dir: str,
    tolerancia_mape_pp_redes: float = TOLERANCIA_MAPE_PP_REDES,
):
    """
    Une, por (region, estrategia), las metricas de `Pipeline_Resultados/<run>`
    (originales) con las de `Pipeline_Resultados_Ventanas/S4/<run>` y marca si
    la reproduccion coincide segun la familia del modelo.

    Devuelve (detalle_df, resumen_df). Nunca escribe nada.
    """
    from .runner import resolve_run

    detalle = []
    for c in configs:
        r = resolve_run(c)
        if c.modelo in BASELINES_AUTO:
            detalle.append({
                "modelo": c.modelo, "run_name": r.run_name, "region": None, "estrategia": None,
                "estado": "NO_COMPARABLE",
            })
            continue
        orig_dir = os.path.join(pipeline_resultados_dir, nombre_corrida_original(r.run_name, c.modelo))
        nuevo_dir = os.path.join(ventanas_base_dir, "S4", r.run_name)
        p_o, p_n = os.path.join(orig_dir, "metricas.csv"), os.path.join(nuevo_dir, "metricas.csv")
        if not (os.path.exists(p_o) and os.path.exists(p_n)):
            detalle.append({
                "modelo": c.modelo, "run_name": r.run_name, "region": None, "estrategia": None,
                "estado": "FALTA_ORIGINAL" if not os.path.exists(p_o) else "FALTA_REPRODUCCION",
            })
            continue
        o = pd.read_csv(p_o, encoding="utf-8-sig")
        n = pd.read_csv(p_n, encoding="utf-8-sig")
        clave = [k for k in ["serie", "modelo"] if k in o.columns and k in n.columns]
        m = o.merge(n, on=clave, suffixes=("_orig", "_nuevo"), how="outer", indicator=True)
        for _, fila in m.iterrows():
            reg = fila["serie"].split("_")[0] if "serie" in m.columns and isinstance(fila["serie"], str) else None
            d = {"modelo": c.modelo, "run_name": r.run_name, "region": reg,
                 "estrategia": fila.get("modelo") if "modelo" in clave else None}
            if fila["_merge"] != "both":
                d["estado"] = "SOLO_ORIGINAL" if fila["_merge"] == "left_only" else "SOLO_REPRODUCCION"
                detalle.append(d)
                continue
            for met in METRICAS:
                d[f"{met}_orig"], d[f"{met}_nuevo"] = fila[f"{met}_orig"], fila[f"{met}_nuevo"]
            d["dMAPE_pp"] = fila["MAPE_nuevo"] - fila["MAPE_orig"]
            den = max(abs(fila["MAPE_orig"]), 1e-12)
            d["dMAPE_rel"] = abs(fila["MAPE_nuevo"] - fila["MAPE_orig"]) / den
            if c.modelo in _DETERMINISTICOS:
                d["estado"] = "OK" if d["dMAPE_rel"] <= TOLERANCIA_REL["deterministico"] else "DIFIERE"
            elif c.modelo in _ARBOLES:
                d["estado"] = "OK" if d["dMAPE_rel"] <= TOLERANCIA_REL["arboles"] else "DIFIERE"
            else:
                d["estado"] = "OK" if abs(d["dMAPE_pp"]) <= tolerancia_mape_pp_redes else "DIFIERE"
            detalle.append(d)

    detalle_df = pd.DataFrame(detalle)
    if detalle_df.empty:
        return detalle_df, detalle_df
    if "dMAPE_pp" not in detalle_df.columns:
        detalle_df["dMAPE_pp"] = np.nan
    resumen = (
        detalle_df.groupby(["modelo", "run_name"], dropna=False)
        .agg(
            filas=("estado", "size"),
            ok=("estado", lambda s: int((s == "OK").sum())),
            difiere=("estado", lambda s: int((s == "DIFIERE").sum())),
            otros=("estado", lambda s: int((~s.isin(["OK", "DIFIERE"])).sum())),
            max_abs_dMAPE_pp=("dMAPE_pp", lambda s: float(np.nanmax(np.abs(s))) if s.notna().any() else np.nan),
        )
        .reset_index()
    )
    no_comparable = resumen["modelo"].isin(BASELINES_AUTO)
    resumen["reproduccion_satisfactoria"] = (resumen["difiere"] == 0) & (resumen["otros"] == 0)
    resumen.loc[no_comparable, "reproduccion_satisfactoria"] = None
    return detalle_df, resumen


# =========================================================
# 4. CONSOLIDADO MODELO x REGION x SEMANA
# =========================================================

def _leer_corridas(ventanas_base_dir: str, ventanas: list) -> pd.DataFrame:
    filas = []
    for v in ventanas:
        d_v = os.path.join(ventanas_base_dir, v)
        if not os.path.isdir(d_v):
            continue
        for run_name in sorted(os.listdir(d_v)):
            d = os.path.join(d_v, run_name)
            p_m, p_c = os.path.join(d, "metricas.csv"), os.path.join(d, "config.json")
            if not (os.path.isfile(p_m) and os.path.isfile(p_c)):
                continue
            with open(p_c, encoding="utf-8") as f:
                cfg = json.load(f)
            m = pd.read_csv(p_m, encoding="utf-8-sig")
            if "region" not in m.columns:
                continue
            m = m.copy()
            m.insert(0, "ventana", v)
            m.insert(1, "run_name", run_name)
            m.insert(2, "modelo_config", cfg.get("modelo"))
            m.insert(3, "exogenas", "+".join(cfg.get("exogenas") or []) or "(ninguna)")
            m.insert(4, "train_hours", cfg.get("train_hours"))
            m = m.rename(columns={"modelo": "estrategia"})
            filas.append(m)
    if not filas:
        return pd.DataFrame()
    return pd.concat(filas, ignore_index=True)


def consolidar_ventanas(
    ventanas_base_dir: str,
    out_dir: Optional[str] = None,
    ventanas: Optional[list] = None,
) -> dict:
    """
    Lee todas las corridas de Pipeline_Resultados_Ventanas/<S#>/ y devuelve
    (y escribe, si se da `out_dir`):

      largo            modelo x exogenas x estrategia x region x semana, con MAE/RMSE/MAPE/sMAPE
      por_region       media y desviacion estandar (ddof=1) entre semanas, por region
      por_semana       promedio entre regiones, por semana
      global_          media y desviacion estandar entre semanas del promedio regional
      inventario       cuantas regiones tiene cada corrida por semana
    """
    ventanas = list(ventanas) if ventanas is not None else sorted(windows.VENTANAS)
    largo = _leer_corridas(ventanas_base_dir, ventanas)
    if largo.empty:
        return {"largo": largo}

    clave = ["modelo_config", "exogenas", "train_hours", "estrategia"]
    metricas = [m for m in METRICAS if m in largo.columns]

    por_region = largo.groupby(clave + ["region"], dropna=False)[metricas].agg(["mean", "std", "count"])
    por_region.columns = [f"{a}_{b}" for a, b in por_region.columns]
    por_region = por_region.reset_index()

    por_semana = largo.groupby(clave + ["ventana"], dropna=False).agg(
        n_regiones=("region", "nunique"), **{f"{m}_media_regiones": (m, "mean") for m in metricas},
    ).reset_index()

    g = por_semana.groupby(clave, dropna=False)
    global_ = g.agg(
        n_semanas=("ventana", "nunique"),
        regiones_por_semana_min=("n_regiones", "min"),
        **{f"{m}_media": (f"{m}_media_regiones", "mean") for m in metricas},
        **{f"{m}_std": (f"{m}_media_regiones", "std") for m in metricas},
    ).reset_index()
    global_["completo_4_semanas_8_regiones"] = (global_["n_semanas"] == len(ventanas)) & (global_["regiones_por_semana_min"] == len(REGIONS_ALL))

    inventario = largo.groupby(["run_name", "ventana"]).agg(n_regiones=("region", "nunique")).reset_index()
    inventario = inventario.pivot(index="run_name", columns="ventana", values="n_regiones").reset_index()

    res = {"largo": largo, "por_region": por_region, "por_semana": por_semana, "global_": global_, "inventario": inventario}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        nombres = {
            "largo": "metricas_ventanas_largo.csv",
            "por_region": "resumen_modelo_region_entre_semanas.csv",
            "por_semana": "resumen_modelo_semana.csv",
            "global_": "resumen_modelo_global_entre_semanas.csv",
            "inventario": "inventario_corridas_ventanas.csv",
        }
        for k, nombre in nombres.items():
            res[k].to_csv(os.path.join(out_dir, nombre), index=False, encoding="utf-8-sig")
    return res


# =========================================================
# 5. TIEMPOS DE EJECUCION
# =========================================================

def guardar_tiempos(registros: list, ruta_csv: str) -> pd.DataFrame:
    """
    Agrega al CSV `ruta_csv` una fila por corrida de `run_matrix()` (modelo,
    ventana, RUN_NAME, estado, duracion en segundos y MAPE promedio) con la
    hora de registro. Nunca reescribe lo ya guardado.
    """
    from datetime import datetime, timezone

    from .matrix import resumen_dataframe

    df = resumen_dataframe(registros)
    df["registrado_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    df.to_csv(ruta_csv, mode="a", header=not os.path.exists(ruta_csv), index=False, encoding="utf-8")
    return df


# =========================================================
# 6. ESTADO DEL PLAN (completas / parciales / sin iniciar)
# =========================================================

def estado_plan(ventanas_base_dir: str, ejecuta: Optional[str] = None, ventanas: Optional[list] = None) -> pd.DataFrame:
    """
    Cruza el plan con lo que existe en `ventanas_base_dir`: por cada corrida
    planeada, `completa` (8 regiones, sin NaN), `parcial` (carpeta incompleta;
    `regiones_ok` dice cuantas) o `no_iniciada`. Solo lectura.
    """
    from .validator import validar_resultado

    plan = plan_dataframe(ventanas=ventanas)
    if ejecuta is not None:
        plan = plan[plan["ejecuta_en"] == ejecuta].copy()
    estados, regiones_ok = [], []
    for _, fila in plan.iterrows():
        run_dir = os.path.join(ventanas_base_dir, fila["ventana"], fila["run_name"])
        if not os.path.isdir(run_dir):
            estados.append("no_iniciada"); regiones_ok.append(0); continue
        rep = validar_resultado(run_dir)
        estados.append("completa" if rep.es_completo else "parcial")
        regiones_ok.append(len(rep.regiones_encontradas))
    plan["estado"] = estados
    plan["regiones_ok"] = regiones_ok
    return plan

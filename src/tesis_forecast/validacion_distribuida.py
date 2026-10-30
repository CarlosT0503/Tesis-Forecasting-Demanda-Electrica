"""
Validacion automatica de una corrida traida de otro trabajador (solo lectura: nunca modifica la carpeta que valida).

Comprueba, contra el plan (`ventanas_tools.plan_dataframe`) y la ventana (`windows`):
  estructura   archivos obligatorios presentes, no vacios y legibles (CSV/JSON)
  regiones     las ocho regiones en metricas.csv
  predicciones 168 predicciones de prueba por region (y modelo), fechas = la ventana de prueba, valores finitos
  metricas     MAPE, sMAPE, MAE y RMSE finitas y no negativas
  configuracion config.json coincide con el plan (modelo, ventana, exogenas, horas, intentos de Optuna, semilla)
  optuna       por region (y componente) intentos COMPLETE = presupuesto; con valores finitos; avisa de exceso
  versiones    el ambiente del trabajador (versiones y datos) es compatible con el de referencia
  integridad   huella SHA-256 de cada archivo; si ya se habia validado, avisa si cambio
  errores      errores de region del ultimo intento en log.txt

Cada hallazgo es un ERROR (la corrida no se consolida) o un AVISO (se consolida y queda registrado).
"""

import json
import math
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from . import windows
from .distribuido import hash_archivo
from .regions import REGIONS_ALL

ARCHIVOS_OBLIGATORIOS = ("config.json", "metricas.csv", "series.csv", "trials.csv", "config_usada.csv")
ARCHIVOS_ESPERADOS = ARCHIVOS_OBLIGATORIOS + ("log.txt", "ventana_manifest.json")
COLUMNAS_METRICAS = ("MAPE", "sMAPE", "MAE", "RMSE")
LIBRERIAS_CRITICAS = ("tensorflow", "keras")                 # distinta version mayor.menor -> ERROR
LIBRERIAS_RELEVANTES = ("numpy", "pandas", "statsmodels", "optuna", "scipy")   # distinta mayor.menor -> AVISO
CAMPOS_CONFIG = ("modelo", "ventana", "train_hours", "forecast_horizon", "optuna_n_trials", "seed", "run_name")


@dataclass
class Informe:
    ventana: str
    run_name: str
    trabajador: str = ""
    hallazgos: list = field(default_factory=list)
    huellas: dict = field(default_factory=dict)
    resumen: dict = field(default_factory=dict)

    def error(self, codigo, mensaje):
        self.hallazgos.append({"nivel": "ERROR", "codigo": codigo, "mensaje": mensaje})

    def aviso(self, codigo, mensaje):
        self.hallazgos.append({"nivel": "AVISO", "codigo": codigo, "mensaje": mensaje})

    @property
    def errores(self):
        return [h for h in self.hallazgos if h["nivel"] == "ERROR"]

    @property
    def ok(self):
        return not self.errores

    def a_dict(self):
        return {"ventana": self.ventana, "run_name": self.run_name, "trabajador": self.trabajador, "ok": self.ok,
                "n_errores": len(self.errores), "hallazgos": self.hallazgos, "huellas": self.huellas, "resumen": self.resumen}


def _mayor_menor(v):
    partes = str(v).split(".")
    return ".".join(partes[:2]) if v else None


def _plan_fila(ventana, run_name):
    from . import ventanas_tools as vt
    plan = vt.plan_dataframe()
    f = plan[(plan["ventana"] == ventana) & (plan["run_name"] == run_name)]
    return None if f.empty else f.iloc[0].to_dict()


def _leer_csv(ruta):
    return pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)


def validar_corrida(run_dir, ventana, run_name, *, trabajador="", ambiente=None, ambiente_referencia=None,
                    huellas_previas=None, regiones=None):
    """Valida UNA corrida. `ambiente` y `ambiente_referencia` son los JSON de `distribuido.registrar_ambiente`."""
    regiones = list(regiones or REGIONS_ALL)
    inf = Informe(ventana=ventana, run_name=run_name, trabajador=trabajador)
    plan = _plan_fila(ventana, run_name)
    if plan is None:
        inf.error("plan", "la corrida no existe en el plan: %s/%s" % (ventana, run_name))
        return inf
    if not os.path.isdir(run_dir):
        inf.error("estructura", "no existe la carpeta %s" % run_dir)
        return inf
    inf.resumen["modelo"] = plan["modelo"]

    # ---- estructura e integridad
    cargados = {}
    for nombre in ARCHIVOS_ESPERADOS:
        ruta = os.path.join(run_dir, nombre)
        if not os.path.isfile(ruta):
            (inf.error if nombre in ARCHIVOS_OBLIGATORIOS else inf.aviso)("estructura", "falta %s" % nombre)
            continue
        if os.path.getsize(ruta) == 0:
            # el log es auxiliar (en Colab puede quedar en 0 bytes si el Drive montado no lo vacio): solo se avisa
            (inf.aviso if nombre == "log.txt" else inf.error)("estructura", "%s esta vacio" % nombre)
            continue
        try:
            cargados[nombre] = json.load(open(ruta, encoding="utf-8-sig")) if nombre.endswith(".json") else (
                _leer_csv(ruta) if nombre.endswith(".csv") else None)
        except (ValueError, OSError, pd.errors.ParserError) as e:
            inf.error("estructura", "%s no se puede leer: %s" % (nombre, str(e)[:120]))
    for nombre in sorted(os.listdir(run_dir)):
        ruta = os.path.join(run_dir, nombre)
        if os.path.isfile(ruta):
            inf.huellas[nombre] = {"sha256": hash_archivo(ruta), "bytes": os.path.getsize(ruta)}
    if huellas_previas:
        cambiados = [n for n, h in inf.huellas.items() if n in huellas_previas and huellas_previas[n]["sha256"] != h["sha256"]]
        if cambiados:
            inf.aviso("integridad", "archivos que cambiaron desde la validacion anterior: %s" % cambiados)
    if not any(n.endswith(".db") for n in inf.huellas):
        inf.aviso("optuna", "no hay base de Optuna (.db) en la carpeta")

    # ---- configuracion
    cfg = cargados.get("config.json")
    if isinstance(cfg, dict):
        for campo in CAMPOS_CONFIG:
            esperado, real = plan.get(campo), cfg.get(campo)
            if campo == "optuna_n_trials":
                esperado = int(esperado) if esperado == esperado else None
            if real is None or (esperado is not None and str(real) != str(esperado) and not (
                    isinstance(esperado, (int, float)) and float(real) == float(esperado))):
                if campo == "seed" and real == 42:
                    continue
                inf.error("configuracion", "config.json: %s = %r, el plan espera %r" % (campo, real, esperado))
        if isinstance(plan["exogenas"], str):   # el plan las guarda como "Temperatura+IGAE" (o como lista escrita)
            exo_plan = sorted(e for e in re.split(r"[+,\s\[\]']+", plan["exogenas"]) if e)
        else:
            exo_plan = sorted(plan["exogenas"])
        if sorted(cfg.get("exogenas") or []) != exo_plan:
            inf.error("configuracion", "config.json: exogenas %s != plan %s" % (cfg.get("exogenas"), exo_plan))
        v = windows.get_ventana(ventana)
        if str(cfg.get("test_start", ""))[:10] != str(v.test_start)[:10] or str(cfg.get("test_end", ""))[:10] != str(v.test_end)[:10]:
            inf.error("configuracion", "ventana de prueba %s..%s != %s..%s" % (cfg.get("test_start"), cfg.get("test_end"), v.test_start, v.test_end))
        inf.resumen["git_commit"] = cfg.get("git_commit")   # las corridas de Colab no lo registran: el commit va en el ambiente

    # ---- regiones y metricas
    met = cargados.get("metricas.csv")
    if isinstance(met, pd.DataFrame):
        col_reg = "region" if "region" in met else None
        if col_reg is None:
            inf.error("regiones", "metricas.csv no tiene columna region")
        else:
            presentes = sorted(met[col_reg].dropna().unique())
            faltan = [r for r in regiones if r not in presentes]
            if faltan:
                inf.error("regiones", "faltan regiones en metricas.csv: %s" % faltan)
            inf.resumen["regiones"] = len(set(presentes) & set(regiones))
        for c in COLUMNAS_METRICAS:
            if c not in met:
                inf.error("metricas", "metricas.csv no tiene la columna %s" % c)
                continue
            x = pd.to_numeric(met[c], errors="coerce")
            malas = int((~x.map(lambda t: isinstance(t, float) and math.isfinite(t) and t >= 0)).sum())
            if malas:
                inf.error("metricas", "%s: %d valores no finitos o negativos" % (c, malas))

    # ---- predicciones
    ser = cargados.get("series.csv")
    if isinstance(ser, pd.DataFrame) and isinstance(cfg, dict):
        needed = {"region", "fecha", "tipo", "subset", "modelo", "valor"}
        if not needed <= set(ser.columns):
            inf.error("predicciones", "series.csv sin columnas %s" % sorted(needed - set(ser.columns)))
        else:
            pred = ser[(ser["tipo"] == "prediccion") & (ser["subset"] == "test")]
            h = int(cfg.get("forecast_horizon") or 168)
            fechas_ok = set(pd.date_range(cfg.get("test_start"), cfg.get("test_end"), freq="h").astype(str))
            for r in regiones:
                pr = pred[pred["region"] == r]
                if pr.empty:
                    inf.error("predicciones", "%s: sin predicciones de prueba" % r)
                    continue
                for modelo, g in pr.groupby("modelo"):
                    if len(g) != h:
                        inf.error("predicciones", "%s/%s: %d predicciones (se esperaban %d)" % (r, modelo, len(g), h))
                    if g["fecha"].astype(str).str[:19].nunique() != len(g):
                        inf.error("predicciones", "%s/%s: fechas repetidas" % (r, modelo))
                    if set(g["fecha"].astype(str).str[:19]) != fechas_ok:
                        inf.error("predicciones", "%s/%s: las fechas no coinciden con la ventana de prueba" % (r, modelo))
                    v = pd.to_numeric(g["valor"], errors="coerce")
                    if not v.map(lambda t: isinstance(t, float) and math.isfinite(t)).all():
                        inf.error("predicciones", "%s/%s: valores no finitos" % (r, modelo))
            inf.resumen["modelos_en_series"] = sorted(pred["modelo"].unique().tolist())

    # ---- Optuna
    tr = cargados.get("trials.csv")
    esperado_trials = int(plan["optuna_n_trials"]) if plan.get("optuna_n_trials") == plan.get("optuna_n_trials") else None
    if isinstance(tr, pd.DataFrame) and esperado_trials:
        claves = [c for c in ("serie", "modelo", "componente", "estrategia", "estudio") if c in tr.columns]
        estado = tr["state"] if "state" in tr else pd.Series(["COMPLETE"] * len(tr))
        valor = pd.to_numeric(tr["value"], errors="coerce") if "value" in tr else pd.Series([float("nan")] * len(tr))
        if not claves:
            inf.aviso("optuna", "trials.csv no identifica serie/modelo: no se puede contar por region")
        else:
            for clave, idx in tr.groupby(claves).groups.items():
                nombre = "/".join(map(str, clave if isinstance(clave, tuple) else (clave,)))
                if "modelo" in claves and str((clave if isinstance(clave, tuple) else (clave,))[claves.index("modelo")]).startswith("AR"):
                    continue   # Ensemble STL: el residuo AR se elige por AIC/BIC (168 rezagos), no con Optuna
                completos = int((estado.loc[idx] == "COMPLETE").sum())
                finitos = int(valor.loc[idx][estado.loc[idx] == "COMPLETE"].map(lambda t: math.isfinite(t)).sum())
                if completos < esperado_trials:
                    inf.error("optuna", "%s: %d intentos COMPLETE de %d" % (nombre, completos, esperado_trials))
                elif completos > esperado_trials:
                    inf.aviso("optuna", "%s: %d intentos COMPLETE, presupuesto %d (exceso respecto al presupuesto de intentos)" % (nombre, completos, esperado_trials))
                if completos and finitos == 0:
                    inf.error("optuna", "%s: ningun intento con valor finito (el ajuste no opero; ver REPRODUCIBILIDAD.md, XGBoost)" % nombre)
                elif finitos < completos:
                    inf.aviso("optuna", "%s: %d de %d intentos con valor infinito/NaN" % (nombre, completos - finitos, completos))
            inf.resumen["grupos_optuna"] = int(tr.groupby(claves).ngroups)

    # ---- errores del ultimo intento
    ruta_log = os.path.join(run_dir, "log.txt")
    if os.path.isfile(ruta_log):
        try:
            from . import errores_log
            errs = errores_log.extraer_errores(open(ruta_log, encoding="utf-8", errors="replace").read(), solo_ultima_ejecucion=True)
            if errs:
                inf.aviso("errores", "el ultimo intento registro %d error(es) de region (la corrida igual tiene las 8 regiones)" % len(errs))
        except Exception:
            pass

    # ---- versiones y datos
    if ambiente is None:
        inf.aviso("versiones", "sin ambiente registrado del trabajador %s: no se pueden comparar versiones" % (trabajador or "?"))
    else:
        if (ambiente.get("datos") or {}).get("ok") is False:
            inf.error("versiones", "los datos del trabajador no coinciden con la lista de referencia: %s" % {
                k: (ambiente["datos"].get(k) or [])[:3] for k in ("distintos", "faltan")})
        if ambiente_referencia:
            va, vr = ambiente.get("versiones") or {}, ambiente_referencia.get("versiones") or {}
            for lib in LIBRERIAS_CRITICAS + LIBRERIAS_RELEVANTES:
                if va.get(lib) is None or vr.get(lib) is None or _mayor_menor(va[lib]) == _mayor_menor(vr[lib]):
                    continue
                (inf.error if lib in LIBRERIAS_CRITICAS else inf.aviso)(
                    "versiones", "%s %s (trabajador) frente a %s (referencia)" % (lib, va[lib], vr[lib]))
    return inf

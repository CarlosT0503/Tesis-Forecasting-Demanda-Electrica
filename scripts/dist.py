"""
Entrenamiento distribuido (varios Colab + la PC): sincronizar, validar, asignar, consolidar y reportar, desde la PC.

    python scripts/dist.py asignar     # reparte lo pendiente entre trabajadores (exclusivo) -> docs/asignacion/<T>.csv
    python scripts/dist.py sync        # descarga a staging los resultados de cada trabajador (solo lectura en origen)
    python scripts/dist.py validar     # valida lo descargado; detecta conflictos
    python scripts/dist.py consolidar  # muestra (o con --confirmar, copia) lo validado al Drive principal, sin sobrescribir
    python scripts/dist.py reporte     # reporte central (progreso, errores, validado, pendiente)
    python scripts/dist.py ciclo       # sync + validar + reporte (para correr cada N minutos con --cada)

Configuracion: `resultados_locales/distribuido/config.json` (ver docs/distribuido_config.ejemplo.json). Solo contiene NOMBRES
de remotos de rclone y rutas; **las credenciales viven en la configuracion de rclone del usuario (fuera del repositorio) y
nunca se leen ni se imprimen aqui**. Un trabajador puede ser un remoto de rclone ("remoto:Ruta") o una carpeta local
("local:C:/ruta"), util para pruebas y para Drive montado.

Garantias: nunca se borra nada en origen ni en destino; el staging es un espejo local (se puede reconstruir); la
consolidacion usa copia sin sobrescribir y, ante cualquier diferencia con lo existente, informa CONFLICTO y se detiene.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from unittest import mock  # noqa: E402

for _n in ["optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras", "tensorflow.keras.models",
           "tensorflow.keras.layers", "tensorflow.keras.callbacks", "tensorflow.keras.optimizers", "tensorflow.keras.backend"]:
    try:
        __import__(_n)
    except ImportError:
        sys.modules[_n] = mock.MagicMock()

import pandas as pd  # noqa: E402
from tesis_forecast import distribuido as dist  # noqa: E402
from tesis_forecast import ventanas_tools as vt  # noqa: E402
from tesis_forecast import validacion_distribuida as vd  # noqa: E402

BASE = os.path.join(RAIZ, "resultados_locales", "distribuido")
RUTA_CONFIG = os.path.join(BASE, "config.json")
CARPETA_RESULTADOS = "Pipeline_Resultados_Ventanas"
DURACION_DEFECTO_MIN = {"fcnn": 25.0}      # dato del autor (T3, Colab); el resto se estima con los tiempos sincronizados


# ---------------------------------------------------------------------------------------------------------
# configuracion y backends
# ---------------------------------------------------------------------------------------------------------

def cargar_config(ruta=None):
    ruta = ruta or RUTA_CONFIG
    if not os.path.isfile(ruta):
        raise SystemExit("falta %s (copia docs/distribuido_config.ejemplo.json y ajusta los nombres de remotos)" % ruta)
    cfg = json.load(open(ruta, encoding="utf-8"))
    if sum(1 for t in cfg["trabajadores"] if t.get("rol") == "principal") != 1:
        raise SystemExit("la configuracion debe tener exactamente un trabajador con rol 'principal'")
    return cfg


def staging(cfg, nombre):
    return os.path.join(cfg.get("staging") or os.path.join(BASE, "staging"), nombre)


def ruta_rclone(cfg):
    for c in (cfg.get("rclone"), shutil.which("rclone"), os.path.join(BASE, "bin", "rclone.exe")):
        if c and os.path.isfile(c):
            return c
    raise SystemExit("rclone no esta instalado (ver docs/DISTRIBUIDO_INSTRUCCIONES.md, paso 1)")


def es_local(spec):
    return spec.startswith("local:")


def raiz_remota(trab):
    """'remoto:Ruta' o 'local:C:/ruta' -> (tipo, spec) apuntando a la carpeta Pipeline_Resultados_Ventanas del trabajador."""
    return trab["origen"].rstrip("/\\")


def _copiar_arbol_local(origen, destino, sobrescribir_cambios=True, incluir=None):
    n = 0
    for raiz, dirs, archivos in os.walk(origen):
        rel = os.path.relpath(raiz, origen)
        dirs[:] = [d for d in dirs if d != "_tmp"]
        for a in archivos:
            s = os.path.join(raiz, a)
            d = os.path.join(destino, rel, a) if rel != "." else os.path.join(destino, a)
            if os.path.exists(d):
                if not sobrescribir_cambios or (os.path.getsize(s) == os.path.getsize(d) and int(os.path.getmtime(s)) <= int(os.path.getmtime(d))):
                    continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    return n


def sincronizar(cfg, trab, rclone=None):
    """Espejo local (staging) del trabajador. Solo lee el origen. Devuelve archivos copiados (o None si no se cuenta)."""
    destino = staging(cfg, trab["nombre"])
    os.makedirs(destino, exist_ok=True)
    origen = raiz_remota(trab)
    if es_local(origen):
        return _copiar_arbol_local(origen[len("local:"):], destino)
    cmd = [rclone or ruta_rclone(cfg), "copy", origen, destino, "--exclude", "_tmp/**", "--exclude", "**/_tmp/**",
           "--transfers", "4", "--checkers", "8", "--stats-one-line", "-v"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("rclone fallo (%s): %s" % (trab["nombre"], (r.stderr or r.stdout)[-300:]))
    return None


# ---------------------------------------------------------------------------------------------------------
# lectura de lo descargado
# ---------------------------------------------------------------------------------------------------------

def corridas_en(carpeta):
    """[(ventana, run_name, ruta)] de las carpetas de corrida con config.json o metricas.csv."""
    out = []
    if not os.path.isdir(carpeta):
        return out
    for v in sorted(os.listdir(carpeta)):
        dv = os.path.join(carpeta, v)
        if not (v.startswith("S") and os.path.isdir(dv)):
            continue
        for r in sorted(os.listdir(dv)):
            dr = os.path.join(dv, r)
            if os.path.isdir(dr):
                out.append((v, r, dr))
    return out


def leer_json(ruta):
    try:
        return json.load(open(ruta, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ambiente_de(cfg, nombre):
    return leer_json(os.path.join(staging(cfg, nombre), "_auditoria", "ambiente_%s.json" % nombre))


# ---------------------------------------------------------------------------------------------------------
# validar
# ---------------------------------------------------------------------------------------------------------

def validar_todo(cfg, ambito="colab"):
    """Valida lo descargado. `ambito="colab"`: solo las corridas del plan de Colab y solo si estan completas (8 regiones);
    las demas se reportan como 'incompleta' o se ignoran (fuera de alcance). `ambito="todo"`: solo para pruebas."""
    from tesis_forecast.validator import validar_resultado
    base_val = os.path.join(BASE, "validacion")
    os.makedirs(base_val, exist_ok=True)
    plan = vt.plan_dataframe()
    plan_colab = plan[plan["ejecuta_en"] == "colab"]
    en_alcance = set(zip(plan_colab["ventana"], plan_colab["run_name"])) if ambito == "colab" else None
    principal = next(t["nombre"] for t in cfg["trabajadores"] if t["rol"] == "principal")
    amb_ref = ambiente_de(cfg, principal)
    por_corrida = {}
    for t in cfg["trabajadores"]:
        for v, r, ruta in corridas_en(staging(cfg, t["nombre"])):
            por_corrida.setdefault((v, r), []).append((t["nombre"], ruta))
    filas = []
    for (v, r), fuentes in sorted(por_corrida.items()):
        if en_alcance is not None and (v, r) not in en_alcance:
            continue
        # conflicto: la misma corrida en mas de un trabajador con contenido distinto
        conflicto = None
        if len(fuentes) > 1:
            huellas = {}
            for nombre, ruta in fuentes:
                huellas[nombre] = tuple(vd.hash_archivo(os.path.join(ruta, a)) if os.path.isfile(os.path.join(ruta, a)) else None
                                        for a in ("config.json", "metricas.csv", "series.csv"))
            if len(set(huellas.values())) > 1:
                conflicto = "la corrida esta en %s con contenido distinto" % sorted(huellas)
        for nombre, ruta in fuentes:
            rep = validar_resultado(ruta)
            if ambito == "colab" and not rep.es_completo:
                d = {"ventana": v, "run_name": r, "trabajador": nombre, "ok": False, "estado": "incompleta", "n_errores": 0,
                     "hallazgos": [], "huellas": {}, "resumen": {"regiones": len(rep.regiones_encontradas)},
                     "validado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with open(os.path.join(base_val, "%s__%s__%s.json" % (nombre, v, r)), "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2, ensure_ascii=False)
                filas.append({"trabajador": nombre, "ventana": v, "run_name": r, "estado": "incompleta", "ok": False,
                              "n_errores": 0, "n_avisos": 0, "primer_error": "%d/8 regiones" % len(rep.regiones_encontradas)})
                continue
            previa = leer_json(os.path.join(base_val, "%s__%s__%s.json" % (nombre, v, r)))
            inf = vd.validar_corrida(ruta, v, r, trabajador=nombre, ambiente=ambiente_de(cfg, nombre), ambiente_referencia=amb_ref,
                                     huellas_previas=(previa or {}).get("huellas"))
            if conflicto:
                inf.error("conflicto", conflicto)
            d = inf.a_dict()
            d["estado"] = "validada" if d["ok"] else "con_errores"
            d["validado_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with open(os.path.join(base_val, "%s__%s__%s.json" % (nombre, v, r)), "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            filas.append({"trabajador": nombre, "ventana": v, "run_name": r, "estado": d["estado"], "ok": d["ok"], "n_errores": d["n_errores"],
                          "n_avisos": sum(h["nivel"] == "AVISO" for h in d["hallazgos"]),
                          "primer_error": next((h["mensaje"] for h in d["hallazgos"] if h["nivel"] == "ERROR"), "")})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(base_val, "resumen_validacion.csv"), index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------------------------------------
# asignar
# ---------------------------------------------------------------------------------------------------------

def duraciones_estimadas(cfg):
    """Minutos por (modelo): mediana de los tiempos sincronizados de los trabajadores; por defecto el dato del autor."""
    obs = {}
    for t in cfg["trabajadores"]:
        aud = os.path.join(staging(cfg, t["nombre"]), "_auditoria")
        if not os.path.isdir(aud):
            continue
        for a in os.listdir(aud):
            if a.startswith("tiempos_colab_") and a.endswith(".csv"):
                d = pd.read_csv(os.path.join(aud, a))
                if {"modelo", "duracion_s", "status"} <= set(d.columns):
                    for m, s in d[d["status"] == "completed"].groupby("modelo")["duracion_s"]:
                        obs.setdefault(m, []).extend(s.tolist())
    est = {m: float(pd.Series(v).median()) / 60 for m, v in obs.items()}
    for m, v in DURACION_DEFECTO_MIN.items():
        est.setdefault(m, v)
    return est


def estado_desde_faltan(ruta_csv):
    """Estado del plan de Colab a partir de `faltan_colab.csv` (lo que el principal aun NO tiene completo)."""
    plan = vt.plan_dataframe()
    plan = plan[plan["ejecuta_en"] == "colab"].copy()
    faltan = pd.read_csv(ruta_csv)
    f = {(r["ventana"], r["run_name"]): (r.get("estado", "no_iniciada"), r.get("regiones_ok", 0)) for _, r in faltan.iterrows()}
    plan["estado"] = [f.get((v, n), ("completa", 8))[0] for v, n in zip(plan["ventana"], plan["run_name"])]
    plan["regiones_ok"] = [f.get((v, n), ("completa", 8))[1] for v, n in zip(plan["ventana"], plan["run_name"])]
    return plan


def asignar(cfg, salida, reservar_principal=None, solo_trabajadores=None, familias_pc=None, faltan_csv=None):
    principal = next(t["nombre"] for t in cfg["trabajadores"] if t["rol"] == "principal")
    estado = estado_desde_faltan(faltan_csv) if faltan_csv else vt.estado_plan(staging(cfg, principal), ejecuta="colab")
    ya = estado[estado["estado"] == "completa"]
    parciales = estado[estado["estado"] == "parcial"]
    pend = estado[estado["estado"] != "completa"].copy()
    est = duraciones_estimadas(cfg)
    medio = sum(est.values()) / len(est) if est else 25.0
    pend["min_est"] = pend["modelo"].map(lambda m: est.get(m, medio))
    pend["trabajador"] = ""
    # 1) lo parcial se queda con el principal (ya tiene sus regiones y su base de Optuna)
    pend.loc[pend["estado"] == "parcial", "trabajador"] = principal
    # 2) el lote actual del principal: "T3-fcnn:B" = nivel T3, modelo fcnn, sesion B (S2 y S3)
    if reservar_principal:
        from tesis_forecast import colab_aislado as ca
        for item in reservar_principal:
            lote, sesion = item.split(":")
            nivel, modelo = lote.split("-", 1)
            m = (pend["nivel"] == nivel) & (pend["modelo"] == modelo) & (pend["ventana"].isin(ca.SESIONES[sesion]))
            pend.loc[m & (pend["trabajador"] == ""), "trabajador"] = principal
    # 3) el resto: mas largo primero al trabajador con menos carga (LPT), segun su capacidad
    trabajadores = [t for t in cfg["trabajadores"] if (not solo_trabajadores or t["nombre"] in solo_trabajadores) and t.get("asignable", True)]
    carga = {t["nombre"]: float(pend[pend["trabajador"] == t["nombre"]]["min_est"].sum()) / float(t.get("capacidad", 1.0)) for t in trabajadores}
    for nombre in [principal]:
        carga.setdefault(nombre, float(pend[pend["trabajador"] == nombre]["min_est"].sum()))
    cap = {t["nombre"]: float(t.get("capacidad", 1.0)) for t in trabajadores}
    for idx, fila in pend[pend["trabajador"] == ""].sort_values("min_est", ascending=False).iterrows():
        candidatos = [t["nombre"] for t in trabajadores if not (t["nombre"] == "PC" and familias_pc is not None and fila["modelo"] not in familias_pc)]
        if not candidatos:
            raise SystemExit("no hay trabajadores asignables")
        elegido = min(candidatos, key=lambda n: carga[n])
        pend.at[idx, "trabajador"] = elegido
        carga[elegido] += fila["min_est"] / cap.get(elegido, 1.0)
    # comprobaciones de exclusividad
    assert pend["trabajador"].ne("").all(), "queda alguna configuracion sin asignar"
    assert not set(zip(pend["ventana"], pend["run_name"])) & set(zip(ya["ventana"], ya["run_name"])), "se asigno una corrida ya completa"
    os.makedirs(salida, exist_ok=True)
    escritos = {}
    for nombre, g in pend.groupby("trabajador"):
        g = g.sort_values(["nivel", "ventana", "modelo"])
        ruta = os.path.join(salida, "%s.csv" % nombre)
        g[["ventana", "nivel", "run_name", "modelo", "trabajador", "min_est"]].rename(columns={"min_est": "duracion_estimada_min"}).to_csv(
            ruta, index=False, encoding="utf-8")
        escritos[nombre] = ruta
    resumen = {"generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "total_plan_colab": len(estado),
               "completas_en_principal": len(ya), "parciales_en_principal": len(parciales), "pendientes": len(pend),
               "por_trabajador": {n: {"configuraciones": int((pend["trabajador"] == n).sum()),
                                      "horas_estimadas": round(float(pend[pend["trabajador"] == n]["min_est"].sum()) / 60 / cap.get(n, 1.0), 1)}
                                  for n in sorted(pend["trabajador"].unique())},
               "duraciones_min_por_modelo": {k: round(v, 1) for k, v in est.items()},
               "archivos": {n: {"ruta": os.path.relpath(r, RAIZ), "sha256": dist.hash_archivo(r)} for n, r in escritos.items()}}
    with open(os.path.join(salida, "manifiesto.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    return pend, resumen


# ---------------------------------------------------------------------------------------------------------
# consolidar
# ---------------------------------------------------------------------------------------------------------

def _archivos(ruta):
    return sorted(a for a in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, a)))


def consolidar(cfg, confirmar=False, rclone=None):
    """Copia al Drive principal las corridas validadas de OTROS trabajadores. Nunca sobrescribe; ante diferencias, CONFLICTO."""
    val = os.path.join(BASE, "validacion")
    principal = next(t for t in cfg["trabajadores"] if t["rol"] == "principal")
    destino = principal.get("destino_consolidacion") or raiz_remota(principal)
    plan = []
    for a in sorted(os.listdir(val)) if os.path.isdir(val) else []:
        if not a.endswith(".json") or a.startswith("resumen"):
            continue
        d = leer_json(os.path.join(val, a))
        if not d or d["trabajador"] == principal["nombre"]:
            continue
        v, r, nombre = d["ventana"], d["run_name"], d["trabajador"]
        fila = {"trabajador": nombre, "ventana": v, "run_name": r, "accion": "", "detalle": ""}
        if not d["ok"]:
            fila.update(accion="OMITIDA", detalle="no paso la validacion (%s, %d errores)" % (d.get("estado", "con_errores"), d["n_errores"]))
        else:
            origen = os.path.join(staging(cfg, nombre), v, r)
            existe, iguales = _estado_en_destino(cfg, destino, v, r, origen, rclone)
            if existe and iguales:
                fila.update(accion="YA_CONSOLIDADA", detalle="el destino ya tiene los mismos archivos")
            elif existe:
                fila.update(accion="CONFLICTO", detalle="el destino ya tiene esta corrida y NO coincide: no se toca; requiere decision")
            else:
                fila.update(accion="COPIAR" if confirmar else "COPIARIA", detalle="no existe en el destino")
                if confirmar:
                    _copiar_a_destino(cfg, destino, v, r, origen, d, rclone)
                    fila["accion"] = "COPIADA"
        plan.append(fila)
    df = pd.DataFrame(plan)
    os.makedirs(BASE, exist_ok=True)
    df.to_csv(os.path.join(BASE, "consolidacion_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")), index=False, encoding="utf-8-sig")
    return df


def _estado_en_destino(cfg, destino, v, r, origen, rclone):
    archivos = _archivos(origen)
    if es_local(destino):
        base = os.path.join(destino[len("local:"):], v, r)
        if not os.path.isdir(base):
            return False, False
        return True, all(os.path.isfile(os.path.join(base, a)) and dist.hash_archivo(os.path.join(base, a), "md5") == dist.hash_archivo(os.path.join(origen, a), "md5")
                         for a in archivos) and set(archivos) == set(_archivos(base))
    rc = rclone or ruta_rclone(cfg)
    ls = subprocess.run([rc, "md5sum", "%s/%s/%s" % (destino, v, r)], capture_output=True, text=True)
    if ls.returncode != 0 or not ls.stdout.strip():
        return False, False
    remoto = {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in ls.stdout.splitlines() if l.strip()}
    return True, remoto == {a: dist.hash_archivo(os.path.join(origen, a), "md5") for a in archivos}


def _copiar_a_destino(cfg, destino, v, r, origen, informe, rclone):
    traza = {"consolidado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "origen_trabajador": informe["trabajador"],
             "ventana": v, "run_name": r, "validacion": {"ok": informe["ok"], "hallazgos": informe["hallazgos"]},
             "huellas": informe["huellas"], "commit_repositorio_pc": dist._commit(RAIZ)}
    if es_local(destino):
        base = os.path.join(destino[len("local:"):], v, r)
        shutil.copytree(origen, base)             # falla si ya existe: no sobrescribe
        os.makedirs(os.path.join(destino[len("local:"):], "_consolidacion"), exist_ok=True)
        json.dump(traza, open(os.path.join(destino[len("local:"):], "_consolidacion", "%s__%s.json" % (v, r)), "w", encoding="utf-8"), indent=2)
        return
    rc = rclone or ruta_rclone(cfg)
    subprocess.run([rc, "copy", origen, "%s/%s/%s" % (destino, v, r), "--ignore-existing"], check=True, capture_output=True)
    tmp = os.path.join(BASE, "_traza_tmp.json")
    json.dump(traza, open(tmp, "w", encoding="utf-8"), indent=2)
    subprocess.run([rc, "copyto", tmp, "%s/_consolidacion/%s__%s.json" % (destino, v, r), "--ignore-existing"], check=True, capture_output=True)


# ---------------------------------------------------------------------------------------------------------
# reporte
# ---------------------------------------------------------------------------------------------------------

def reporte(cfg, ruta_asignacion=None):
    ruta_asignacion = ruta_asignacion or os.path.join(RAIZ, "docs", "asignacion")
    val = leer_resumen_validacion()
    estado_all = {}
    lineas = ["# Reporte central del entrenamiento distribuido", "", "Generado: %s UTC" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), ""]
    lineas += ["| Trabajador | Asignadas | Completas (8 regiones) | Parciales | Validadas | Con errores | Pendientes | Horas estimadas restantes |",
               "|---|---|---|---|---|---|---|---|"]
    detalle, fallos = [], []
    est = duraciones_estimadas(cfg)
    for t in cfg["trabajadores"]:
        n = t["nombre"]
        ruta_csv = os.path.join(ruta_asignacion, "%s.csv" % n)
        asign = dist.cargar_asignacion(ruta_csv) if os.path.isfile(ruta_csv) else []
        estado = vt.estado_plan(staging(cfg, n), ejecuta="colab") if os.path.isdir(staging(cfg, n)) else None
        claves = {(a["ventana"], a["run_name"]) for a in asign}
        if estado is not None:
            if claves:
                estado = estado[[(v, r) in claves for v, r in zip(estado["ventana"], estado["run_name"])]]
            completas = int((estado["estado"] == "completa").sum())
            parciales = int((estado["estado"] == "parcial").sum())
            regiones_parciales = int(estado[estado["estado"] == "parcial"]["regiones_ok"].sum())
        else:
            completas = parciales = regiones_parciales = 0
        validadas = int(((val["trabajador"] == n) & (val["estado"] == "validada")).sum()) if len(val) else 0
        con_err = int(((val["trabajador"] == n) & (val["estado"] == "con_errores")).sum()) if len(val) else 0
        total = len(asign) if asign else (len(estado) if estado is not None else 0)
        pendientes = max(total - completas, 0)
        horas = 0.0
        if estado is not None and len(estado):
            pend = estado[estado["estado"] != "completa"]
            horas = sum(est.get(m, 25.0) for m in pend["modelo"]) / 60 / float(t.get("capacidad", 1.0))
        lineas.append("| %s | %d | %d | %d (%d regiones) | %d | %d | %d | %.1f |" % (n, total, completas, parciales, regiones_parciales, validadas, con_err, pendientes, horas))
        reg = os.path.join(staging(cfg, n), "_auditoria", "subprocesos_colab_%s.csv" % n)
        if os.path.isfile(reg):
            d = pd.read_csv(reg)
            f = d[d["clase"] != "ok"]
            fallos += [(n, r["run_name"], r["clase"], str(r.get("mensaje", ""))[:120]) for _, r in f.tail(5).iterrows()]
        ult = max([os.path.getmtime(os.path.join(staging(cfg, n), a)) for a in os.listdir(staging(cfg, n))] or [0]) if os.path.isdir(staging(cfg, n)) else 0
        detalle.append((n, datetime.fromtimestamp(ult).strftime("%Y-%m-%d %H:%M") if ult else "nunca"))
    lineas += ["", "## Ultima sincronizacion", ""] + ["- %s: %s" % x for x in detalle]
    lineas += ["", "## Errores recientes de los trabajadores (ultimos 5 por trabajador)", ""]
    lineas += ["- %s | %s | %s | %s" % x for x in fallos] or ["- ninguno registrado"]
    if len(val):
        lineas += ["", "## Corridas con errores de validacion", ""]
        malas = val[val["estado"] == "con_errores"]
        lineas += ["- %s %s/%s: %s" % (r["trabajador"], r["ventana"], r["run_name"], r["primer_error"]) for _, r in malas.iterrows()] or ["- ninguna"]
    texto = "\n".join(lineas) + "\n"
    os.makedirs(BASE, exist_ok=True)
    with open(os.path.join(BASE, "reporte_central.md"), "w", encoding="utf-8") as f:
        f.write(texto)
    return texto


def leer_resumen_validacion():
    ruta = os.path.join(BASE, "validacion", "resumen_validacion.csv")
    return pd.read_csv(ruta) if os.path.isfile(ruta) else pd.DataFrame(columns=["trabajador", "ventana", "run_name", "estado", "ok", "n_errores", "n_avisos", "primer_error"])


# ---------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("accion", choices=["asignar", "sync", "validar", "consolidar", "reporte", "ciclo"])
    ap.add_argument("--config", default=RUTA_CONFIG)
    ap.add_argument("--solo", nargs="*", help="limitar sync/validar a estos trabajadores")
    ap.add_argument("--confirmar", action="store_true", help="consolidar: copiar de verdad (sin esto es un ensayo)")
    ap.add_argument("--reservar-principal", nargs="*", default=[], help="lotes que se quedan en el principal, p. ej. T3-fcnn:B")
    ap.add_argument("--familias-pc", nargs="*", help="modelos que puede recibir el trabajador PC (si no, la PC no recibe nada)")
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "asignacion"))
    ap.add_argument("--cada", type=float, help="ciclo: repetir cada N minutos")
    ap.add_argument("--faltan-csv", help="asignar: usar faltan_colab.csv del principal en vez de su staging")
    args = ap.parse_args(argv)
    cfg = cargar_config(args.config)
    if args.accion == "asignar":
        pend, res = asignar(cfg, args.salida, args.reservar_principal or None, None, args.familias_pc if args.familias_pc is not None else [], args.faltan_csv)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.accion in ("sync", "ciclo"):
        while True:
            for t in cfg["trabajadores"]:
                if args.solo and t["nombre"] not in args.solo:
                    continue
                try:
                    n = sincronizar(cfg, t)
                    print("[%s] sync %s: %s" % (datetime.now().strftime("%H:%M:%S"), t["nombre"], "ok" if n is None else "%d archivos nuevos o cambiados" % n))
                except Exception as e:  # un trabajador caido no detiene a los demas
                    print("[%s] sync %s FALLO: %s" % (datetime.now().strftime("%H:%M:%S"), t["nombre"], str(e)[:200]))
            if args.accion == "ciclo":
                df = validar_todo(cfg)
                print("validadas: %d, con errores: %d, incompletas: %d" % tuple(int((df["estado"] == e).sum()) if len(df) else 0 for e in ("validada", "con_errores", "incompleta")))
                reporte(cfg)
            if not args.cada:
                break
            time.sleep(args.cada * 60)
        return 0
    if args.accion == "validar":
        df = validar_todo(cfg)
        print(df.to_string(index=False) if len(df) else "no hay corridas en staging")
        return 0
    if args.accion == "consolidar":
        df = consolidar(cfg, args.confirmar)
        print(df.to_string(index=False) if len(df) else "nada que consolidar")
        return 0
    if args.accion == "reporte":
        print(reporte(cfg))
        return 0


if __name__ == "__main__":
    sys.exit(main())

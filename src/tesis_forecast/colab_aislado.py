"""
Ejecucion aislada de las corridas de Colab: UNA configuracion = UN subproceso.

Por que: las cuatro redes (FCNN, LSTM directa, LSTM residual y Ensemble STL) se ejecutaban todas en el mismo proceso
Python del cuaderno (hasta 24 corridas seguidas). `clear_session()` no devuelve al sistema operativo la memoria nativa
de TensorFlow, y si el kernel moria por falta de RAM se perdia la celda entera sin dejar rastro. Con un subproceso por
configuracion:
  - al terminar cada corrida el sistema operativo recupera TODA su memoria;
  - si el OOM-killer mata a un subproceso, el cuaderno sobrevive, lo registra y continua;
  - se distingue un fallo de RAM de uno de GPU o de codigo (ver `clasificar_fallo`);
  - los reintentos estan acotados y cada reintento reanuda por region (checkpoint), sin repetir regiones completas.

NO cambia hiperparametros, ventanas, datos ni metricas: cada subproceso llama al mismo `run_matrix` de siempre.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from . import io_drive, telemetria
from . import ventanas_tools as vt
from .atomic_io import escribir_csv_atomico
from .config import ExperimentConfig
from .matrix import MatrixRunRecord, resumen_dataframe
from .runner import resolve_run
from .validator import validar_resultado

SESIONES = {"A": ["S4", "S1"], "B": ["S2", "S3"]}
MODELOS_COLAB = ("fcnn", "lstm_resid", "ensemble_stl", "lstm_direct")
NIVELES = ("T1", "T2", "T3")
LOTES_VALIDOS = ["TODO", "T2", "T3"] + ["T3-" + m for m in MODELOS_COLAB]
CONTEXTO_ACTUAL = {"txt": ""}  # corrida en curso (lo lee la telemetria)
SCRIPT_HIJO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "run_config_aislada.py"))

# Clases de fallo
RAM_SISTEMA = "ram_sistema"          # el subproceso murio por SIGKILL con la RAM del sistema agotada (OOM-killer)
RAM_PYTHON = "ram_python"            # MemoryError / std::bad_alloc / "Cannot allocate memory"
GPU_MEMORIA = "gpu_memoria"          # ResourceExhaustedError / CUDA OOM (la sesion sigue viva, hay traceback)
SIGKILL_OTRO = "sigkill_otro"        # SIGKILL con RAM holgada (alguien/algo mato el proceso)
INTERRUMPIDO = "interrumpido"        # SIGTERM / SIGINT
FALLO_NATIVO = "fallo_nativo"        # SIGSEGV / SIGABRT en una libreria nativa
TIMEOUT = "timeout"
INCOMPLETA = "incompleta"            # el subproceso termino bien pero la corrida no quedo con las 8 regiones
ERROR_CODIGO = "error_codigo"        # excepcion de Python distinta de memoria
CLASES_RAM = {RAM_SISTEMA, RAM_PYTHON}

_PATRONES_GPU = [
    "ResourceExhaustedError", "OOM when allocating tensor", "CUDA_ERROR_OUT_OF_MEMORY", "CUDA out of memory",
    "Out of memory while trying to allocate", "failed to allocate memory on device",
]
_PATRONES_RAM_PYTHON = ["MemoryError", "std::bad_alloc", "Cannot allocate memory", "bad allocation", "Unable to allocate"]


def clasificar_fallo(returncode, texto="", ram_disp_min_gb=None, timeout=False):
    """Clasifica por que fallo un subproceso. Funcion pura (se prueba sin Colab).

    returncode: codigo de salida (negativo en POSIX = senal; 137 = 128 + SIGKILL).
    texto: cola del log del subproceso y/o la excepcion registrada.
    ram_disp_min_gb: minimo de RAM disponible del sistema observado mientras corria (None = sin dato).
    """
    if timeout:
        return TIMEOUT
    texto = texto or ""
    if returncode in (-9, 137):
        if ram_disp_min_gb is None or ram_disp_min_gb < 2.0:
            return RAM_SISTEMA
        return SIGKILL_OTRO
    if any(p.lower() in texto.lower() for p in _PATRONES_GPU):
        return GPU_MEMORIA
    if any(p.lower() in texto.lower() for p in _PATRONES_RAM_PYTHON):
        return RAM_PYTHON
    if returncode in (-15, 143, -2, 130):
        return INTERRUMPIDO
    if returncode in (-11, 139, -6, 134):
        return FALLO_NATIVO
    return ERROR_CODIGO


# ---------------------------------------------------------------------------
# Cola de trabajo (sesion + lote)
# ---------------------------------------------------------------------------

def construir_cola(sesion, lote="TODO"):
    """Lista de (nivel, ventana, [ExperimentConfig]) de la sesion A o B, filtrada por lote.

    lote: "TODO" | "T2" | "T3" | "T3-fcnn" | "T3-lstm_resid" | "T3-ensemble_stl" | "T3-lstm_direct".
    Orden (el de siempre): T2 antes que T3 y, dentro de cada nivel, una semana tras otra.
    """
    if sesion not in SESIONES:
        raise ValueError("sesion debe ser 'A' o 'B'")
    if lote not in LOTES_VALIDOS:
        raise ValueError("lote debe ser uno de %s" % LOTES_VALIDOS)
    modelo_lote = lote.split("-", 1)[1] if lote.startswith("T3-") else None
    nivel_lote = None if lote == "TODO" else lote.split("-", 1)[0]
    cola = []
    for nivel in NIVELES:
        if nivel_lote is not None and nivel != nivel_lote:
            continue
        for v in SESIONES[sesion]:
            configs = vt.filtrar_por_entorno(vt.construir_configs(nivel, v), "colab")
            if modelo_lote is not None:
                configs = [c for c in configs if c.modelo == modelo_lote]
            if configs:
                cola.append((nivel, v, configs))
    return cola


# ---------------------------------------------------------------------------
# Un subproceso
# ---------------------------------------------------------------------------

def _regiones_ok(run_dir):
    return len(validar_resultado(run_dir).regiones_encontradas) if os.path.isdir(run_dir) else 0


def _comando_hijo(python, cfg_json, resultado_json, data_dir, base_ventanas, stubs):
    cmd = [python, "-u", SCRIPT_HIJO, "--config-json", cfg_json, "--resultado", resultado_json,
           "--data-dir", data_dir, "--base-ventanas", base_ventanas]
    if stubs:
        cmd.append("--stubs-pesados")
    return cmd


def _num(x):
    return 0.0 if x != x else float(x)


def _cola_texto(ruta, n=6000):
    try:
        with open(ruta, "rb") as f:
            f.seek(0, os.SEEK_END)
            tam = f.tell()
            f.seek(max(0, tam - n))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminar_arbol(proc, forzar):
    """Termina al subproceso Y a todos sus descendientes (en Windows el python de un entorno virtual es un
    lanzador que crea otro proceso: matar solo al lanzador dejaria huerfano al que de verdad trabaja)."""
    import signal

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL if forzar else signal.SIGTERM)
        else:
            subprocess.run(["taskkill", "/T"] + (["/F"] if forzar else []) + ["/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass
    if forzar:
        try:
            proc.kill()
        except OSError:
            pass


def _lanzar_hijo(cmd, log_path, env, timeout_s, sondeo_s=5):
    """Ejecuta el subproceso y lo vigila. Devuelve (returncode, timeout, pico_hijo_mb, ram_disp_min_gb)."""
    pico_mb, ram_min = 0.0, float("inf")
    timeout = False
    popen_kw = {"start_new_session": True} if os.name == "posix" else {}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, **popen_kw)
        t0 = time.time()
        try:
            while True:
                try:
                    proc.wait(timeout=sondeo_s)
                    break
                except subprocess.TimeoutExpired:
                    pass
                m = telemetria.snapshot(proc.pid, gpu=False)
                pico_mb = max(pico_mb, _num(m["rss_proceso_mb"]) + _num(m["rss_hijos_mb"]))
                if m["ram_disponible_gb"] == m["ram_disponible_gb"]:
                    ram_min = min(ram_min, m["ram_disponible_gb"])
                if timeout_s is not None and time.time() - t0 > timeout_s:
                    timeout = True
                    _terminar_arbol(proc, forzar=True)
                    proc.wait()
                    break
        except KeyboardInterrupt:  # el usuario detuvo la celda: no dejar el subproceso ocupando RAM/GPU
            _terminar_arbol(proc, forzar=False)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _terminar_arbol(proc, forzar=True)
                proc.wait()
            raise
    return proc.returncode, timeout, pico_mb, (None if ram_min == float("inf") else ram_min)


def registrar_intento(ruta_csv, fila):
    telemetria.agregar_fila(ruta_csv, fila)


def ejecutar_config_aislada(
    config, *, data_dir, vent_dir, sesion="A", max_intentos=3, max_sin_progreso=2, timeout_s=None,
    python=None, stubs=False, allow_growth_gpu=True, tmp_dir=None, ruta_registro=None, verbose=True,
    comando_hijo=None, sondeo_s=5,
):
    """Corre UNA configuracion en un subproceso, con reintentos acotados que reanudan por region.

    - Si la corrida ya esta completa no se lanza nada.
    - Se detiene tras `max_intentos` intentos, o tras `max_sin_progreso` intentos seguidos que no agregaron ninguna
      region nueva (evita reintentos infinitos ante un error determinista).
    - Devuelve (MatrixRunRecord, clase_final) donde clase_final es None si termino bien.
    """
    python = python or sys.executable
    tmp_dir = tmp_dir or os.path.join(tempfile.gettempdir(), "subprocesos_tesis")
    os.makedirs(tmp_dir, exist_ok=True)
    ruta_registro = ruta_registro or os.path.join(vent_dir, "_auditoria", "subprocesos_colab_%s.csv" % sesion)

    try:
        resolved = resolve_run(config)
    except Exception as e:  # configuracion invalida
        return MatrixRunRecord(config, None, "failed", None, "%s: %s" % (type(e).__name__, e), None), ERROR_CODIGO

    run_dir = resolved.run_dir
    nombre = resolved.run_name
    if os.path.isdir(run_dir) and validar_resultado(run_dir).es_completo:
        if verbose:
            print("  YA COMPLETO, se salta: %s/%s" % (config.ventana, nombre))
        return MatrixRunRecord(config, nombre, "skipped_ya_completo", None, None, run_dir), None

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    if allow_growth_gpu:
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # solo cambia COMO se reserva la memoria de GPU, no los calculos

    t_total = time.time()
    antes = _regiones_ok(run_dir)
    regiones_iniciales = antes
    sin_progreso, clase, ultimo_msg = 0, None, ""
    for intento in range(1, max_intentos + 1):
        base = "%s__%s__intento%d" % (config.ventana, nombre, intento)
        cfg_json = os.path.join(tmp_dir, base + ".config.json")
        res_json = os.path.join(tmp_dir, base + ".resultado.json")
        log_path = os.path.join(tmp_dir, base + ".log")
        with open(cfg_json, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f)
        if os.path.exists(res_json):
            os.remove(res_json)

        t0 = time.time()
        if verbose:
            print("  [%s] %s/%s intento %d/%d (regiones ya guardadas: %d/8)" % (
                datetime.now().strftime("%H:%M:%S"), config.ventana, nombre, intento, max_intentos, antes))
        CONTEXTO_ACTUAL["txt"] = "%s/%s intento %d" % (config.ventana, nombre, intento)
        if comando_hijo is not None:  # solo pruebas: sustituye al script real
            cmd = comando_hijo(cfg_json, res_json, data_dir, vent_dir)
        else:
            cmd = _comando_hijo(python, cfg_json, res_json, data_dir, vent_dir, stubs)
        try:
            rc, timeout, pico_mb, ram_min = _lanzar_hijo(cmd, log_path, env, timeout_s, sondeo_s)
        except OSError as e:  # no se pudo ni lanzar el subproceso
            rc, timeout, pico_mb, ram_min = 127, False, 0.0, None
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("No se pudo lanzar el subproceso: %s" % e)

        resultado = {}
        if os.path.exists(res_json):
            try:
                with open(res_json, encoding="utf-8") as f:
                    resultado = json.load(f)
            except (OSError, ValueError):
                resultado = {}
        texto = _cola_texto(log_path) + "\n" + str(resultado.get("traceback_cola", ""))
        despues = _regiones_ok(run_dir)
        completa = os.path.isdir(run_dir) and validar_resultado(run_dir).es_completo

        err = resultado.get("errores_region") or {}
        primero = err.get("primero") or {}
        texto_err = " ".join(str(primero.get(k, "")) for k in ("tipo", "mensaje", "traza_cola"))
        if rc == 0 and completa:
            clase = None
            ultimo_msg = ""
        elif rc in (0, 3):  # termino sin morir pero la corrida no tiene las 8 regiones
            # un error tragado por el modelo puede ser de memoria: se distingue para no esconder un fallo de GPU/RAM
            causa = clasificar_fallo(1, texto_err)
            clase = causa if causa in (GPU_MEMORIA, RAM_PYTHON) else INCOMPLETA
            ultimo_msg = (err.get("texto") or "el subproceso termino pero la corrida tiene %d/8 regiones" % despues)[:400]
        else:
            clase = clasificar_fallo(rc, texto + " " + texto_err, ram_min, timeout)
            lineas = [l for l in texto.strip().splitlines() if l.strip()]
            ultimo_msg = (err.get("texto") or resultado.get("error") or (lineas[-1] if lineas else "sin salida"))[:400]

        registrar_intento(ruta_registro, {
            "registrado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sesion": sesion, "ventana": config.ventana, "modelo": config.modelo, "run_name": nombre,
            "intento": intento, "returncode": rc, "clase": clase or "ok",
            "duracion_s": round(time.time() - t0, 1), "regiones_antes": antes, "regiones_despues": despues,
            "pico_rss_hijo_mb": round(pico_mb), "ram_disponible_min_gb": None if ram_min is None else round(ram_min, 2),
            "pico_rss_hijo_reportado_mb": resultado.get("pico_rss_mb"), "log": log_path, "mensaje": ultimo_msg,
        })
        if verbose:
            print("      -> %s | rc=%s | regiones %d->%d | %.0f s | pico %.0f MB%s" % (
                clase or "OK", rc, antes, despues, time.time() - t0, pico_mb, (" | " + ultimo_msg) if ultimo_msg else ""))
            if clase is not None and primero.get("traza_cola"):
                print("      traza del primer error (completa en %s/log.txt):" % run_dir)
                for linea_traza in str(primero["traza_cola"]).splitlines():
                    print("        " + linea_traza)

        CONTEXTO_ACTUAL["txt"] = ""
        if clase is None:
            return MatrixRunRecord(config, nombre, "completed", resultado.get("mape_promedio"), None, run_dir,
                                   duracion_s=time.time() - t_total), None

        sin_progreso = 0 if despues > antes else sin_progreso + 1
        # fallo determinista: sin avanzar y varias regiones con exactamente el mismo error -> repetir no sirve
        determinista = despues == antes and err.get("n", 0) >= 2 and err.get("mensajes_unicos") == 1
        antes = despues
        if determinista:
            if verbose:
                print("      las %d regiones fallaron con el mismo error y sin avanzar: se abandona sin reintentar (fallo determinista)." % err["n"])
            break
        if sin_progreso >= max_sin_progreso:
            if verbose:
                print("      %d intentos seguidos sin avanzar: se abandona esta configuracion (se puede relanzar despues)." % sin_progreso)
            break

    error = "%s: %s (intentos=%d, regiones %d->%d)" % (clase, ultimo_msg, intento, regiones_iniciales, antes)
    return MatrixRunRecord(config, nombre, "failed", None, error, run_dir, duracion_s=time.time() - t_total), clase


# ---------------------------------------------------------------------------
# Cola completa de una sesion
# ---------------------------------------------------------------------------

def ejecutar_cola(cola, *, data_dir, vent_dir, sesion="A", max_intentos=3, max_sin_progreso=2,
                  max_fallos_ram_consecutivos=3, timeout_s=None, ruta_tiempos=None, **kw):
    """Recorre la cola; cada configuracion en su subproceso. Devuelve (registros, motivo_de_parada|None).

    Se detiene antes de terminar si `max_fallos_ram_consecutivos` configuraciones seguidas fallan por RAM: el entorno
    no alcanza y seguir solo gastaria horas (hay que reiniciar el entorno / usar mas RAM / un lote mas chico).
    """
    ruta_tiempos = ruta_tiempos or os.path.join(vent_dir, "_auditoria", "tiempos_colab_%s.csv" % sesion)
    registros, ram_seguidos, motivo = [], 0, None
    for nivel, ventana, configs in cola:
        print("\n######## %s - %s: %d configuraciones ########" % (nivel, ventana, len(configs)))
        for i, config in enumerate(configs, 1):
            print("\n# %s %s %d/%d: %s" % (nivel, ventana, i, len(configs), config.modelo))
            registro, clase = ejecutar_config_aislada(
                config, data_dir=data_dir, vent_dir=vent_dir, sesion=sesion, max_intentos=max_intentos,
                max_sin_progreso=max_sin_progreso, timeout_s=timeout_s, **kw)
            registros.append(registro)
            try:
                vt.guardar_tiempos([registro], ruta_tiempos)
            except OSError as e:
                print("      aviso: no se pudo guardar el tiempo (%s)" % e)
            ram_seguidos = ram_seguidos + 1 if clase in CLASES_RAM else 0
            if ram_seguidos >= max_fallos_ram_consecutivos:
                motivo = ("%d configuraciones seguidas fallaron por RAM. Detengo la cola. Reinicia el entorno de "
                          "ejecucion, usa RAM alta o ejecuta un lote mas pequeno (variable LOTE)." % ram_seguidos)
                print("\n*** " + motivo)
                return registros, motivo
    return registros, motivo


def ruta_telemetria(vent_dir, sesion):
    pref = os.path.join(vent_dir, "_auditoria", "telemetria_colab_%s.csv" % sesion)
    alt = os.path.join(tempfile.gettempdir(), "telemetria_colab_%s.csv" % sesion)
    return telemetria.resolver_ruta(pref, alt)


def ejecutar_sesion(sesion, lote="TODO", *, vent_dir, data_dir, max_intentos=3, intervalo_telemetria_s=120, **kw):
    """Punto de entrada de la celda de lanzamiento: cola + telemetria + resumen."""
    cola = construir_cola(sesion, lote)
    total = sum(len(c) for _, _, c in cola)
    ruta_tel = ruta_telemetria(vent_dir, sesion)
    print("Sesion %s - lote %s - %d corridas asignadas a Colab (las ya completas se saltan)" % (sesion, lote, total))
    print("Telemetria de memoria (RAM/GPU): %s" % ruta_tel)
    monitor = telemetria.Monitor(ruta_tel, intervalo_s=intervalo_telemetria_s, etiqueta=lambda: CONTEXTO_ACTUAL["txt"]).iniciar()
    try:
        registros, motivo = ejecutar_cola(cola, data_dir=data_dir, vent_dir=vent_dir, sesion=sesion,
                                          max_intentos=max_intentos, **kw)
    finally:
        monitor.detener()
    resumen = resumen_dataframe(registros) if registros else pd.DataFrame()
    if len(resumen):
        print("\n" + str(resumen["status"].value_counts()))
    return resumen, motivo


# ---------------------------------------------------------------------------
# Informe de progreso (solo lectura)
# ---------------------------------------------------------------------------

def informe_progreso(vent_dir, ver=True):
    """Estado de las 112 corridas de Colab por sesion, faltantes, fallos recientes y memoria. No modifica resultados."""
    estado = vt.estado_plan(vent_dir, ejecuta="colab")
    estado["sesion"] = estado["ventana"].map({v: s for s, vs in SESIONES.items() for v in vs})
    por_sesion = estado.groupby(["sesion", "estado"]).size().unstack(fill_value=0)
    for col in ("completa", "parcial", "no_iniciada"):
        if col not in por_sesion.columns:
            por_sesion[col] = 0
    por_sesion = por_sesion[["completa", "parcial", "no_iniciada"]]
    faltan = estado[estado["estado"] != "completa"][
        ["sesion", "ventana", "nivel", "modelo", "run_name", "estado", "regiones_ok"]].reset_index(drop=True)

    # Intentos de Optuna por region (esperado: 10 en LSTM directa, 5 en las demas). Detecta presupuestos excedidos.
    anomalias = []
    for _, fila in estado.iterrows():
        ruta = os.path.join(vent_dir, fila["ventana"], fila["run_name"], "trials.csv")
        if not os.path.exists(ruta):
            continue
        try:
            t = pd.read_csv(ruta, encoding="utf-8-sig")
            por_region = t.groupby("serie").size() if "serie" in t.columns else pd.Series(dtype=int)
        except Exception:
            continue
        esperado = int(fila["optuna_n_trials"]) if "optuna_n_trials" in estado.columns and pd.notna(fila.get("optuna_n_trials")) else None
        if esperado is None:
            continue
        # FCNN/Ensemble guardan mas de una fila por intento (varios estudios por region): solo se vigila lstm_direct
        if fila["modelo"] == "lstm_direct":
            exceso = por_region[por_region != esperado]
            for serie, n in exceso.items():
                anomalias.append({"ventana": fila["ventana"], "run_name": fila["run_name"], "serie": serie, "trials": int(n), "esperado": esperado})
    anomalias = pd.DataFrame(anomalias, columns=["ventana", "run_name", "serie", "trials", "esperado"])

    fallos = []
    tele = {}
    audit = os.path.join(vent_dir, "_auditoria")
    for s in SESIONES:
        f = os.path.join(audit, "subprocesos_colab_%s.csv" % s)
        if os.path.exists(f):
            d = pd.read_csv(f)
            fallos.append(d[d["clase"] != "ok"].tail(10))
        t = os.path.join(audit, "telemetria_colab_%s.csv" % s)
        if os.path.exists(t):
            tele[s] = pd.read_csv(t).tail(3)
    fallos = pd.concat(fallos, ignore_index=True) if fallos else pd.DataFrame()

    if ver:
        print("Corridas de Colab completas: %d de %d\n" % ((estado["estado"] == "completa").sum(), len(estado)))
        print(por_sesion.to_string())
        print("\nFaltan %d corridas (primeras 25):" % len(faltan))
        print(faltan.head(25).to_string())
        if len(anomalias):
            print("\nAVISO: LSTM directa con un numero de intentos de Optuna distinto del presupuesto (10):")
            print(anomalias.to_string())
        if len(fallos):
            print("\nUltimos fallos de subproceso:")
            print(fallos[["registrado_utc", "sesion", "ventana", "run_name", "intento", "clase", "regiones_antes", "regiones_despues", "mensaje"]].to_string())
        for s, d in tele.items():
            print("\nTelemetria (ultimas muestras) sesion %s:" % s)
            print(d.to_string())
    return {"estado": estado, "por_sesion": por_sesion, "faltan": faltan, "anomalias": anomalias, "fallos": fallos, "telemetria": tele}


def guardar_faltantes(vent_dir, faltan):
    ruta = os.path.join(vent_dir, "_auditoria", "faltan_colab.csv")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    escribir_csv_atomico(faltan, ruta, index=False, encoding="utf-8")
    return ruta

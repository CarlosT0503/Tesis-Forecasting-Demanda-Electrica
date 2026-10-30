"""
Supervisor de la cola local de experimentos (PC). Corre en segundo plano y no
depende de ninguna sesion de chat.

Prioridad de la cola (cada trabajo = una ventana x nivel x modelo, reanudable):
  1. XGBoost y LightGBM, T1 (S4, S1, S2, S3)
  2. SARIMAX T1 (S4, S1, S2, S3)
  3. XGBoost y LightGBM, T3
  4. SARIMAX T3          <- arranca solo cuando ya no queda nada mas prioritario

Reglas de estabilidad:
  - 1 hilo de BLAS/OpenMP por proceso (lo fija run_local_estadisticos.py).
  - Maximo `--max-workers` procesos a la vez (por defecto 3) y como mucho 2 SARIMAX
    mientras queden trabajos de arboles.
  - No lanza trabajos nuevos si la RAM libre < `--min-ram-gb`, si la PC va con
    bateria por debajo de `--min-bateria` %, o si hubo un fallo reciente.
  - Mantiene la PC despierta (SetThreadExecutionState) mientras todo este sano y
    deja de hacerlo si salta una de esas protecciones. No cambia ajustes de energia.
  - Un fallo se reintenta hasta 2 veces; despues se salta. Nunca borra ni
    sobrescribe resultados: cada trabajo reanuda desde el ultimo checkpoint.

Estado en vivo: resultados_locales/cola/estado_cola.json y supervisor.log.
Detener: terminar el proceso del supervisor (los workers reanudan luego sin perder lo guardado).
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from unittest import mock

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for _n in [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.models", "tensorflow.keras.layers",
    "tensorflow.keras.callbacks", "tensorflow.keras.optimizers", "tensorflow.keras.backend",
    "optuna", "optuna.samplers", "xgboost", "lightgbm",
]:
    try:
        __import__(_n)
    except ImportError:
        sys.modules[_n] = mock.MagicMock()

from tesis_forecast import ventanas_tools as vt  # noqa: E402

BASE = os.path.join(RAIZ, "resultados_locales", "Pipeline_Resultados_Ventanas")
DIR_COLA = os.path.join(RAIZ, "resultados_locales", "cola")
VENTANAS = ["S4", "S1", "S2", "S3"]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]


def ram_libre_gb() -> float:
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 1024 ** 3


def energia():
    """(en_corriente, % de bateria)."""
    p = _PowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))
    pct = p.BatteryLifePercent if p.BatteryLifePercent <= 100 else 100
    return p.ACLineStatus == 1, int(pct)


def log(msg):
    linea = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    print(linea, flush=True)
    with open(os.path.join(DIR_COLA, "supervisor.log"), "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def construir_trabajos():
    """Lista ordenada por prioridad de (clase, nivel, ventana, modelo)."""
    trabajos = []
    for v in VENTANAS:
        for m in ("xgboost", "lightgbm"):
            trabajos.append(("arboles", "T1", v, m))
    for v in VENTANAS:
        trabajos.append(("sarimax", "T1", v, "sarimax"))
    for v in VENTANAS:
        for m in ("xgboost", "lightgbm"):
            trabajos.append(("arboles", "T3", v, m))
    for v in VENTANAS:
        trabajos.append(("sarimax", "T3", v, "sarimax"))
    return trabajos


def pendiente(nivel, ventana, modelo) -> bool:
    """True si alguna corrida del trabajo no esta completa."""
    plan = vt.estado_plan(BASE, ejecuta="local", ventanas=[ventana])
    sel = plan[(plan["nivel"] == nivel) & (plan["modelo"] == modelo)]
    return bool((sel["estado"] != "completa").any())


def lanzar(py, nivel, ventana, modelo):
    nombre = "%s_%s_%s" % (modelo, nivel, ventana)
    out = open(os.path.join(DIR_COLA, nombre + ".log"), "a", encoding="utf-8")
    err = open(os.path.join(DIR_COLA, nombre + ".err"), "a", encoding="utf-8")
    cmd = [py, os.path.join("scripts", "run_local_estadisticos.py"), "--ventanas", ventana,
           "--niveles", nivel, "--modelos", modelo]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.Popen(cmd, cwd=RAIZ, stdout=out, stderr=err, creationflags=flags)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--max-sarimax-con-arboles", type=int, default=2)
    ap.add_argument("--min-ram-gb", type=float, default=1.2)
    ap.add_argument("--ram-margen-worker-gb", type=float, default=0.8)
    ap.add_argument("--min-bateria", type=int, default=30)
    ap.add_argument("--python", default=os.path.join(RAIZ, ".venv_local", "Scripts", "python.exe"))
    args = ap.parse_args()

    os.makedirs(DIR_COLA, exist_ok=True)
    pid_file = os.path.join(DIR_COLA, "supervisor.pid")
    if os.path.exists(pid_file):
        try:
            viejo = int(open(pid_file).read().strip())
            if subprocess.run(["tasklist", "/FI", "PID eq %d" % viejo], capture_output=True, text=True).stdout.count(str(viejo)):
                print("Ya hay un supervisor activo (PID %d). Salgo." % viejo)
                return 1
        except Exception:
            pass
    open(pid_file, "w").write(str(os.getpid()))

    cola = construir_trabajos()
    hechos, en_curso, fallos = set(), {}, {}
    log("Supervisor iniciado (PID %d): max_workers=%d, RAM libre %.1f GB" % (os.getpid(), args.max_workers, ram_libre_gb()))

    while True:
        # 1) cerrar trabajos terminados
        for clave in list(en_curso):
            p, t0 = en_curso[clave]
            rc = p.poll()
            if rc is None:
                continue
            del en_curso[clave]
            dur = (time.time() - t0) / 60
            if rc == 0:
                hechos.add(clave)
                log("TERMINADO %s (%.1f min)" % (clave, dur))
            else:
                fallos[clave] = fallos.get(clave, 0) + 1
                log("FALLO rc=%s %s (%.1f min), intento %d/2" % (rc, clave, dur, fallos[clave]))
                if fallos[clave] >= 3:
                    hechos.add(clave)
                    log("ABANDONADO %s tras 3 intentos (se puede relanzar luego)" % (clave,))

        # 2) proteccion y lanzamiento
        sano, motivo = True, ""
        ram = ram_libre_gb()
        ac, bat = energia()
        if ram < args.min_ram_gb:
            sano, motivo = False, "RAM libre %.1f GB < %.1f" % (ram, args.min_ram_gb)
        elif (not ac) and bat < args.min_bateria:
            sano, motivo = False, "bateria %d%% sin corriente" % bat
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if sano else 0))

        pendientes_arboles = any(c[0] == "arboles" and c not in hechos and c not in en_curso for c in cola)
        n_sar = sum(1 for c in en_curso if c[0] == "sarimax")
        tope_sar = args.max_sarimax_con_arboles if (pendientes_arboles or any(c[0] == "arboles" for c in en_curso)) else args.max_workers

        while sano and len(en_curso) < args.max_workers:
            # cada worker adicional exige RAM libre de sobra (el suelo + el margen de un worker),
            # medida despues de que arranco el anterior; asi no se fuerza un tercer worker sin memoria
            ram_ahora = ram_libre_gb()
            if len(en_curso) >= 1 and ram_ahora < args.min_ram_gb + args.ram_margen_worker_gb:
                break
            elegido = None
            for clave in cola:
                if clave in hechos or clave in en_curso:
                    continue
                if clave[0] == "sarimax" and n_sar >= tope_sar:
                    continue
                if not pendiente(clave[1], clave[2], clave[3]):
                    hechos.add(clave)
                    log("OMITIDO %s (ya completo)" % (clave,))
                    continue
                elegido = clave
                break
            if elegido is None:
                break
            en_curso[elegido] = (lanzar(args.python, elegido[1], elegido[2], elegido[3]), time.time())
            if elegido[0] == "sarimax":
                n_sar += 1
            log("LANZADO %s (workers=%d, RAM libre %.1f GB)" % (elegido, len(en_curso), ram_libre_gb()))
            time.sleep(20)  # escalonar arranques para no pisar la RAM

        estado = {
            "actualizado": datetime.now().isoformat(timespec="seconds"),
            "en_curso": {"%s/%s/%s" % (k[1], k[2], k[3]): {"pid": v[0].pid, "min": round((time.time() - v[1]) / 60, 1)} for k, v in en_curso.items()},
            "terminados": len([h for h in hechos]), "total_trabajos": len(cola),
            "ram_libre_gb": round(ram, 2), "en_corriente": ac, "bateria_pct": bat,
            "pausado_por": None if sano else motivo,
        }
        with open(os.path.join(DIR_COLA, "estado_cola.json"), "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=1)

        if not en_curso and all(c in hechos for c in cola):
            log("COLA COMPLETA")
            break
        if not en_curso and not sano:
            log("En pausa: %s" % motivo)
        time.sleep(30)

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    os.remove(pid_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

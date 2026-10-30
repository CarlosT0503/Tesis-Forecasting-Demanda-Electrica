"""
Pruebas de la telemetria de RAM/GPU: muestras, respaldo sin psutil, registro persistente y ruta de Drive.

    python tests/test_telemetria.py
"""

import math
import os
import shutil
import sys
import tempfile
import time
from unittest import mock

import pandas as pd

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))

from tesis_forecast import telemetria  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def main():
    tmp = tempfile.mkdtemp(prefix="test_tele_")
    try:
        print("1. snapshot")
        s = telemetria.snapshot()
        check(set(telemetria.COLUMNAS) - {"contexto"} <= set(s), "trae todas las columnas")
        check(s["ram_total_gb"] > 0 and 0 <= s["ram_disponible_gb"] <= s["ram_total_gb"], "RAM total/disponible coherentes (%.1f/%.1f GB)" % (s["ram_total_gb"], s["ram_disponible_gb"]))
        check(s["rss_proceso_mb"] > 10, "RSS del proceso (%s MB)" % s["rss_proceso_mb"])
        check(all(math.isnan(s[k]) for k in ("gpu_usada_mb", "gpu_total_mb")) or s["gpu_total_mb"] > 0, "GPU: NaN sin nvidia-smi o valores reales con GPU")
        check(math.isnan(telemetria.snapshot(gpu=False)["gpu_usada_mb"]), "gpu=False omite nvidia-smi")

        print("2. Respaldo sin psutil (como en un entorno minimo)")
        with mock.patch.object(telemetria, "psutil", None):
            s2 = telemetria.snapshot()
        check(s2["ram_total_gb"] > 0 and s2["ram_disponible_gb"] > 0, "RAM sin psutil: %.1f/%.1f GB" % (s2["ram_total_gb"], s2["ram_disponible_gb"]))

        print("3. Se mide el arbol de procesos hijos")
        import subprocess

        hijo = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(80*1024*1024); time.sleep(6)"])
        time.sleep(2.5)
        m = telemetria.snapshot(os.getpid())
        hijo.kill()
        hijo.wait()
        if telemetria.psutil is not None:
            check(m["rss_hijos_mb"] >= 60, "rss_hijos_mb refleja al subproceso (%s MB)" % m["rss_hijos_mb"])
        else:
            check(math.isnan(m["rss_hijos_mb"]) or m["rss_hijos_mb"] >= 0, "sin psutil los hijos no se miden (se reporta NaN)")

        print("4. Monitor: registro persistente fila a fila")
        ruta = os.path.join(tmp, "audit", "telemetria.csv")
        ctx = {"t": "S4/corrida/intento 1"}
        mon = telemetria.Monitor(ruta, intervalo_s=0.3, etiqueta=lambda: ctx["t"]).iniciar()
        time.sleep(1.4)
        filas_en_vuelo = len(pd.read_csv(ruta))  # el archivo se puede leer MIENTRAS el monitor corre
        mon.detener()
        df = pd.read_csv(ruta)
        check(filas_en_vuelo >= 3, "ya habia %d filas en disco con el monitor activo" % filas_en_vuelo)
        check(len(df) >= filas_en_vuelo + 1, "al detener agrega una fila final (%d filas)" % len(df))
        check(list(df.columns[:3]) == ["utc", "ram_total_gb", "ram_disponible_gb"] and "contexto" in df.columns, "encabezado una sola vez, con la columna 'contexto'")
        check((df["contexto"] == "S4/corrida/intento 1").all(), "cada fila guarda la corrida en curso")
        n = len(df)
        time.sleep(0.8)
        check(len(pd.read_csv(ruta)) == n, "tras detener, no se agregan mas filas")

        print("5. El monitor nunca tumba la corrida")
        malo = telemetria.Monitor(os.path.join(tmp, "x", "t.csv"), intervalo_s=0.2, etiqueta=lambda: 1 / 0)
        with mock.patch.object(telemetria, "agregar_fila", side_effect=OSError("Drive desconectado")):
            malo.iniciar()
            time.sleep(0.6)
            malo.detener()
        check(True, "errores de etiqueta y de escritura se absorben")

        print("6. resolver_ruta: Drive disponible o respaldo local")
        pref, alt = os.path.join(tmp, "drive", "_auditoria", "t.csv"), os.path.join(tmp, "local", "t.csv")
        check(telemetria.resolver_ruta(pref, alt) == pref, "con Drive escribible usa la ruta de Drive")
        with mock.patch.object(telemetria.os, "makedirs", side_effect=[OSError("solo lectura"), None]):
            check(telemetria.resolver_ruta(pref, alt) == alt, "sin Drive usa la ruta local")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Telemetria de memoria (RAM del sistema, RAM de los procesos y GPU) con registro persistente.

Sirve para distinguir un fallo de **RAM del sistema** (el kernel/subproceso muere con SIGKILL, sin traceback) de un
fallo de **memoria de GPU** (excepcion `ResourceExhaustedError`/"OOM when allocating tensor") y para ver la curva de
memoria antes de una caida. Solo lee contadores: no afecta a ningun calculo.

- `snapshot()` toma una muestra (psutil si existe; si no, /proc en Linux o la API de Windows).
- `Monitor` es un hilo daemon que agrega una fila a un CSV cada `intervalo_s`. Cada fila se escribe abriendo y
  cerrando el archivo, para que quede en Drive aunque la sesion muera justo despues.
"""

import csv
import ctypes
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

try:  # psutil viene preinstalado en Colab; en otros entornos puede faltar
    import psutil
except ImportError:  # pragma: no cover - depende del entorno
    psutil = None

COLUMNAS = [
    "utc", "ram_total_gb", "ram_disponible_gb", "ram_usada_pct", "rss_proceso_mb", "rss_hijos_mb",
    "gpu_usada_mb", "gpu_total_mb", "gpu_util_pct", "contexto",
]
_GB = 1024 ** 3


def _ram_sistema():
    """(total_bytes, disponible_bytes) o (nan, nan)."""
    if psutil is not None:
        m = psutil.virtual_memory()
        return float(m.total), float(m.available)
    if os.path.exists("/proc/meminfo"):
        datos = {}
        with open("/proc/meminfo") as f:
            for linea in f:
                k, _, v = linea.partition(":")
                datos[k.strip()] = float(v.split()[0]) * 1024
        return datos.get("MemTotal", float("nan")), datos.get("MemAvailable", float("nan"))
    if os.name == "nt":  # pragma: no cover - solo Windows
        class _M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = _M()
        s.dwLength = ctypes.sizeof(_M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return float(s.ullTotalPhys), float(s.ullAvailPhys)
    return float("nan"), float("nan")


def _rss_propio_y_hijos(pid=None):
    """(rss_proceso_bytes, rss_de_todos_sus_descendientes_bytes) o (nan, nan)."""
    if psutil is not None:
        try:
            p = psutil.Process(pid or os.getpid())
            propio = p.memory_info().rss
            hijos = 0
            for h in p.children(recursive=True):
                try:
                    hijos += h.memory_info().rss
                except psutil.Error:
                    pass
            return float(propio), float(hijos)
        except psutil.Error:
            return float("nan"), float("nan")
    ruta_proc = "/proc/%s/status" % (pid or "self")
    if os.path.exists(ruta_proc):
        with open(ruta_proc) as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    return float(linea.split()[1]) * 1024, float("nan")
    if os.name == "nt" and pid is None:  # pragma: no cover - solo Windows sin psutil
        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize), float("nan")
    return float("nan"), float("nan")


def _gpu():
    """(usada_mb, total_mb, util_pct) segun nvidia-smi, o (nan, nan, nan) si no hay GPU o no responde."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            raise ValueError("sin salida")
        usada, total, util = [float(x) for x in out.stdout.strip().splitlines()[0].split(",")]
        return usada, total, util
    except (OSError, ValueError, subprocess.SubprocessError):
        return float("nan"), float("nan"), float("nan")


def snapshot(pid=None, gpu=True):
    """Una muestra de memoria. `pid` (opcional) es el proceso cuyo arbol se mide; por defecto, este. `gpu=False` omite nvidia-smi."""
    total, disp = _ram_sistema()
    propio, hijos = _rss_propio_y_hijos(pid)
    usada, gtotal, util = _gpu() if gpu else (float("nan"),) * 3
    mb = 1024 ** 2
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ram_total_gb": round(total / _GB, 2),
        "ram_disponible_gb": round(disp / _GB, 2),
        "ram_usada_pct": round(100.0 * (1 - disp / total), 1) if total == total and total > 0 else float("nan"),
        "rss_proceso_mb": round(propio / mb, 0),
        "rss_hijos_mb": round(hijos / mb, 0),
        "gpu_usada_mb": usada,
        "gpu_total_mb": gtotal,
        "gpu_util_pct": util,
    }


def agregar_fila(ruta, fila):
    """Agrega una fila al CSV `ruta` (con encabezado si es nuevo). Abre y cierra el archivo en cada llamada."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def resolver_ruta(preferida, alternativa):
    """Devuelve `preferida` si su carpeta existe y se puede escribir (Drive montado); si no, `alternativa`."""
    carpeta = os.path.dirname(os.path.abspath(preferida))
    try:
        os.makedirs(carpeta, exist_ok=True)
        prueba = os.path.join(carpeta, ".escritura_ok")
        with open(prueba, "w") as f:
            f.write("ok")
        os.remove(prueba)
        return preferida
    except OSError:
        os.makedirs(os.path.dirname(os.path.abspath(alternativa)), exist_ok=True)
        return alternativa


class Monitor:
    """Hilo daemon que registra una muestra cada `intervalo_s` en `ruta_csv`, ademas de una al arrancar y otra al parar.

    `etiqueta()` (opcional) devuelve un texto (p. ej. la corrida en curso) que se guarda en la columna `contexto`.
    """

    def __init__(self, ruta_csv, intervalo_s=120, etiqueta=None, pid=None):
        self.ruta_csv = ruta_csv
        self.intervalo_s = intervalo_s
        self.etiqueta = etiqueta
        self.pid = pid
        self._parar = threading.Event()
        self._hilo = None
        self.ultima = None

    def _muestrear(self):
        fila = snapshot(self.pid)
        try:
            fila["contexto"] = self.etiqueta() if self.etiqueta else ""
        except Exception:
            fila["contexto"] = ""
        self.ultima = fila
        try:
            agregar_fila(self.ruta_csv, fila)
        except OSError:
            pass  # el registro nunca debe tumbar la corrida

    def _bucle(self):
        while not self._parar.wait(self.intervalo_s):
            self._muestrear()

    def iniciar(self):
        self._muestrear()
        self._hilo = threading.Thread(target=self._bucle, name="telemetria", daemon=True)
        self._hilo.start()
        return self

    def detener(self):
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=10)
        self._muestrear()

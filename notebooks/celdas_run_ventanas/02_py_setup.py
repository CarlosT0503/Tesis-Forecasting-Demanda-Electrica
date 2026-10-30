import os
import sys
import subprocess

from google.colab import drive
drive.mount("/content/drive")

!pip install -q optuna==5.0.0 lightgbm==4.6.0

REPO_URL = "https://github.com/CarlosT0503/Tesis-Forecasting-Demanda-Electrica.git"
REPO_DIR = "/content/tesis_repo"
if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull

SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Rutas (deben coincidir con io_drive.py)
DATA_DIR = "/content/drive/MyDrive/Bases de datos Tesis"
ORIG_DIR = "/content/drive/MyDrive/Pipeline_Resultados"
VENT_DIR = "/content/drive/MyDrive/Pipeline_Resultados_Ventanas"
os.makedirs(os.path.join(VENT_DIR, "_auditoria"), exist_ok=True)

from tesis_forecast import io_drive, windows
from tesis_forecast import ventanas_tools as vt
from tesis_forecast.matrix import run_matrix, resumen_dataframe

commit = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Version del repositorio:", commit)
assert io_drive.BASE_DIR_VENTANAS == VENT_DIR, "VENT_DIR no coincide con io_drive.BASE_DIR_VENTANAS"
assert io_drive.BASE_DIR == ORIG_DIR, "ORIG_DIR no coincide con io_drive.BASE_DIR"
assert hasattr(vt, "estado_plan") and "xgboost" in vt.MODELOS_LOCALES, "Codigo desactualizado: reinicia el entorno y vuelve a ejecutar esta celda"
from tesis_forecast import colab_aislado as ca, telemetria
assert hasattr(ca, "ejecutar_sesion") and hasattr(ca, "informe_progreso"), "Codigo desactualizado: Entorno de ejecucion > Reiniciar sesion y vuelve a ejecutar esta celda"
muestra = telemetria.snapshot()
print("Memoria al iniciar: RAM %.1f GB libres de %.1f | GPU %s de %s MB usados" % (
    muestra["ram_disponible_gb"], muestra["ram_total_gb"], muestra["gpu_usada_mb"], muestra["gpu_total_mb"]))
print("Ventanas:", {k: (str(v.test_start), str(v.test_end)) for k, v in windows.VENTANAS.items()})

# Colab ADICIONALES (W2, W3): si tu Drive no tiene la carpeta de datos, se descargan (solo lectura) desde la carpeta compartida
# a /content. En el Colab principal no cambia nada: se sigue usando su carpeta del Drive.
from tesis_forecast import distribuido as dist
DATA_DIR = dist.preparar_datos(DATA_DIR, REPO_DIR)

"""
Montaje de Google Drive y resolucion de la carpeta base de resultados.

En el notebook legacy, el pipeline de XGBoost (celda 49) escribia primero en
una carpeta local /content/drive "falsa" (creada por os.makedirs porque Drive
todavia no estaba montado), y una celda posterior (51) montaba Drive de verdad
y copiaba los resultados desde ahi. Ese parche ya no es necesario: aqui el
runner exige montar Drive ANTES de crear cualquier carpeta de resultados, asi
que el problema que la celda 51 corregia no vuelve a ocurrir.
"""

import os

BASE_DIR = "/content/drive/MyDrive/Pipeline_Resultados"

# Resultados de la extension a cuatro semanas de prueba: una subcarpeta por
# ventana (S1..S4), separada de Pipeline_Resultados/ para no mezclar ni pisar
# nada de la semana original (y con ello cada estudio Optuna, que vive dentro
# de la carpeta de la corrida, queda tambien separado por semana).
BASE_DIR_VENTANAS = "/content/drive/MyDrive/Pipeline_Resultados_Ventanas"


def mount_drive() -> bool:
    """
    Monta Google Drive si estamos en Colab. Fuera de Colab es un no-op
    (permite probar el resto del pipeline localmente).
    """
    try:
        from google.colab import drive
    except ImportError:
        print("No se detecto Google Colab: se omite el montaje de Drive.")
        return False

    drive.mount("/content/drive")
    print("Google Drive montado correctamente.")
    return True


def resolve_base_dir() -> str:
    """Carpeta raiz de resultados, la misma que usaba el pipeline legacy."""
    return BASE_DIR


def resolve_ventanas_base_dir(ventana_id: str) -> str:
    """Carpeta de resultados de una ventana semanal (S1..S4)."""
    return os.path.join(BASE_DIR_VENTANAS, ventana_id)

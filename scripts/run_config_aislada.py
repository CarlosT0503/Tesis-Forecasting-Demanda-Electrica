"""
Ejecuta UNA configuracion experimental en este proceso y escribe un resumen JSON. Lo lanza
`tesis_forecast.colab_aislado` (un subproceso por configuracion) para que, al terminar, el sistema operativo
recupere toda la memoria. Hace exactamente lo que hacia la celda de lanzamiento: `run_matrix([config])`, es decir,
salta lo completo, reanuda por region lo parcial y no cambia ningun hiperparametro, ventana, dato ni metrica.

    python scripts/run_config_aislada.py --config-json cfg.json --resultado res.json --data-dir <DATA_DIR> \
        --base-ventanas <Pipeline_Resultados_Ventanas>

Codigos de salida: 0 = corrida completa (o ya estaba completa); 1 = fallo (excepcion, o ninguna region produjo metricas);
3 = el proceso termino sin error pero la corrida NO quedo con las 8 regiones (algunas regiones fallaron y el modelo siguio).
El JSON de resumen trae `errores_region` (errores por region que los modelos registraron en log.txt) y `regiones_ok`.

`--stubs-pesados` sustituye TensorFlow/Optuna/XGBoost/LightGBM por stubs: SOLO para pruebas locales con modelos que
no los usan (nunca en Colab: con stubs una red neuronal "entrenaria" sin entrenar).
"""

import argparse
import json
import os
import sys
import time
import traceback
from unittest import mock

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))


def _pico_rss_mb():
    try:
        import resource  # POSIX

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024.0 if sys.platform != "darwin" else kb / 1024.0 / 1024.0)
    except ImportError:
        pass
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 1024.0 ** 2)
    except Exception:
        pass
    try:  # sin resource ni psutil (p. ej. Windows minimo): RSS actual
        from tesis_forecast import telemetria

        rss = telemetria.snapshot(gpu=False)["rss_proceso_mb"]
        return None if rss != rss else int(rss)
    except Exception:
        return None


def _escribir_json(ruta, datos):
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False)
    os.replace(tmp, ruta)  # misma carpeta: atomico


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--resultado", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base-ventanas", default=None, help="sustituye io_drive.BASE_DIR_VENTANAS (pruebas)")
    ap.add_argument("--stubs-pesados", action="store_true")
    args = ap.parse_args()

    if args.stubs_pesados:
        for nombre in [
            "optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras",
            "tensorflow.keras.models", "tensorflow.keras.layers", "tensorflow.keras.callbacks",
            "tensorflow.keras.optimizers", "tensorflow.keras.backend",
        ]:
            try:
                __import__(nombre)
            except ImportError:
                sys.modules[nombre] = mock.MagicMock()

    t0 = time.time()
    resultado = {"status": "failed", "run_name": None, "mape_promedio": None, "error": None, "traceback_cola": "",
                 "duracion_s": None, "pico_rss_mb": None}
    codigo = 1
    try:
        from tesis_forecast import errores_log, io_drive
        from tesis_forecast.atomic_io import limpiar_temporales
        from tesis_forecast.config import ExperimentConfig
        from tesis_forecast.matrix import run_matrix
        from tesis_forecast.runner import resolve_run
        from tesis_forecast.validator import validar_resultado

        if args.base_ventanas:
            io_drive.BASE_DIR_VENTANAS = args.base_ventanas
        with open(args.config_json, encoding="utf-8") as f:
            config = ExperimentConfig(**json.load(f))

        resolved = resolve_run(config)
        resultado["run_name"] = resolved.run_name
        borrados = limpiar_temporales(resolved.run_dir)  # restos de una escritura interrumpida
        if borrados:
            print("Temporales de una sesion anterior eliminados: %d" % len(borrados))

        registro = run_matrix([config], data_dir=args.data_dir, mount_drive=False)[0]
        resultado.update(status=registro.status, mape_promedio=registro.mape_promedio, error=registro.error)
        reporte = validar_resultado(resolved.run_dir)
        resultado["regiones_ok"] = len(reporte.regiones_encontradas)
        resultado["errores_region"] = errores_log.resumir(errores_log.extraer_errores(os.path.join(resolved.run_dir, "log.txt")))
        if registro.status not in ("completed", "skipped_ya_completo"):
            codigo = 1
        elif reporte.es_completo:
            codigo = 0
        else:  # termino sin excepcion pero faltan regiones: NO es una corrida completada
            codigo = 3
            resultado["status"] = "incompleta"
            resultado["error"] = "La corrida termino con %d/8 regiones. %s" % (resultado["regiones_ok"], resultado["errores_region"]["texto"])
    except BaseException as e:  # incluye MemoryError y KeyboardInterrupt
        resultado["error"] = "%s: %s" % (type(e).__name__, e)
        resultado["traceback_cola"] = traceback.format_exc()[-4000:]
        print(resultado["traceback_cola"], file=sys.stderr)
        codigo = 1
    finally:
        resultado["duracion_s"] = round(time.time() - t0, 1)
        resultado["pico_rss_mb"] = _pico_rss_mb()
        try:
            _escribir_json(args.resultado, resultado)
        except OSError:
            pass
    return codigo


if __name__ == "__main__":
    sys.exit(main())

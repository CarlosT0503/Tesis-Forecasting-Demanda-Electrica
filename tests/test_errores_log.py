"""
Errores por region: trazas completas en log.txt, corrida fallida si ninguna region produjo resultados, y clasificacion en el
orquestador de Colab. Sin TensorFlow ni entrenamientos (las funciones de red se sustituyen por otras que lanzan un error).

  1. errores_log: marca + traza, lectura solo de la ultima ejecucion, resumen, proceso muerto a mitad de traza.
  2. Los 4 modelos neuronales dejan la traza completa cuando tragan un error (FCNN con sus 3 puntos, LSTM directa, LSTM
     residual, Ensemble STL) y siguen sin detener las demas regiones.
  3. runner: ninguna region con metricas -> RuntimeError (corrida fallida), conservando los CSV; con regiones heredadas
     del checkpoint o con metricas -> completada. El mensaje trae el primer error real.
  4. Orquestador: rc=3 / rc=0 incompleta con errores, errores de GPU o RAM tragados por el modelo, y abandono sin
     reintento cuando todas las regiones fallan con el mismo error; los casos anteriores siguen igual.

    python tests/test_errores_log.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from unittest import mock

import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))

for nombre in [
    "optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras", "tensorflow.keras.models",
    "tensorflow.keras.layers", "tensorflow.keras.callbacks", "tensorflow.keras.optimizers", "tensorflow.keras.backend",
]:
    try:
        __import__(nombre)
    except ImportError:
        sys.modules[nombre] = mock.MagicMock()

from tesis_forecast import colab_aislado as ca  # noqa: E402
from tesis_forecast import errores_log, io_drive, runner  # noqa: E402
from tesis_forecast import ventanas_tools as vt  # noqa: E402
from tesis_forecast.config import ExperimentConfig  # noqa: E402
from tesis_forecast.data.exogenas import cargar_exogenas_horarias  # noqa: E402
from tesis_forecast.matrix import run_matrix  # noqa: E402
from tesis_forecast.models import ensemble_stl, fcnn_model, lstm_direct, lstm_resid_model  # noqa: E402
from tesis_forecast.validator import validar_resultado  # noqa: E402

DATOS = os.path.join(RAIZ, "datos_locales")
HIJO_FALSO = os.path.join(RAIZ, "tests", "_hijo_falso.py")
FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def capturar(fun, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            r = fun(*a, **k)
        except BaseException as e:  # noqa: BLE001
            r = e
    return r, buf.getvalue()


def main():
    tmp = tempfile.mkdtemp(prefix="test_err_")
    try:
        print("1. errores_log")
        def lanza(n):
            try:
                {}["clave_inexistente"]
            except KeyError as e:
                errores_log.registrar_error("CEN_DEMANDA", e)
        _, salida = capturar(lanza, 1)
        check(salida.startswith(errores_log.MARCA + " CEN | KeyError | "), "marca con region, tipo y mensaje")
        check("Traceback (most recent call last):" in salida and "clave_inexistente" in salida and errores_log.FIN in salida, "traza completa y marca de cierre")
        log = os.path.join(tmp, "log.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write("Experimento: corrida_A\nruido\n" + salida + "Experimento: corrida_A\nmas ruido\n")
            f.write(errores_log.MARCA + " NES | ValueError | boom\nTraceback...\n  File x, line 3\nValueError: boom\n" + errores_log.FIN + "\n")
            f.write(errores_log.MARCA + " NOR | ValueError | boom\nTraceback...\nValueError: boom\n" + errores_log.FIN + "\n")
        ult = errores_log.extraer_errores(log)
        check([e["region"] for e in ult] == ["NES", "NOR"], "solo cuenta la ultima ejecucion (los errores de intentos anteriores quedan fuera): %s" % [e["region"] for e in ult])
        check([e["region"] for e in errores_log.extraer_errores(log, solo_ultima_ejecucion=False)] == ["CEN", "NES", "NOR"], "con solo_ultima_ejecucion=False cuenta todos")
        r = errores_log.resumir(ult)
        check(r["n"] == 2 and r["mensajes_unicos"] == 1 and r["primero"]["region"] == "NES" and "NES ValueError: boom" in r["texto"], "resumen: 2 errores, 1 mensaje unico, primero NES")
        with open(log, "a", encoding="utf-8") as f:
            f.write(errores_log.MARCA + " ORI | OSError | se corto\nTraceback a medias")
        check(errores_log.extraer_errores(log)[-1]["region"] == "ORI", "una traza cortada por la muerte del proceso tambien se recupera")
        check(errores_log.extraer_errores(os.path.join(tmp, "no_existe.txt")) == [] and errores_log.resumir([])["n"] == 0, "log inexistente o sin errores -> vacio")

        print("2. Los 4 modelos neuronales dejan la traza completa y siguen con la region siguiente")
        if not os.path.isdir(DATOS):
            print("  (se omite: no hay datos locales)")
        else:
            glob = cargar_exogenas_horarias(data_dir=DATOS)
            exog = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]
            casos = [
                ("ensemble_stl: run() traga el error de la region", ensemble_stl, "evaluar_region", None),
                ("fcnn: error de la region", fcnn_model, "evaluar_region", None),
                ("fcnn: error dentro de la FCNN multivariada", fcnn_model, "tunear_fcnn_optuna", None),
                ("lstm_direct: error de la region", lstm_direct, "tune_lstm", None),
                ("lstm_resid: error dentro de la serie", lstm_resid_model, "tunear_lstm", None),
            ]
            for etiqueta, mod, atributo, _ in casos:
                salida_dir = os.path.join(tmp, "m_" + mod.__name__.split(".")[-1] + atributo)
                os.makedirs(salida_dir)
                def roto(*a, **k):
                    raise ValueError("boom_" + atributo)
                with mock.patch.object(mod, atributo, side_effect=roto):
                    r, salida = capturar(mod.run, glob, ["CEN", "NES"], train_hours=3600, forecast_horizon=168, exog_cols=exog,
                                         optuna_n_trials=1, data_dir=DATOS, output_dir=salida_dir)
                marcas = [l for l in salida.splitlines() if l.startswith(errores_log.MARCA)]
                check(not isinstance(r, BaseException), "%s: run() no se detiene (devuelve tablas)" % etiqueta)
                esperadas = 4 if atributo == "tunear_fcnn_optuna" else 2  # la FCNN tiene 2 estrategias por region
                check(len(marcas) == esperadas and all("ValueError | boom_%s" % atributo in m for m in marcas)
                      and {m.split()[1] for m in marcas} == {"CEN", "NES"}, "%s: marca por cada region y estrategia (%d)" % (etiqueta, esperadas))
                check(salida.count("Traceback (most recent call last):") >= 2 and ("boom_" + atributo) in salida, "%s: traza completa impresa" % etiqueta)
                check(salida.count(errores_log.FIN) == len(marcas), "%s: cada traza esta cerrada" % etiqueta)

        print("3. runner: ninguna region con metricas -> corrida fallida (CSV conservados)")
        if os.path.isdir(DATOS):
            base = os.path.join(tmp, "orig")
            original = io_drive.BASE_DIR
            io_drive.BASE_DIR = base
            try:
                cfg = ExperimentConfig(modelo="ensemble_stl", sufijo_run_name="t_fallo")
                runner_orig = runner.MODEL_RUNNERS["ensemble_stl"]

                def run_todo_falla(**kw):
                    for region in ("CEN", "NES"):
                        print("Evaluando " + region)
                        errores_log.registrar_error(region, RuntimeError("tensor con forma rara"))
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

                with mock.patch.dict(runner.MODEL_RUNNERS, {"ensemble_stl": run_todo_falla}):
                    r, _ = capturar(run_matrix, [cfg], data_dir=DATOS, mount_drive=False)
                rec = r[0]
                check(rec.status == "failed" and rec.mape_promedio is None, "estado 'failed' (antes: 'completed')")
                check("Ninguna region produjo metricas" in rec.error and "CEN RuntimeError: tensor con forma rara" in rec.error, "el error trae el primer error real: %s" % rec.error[:110])
                rd = runner.resolve_run(cfg).run_dir
                check(os.path.exists(os.path.join(rd, "log.txt")) and "ERROR: la corrida no completo" in open(os.path.join(rd, "log.txt"), encoding="utf-8").read(), "queda log.txt con la traza")
                check(not validar_resultado(rd).es_completo, "validar_resultado: no completa")

                def run_con_metricas(**kw):
                    m = pd.DataFrame({"serie": ["CEN_DEMANDA"], "region": ["CEN"], "MAE": [1.0], "RMSE": [1.0], "MAPE": [1.0], "sMAPE": [1.0]})
                    return pd.DataFrame({"serie": ["CEN_DEMANDA"]}), m, pd.DataFrame(), pd.DataFrame()

                cfg2 = ExperimentConfig(modelo="ensemble_stl", sufijo_run_name="t_parcial")
                with mock.patch.dict(runner.MODEL_RUNNERS, {"ensemble_stl": run_con_metricas}):
                    r2, _ = capturar(run_matrix, [cfg2], data_dir=DATOS, mount_drive=False)
                check(r2[0].status == "completed" and r2[0].mape_promedio == 1.0, "con al menos una region con metricas (p. ej. heredada del checkpoint) sigue 'completed'")
                check(runner.MODEL_RUNNERS["ensemble_stl"] is runner_orig, "el registro de modelos se restaura")
            finally:
                io_drive.BASE_DIR = original

        print("4. Orquestador")
        orig_v = io_drive.BASE_DIR_VENTANAS
        vent = os.path.join(tmp, "Pipeline_Resultados_Ventanas")
        io_drive.BASE_DIR_VENTANAS = vent
        try:
            cfgs = [c for c in vt.construir_configs("T3", "S3") if c.modelo == "lstm_direct"]

            def correr(config, secuencia, resultado_json=None, **kw):
                run_dir = runner.resolve_run(config).run_dir
                contador = os.path.join(tmp, "cont_%s.txt" % config.exogenas[0])
                if os.path.exists(contador):
                    os.remove(contador)

                def cmd(cfg_json, res_json, dd, vd):
                    if resultado_json is not None:
                        with open(res_json + ".plantilla", "w", encoding="utf-8") as f:
                            json.dump(resultado_json, f)
                    return [sys.executable, HIJO_FALSO, run_dir, secuencia, contador, res_json] + ([res_json + ".plantilla"] if resultado_json is not None else [])
                kw.setdefault("sondeo_s", 0.5)
                r = ca.ejecutar_config_aislada(config, data_dir=tmp, vent_dir=vent, sesion="E", comando_hijo=cmd,
                                               tmp_dir=os.path.join(tmp, "sub"), verbose=False, **kw)
                return r, (int(open(contador).read()) if os.path.exists(contador) else 0)

            reg_csv = os.path.join(vent, "_auditoria", "subprocesos_colab_E.csv")
            e_igual = {"n": 3, "regiones": ["BCA", "CEN", "NES"], "mensajes_unicos": 1, "texto": "3 error(es) en regiones ['BCA', 'CEN', 'NES']; primero: BCA ValueError: columna rara",
                       "primero": {"region": "BCA", "tipo": "ValueError", "mensaje": "columna rara", "traza_cola": "File x, line 9\nValueError: columna rara"}}
            (rec, clase), n = correr(cfgs[0], "rc3", resultado_json={"errores_region": e_igual, "error": "x"})
            check(rec.status == "failed" and clase == ca.INCOMPLETA and n == 1, "rc=3 con las mismas excepciones y sin avanzar: incompleta y SIN reintento (intentos=%d)" % n)
            check("BCA ValueError: columna rara" in rec.error, "el error de la corrida cita la excepcion real")
            fila = pd.read_csv(reg_csv).iloc[-1]
            check(fila["clase"] == "incompleta" and "columna rara" in str(fila["mensaje"]), "el registro de intentos guarda el mensaje con la causa")

            e_dist = dict(e_igual, mensajes_unicos=2)
            (rec, clase), n = correr(cfgs[1], "rc3,rc3,rc3", resultado_json={"errores_region": e_dist}, max_intentos=3, max_sin_progreso=2)
            check(clase == ca.INCOMPLETA and n == 2, "errores distintos sin avanzar: se aplica la regla de 2 intentos sin progreso (intentos=%d)" % n)

            e_gpu = {"n": 2, "regiones": ["CEN", "NES"], "mensajes_unicos": 2, "texto": "2 error(es)",
                     "primero": {"region": "CEN", "tipo": "ResourceExhaustedError", "mensaje": "OOM when allocating tensor", "traza_cola": ""}}
            (rec, clase), n = correr(cfgs[2], "rc3,rc3", resultado_json={"errores_region": e_gpu}, max_intentos=2)
            check(clase == ca.GPU_MEMORIA, "un error de GPU tragado por el modelo se clasifica 'gpu_memoria', no 'incompleta'")

            e_mem = dict(e_gpu, primero={"region": "CEN", "tipo": "MemoryError", "mensaje": "Unable to allocate 2 GiB", "traza_cola": ""})
            (rec, clase), n = correr(cfgs[3], "rc3,rc3", resultado_json={"errores_region": e_mem}, max_intentos=2)
            check(clase == ca.RAM_PYTHON and clase in ca.CLASES_RAM, "un MemoryError tragado se clasifica 'ram_python' (cuenta como RAM)")

            (rec, clase), n = correr(cfgs[4], "incompleta,incompleta,incompleta")
            check(clase == ca.INCOMPLETA and n == 3, "comportamiento anterior intacto: rc=0 con 5 regiones, avanza y luego 2 sin avanzar (intentos=%d)" % n)
            (rec, clase), n = correr(cfgs[5], "parcial3,ok")
            check(rec.status == "completed" and n == 2, "comportamiento anterior intacto: caida con avance y reintento que completa")

            print("   (Ejecucion real del hijo con una falla natural: sin datos)")
            cfg_real = [c for c in vt.construir_configs("T1", "S4") if c.modelo == "naive"][0]
            r, _ = capturar(ca.ejecutar_config_aislada, cfg_real, data_dir=os.path.join(tmp, "sin_datos"), vent_dir=vent, sesion="E",
                            stubs=True, python=sys.executable, tmp_dir=os.path.join(tmp, "sub"), verbose=True, sondeo_s=1, max_intentos=2)
            rec, clase = r
            check(rec.status == "failed" and clase == ca.ERROR_CODIGO, "falla real del hijo (sin datos): failed / error_codigo")
            check(os.path.exists(os.path.join(tmp, "sub")) and any(f.endswith(".resultado.json") for f in os.listdir(os.path.join(tmp, "sub"))), "el hijo deja su resumen JSON aun al fallar")
        finally:
            io_drive.BASE_DIR_VENTANAS = orig_v
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

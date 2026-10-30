"""
Pruebas de la ejecucion aislada de Colab (un subproceso por configuracion), sin Colab ni TensorFlow:

  1. Clasificacion de fallos (RAM del sistema, RAM de Python, GPU, timeout, ...).
  2. Cola por sesion y lote: A y B cubren las 112 corridas de Colab sin traslape; los lotes parten la cola.
  3. Subprocesos simulados: exito, reintento que reanuda, reintentos acotados (nunca infinitos), cada clase de fallo,
     timeout, corrida ya completa (no lanza nada) y corte de la cola tras fallos de RAM consecutivos.
  4. Ejecucion REAL de extremo a extremo con el script hijo y un modelo sin redes (naive, semana S4, datos locales):
     corrida completa -> se trunca a 3 regiones (lo que dejaria una sesion caida) -> se reanuda: no se repiten las
     regiones completas y el resultado final es identico al de la corrida completa.
  5. Informe de progreso (solo lectura).

    python tests/test_ejecucion_aislada.py
"""

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
    "optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras",
    "tensorflow.keras.models", "tensorflow.keras.layers", "tensorflow.keras.callbacks",
    "tensorflow.keras.optimizers", "tensorflow.keras.backend",
]:
    try:
        __import__(nombre)
    except ImportError:
        sys.modules[nombre] = mock.MagicMock()

from tesis_forecast import colab_aislado as ca  # noqa: E402
from tesis_forecast import io_drive  # noqa: E402
from tesis_forecast import ventanas_tools as vt  # noqa: E402
from tesis_forecast.runner import resolve_run  # noqa: E402
from tesis_forecast.validator import validar_resultado  # noqa: E402

HIJO_FALSO = os.path.join(RAIZ, "tests", "_hijo_falso.py")
DATOS = os.path.join(RAIZ, "datos_locales")
FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def main():
    tmp = tempfile.mkdtemp(prefix="test_aislada_")
    vent = os.path.join(tmp, "Pipeline_Resultados_Ventanas")
    original_base = io_drive.BASE_DIR_VENTANAS
    io_drive.BASE_DIR_VENTANAS = vent
    try:
        print("1. clasificar_fallo")
        C = ca.clasificar_fallo
        check(C(-9, "") == ca.RAM_SISTEMA and C(137, "", 0.4) == ca.RAM_SISTEMA, "SIGKILL con RAM baja o sin dato -> ram_sistema")
        check(C(-9, "", 6.0) == ca.SIGKILL_OTRO, "SIGKILL con RAM holgada -> sigkill_otro")
        check(C(1, "ResourceExhaustedError: OOM when allocating tensor") == ca.GPU_MEMORIA, "OOM de GPU con traceback -> gpu_memoria")
        check(C(1, "CUDA_ERROR_OUT_OF_MEMORY: out of memory") == ca.GPU_MEMORIA, "CUDA OOM -> gpu_memoria")
        check(C(1, "MemoryError: Unable to allocate 3 GiB") == ca.RAM_PYTHON, "MemoryError -> ram_python")
        check(C(1, "std::bad_alloc") == ca.RAM_PYTHON, "bad_alloc -> ram_python")
        check(C(1, "ValueError: x") == ca.ERROR_CODIGO, "excepcion de codigo -> error_codigo")
        check(C(-15, "") == ca.INTERRUMPIDO and C(130, "") == ca.INTERRUMPIDO, "SIGTERM/SIGINT -> interrumpido")
        check(C(-11, "") == ca.FALLO_NATIVO, "SIGSEGV -> fallo_nativo")
        check(C(-9, "", timeout=True) == ca.TIMEOUT, "timeout manda sobre todo")
        check(C(1, "RESOURCEEXHAUSTEDERROR") == ca.GPU_MEMORIA, "no distingue mayusculas")
        check({ca.RAM_SISTEMA, ca.RAM_PYTHON} == ca.CLASES_RAM and ca.GPU_MEMORIA not in ca.CLASES_RAM, "la GPU NO cuenta como fallo de RAM")

        print("2. Cola por sesion y lote")
        def claves(cola):
            return {(c.ventana, resolve_run(c).run_name) for _, _, cs in cola for c in cs}
        a, b = claves(ca.construir_cola("A")), claves(ca.construir_cola("B"))
        check(len(a) == 56 and len(b) == 56 and not (a & b), "A=56 y B=56 corridas, sin traslape")
        plan = vt.plan_dataframe()
        plan_colab = {(r.ventana, r.run_name) for r in plan[plan["ejecuta_en"] == "colab"].itertuples()}
        check((a | b) == plan_colab and len(plan_colab) == 112, "A u B = las 112 corridas de Colab del plan")
        check(all(vt.ejecuta_en(c.modelo) == "colab" for _, _, cs in ca.construir_cola("A") for c in cs), "solo redes (nada de la PC)")
        for sesion in ("A", "B"):
            partes = {l: claves(ca.construir_cola(sesion, l)) for l in ca.LOTES_VALIDOS}
            check(len(partes["T2"]) == 8 and len(partes["T3"]) == 48, "sesion %s: T2=8 y T3=48" % sesion)
            lotes_modelo = [partes["T3-" + m] for m in ca.MODELOS_COLAB]
            check(all(len(p) == 12 for p in lotes_modelo), "sesion %s: cada T3-<modelo> tiene 12 corridas" % sesion)
            union = partes["T2"].union(*lotes_modelo)
            check(union == partes["TODO"] and sum(len(p) for p in lotes_modelo) + len(partes["T2"]) == 56, "sesion %s: los 5 lotes particionan las 56 sin repetir" % sesion)
        niveles = [n for n, _, _ in ca.construir_cola("A")]
        check(niveles == sorted(niveles), "orden: T2 antes que T3")
        try:
            ca.construir_cola("A", "T9")
            check(False, "debia rechazar un lote invalido")
        except ValueError:
            check(True, "lote invalido -> ValueError")

        print("3. Subprocesos simulados")
        # la clase de un SIGKILL depende de la RAM libre observada: se fija para que la prueba no dependa de la maquina
        _snap = lambda pid=None, gpu=True: {"rss_proceso_mb": 0.0, "rss_hijos_mb": 0.0, "ram_disponible_gb": 0.5}
        _parche_ram = mock.patch.object(ca.telemetria, "snapshot", side_effect=_snap)
        _parche_ram.start()
        cfgs = [c for c in vt.construir_configs("T3", "S4") if c.modelo == "lstm_direct"]
        reg = os.path.join(vent, "_auditoria", "subprocesos_colab_T.csv")

        def correr(config, secuencia, **kw):
            run_dir = resolve_run(config).run_dir
            contador = os.path.join(tmp, "contador_%s.txt" % config.exogenas[0])
            if os.path.exists(contador):
                os.remove(contador)
            cmd = lambda cfg, res, dd, vd: [sys.executable, HIJO_FALSO, run_dir, secuencia, contador, res]
            kw.setdefault("sondeo_s", 0.5)
            r = ca.ejecutar_config_aislada(config, data_dir=tmp, vent_dir=vent, sesion="T", comando_hijo=cmd,
                                           tmp_dir=os.path.join(tmp, "sub"), verbose=False, **kw)
            n = int(open(contador).read()) if os.path.exists(contador) else 0
            return r, n

        def filas(run_name):
            d = pd.read_csv(reg)
            return d[d["run_name"] == run_name]

        c0 = cfgs[0]
        (rec, clase), n = correr(c0, "ok")
        check(rec.status == "completed" and clase is None and n == 1, "exito a la primera: completed, 1 subproceso")
        check(validar_resultado(rec.run_dir).es_completo, "la corrida quedo completa (8 regiones)")
        f = filas(rec.run_name)
        check(len(f) == 1 and f.iloc[0]["clase"] == "ok" and f.iloc[0]["regiones_despues"] == 8, "registro CSV: 1 fila 'ok'")
        (rec, clase), n = correr(c0, "kill")
        check(rec.status == "skipped_ya_completo" and n == 0, "ya completa: no se lanza ningun subproceso")

        c1 = cfgs[1]
        (rec, clase), n = correr(c1, "parcial3,ok")
        f = filas(rec.run_name)
        check(rec.status == "completed" and n == 2, "cae con 3 regiones (137) y el reintento termina: 2 subprocesos")
        check(list(f["clase"]) == ["ram_sistema", "ok"] and list(f["regiones_antes"]) == [0, 3] and list(f["regiones_despues"]) == [3, 8],
              "registro: ram_sistema (0->3 regiones) y luego ok (3->8)")

        c2 = cfgs[2]
        (rec, clase), n = correr(c2, "kill,kill,kill", max_intentos=3, max_sin_progreso=2)
        check(rec.status == "failed" and clase == ca.RAM_SISTEMA and n == 2, "2 intentos seguidos sin avanzar: abandona (no llega al 3.o)")
        check("ram_sistema" in rec.error and "intentos=2" in rec.error, "el error queda registrado: %s" % rec.error[:90])

        c3 = cfgs[3]
        (rec, clase), n = correr(c3, "mem,ok")
        f = filas(rec.run_name)
        check(rec.status == "completed" and list(f["clase"]) == ["ram_python", "ok"], "MemoryError -> ram_python y el reintento completa")

        c4 = cfgs[4]
        (rec, clase), n = correr(c4, "gpu,gpu,gpu")
        check(rec.status == "failed" and clase == ca.GPU_MEMORIA and n == 2, "GPU OOM x2 -> gpu_memoria, abandona; NO se cuenta como RAM")

        c5 = cfgs[5]
        (rec, clase), n = correr(c5, "err,err,err")
        check(rec.status == "failed" and clase == ca.ERROR_CODIGO and n == 2, "error de codigo repetido -> error_codigo, abandona tras 2")

        otras = {v: [c for c in vt.construir_configs("T3", v) if c.modelo == "lstm_direct"] for v in ("S1", "S2", "S3")}
        c6 = otras["S1"][0]
        (rec, clase), n = correr(c6, "parcial2,err,ok", max_intentos=3, max_sin_progreso=2)
        check(rec.status == "completed" and n == 3, "con avance entre intentos (0->2 regiones) un fallo no corta: completa en el 3.o")

        c7 = otras["S1"][1]
        (rec, clase), n = correr(c7, "dormir,dormir", timeout_s=10, sondeo_s=0.5)
        check(rec.status == "failed" and clase == ca.TIMEOUT and n == 2, "subproceso colgado: timeout, se mata y no reintenta mas de 2")
        pids = [int(x) for x in open(os.path.join(tmp, "contador_%s.txt.pid" % c7.exogenas[0])).read().split()]
        check(len(pids) == 2 and not any(_vivo(p) for p in pids), "tras el timeout no queda ningun proceso huerfano (pids %s)" % pids)

        c8 = otras["S1"][2]
        (rec, clase), n = correr(c8, "incompleta,incompleta,incompleta")
        check(rec.status == "failed" and clase == ca.INCOMPLETA and n == 3, "sale con 0 pero sin las 8 regiones -> 'incompleta'; avanza 0->5 y luego 2 intentos sin avanzar: abandona (3 subprocesos)")

        c9 = otras["S1"][3]
        (rec, clase), n = correr(c9, "kill,ok", max_intentos=1)
        check(rec.status == "failed" and n == 1, "max_intentos=1: un solo intento")

        print("4. Corte de la cola tras fallos de RAM consecutivos")
        malas = otras["S2"][:4]
        cola = [("T3", "S2", malas)]
        contador = os.path.join(tmp, "contador_cola.txt")
        cmd = lambda cfg, res, dd, vd: [sys.executable, HIJO_FALSO, os.path.join(tmp, "no_se_usa"), "kill", contador, res]
        regs, motivo = ca.ejecutar_cola(cola, data_dir=tmp, vent_dir=vent, sesion="T", max_fallos_ram_consecutivos=3,
                                        comando_hijo=cmd, sondeo_s=0.5, tmp_dir=os.path.join(tmp, "sub"), verbose=False)
        check(motivo is not None and len(regs) == 3, "con 3 fallos de RAM seguidos la cola se detiene (procesadas %d de %d)" % (len(regs), len(cola[0][2])))
        check(os.path.exists(os.path.join(vent, "_auditoria", "tiempos_colab_T.csv")), "los tiempos se guardan por configuracion")

        _parche_ram.stop()

        print("5. Ejecucion REAL de extremo a extremo (script hijo + run_matrix + checkpoint), modelo naive, semana S4")
        if not os.path.isdir(DATOS) or os.environ.get("SIN_PRUEBA_REAL"):
            print("  (se omite: no hay datos locales en datos_locales/ o SIN_PRUEBA_REAL=1)")
        else:
            real = os.path.join(tmp, "real")
            vent_real = os.path.join(real, "Pipeline_Resultados_Ventanas")
            io_drive.BASE_DIR_VENTANAS = vent_real
            cfg = [c for c in vt.construir_configs("T1", "S4") if c.modelo == "naive"][0]
            run_dir = resolve_run(cfg).run_dir
            kw = dict(data_dir=DATOS, vent_dir=vent_real, sesion="R", stubs=True, python=sys.executable,
                      tmp_dir=os.path.join(real, "sub"), verbose=True, sondeo_s=2)
            rec1, clase1 = ca.ejecutar_config_aislada(cfg, **kw)
            check(rec1.status == "completed" and clase1 is None, "corrida completa en un subproceso real")
            check(validar_resultado(run_dir).es_completo, "queda con las 8 regiones y sin NaN")
            base_m = pd.read_csv(os.path.join(run_dir, "metricas.csv")).sort_values("serie").reset_index(drop=True)
            base_s = pd.read_csv(os.path.join(run_dir, "series.csv"), encoding="utf-8-sig")
            check(not [f for f in os.listdir(run_dir) if f.startswith(".tmp_")], "sin temporales en la carpeta de la corrida")
            check(os.path.exists(os.path.join(real, "sub", "S4__%s__intento1.resultado.json" % os.path.basename(run_dir))), "el hijo dejo su resumen JSON")
            res = json.load(open(os.path.join(real, "sub", "S4__%s__intento1.resultado.json" % os.path.basename(run_dir))))
            check(res["status"] == "completed" and (res["pico_rss_mb"] or 0) > 20, "resumen: completed, pico RSS %s MB" % res["pico_rss_mb"])

            # una sesion caida habria dejado solo 3 regiones guardadas
            conservar = ["BCA_DEMANDA", "CEN_DEMANDA", "NES_DEMANDA"]
            base_m[base_m["serie"].isin(conservar)].to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
            base_s[base_s["serie"].isin(conservar)].to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
            # y un temporal huerfano de una escritura interrumpida
            open(os.path.join(run_dir, ".tmp_metricas.csv.zzz.part"), "w").write("basura")
            check(_regiones(run_dir) == 3 and not validar_resultado(run_dir).es_completo, "preparacion: la corrida figura parcial (3/8)")
            log_antes = open(os.path.join(run_dir, "log.txt"), encoding="utf-8").read()
            rec2, clase2 = ca.ejecutar_config_aislada(cfg, **kw)
            check(rec2.status == "completed" and clase2 is None, "la reanudacion termina bien")
            log_desp = open(os.path.join(run_dir, "log.txt"), encoding="utf-8").read()[len(log_antes):]
            check("Checkpoint: 3 region(es) ya completas" in log_desp, "el log de la reanudacion dice que se saltan 3 regiones")
            for r in ("BCA", "CEN", "NES"):
                check(("Serie: %s_DEMANDA" % r) not in log_desp, "la reanudacion NO recalcula %s" % r)
            for r in ("NOR", "NTE", "OCC", "ORI", "PEN"):
                check(("Serie: %s_DEMANDA" % r) in log_desp, "la reanudacion SI calcula %s (faltaba)" % r)
            fin_m = pd.read_csv(os.path.join(run_dir, "metricas.csv")).sort_values("serie").reset_index(drop=True)
            fin_s = pd.read_csv(os.path.join(run_dir, "series.csv"), encoding="utf-8-sig")
            pd.testing.assert_frame_equal(base_m, fin_m, check_like=True)
            check(True, "metricas.csv final identico al de la corrida completa original")
            check(len(fin_s) == len(base_s) and fin_s.sort_values(list(fin_s.columns)).reset_index(drop=True).equals(base_s.sort_values(list(base_s.columns)).reset_index(drop=True)),
                  "series.csv final identico (mismas filas, sin duplicados)")
            check(not [f for f in os.listdir(run_dir) if f.startswith(".tmp_")], "el temporal huerfano se limpio")
            filas_reg = pd.read_csv(os.path.join(vent_real, "_auditoria", "subprocesos_colab_R.csv"))
            check(list(filas_reg["clase"]) == ["ok", "ok"] and list(filas_reg["regiones_antes"]) == [0, 3], "registro de subprocesos: 2 filas ok (regiones 0 y 3 al empezar)")
            rec3, _ = ca.ejecutar_config_aislada(cfg, **kw)
            check(rec3.status == "skipped_ya_completo", "una tercera ejecucion no hace nada (ya completa)")

            print("6. Informe de progreso (solo lectura)")
            io_drive.BASE_DIR_VENTANAS = vent_real
            inf = ca.informe_progreso(vent_real, ver=False)
            check(len(inf["estado"]) == 112 and set(inf["por_sesion"].index) == {"A", "B"}, "112 corridas de Colab, sesiones A y B")
            check(int(inf["por_sesion"].to_numpy().sum()) == 112 and list(inf["por_sesion"].columns) == ["completa", "parcial", "no_iniciada"], "completa/parcial/no_iniciada suman 112")
            check(len(inf["faltan"]) == 112, "sin corridas de Colab hechas: faltan 112")
            ruta = ca.guardar_faltantes(vent_real, inf["faltan"])
            check(os.path.exists(ruta) and len(pd.read_csv(ruta)) == 112, "faltantes guardados: %s" % os.path.basename(ruta))
    finally:
        io_drive.BASE_DIR_VENTANAS = original_base
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


def _vivo(pid):
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if os.name == "nt":
        import subprocess

        salida = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
        return str(pid) in salida
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _regiones(run_dir):
    return len(validar_resultado(run_dir).regiones_encontradas)


if __name__ == "__main__":
    sys.exit(main())

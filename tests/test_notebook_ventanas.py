"""
Pruebas del notebook de Colab (notebooks/run_ventanas.ipynb), sin Colab:

  1. El notebook coincide con sus celdas fuente y TODAS las celdas de codigo
     compilan (ast + compile; las lineas de magia `!`/`%` se omiten).
  2. Las rutas de las celdas coinciden con io_drive.py, y todo `vt.<nombre>` que
     usan las celdas existe en ventanas_tools.
  3. Se EJECUTAN las celdas 2, 3 y 4 en un espacio de nombres controlado:
       - auditoria con regiones faltantes (el caso del KeyError 'problema') y con
         todas presentes (y con los datos reales locales si existen);
       - estado del plan (completa / parcial / no iniciada);
       - lanzamiento: sesiones A y B sin traslape, solo corridas de Colab, todas
         las del plan cubiertas, rutas dentro de Pipeline_Resultados_Ventanas,
         ningun RUN_NAME repetido, y los tiempos se guardan.
  4. Nada de lo anterior escribe en Pipeline_Resultados/ (originales).

    python tests/test_notebook_ventanas.py
"""

import ast
import json
import os
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "tests"))

for nombre in [
    "optuna", "optuna.samplers", "xgboost", "lightgbm", "tensorflow", "tensorflow.keras",
    "tensorflow.keras.models", "tensorflow.keras.layers", "tensorflow.keras.callbacks",
    "tensorflow.keras.optimizers", "tensorflow.keras.backend",
]:
    sys.modules.setdefault(nombre, mock.MagicMock())

import build_run_ventanas_notebook as bld  # noqa: E402
from tesis_forecast import colab_aislado as ca  # noqa: E402
from tesis_forecast import io_drive, runner, windows  # noqa: E402
from tesis_forecast import ventanas_tools as vt  # noqa: E402
from tesis_forecast.matrix import MatrixRunRecord, resumen_dataframe  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def celdas_codigo():
    with open(bld.SALIDA, encoding="utf-8") as f:
        nb = json.load(f)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def ejecutar(codigo, ns):
    exec(compile(bld.sin_magics(codigo), "<celda>", "exec"), ns)


def main():
    print("1. Sintaxis y consistencia del notebook")
    celdas = bld.cargar_celdas()
    bld.validar(celdas)
    check(True, "ast.parse de todas las celdas de codigo")
    todas = celdas_codigo()
    # las celdas de TRABAJADORES adicionales (1b y 4-W) son opcionales y no alteran el flujo del Colab principal
    codigo = [c for c in todas if "TRABAJADOR" not in c]
    check(len(todas) == 8 and len(todas) - len(codigo) == 1, "8 celdas de codigo: 7 del flujo principal + la celda 4-W de trabajadores")
    for i, c in enumerate(codigo, start=1):
        compile(bld.sin_magics(c), "<celda %d>" % i, "exec")
    check(len(codigo) == 7, "%d celdas de codigo compilan (compile())" % len(codigo))
    with open(bld.SALIDA, encoding="utf-8") as f:
        check(json.load(f) == bld.construir(celdas), "run_ventanas.ipynb coincide con notebooks/celdas_run_ventanas/")

    print("\n2. Rutas e importaciones")
    arbol = ast.parse(bld.sin_magics(codigo[0]))
    rutas = {}
    for n in ast.walk(arbol):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    rutas[t.id] = n.value.value
    check(rutas["VENT_DIR"] == io_drive.BASE_DIR_VENTANAS, "VENT_DIR == io_drive.BASE_DIR_VENTANAS")
    check(rutas["ORIG_DIR"] == io_drive.BASE_DIR, "ORIG_DIR == io_drive.BASE_DIR (solo lectura)")
    check(rutas["DATA_DIR"] == "/content/drive/MyDrive/Bases de datos Tesis", "DATA_DIR segun docs/DATOS_REQUERIDOS.md")
    check("Pipeline_Resultados_Ventanas" in rutas["VENT_DIR"] and rutas["VENT_DIR"] != rutas["ORIG_DIR"],
          "resultados nuevos y originales en carpetas distintas")
    usados = set()
    for c in codigo:
        for n in ast.walk(ast.parse(bld.sin_magics(c))):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "vt":
                usados.add(n.attr)
    faltan = [a for a in usados if not hasattr(vt, a)]
    check(not faltan, "todo vt.<x> usado existe en ventanas_tools (%s)" % sorted(usados))
    check(all(hasattr(__import__("tesis_forecast.matrix", fromlist=["x"]), a) for a in ("run_matrix", "resumen_dataframe")),
          "run_matrix y resumen_dataframe importables")

    tmp = tempfile.mkdtemp(prefix="test_nb_")
    try:
        vent = os.path.join(tmp, "Pipeline_Resultados_Ventanas")
        os.makedirs(os.path.join(vent, "_auditoria"))
        base_ns = {"os": os, "vt": vt, "VENT_DIR": vent, "ORIG_DIR": os.path.join(tmp, "Pipeline_Resultados"),
                   "resumen_dataframe": resumen_dataframe, "ca": ca, "__builtins__": __builtins__}

        print("\n3. Celda 2 (auditoria) con regiones faltantes y completas")
        from test_windows import construir_datos  # datos sinteticos: CEN y NES completas, BCA corta
        datos = os.path.join(tmp, "datos")
        os.makedirs(datos)
        construir_datos(datos)
        ns = dict(base_ns, DATA_DIR=datos)
        try:
            ejecutar(codigo[1], dict(ns))  # celda 2 de codigo
            check(True, "celda 2 se ejecuta con regiones faltantes (sin KeyError 'problema')")
        except Exception as e:
            check(False, "celda 2 con regiones faltantes: %s: %s" % (type(e).__name__, e))
        check(os.path.exists(os.path.join(vent, "_auditoria", "plan_corridas_ventanas.csv")), "celda 2 guarda el plan")
        datos_reales = os.path.join(RAIZ, "datos_locales")
        if os.path.isdir(datos_reales):
            try:
                ejecutar(codigo[1], dict(base_ns, DATA_DIR=datos_reales))
                check(True, "celda 2 se ejecuta con los datos reales locales (todas las regiones presentes)")
            except Exception as e:
                check(False, "celda 2 con datos reales: %s: %s" % (type(e).__name__, e))
        else:
            print("  (datos_locales/ no existe: se omite la prueba con datos reales)")

        print("\n4. Celda 3 (estado) y reanudacion")
        plan = vt.plan_dataframe(ventanas=["S4"])
        colab = plan[plan["ejecuta_en"] == "colab"].reset_index(drop=True)
        run_completa, run_parcial = colab.loc[0, "run_name"], colab.loc[1, "run_name"]
        for nombre, regiones in [(run_completa, ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]), (run_parcial, ["BCA", "CEN"])]:
            d = os.path.join(vent, "S4", nombre)
            os.makedirs(d)
            json.dump({"modelo": "x"}, open(os.path.join(d, "config.json"), "w"))
            pd.DataFrame({"region": regiones, "MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0, "sMAPE": 1.0}).to_csv(os.path.join(d, "metricas.csv"), index=False)
            pd.DataFrame({"x": [1]}).to_csv(os.path.join(d, "series.csv"), index=False)
        ns3 = dict(base_ns)
        ejecutar(codigo[2], ns3)
        est = ns3["inf"]["estado"]
        s = est[est["ventana"] == "S4"].set_index("run_name")["estado"]
        check(s[run_completa] == "completa" and s[run_parcial] == "parcial", "estado_plan distingue completa / parcial")
        check((est["estado"] == "no_iniciada").sum() == len(est) - 2 and len(est) == 28 * 4, "las demas corridas de Colab figuran no iniciadas (112 en total)")
        check(os.path.exists(os.path.join(vent, "_auditoria", "faltan_colab.csv")) and len(pd.read_csv(os.path.join(vent, "_auditoria", "faltan_colab.csv"))) == 111,
              "la celda 3 guarda los faltantes (111 = 112 - 1 completa)")

        print("\n5. Celda 4 (lanzamiento) integrada con la ejecucion aislada: sesiones A y B (lote T2, subprocesos simulados)")
        import types

        base_original = io_drive.BASE_DIR_VENTANAS
        io_drive.BASE_DIR_VENTANAS = vent  # las rutas de resultados quedan en el directorio temporal
        try:
            hijo_falso = os.path.join(RAIZ, "tests", "_hijo_falso.py")

            def comando(cfg_json, res_json, data_dir, vent_dir):
                from tesis_forecast.config import ExperimentConfig

                cfg = ExperimentConfig(**json.load(open(cfg_json)))
                return [sys.executable, hijo_falso, runner.resolve_run(cfg).run_dir, "ok", cfg_json + ".contador", res_json]

            ca_prueba = types.SimpleNamespace(
                ejecutar_sesion=lambda *a, **k: ca.ejecutar_sesion(*a, comando_hijo=comando, sondeo_s=0.5, tmp_dir=os.path.join(tmp, "sub"),
                                                                     intervalo_telemetria_s=1, **k),
                informe_progreso=ca.informe_progreso, guardar_faltantes=ca.guardar_faltantes)
            lanzadas = {}
            check('LOTE = "TODO"' in codigo[3] and "MAX_INTENTOS" in codigo[3], "la celda 4 expone SESION, LOTE (por defecto TODO) y MAX_INTENTOS")
            for sesion in ("A", "B"):
                ns4 = dict(base_ns, ca=ca_prueba, DATA_DIR=datos)
                fuente = codigo[3].replace('SESION = "A"', 'SESION = "%s"' % sesion).replace('LOTE = "TODO"', 'LOTE = "T2"')
                ejecutar(fuente, ns4)
                res = ns4["resumen"]
                vistas = res["status"].value_counts().to_dict()
                esperado = {"completed": 7, "skipped_ya_completo": 1} if sesion == "A" else {"completed": 8}  # en A se dejo una corrida ya completa (paso 4)
                check(len(res) == 8 and vistas == esperado, "sesion %s lote T2: 8 corridas, %s (la ya completa se salta y la parcial se reanuda)" % (sesion, vistas))
                check(set(res["ventana"]) == ({"S4", "S1"} if sesion == "A" else {"S2", "S3"}), "sesion %s procesa solo sus ventanas" % sesion)
                check(all(vt.ejecuta_en(m) == "colab" for m in res["modelo"]), "sesion %s: solo modelos de Colab (redes)" % sesion)
                check(os.path.exists(os.path.join(vent, "_auditoria", "tiempos_colab_%s.csv" % sesion)), "sesion %s guarda tiempos" % sesion)
                tel = os.path.join(vent, "_auditoria", "telemetria_colab_%s.csv" % sesion)
                check(os.path.exists(tel) and len(pd.read_csv(tel)) >= 2, "sesion %s guarda telemetria de RAM/GPU en _auditoria" % sesion)
                check(os.path.exists(os.path.join(vent, "_auditoria", "subprocesos_colab_%s.csv" % sesion)), "sesion %s registra cada subproceso" % sesion)
                lanzadas[sesion] = {(r.ventana, r.run_name) for r in res.itertuples()}
            a, b = lanzadas["A"], lanzadas["B"]
            check(not (a & b), "sesiones A y B sin traslape")
            plan_t2 = {(r.ventana, r.run_name) for r in vt.plan_dataframe().query("ejecuta_en == 'colab' and nivel == 'T2'").itertuples()}
            check((a | b) == plan_t2 and len(plan_t2) == 16, "A u B cubren exactamente las 16 corridas T2 de Colab del plan")
            rutas = [os.path.join(vent, v, n) for v, n in plan_t2]
            check(all(os.path.isdir(r) for r in rutas), "todas las rutas dentro de Pipeline_Resultados_Ventanas/")
            check(not os.path.exists(io_drive.BASE_DIR), "nada en Pipeline_Resultados/ (originales)")
            plan_colab = {(r.ventana, r.run_name) for r in vt.plan_dataframe().query("ejecuta_en == 'colab'").itertuples()}
        finally:
            io_drive.BASE_DIR_VENTANAS = base_original

        print("\n6. La PC y Colab no se pisan")
        local = {(r.ventana, r.run_name) for r in vt.plan_dataframe().query("ejecuta_en == 'local'").itertuples()}
        check(not (local & plan_colab) and len(local) + len(plan_colab) == 216, "PC (104) + Colab (112) = 216 sin interseccion")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODO OK" if not FALLOS else "%d FALLO(S): %s" % (len(FALLOS), FALLOS)))
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Prueba de extremo a extremo del entrenamiento distribuido con "Drives" SIMULADOS (carpetas locales; sin rclone ni red):

  descargar (staging) -> validar -> detectar conflictos -> consolidar sin sobrescribir -> reporte, mas la asignacion
  exclusiva y el registro del entorno. Usa corridas locales REALES (LightGBM y XGBoost de S4) como si las hubiera
  entrenado un Colab, y las degrada a proposito (regiones truncadas, archivo alterado) para comprobar que se detectan.

    python tests/test_distribuido.py
"""

import json
import os
import shutil
import sys
import tempfile

import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "src"))
import dist as D  # noqa: E402
from tesis_forecast import distribuido as dd  # noqa: E402

ORIG = os.path.join(RAIZ, "resultados_locales", "Pipeline_Resultados_Ventanas", "S4")
OK_RUN = "LightGBM_train336h_fh168h_Gen"          # valida
DEFECTO_RUN = "XGBoost_train336h_fh168h_Gen"      # todos los intentos de Optuna en inf
TRUNC_RUN = "LightGBM_train336h_fh168h_IGAE"      # se trunca a 3 regiones
ALTERADA_RUN = "LightGBM_train336h_fh168h_Imp"    # se altera un archivo despues de validar
FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def copiar(run, destino_base):
    destino = os.path.join(destino_base, "S4", run)
    shutil.copytree(os.path.join(ORIG, run), destino)
    return destino


def main():
    if not os.path.isdir(os.path.join(ORIG, OK_RUN)):
        print("no hay resultados locales de referencia: prueba omitida")
        return 0
    tmp = tempfile.mkdtemp(prefix="test_dist_")
    base0 = D.BASE
    try:
        D.BASE = os.path.join(tmp, "estado")
        w2 = os.path.join(tmp, "drive_w2", "Pipeline_Resultados_Ventanas")
        pr = os.path.join(tmp, "drive_principal", "Pipeline_Resultados_Ventanas")
        for run in (OK_RUN, DEFECTO_RUN, TRUNC_RUN, ALTERADA_RUN):
            copiar(run, w2)
        copiar("LightGBM_train336h_fh168h_Exp", pr)       # algo ya existente en el principal
        # degradar la corrida truncada: dejar solo 3 regiones en metricas y series
        d = os.path.join(w2, "S4", TRUNC_RUN)
        for nombre in ("metricas.csv", "series.csv"):
            x = pd.read_csv(os.path.join(d, nombre), encoding="utf-8-sig")
            x[x["region"].isin(["BCA", "CEN", "NES"])].to_csv(os.path.join(d, nombre), index=False, encoding="utf-8-sig")
        # ambientes registrados por cada trabajador (versiones, commit, datos)
        for carpeta, nombre in ((w2, "W2"), (pr, "PRINCIPAL")):
            amb = dd.registrar_ambiente(carpeta, nombre, RAIZ, os.path.join(RAIZ, "datos_locales"))
        check(amb["datos"]["ok"] is True and amb["datos"]["coinciden"] == amb["datos"]["archivos_referencia"],
              "los datos de entrada coinciden con la lista de referencia (%d archivos)" % amb["datos"]["archivos_referencia"])
        check(os.path.isfile(os.path.join(w2, "_auditoria", "ambiente_W2.json")) and amb["versiones"]["numpy"] is not None, "se registra el ambiente (versiones y commit)")

        cfg = {"staging": os.path.join(tmp, "staging"), "trabajadores": [
            {"nombre": "PRINCIPAL", "rol": "principal", "origen": "local:" + pr, "destino_consolidacion": "local:" + pr},
            {"nombre": "W2", "rol": "trabajador", "origen": "local:" + w2}]}

        print("1. sincronizacion")
        for t in cfg["trabajadores"]:
            n = D.sincronizar(cfg, t)
            check(n and n > 0, "%s: se descargan archivos (%s)" % (t["nombre"], n))
        check(D.sincronizar(cfg, cfg["trabajadores"][1]) == 0, "una segunda sincronizacion no copia nada (sin cambios)")

        print("2. validacion")
        df = D.validar_todo(cfg, ambito="todo")
        r = {(f["trabajador"], f["run_name"]): f for _, f in df.iterrows()}
        check(bool(r[("W2", OK_RUN)]["ok"]), "LightGBM completa y correcta: valida")
        check(not r[("W2", DEFECTO_RUN)]["ok"] and "ningun intento con valor finito" in str(r[("W2", DEFECTO_RUN)]["primer_error"]),
              "XGBoost con todos los intentos en 'inf': ERROR de Optuna (%s)" % str(r[("W2", DEFECTO_RUN)]["primer_error"])[:50])
        check(not r[("W2", TRUNC_RUN)]["ok"], "corrida truncada a 3 regiones: ERROR")
        inf = json.load(open(os.path.join(D.BASE, "validacion", "W2__S4__%s.json" % TRUNC_RUN), encoding="utf-8"))
        codigos = {h["codigo"] for h in inf["hallazgos"] if h["nivel"] == "ERROR"}
        check({"regiones", "predicciones"} <= codigos, "el error cita regiones y predicciones: %s" % sorted(codigos))
        ok = json.load(open(os.path.join(D.BASE, "validacion", "W2__S4__%s.json" % OK_RUN), encoding="utf-8"))
        check(len(ok["huellas"]) >= 6 and all("sha256" in h for h in ok["huellas"].values()), "huellas SHA-256 de cada archivo")

        print("3. conflictos y cambios")
        # la misma corrida en el principal con contenido distinto -> conflicto en ambos
        copiar(OK_RUN, pr)
        m = os.path.join(pr, "S4", OK_RUN, "metricas.csv")
        x = pd.read_csv(m, encoding="utf-8-sig")
        x.loc[0, "MAE"] = x.loc[0, "MAE"] + 1.0
        x.to_csv(m, index=False, encoding="utf-8-sig")
        D.sincronizar(cfg, cfg["trabajadores"][0])
        df = D.validar_todo(cfg, ambito="todo")
        c = df[df["run_name"] == OK_RUN]
        check(len(c) == 2 and not c["ok"].any() and c["primer_error"].astype(str).str.contains("contenido distinto").all(),
              "la misma corrida con contenido distinto en dos trabajadores: CONFLICTO en ambos")
        shutil.rmtree(os.path.join(pr, "S4", OK_RUN))
        shutil.rmtree(os.path.join(D.staging(cfg, "PRINCIPAL"), "S4", OK_RUN))
        # archivo alterado despues de validar -> aviso de integridad
        D.validar_todo(cfg, ambito="todo")
        a = os.path.join(w2, "S4", ALTERADA_RUN, "log.txt")
        open(a, "a", encoding="utf-8").write("\nlinea agregada despues\n")
        D.sincronizar(cfg, cfg["trabajadores"][1])
        D.validar_todo(cfg, ambito="todo")
        j = json.load(open(os.path.join(D.BASE, "validacion", "W2__S4__%s.json" % ALTERADA_RUN), encoding="utf-8"))
        check(any(h["codigo"] == "integridad" for h in j["hallazgos"]), "un archivo que cambio desde la validacion anterior se avisa")

        print("4. consolidacion (sin sobrescribir)")
        ensayo = D.consolidar(cfg, confirmar=False)
        acc = dict(zip(ensayo["run_name"], ensayo["accion"]))
        check(acc[OK_RUN] == "COPIARIA" and acc[DEFECTO_RUN] == "OMITIDA" and acc[TRUNC_RUN] == "OMITIDA", "ensayo: solo la corrida validada se copiaria")
        check(not os.path.isdir(os.path.join(pr, "S4", OK_RUN)), "el ensayo no escribe nada")
        real = D.consolidar(cfg, confirmar=True)
        check(os.path.isdir(os.path.join(pr, "S4", OK_RUN)) and os.path.isfile(os.path.join(pr, "_consolidacion", "S4__%s.json" % OK_RUN)),
              "con --confirmar se copia la corrida y se escribe su traza de consolidacion")
        traza = json.load(open(os.path.join(pr, "_consolidacion", "S4__%s.json" % OK_RUN), encoding="utf-8"))
        check(traza["origen_trabajador"] == "W2" and traza["validacion"]["ok"], "la traza registra origen y validacion")
        dos = D.consolidar(cfg, confirmar=True)
        check(dict(zip(dos["run_name"], dos["accion"]))[OK_RUN] == "YA_CONSOLIDADA", "una segunda consolidacion no repite nada")
        # destino modificado -> conflicto, y no se toca
        m = os.path.join(pr, "S4", OK_RUN, "config.json")
        previo = open(m, encoding="utf-8").read()
        open(m, "a", encoding="utf-8").write(" ")
        tres = D.consolidar(cfg, confirmar=True)
        check(dict(zip(tres["run_name"], tres["accion"]))[OK_RUN] == "CONFLICTO" and open(m, encoding="utf-8").read() == previo + " ",
              "si el destino difiere: CONFLICTO y no se sobrescribe")

        print("5. asignacion exclusiva")
        cfg3 = {"staging": cfg["staging"], "trabajadores": [
            {"nombre": "PRINCIPAL", "rol": "principal", "origen": "local:" + pr},
            {"nombre": "W2", "rol": "trabajador", "origen": "local:" + w2},
            {"nombre": "W3", "rol": "trabajador", "origen": "local:" + os.path.join(tmp, "x")}]}
        pend, res = D.asignar(cfg3, os.path.join(tmp, "asig"), reservar_principal=["T3-fcnn:B"])
        check(len(pend) == 112 and pend["trabajador"].ne("").all(), "los 112 pendientes de Colab quedan asignados")
        check(pend.duplicated(["ventana", "run_name"]).sum() == 0, "ninguna configuracion se asigna dos veces")
        fcnn_b = pend[(pend["modelo"] == "fcnn") & (pend["nivel"] == "T3") & (pend["ventana"].isin(["S2", "S3"]))]
        check(len(fcnn_b) == 12 and (fcnn_b["trabajador"] == "PRINCIPAL").all(), "el lote actual del principal (T3-fcnn B, 12) se queda en el principal")
        cargas = {k: v["horas_estimadas"] for k, v in res["por_trabajador"].items()}
        check(max(cargas.values()) - min(cargas.values()) < 0.45 * max(cargas.values()) + 1, "la carga queda razonablemente balanceada: %s" % cargas)
        cs = [pd.read_csv(os.path.join(tmp, "asig", f)) for f in ("PRINCIPAL.csv", "W2.csv", "W3.csv")]
        check(sum(len(c) for c in cs) == 112 and len(set(zip(*[pd.concat(cs)[k] for k in ("ventana", "run_name")]))) == 112, "los CSV de asignacion son disjuntos y suman 112")
        check(all(set(c["trabajador"]) == {n} for c, n in zip(cs, ("PRINCIPAL", "W2", "W3"))), "cada CSV lleva solo su trabajador")
        try:
            dd.ejecutar_asignacion(os.path.join(tmp, "asig", "W2.csv"), trabajador="W3", vent_dir=tmp, data_dir=tmp)
            check(False, "un trabajador no puede ejecutar la asignacion de otro")
        except ValueError:
            check(True, "un trabajador no puede ejecutar la asignacion de otro")

        # desde faltan_colab.csv: 38 completas (no figuran) y 74 pendientes, 5 de ellas parciales
        plan = D.vt.plan_dataframe()
        plan = plan[plan["ejecuta_en"] == "colab"].reset_index(drop=True)
        faltan = plan.iloc[38:].copy()
        faltan["estado"] = ["parcial" if i < 5 else "no_iniciada" for i in range(len(faltan))]
        faltan["regiones_ok"] = [3 if i < 5 else 0 for i in range(len(faltan))]
        ruta_faltan = os.path.join(tmp, "faltan_colab.csv")
        faltan.to_csv(ruta_faltan, index=False)
        pend2, res2 = D.asignar(cfg3, os.path.join(tmp, "asig2"), reservar_principal=["T3-fcnn:B"], faltan_csv=ruta_faltan)
        check(len(pend2) == 74 and res2["completas_en_principal"] == 38, "desde faltan_colab.csv: 38 completas y 74 pendientes")
        parc = pend2[pend2["estado"] == "parcial"]
        check(len(parc) == 5 and (parc["trabajador"] == "PRINCIPAL").all(), "lo parcial se queda con el principal (no se duplica)")
        check(not set(zip(pend2["ventana"], pend2["run_name"])) & set(zip(plan.iloc[:38]["ventana"], plan.iloc[:38]["run_name"])), "nada de lo completo se reasigna")

        print("6. reporte central")
        texto = D.reporte(cfg3, os.path.join(tmp, "asig"))
        check("PRINCIPAL" in texto and "W2" in texto and "Reporte central" in texto, "el reporte central se genera")
        check(os.path.isfile(os.path.join(D.BASE, "reporte_central.md")), "y se guarda en disco")
    finally:
        D.BASE = base0
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

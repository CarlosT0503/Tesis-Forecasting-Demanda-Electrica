"""
Auditoria de solo lectura de los trials de LSTM directa (`scripts/auditar_trials_lstm_directa.py`).

Crea bases Optuna reales en una carpeta temporal con la estructura de las ventanas y comprueba que el informe
distingue: region con 10 intentos, region con 20 donde el mejor esta entre los primeros 10 (el exceso no cambia la
seleccion), region con 20 donde el mejor esta en los intentos 11-20 (si la cambia), y que la base no se modifica.

    python tests/test_auditar_trials_lstm_directa.py
"""

import hashlib
import os
import shutil
import sys
import tempfile

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

try:
    import optuna
except ImportError:
    print("optuna no esta instalado: prueba omitida")
    sys.exit(0)

optuna.logging.set_verbosity(optuna.logging.WARNING)
import auditar_trials_lstm_directa as aud  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def _estudio(db, nombre, valores):
    """Estudio real con un trial por valor, en orden; units/dropout categoricos como en la LSTM directa."""
    estudio = optuna.create_study(direction="minimize", study_name=nombre, storage=f"sqlite:///{db}", load_if_exists=True)
    it = iter(valores)

    def objetivo(trial):
        trial.suggest_categorical("units", [32, 64, 128])
        trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
        return next(it)

    estudio.optimize(objetivo, n_trials=len(valores))


def _md5(ruta):
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="aud_lstm_")
    try:
        base = os.path.join(tmp, "Pipeline_Resultados_Ventanas")
        corrida = os.path.join(base, "S4", "LSTM_Directa_train2160h_fh168h_Temp")
        os.makedirs(corrida)
        db = os.path.join(corrida, "optuna_lstm_directa.db")

        primeros = [9.0, 8.0, 7.5, 9.5, 8.5, 9.1, 8.8, 9.9, 8.2, 9.3]
        _estudio(db, "demanda_CEN_lstm_directa", primeros)  # 10 intentos
        _estudio(db, "demanda_OCC_lstm_directa", primeros + [8.0, 9.0, 9.2, 8.1, 9.4, 9.0, 8.9, 9.1, 8.4, 9.6])  # 20, mejor entre los 10
        _estudio(db, "demanda_NTE_lstm_directa", primeros + [8.0, 6.5, 9.2, 8.1, 9.4, 9.0, 8.9, 9.1, 8.4, 9.6])  # 20, mejor en el 12
        # Estudio ya con 10 completos, reanudado como lo hacia el codigo anterior: otros 10 encima
        _estudio(db, "demanda_ORI_lstm_directa", primeros)
        _estudio(db, "demanda_ORI_lstm_directa", [7.0] * 10)  # el mejor, 7.0, queda en el trial 10

        antes = _md5(db)
        filas = aud.auditar(base, 10)
        despues = _md5(db)
        check(antes == despues, "la base no se modifica (hash identico antes y despues)")

        por_region = {f["region"]: f for f in filas}
        check(set(por_region) == {"CEN", "OCC", "NTE", "ORI"}, "un registro por estudio/region")
        check(all(f["ventana"] == "S4" and f["corrida"].startswith("LSTM_Directa") for f in filas), "ventana y corrida leidas de la ruta")

        cen = por_region["CEN"]
        check(cen["n_completos"] == 10 and cen["excede_presupuesto"] is False, "CEN: 10 completos, no excede")
        check(cen["mejor_fuera_del_presupuesto"] is False and cen["mejora_por_exceso"] == 0, "CEN: sin diferencia")

        occ = por_region["OCC"]
        check(occ["n_completos"] == 20 and occ["excede_presupuesto"] is True, "OCC: 20 completos, excede")
        check(occ["trial_mejor_primeros"] == 2 and occ["trial_mejor_todos"] == 2, "OCC: el mejor (7.5) sigue en el trial 2")
        check(occ["mejor_fuera_del_presupuesto"] is False and occ["mismos_params"] is True, "OCC: el exceso NO cambia la seleccion")

        nte = por_region["NTE"]
        check(nte["excede_presupuesto"] is True and nte["trial_mejor_todos"] == 11, "NTE: el mejor global es el trial 11 (6.5)")
        check(nte["trial_mejor_primeros"] == 2 and nte["mejor_valor_primeros"] == 7.5, "NTE: el mejor de los primeros 10 es 7.5")
        check(nte["mejor_fuera_del_presupuesto"] is True, "NTE: el exceso SI cambia la seleccion")
        check(abs(nte["mejora_por_exceso"] - 1.0) < 1e-9, "NTE: mejora por exceso = 7.5 - 6.5 = 1.0")
        check(nte["params_mejor_todos"].startswith("{") and "units" in nte["params_mejor_todos"], "parametros decodificados (no indices)")
        check(nte["params_mejor_todos"].count(".") >= 1 or "dropout" in nte["params_mejor_todos"], "dropout presente")

        ori = por_region["ORI"]
        check(ori["n_completos"] == 20 and ori["trial_mejor_todos"] == 10 and ori["mejor_valor_todos"] == 7.0,
              "ORI: 10 + 10 reanudados, mejor 7.0 en el trial 10")
        check(ori["mejor_fuera_del_presupuesto"] is True, "ORI: cambia la seleccion (7.0 < 7.5)")

        texto = aud.resumen(filas)
        check("con mas intentos completos que el presupuesto: 3" in texto, "el resumen cuenta 3 estudios con exceso")
        check("cambia la seleccion): 2" in texto, "el resumen cuenta 2 que cambian la seleccion")

        # informe CSV: se escribe fuera de la carpeta de corridas; dentro de ella se rechaza
        salida = os.path.join(tmp, "informe.csv")
        aud.escribir_csv(filas, salida, base)
        check(os.path.getsize(salida) > 0, "informe CSV escrito fuera de las carpetas de corrida")
        try:
            aud.escribir_csv(filas, os.path.join(corrida, "informe.csv"), base)
            check(False, "rechaza escribir dentro de una carpeta de corrida")
        except ValueError:
            check(True, "rechaza escribir dentro de una carpeta de corrida")
        os.makedirs(os.path.join(base, "_auditoria"))
        aud.escribir_csv(filas, os.path.join(base, "_auditoria", "informe.csv"), base)
        check(True, "permite escribir en _auditoria")

        check(aud.main(["--base-ventanas", os.path.join(tmp, "vacia")]) == 2, "sin bases: codigo de salida 2")
        check(aud.main(["--base-ventanas", base]) == 0, "con bases: codigo de salida 0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

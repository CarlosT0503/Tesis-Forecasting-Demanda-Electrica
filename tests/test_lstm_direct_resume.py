"""
LSTM directa: al reanudar una region, solo se ejecutan los intentos de Optuna que faltan (presupuesto original).

Usa Optuna real con un .db SQLite temporal y sustituye el objetivo (que entrenaria la red) por una funcion barata;
TensorFlow se sustituye por stubs. Lo que se prueba es la logica de reanudacion de `tune_lstm`, no la red.

    python tests/test_lstm_direct_resume.py
"""

import os
import shutil
import sys
import tempfile
from unittest import mock

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))

for nombre in [
    "tensorflow", "tensorflow.keras", "tensorflow.keras.models", "tensorflow.keras.layers",
    "tensorflow.keras.callbacks", "tensorflow.keras.optimizers", "tensorflow.keras.backend",
]:
    sys.modules.setdefault(nombre, mock.MagicMock())

try:
    import optuna
except ImportError:
    print("optuna no esta instalado: prueba omitida")
    sys.exit(0)

optuna.logging.set_verbosity(optuna.logging.WARNING)
from tesis_forecast.models import lstm_direct  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def completos(db, nombre="CEN_DEMANDA"):
    st = optuna.load_study(study_name=f"{nombre}_lstm_directa", storage=f"sqlite:///{db}")
    return len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]), st


def main():
    tmp = tempfile.mkdtemp(prefix="test_lstm_")
    try:
        llamadas = {"n": 0, "interrumpir_en": None}

        def objetivo(trial, *args, **kw):
            llamadas["n"] += 1
            if llamadas["interrumpir_en"] is not None and llamadas["n"] == llamadas["interrumpir_en"]:
                raise KeyboardInterrupt("la sesion se corta a mitad de un intento")
            return trial.suggest_float("x", 0.0, 1.0) + trial.suggest_categorical("w", [24, 48, 168]) / 1000.0

        with mock.patch.object(lstm_direct, "objective_lstm", side_effect=objetivo):
            args = (None, None, "CEN_DEMANDA", ["Temperatura"], 168)

            print("1. Region nueva: se corren exactamente los 10 intentos del presupuesto")
            db1 = os.path.join(tmp, "a.db")
            lstm_direct.tune_lstm(*args, db1, 10)
            check(llamadas["n"] == 10 and completos(db1)[0] == 10, "10 intentos completos (llamadas=%d)" % llamadas["n"])

            print("2. Reanudar con el presupuesto cumplido: NO corre ningun intento mas")
            llamadas["n"] = 0
            params, df = lstm_direct.tune_lstm(*args, db1, 10)
            check(llamadas["n"] == 0 and completos(db1)[0] == 10, "0 intentos nuevos; siguen 10 en el estudio")
            check(len(df) == 10 and set(params) == {"x", "w"}, "devuelve los mejores parametros y los 10 intentos")

            print("3. Region cortada a mitad (3 de 10 hechos): al reanudar corre solo los 7 que faltan")
            db2 = os.path.join(tmp, "b.db")
            llamadas["n"] = 0
            lstm_direct.tune_lstm(*args, db2, 3)
            check(completos(db2)[0] == 3, "estado de partida: 3 intentos completos")
            llamadas["n"] = 0
            lstm_direct.tune_lstm(*args, db2, 10)
            check(llamadas["n"] == 7 and completos(db2)[0] == 10, "se ejecutaron 7 y el total es 10 (antes habria sido 13); llamadas=%d" % llamadas["n"])

            print("4. La sesion muere DENTRO de un intento (KeyboardInterrupt): el intento fallido no cuenta")
            db3 = os.path.join(tmp, "c.db")
            llamadas["n"], llamadas["interrumpir_en"] = 0, 4
            try:
                lstm_direct.tune_lstm(*args, db3, 10)
            except KeyboardInterrupt:
                pass
            check(completos(db3)[0] == 3, "quedaron 3 completos y 1 fallido")
            llamadas["n"], llamadas["interrumpir_en"] = 0, None
            lstm_direct.tune_lstm(*args, db3, 10)
            c, st = completos(db3)
            check(c == 10 and llamadas["n"] == 7, "al reanudar completa los 7 que faltan: %d completos con %d llamadas" % (c, llamadas["n"]))

            print("5. Un intento que quedo en estado RUNNING (proceso muerto) tampoco cuenta")
            db4 = os.path.join(tmp, "d.db")
            llamadas["n"] = 0
            lstm_direct.tune_lstm(*args, db4, 2)
            st4 = optuna.load_study(study_name="CEN_DEMANDA_lstm_directa", storage=f"sqlite:///{db4}")
            st4.ask()  # deja un intento RUNNING sin cerrar
            llamadas["n"] = 0
            lstm_direct.tune_lstm(*args, db4, 5)
            check(completos(db4)[0] == 5 and llamadas["n"] == 3, "5 completos; se ejecutaron 3 (llamadas=%d)" % llamadas["n"])

            print("6. Un presupuesto ya excedido (estudio heredado con mas de 10) no agrega intentos")
            db5 = os.path.join(tmp, "e.db")
            lstm_direct.tune_lstm(*args, db5, 13)
            llamadas["n"] = 0
            lstm_direct.tune_lstm(*args, db5, 10)
            check(llamadas["n"] == 0, "no se ejecuta nada mas")

        print("7. _intentos_pendientes")
        class T:
            def __init__(self, s):
                self.state = s

        S = optuna.trial.TrialState
        est = mock.MagicMock()
        est.trials = [T(S.COMPLETE)] * 4 + [T(S.FAIL), T(S.RUNNING)]
        check(lstm_direct._intentos_pendientes(est, 10) == 6, "4 completos de 10 -> faltan 6 (FAIL y RUNNING no cuentan)")
        check(lstm_direct._intentos_pendientes(est, 3) == 0, "presupuesto menor al ya hecho -> 0 (nunca negativo)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

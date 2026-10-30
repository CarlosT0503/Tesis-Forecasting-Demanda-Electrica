"""
Pruebas de la escritura atomica de CSV y de su uso en los modelos neuronales y en el runner (sin TensorFlow:
se sustituye por stubs; solo se ejercita el guardado).

    python tests/test_atomic_io.py
"""

import os
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
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

from tesis_forecast import atomic_io  # noqa: E402
from tesis_forecast.atomic_io import escribir_csv_atomico, limpiar_temporales  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def listar(carpeta):
    return sorted(os.listdir(carpeta))


def main():
    tmp = tempfile.mkdtemp(prefix="test_atomic_")
    try:
        df = pd.DataFrame({"serie": ["CEN_DEMANDA", "NES_DEMANDA"], "MAPE": [1.5, 2.5], "texto": ["áé", "ñü"]})

        print("1. Contenido identico al de DataFrame.to_csv")
        a, b = os.path.join(tmp, "atomico.csv"), os.path.join(tmp, "directo.csv")
        escribir_csv_atomico(df, a, index=False, encoding="utf-8-sig")
        df.to_csv(b, index=False, encoding="utf-8-sig")
        check(open(a, "rb").read() == open(b, "rb").read(), "los bytes coinciden con to_csv (BOM utf-8-sig y fin de linea incluidos)")
        check(pd.read_csv(a, encoding="utf-8-sig").equals(df), "se relee igual")
        check(listar(tmp) == ["atomico.csv", "directo.csv"], "no queda ningun temporal tras escribir")

        print("2. Un fallo a mitad de escritura deja intacto el archivo anterior")
        viejo = open(a, "rb").read()

        class Roto:
            def to_csv(self, f, **kw):
                f.write("encabezado,a medias\n1,2")
                raise OSError("disco lleno simulado")

        try:
            escribir_csv_atomico(Roto(), a, index=False)
            check(False, "debia propagar el error")
        except OSError:
            check(True, "el error se propaga")
        check(open(a, "rb").read() == viejo, "el CSV definitivo anterior sigue intacto")
        check(listar(tmp) == ["atomico.csv", "directo.csv"], "el temporal parcial se borro")

        print("3. Un fallo en el reemplazo tambien deja todo intacto")
        with mock.patch.object(atomic_io.os, "replace", side_effect=OSError("replace fallo")):
            try:
                escribir_csv_atomico(df.assign(MAPE=99.0), a, index=False, encoding="utf-8-sig")
            except OSError:
                pass
        check(open(a, "rb").read() == viejo, "el definitivo no cambio")
        check(listar(tmp) == ["atomico.csv", "directo.csv"], "sin temporales")

        print("4. El temporal se crea en la MISMA carpeta que el definitivo (mismo sistema de archivos)")
        carpetas = []
        real_mkstemp = tempfile.mkstemp

        def espia(*args, **kw):
            carpetas.append(os.path.abspath(kw.get("dir")))
            return real_mkstemp(*args, **kw)

        sub = os.path.join(tmp, "sub")
        os.makedirs(sub)
        with mock.patch.object(atomic_io.tempfile, "mkstemp", side_effect=espia):
            escribir_csv_atomico(df, os.path.join(sub, "x.csv"), index=False)
        check(carpetas == [os.path.abspath(sub)], "tempfile.mkstemp(dir=<carpeta destino>)")

        print("5. Si no se puede crear el temporal, se escribe directo (no se pierde el guardado)")
        destino = os.path.join(tmp, "fallback.csv")
        with mock.patch.object(atomic_io.tempfile, "mkstemp", side_effect=OSError("sin permiso")):
            escribir_csv_atomico(df, destino, index=False)
        check(pd.read_csv(destino).equals(df), "fallback escribe el CSV completo")

        print("6. limpiar_temporales solo borra restos propios")
        restos = [os.path.join(tmp, ".tmp_series.csv.abc.part"), os.path.join(tmp, ".tmp_metricas.csv.xyz.part")]
        ajenos = [os.path.join(tmp, "series.csv"), os.path.join(tmp, "notas.part"), os.path.join(tmp, ".tmp_otro.txt")]
        for r in restos + ajenos:
            open(r, "w").write("x")
        borrados = limpiar_temporales(tmp)
        check(sorted(borrados) == sorted(restos), "borra los .tmp_*.part")
        check(all(os.path.exists(r) for r in ajenos), "no toca archivos ajenos ni definitivos")
        check(limpiar_temporales(os.path.join(tmp, "no_existe")) == [], "carpeta inexistente: lista vacia")

        print("7. Los cuatro modelos neuronales guardan sus avances de forma atomica")
        from tesis_forecast.models import ensemble_stl, fcnn_model, lstm_direct, lstm_resid_model

        for nombre, mod in [("fcnn", fcnn_model), ("lstm_direct", lstm_direct), ("lstm_resid", lstm_resid_model), ("ensemble_stl", ensemble_stl)]:
            carpeta = os.path.join(tmp, nombre)
            os.makedirs(carpeta)
            res = mod._ResultsAccumulator()
            res.series = [{"serie": "CEN_DEMANDA", "fecha": pd.Timestamp("2026-05-17 00:00"), "tipo": "prediccion",
                           "subset": "test", "modelo": "M", "valor": 1.0}]
            if nombre in ("fcnn", "ensemble_stl", "lstm_resid"):
                res.series = [{"serie": "CEN_DEMANDA", "fecha": np.array([pd.Timestamp("2026-05-17 00:00")]), "tipo": "prediccion",
                               "subset": "test", "modelo": "M", "valor": np.array([1.0])}]
            res.metrics = [{"serie": "CEN_DEMANDA", "modelo": "M", "tuneado": True, "MAPE": 1.0, "sMAPE": 1.0, "MAE": 1.0, "RMSE": 1.0}]
            res.trials = [pd.DataFrame({"number": [0], "value": [1.0], "serie": ["CEN_DEMANDA"]})]
            res.config_usada = [{"serie": "CEN_DEMANDA", "modelo": "M", "parametros": "{}"}]
            llamadas = []
            original = mod.escribir_csv_atomico

            def espia_csv(df_, ruta, _l=llamadas, _o=original, **kw):
                _l.append(os.path.basename(ruta))
                return _o(df_, ruta, **kw)

            with mock.patch.object(mod, "escribir_csv_atomico", side_effect=espia_csv):
                mod._guardar_avance_csv(res, carpeta)
            check(sorted(llamadas) == ["config_usada.csv", "metricas.csv", "series.csv", "trials.csv"],
                  "%s: los 4 CSV pasan por escribir_csv_atomico" % nombre)
            check(listar(carpeta) == ["config_usada.csv", "metricas.csv", "series.csv", "trials.csv"], "%s: sin temporales" % nombre)
            # una caida durante el guardado siguiente no corrompe el anterior
            antes = {f: open(os.path.join(carpeta, f), "rb").read() for f in listar(carpeta)}
            res.metrics.append({"serie": "NES_DEMANDA", "modelo": "M", "tuneado": True, "MAPE": 2.0, "sMAPE": 2.0, "MAE": 2.0, "RMSE": 2.0})
            with mock.patch.object(atomic_io.os, "replace", side_effect=OSError("cae la sesion")):
                try:
                    mod._guardar_avance_csv(res, carpeta)
                except OSError:
                    pass
            despues = {f: open(os.path.join(carpeta, f), "rb").read() for f in listar(carpeta) if not f.startswith(".tmp_")}
            check(antes == despues, "%s: tras una caida los CSV previos quedan intactos y completos" % nombre)
            check(not [f for f in listar(carpeta) if f.startswith(".tmp_")], "%s: la caida no deja temporales" % nombre)

        print("8. El runner escribe los CSV finales de forma atomica")
        src = open(os.path.join(RAIZ, "src", "tesis_forecast", "runner.py"), encoding="utf-8").read()
        check(".to_csv(os.path.join(run_dir" not in src and src.count("escribir_csv_atomico(") >= 4,
              "runner.py ya no usa to_csv directo para series/metricas/trials/config_usada")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

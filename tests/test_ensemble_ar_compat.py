"""
Regresion del ajuste AR de Ensemble STL (`seleccionar_ar_por_aic`) ante statsmodels >= 0.15, donde `AutoReg` ya no acepta
`old_names` (TypeError en los 168 candidatos -> "No se pudo ajustar ningun AR valido." en todas las regiones).

  1. Compatibilidad: con un constructor que, como statsmodels 0.15, RECHAZA `old_names`, el ajuste funciona.
  2. Equivalencia: el resultado es identico al del codigo anterior (con old_names=False) cuando la version instalada lo
     acepta: mismo rezago, mismo AIC, mismos parametros y misma prediccion.
  3. Diagnostico: si todos los candidatos fallan, el error dice por que (excepcion y conteo, AIC no finito, observaciones).
  4. Un fallo parcial de candidatos no impide elegir el mejor de los validos.
  5. (Opcional) con statsmodels 0.15.0 REAL en una carpeta aparte: definir SM015_PATH=<carpeta con statsmodels 0.15.0>.

    python tests/test_ensemble_ar_compat.py
"""

import os
import subprocess
import sys
import warnings
from unittest import mock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
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

import statsmodels  # noqa: E402
from statsmodels.tsa.ar_model import AutoReg as AutoRegReal  # noqa: E402

from tesis_forecast.models import ensemble_stl as E  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


def ar_sintetico(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 50)
    y = np.zeros(n + 50)
    for t in range(2, n + 50):
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + e[t]
    return y[50:]


def constructor_015(*args, **kwargs):
    """Como AutoReg de statsmodels 0.15: la firma ya no tiene `old_names`."""
    if "old_names" in kwargs:
        raise TypeError("AutoReg.__init__() got an unexpected keyword argument 'old_names'")
    return AutoRegReal(*args, **kwargs)


def seleccionar_codigo_anterior(y, max_lag):
    """Copia de la logica ANTERIOR (con old_names=False) para comparar. Solo valida si la version instalada lo acepta."""
    y = pd.Series(y).astype(float).dropna().to_numpy()
    mejor, mejor_lag, mejor_aic = None, None, np.inf
    for lag in range(1, max_lag + 1):
        try:
            f = AutoRegReal(y, lags=lag, trend="c", old_names=False).fit()
            if np.isfinite(f.aic) and f.aic < mejor_aic:
                mejor, mejor_lag, mejor_aic = f, lag, f.aic
        except Exception:  # noqa: BLE001
            pass
    return mejor, mejor_lag


def main():
    print("statsmodels instalado:", statsmodels.__version__)
    y = ar_sintetico()

    print("1. Compatibilidad con un AutoReg sin old_names (statsmodels >= 0.15)")
    with mock.patch.object(E, "AutoReg", constructor_015):
        modelo, lag, df = E.seleccionar_ar_por_aic(y, max_lag=12)
        check(modelo is not None and 1 <= lag <= 12 and len(df) == 12 and df["AIC"].notna().all(), "elige un AR valido (rezago %s) con los 12 candidatos ajustados" % lag)
        check("error" not in df.columns, "ningun candidato fallo")
        pred, _, lag2, _ = E.forecast_ar_resid(y, 24)
        check(len(pred) == 24 and np.isfinite(pred).all(), "forecast_ar_resid devuelve 24 predicciones finitas")
    fuente = open(os.path.join(RAIZ, "src", "tesis_forecast", "models", "ensemble_stl.py"), encoding="utf-8").read()
    check("old_names=" not in fuente.replace("`old_names`", "").replace("# Sin", ""), "ensemble_stl.py ya no pasa old_names a AutoReg")

    print("2. Equivalencia numerica con el codigo anterior")
    try:
        AutoRegReal(y[:200], lags=1, trend="c", old_names=False)
        acepta = True
    except TypeError:
        acepta = False
    if not acepta:
        print("  (se omite: la version instalada ya rechaza old_names; la equivalencia se comprueba con 0.14.x)")
    else:
        for max_lag, serie, nombre in ((20, y, "AR sintetico"),):
            nuevo, lag_n, _ = E.seleccionar_ar_por_aic(serie, max_lag=max_lag)
            viejo, lag_v = seleccionar_codigo_anterior(serie, max_lag)
            check(lag_n == lag_v and nuevo.aic == viejo.aic and np.array_equal(np.asarray(nuevo.params), np.asarray(viejo.params)),
                  "%s: mismo rezago (%s), AIC y parametros identicos bit a bit" % (nombre, lag_n))
            p_n = nuevo.predict(start=len(serie), end=len(serie) + 167, dynamic=False)
            p_v = viejo.predict(start=len(serie), end=len(serie) + 167, dynamic=False)
            check(np.array_equal(p_n, p_v), "%s: prediccion a 168 h identica" % nombre)
        datos = os.path.join(RAIZ, "datos_locales", "CEN_long.csv")
        if os.path.exists(datos):
            from tesis_forecast.data.exogenas import cargar_exogenas_horarias  # noqa: F401

            df_c = E.cargar_regiones(["CEN"], os.path.dirname(datos))["CEN"]
            serie_c, _ = E.extraer_serie_horaria(df_c, E.COL_DEMANDA)
            resid_c = E.descomponer_stl(serie_c[-(3600 + 168):][:-168])[2]
            nuevo, lag_n, _ = E.seleccionar_ar_por_aic(resid_c, max_lag=48)
            viejo, lag_v = seleccionar_codigo_anterior(resid_c, 48)
            check(lag_n == lag_v and nuevo.aic == viejo.aic and np.array_equal(np.asarray(nuevo.params), np.asarray(viejo.params)),
                  "residuo STL real de CEN (S4), rezagos 1..48: identico (rezago %s)" % lag_n)

    print("3. Si todos los candidatos fallan, el error explica por que")
    def falla(motivo):
        def ctor(*a, **k):
            raise motivo
        return ctor
    with mock.patch.object(E, "AutoReg", falla(TypeError("AutoReg.__init__() got an unexpected keyword argument 'old_names'"))):
        try:
            E.seleccionar_ar_por_aic(y, max_lag=168)
            check(False, "debia lanzar RuntimeError")
        except RuntimeError as e:
            m = str(e)
            check("No se pudo ajustar ningun AR valido" in m, "conserva el mensaje original")
            check("TypeError" in m and "old_names" in m and "x168" in m, "dice la excepcion y que ocurrio en los 168 candidatos: %s" % m[-120:])
            check("1500 observaciones" in m, "indica la longitud del residuo")
    try:
        E.seleccionar_ar_por_aic(np.zeros(500), max_lag=20)
        check(False, "residuo constante debia fallar")
    except RuntimeError as e:
        check("AIC no finito" in str(e) and "excepcion" not in str(e).split("AIC")[0].split("evaluados).")[-1], "residuo constante: explica el AIC no finito (sin excepciones): %s" % str(e)[-90:])
    try:
        E.seleccionar_ar_por_aic(np.full(300, np.nan), max_lag=10)
        check(False, "residuo vacio debia fallar")
    except RuntimeError as e:
        check("0 observaciones" in str(e) or "IndexError" in str(e) or "excepcion" in str(e), "residuo todo NaN: el mensaje muestra la causa: %s" % str(e)[:140])

    print("4. Fallo parcial: se elige el mejor de los validos")
    def parcial(*a, **k):
        lags = k.get("lags", a[1] if len(a) > 1 else None)
        if lags % 2 == 0:
            raise ValueError("rezago par no admitido (simulado)")
        return AutoRegReal(*a, **k)
    with mock.patch.object(E, "AutoReg", parcial):
        modelo, lag, df = E.seleccionar_ar_por_aic(y, max_lag=10)
        check(lag % 2 == 1 and int(df["error"].notna().sum()) == 5 and modelo is not None, "elige entre los impares (rezago %d) y registra los 5 errores" % lag)
        check(str(df.loc[df["error"].notna(), "error"].iloc[0]).startswith("ValueError: "), "el error guardado incluye el tipo de excepcion")

    print("5. statsmodels 0.15.0 REAL (opcional)")
    ruta015 = os.environ.get("SM015_PATH")
    if not ruta015 or not os.path.isdir(ruta015):
        print("  (se omite: defina SM015_PATH con una carpeta que contenga statsmodels 0.15.0)")
    else:
        codigo = (
            "import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, r'%s')\n"
            "from unittest import mock\n"
            "for n in ['optuna','optuna.samplers','xgboost','lightgbm','tensorflow','tensorflow.keras','tensorflow.keras.models',"
            "'tensorflow.keras.layers','tensorflow.keras.callbacks','tensorflow.keras.optimizers','tensorflow.keras.backend']:\n"
            "    try: __import__(n)\n"
            "    except ImportError: sys.modules[n] = mock.MagicMock()\n"
            "import statsmodels, numpy as np\n"
            "from tesis_forecast.models import ensemble_stl as E\n"
            "y = np.random.default_rng(0).standard_normal(3600)\n"
            "m, lag, df = E.seleccionar_ar_por_aic(y, max_lag=24)\n"
            "print('VERSION', statsmodels.__version__, 'LAG', lag, 'VALIDOS', int(df['AIC'].notna().sum()))\n"
        ) % os.path.join(RAIZ, "src")
        env = dict(os.environ, PYTHONPATH=ruta015)
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", codigo], capture_output=True, text=True, env=env)
        linea = [l for l in r.stdout.splitlines() if l.startswith("VERSION")]
        check(bool(linea) and linea[0].startswith("VERSION 0.15") and "VALIDOS 24" in linea[0], "0.15.0 real: %s" % (linea[0] if linea else (r.stderr[-200:])))

    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

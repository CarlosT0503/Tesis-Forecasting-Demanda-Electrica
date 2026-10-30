"""
Tests del fix del bug de desfase horario +1h (convertir_hora_0_23()/
_hora_a_0_23() condicional -> incondicional).

NO entrena ningun modelo real: solo ejercita las funciones puras de
extraccion/normalizacion/alineacion de fechas (extraer_serie_horaria,
preparar_exogena_horaria/_normalizar_exogena, construir_matriz_exogena_region/
merge_exogenas, alinear_exogenas_con_fechas/alinear_exogenas_a_fechas).
Ningun .fit() de sklearn/statsmodels/keras se llama en este archivo.

fcnn_model.py/ensemble_stl.py/lstm_resid_model.py importan tensorflow/optuna
a nivel de modulo -- si no estan instalados en este entorno (no lo estan
localmente), los tests que los necesitan se SALTAN explicitamente (mismo
patron que test_fase3_temp_igae.py con runner.py), sin fallar en bloque.
naive_trend_seasonal_model.py/ar_resid_trend_seasonal_model.py NO requieren
tensorflow/optuna (solo sklearn/statsmodels), asi que sus tests corren
siempre.

Uso:
    python tests/test_fix_horario.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.models import naive_trend_seasonal_model
from tesis_forecast.models import ar_resid_trend_seasonal_model

try:
    from tesis_forecast.models import fcnn_model
    from tesis_forecast.models import ensemble_stl
    from tesis_forecast.models import lstm_resid_model
    MODELOS_PESADOS_DISPONIBLES = True
    _IMPORT_ERROR_PESADOS = None
except ImportError as e:
    fcnn_model = ensemble_stl = lstm_resid_model = None
    MODELOS_PESADOS_DISPONIBLES = False
    _IMPORT_ERROR_PESADOS = e

COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

# Los 5 modulos afectados por el bug -- (modulo, nombre) para los que SIEMPRE
# se pueden importar (univariados, sin tensorflow/optuna).
MODULOS_LIVIANOS = [
    (naive_trend_seasonal_model, "naive_trend_seasonal_model"),
    (ar_resid_trend_seasonal_model, "ar_resid_trend_seasonal_model"),
]


def _modulos_afectados_disponibles():
    """Los 5 modulos afectados, incluyendo los pesados SOLO si se pudieron importar."""
    modulos = list(MODULOS_LIVIANOS)
    if MODELOS_PESADOS_DISPONIBLES:
        modulos = [
            (fcnn_model, "fcnn_model"),
            (ensemble_stl, "ensemble_stl"),
            (lstm_resid_model, "lstm_resid_model"),
        ] + modulos
    return modulos


# =========================================================
# 1-3. convertir_hora_0_23() -- pruebas unitarias directas
# =========================================================

def test_hora_1_a_0():
    for modulo, nombre in _modulos_afectados_disponibles():
        resultado = modulo.convertir_hora_0_23(pd.Series([1]))
        assert resultado.iloc[0] == 0, f"{nombre}: Hora=1 deberia dar hora_0_23=0, dio {resultado.iloc[0]}"
    print("OK test_hora_1_a_0")


def test_hora_24_a_23():
    for modulo, nombre in _modulos_afectados_disponibles():
        resultado = modulo.convertir_hora_0_23(pd.Series([24]))
        assert resultado.iloc[0] == 23, f"{nombre}: Hora=24 deberia dar hora_0_23=23, dio {resultado.iloc[0]}"
    print("OK test_hora_24_a_23")


def test_fila_anomala_no_rompe_la_columna_completa():
    """
    EL BUG ORIGINAL: la version condicional evaluaba `hora.min()>=1 and
    hora.max()<=24` sobre TODA la columna antes de decidir si restaba 1 --
    una sola fila fuera de rango (ej. Hora=99) hacia que NINGUNA fila de la
    columna restara 1, aunque el resto fuera perfectamente valido (1..24).

    Con el fix (incondicional, por fila), una fila anomala nunca contamina
    a las demas: cada fila resta 1 de forma independiente.
    """
    entrada = pd.Series([1, 2, 24, 99])  # la ultima fila es la anomalia

    for modulo, nombre in _modulos_afectados_disponibles():
        resultado = modulo.convertir_hora_0_23(entrada)

        # Las filas VALIDAS deben restar 1 SIN IMPORTAR la fila anomala --
        # bajo el bug original, estas 3 aserciones habrian fallado (el
        # resultado habria sido [1, 2, 24, 99], sin ningun -1 aplicado).
        assert resultado.iloc[0] == 0, f"{nombre}: Hora=1 no deberia verse afectada por la fila anomala"
        assert resultado.iloc[1] == 1, f"{nombre}: Hora=2 no deberia verse afectada por la fila anomala"
        assert resultado.iloc[2] == 23, f"{nombre}: Hora=24 no deberia verse afectada por la fila anomala"

    print("OK test_fila_anomala_no_rompe_la_columna_completa")


# =========================================================
# 4. Los 5 modelos afectados producen timestamps correctos
# =========================================================

def test_5_modelos_producen_timestamps_correctos():
    fecha_base = pd.Timestamp("2026-05-17")
    horas_raw = list(range(1, 25))

    df = pd.DataFrame({
        "fecha": [fecha_base] * 24,
        "Hora": horas_raw,
        COL_DEMANDA: np.arange(24, dtype=float),
    })

    modulos = _modulos_afectados_disponibles()
    if not MODELOS_PESADOS_DISPONIBLES:
        print(f"AVISO: fcnn/ensemble_stl/lstm_resid SALTADOS en este test (tensorflow/optuna no disponibles: {_IMPORT_ERROR_PESADOS})")

    for modulo, nombre in modulos:
        _, fechas = modulo.extraer_serie_horaria(df, COL_DEMANDA)

        primer_ts = pd.Timestamp(fechas[0])
        ultimo_ts = pd.Timestamp(fechas[-1])

        assert primer_ts == fecha_base, (
            f"{nombre}: primer timestamp deberia ser {fecha_base} 00:00 (Hora=1 -> hora_0_23=0), obtuvo {primer_ts}"
        )
        assert ultimo_ts == fecha_base + pd.Timedelta(hours=23), (
            f"{nombre}: ultimo timestamp deberia ser {fecha_base} 23:00 (Hora=24 -> hora_0_23=23), obtuvo {ultimo_ts}"
        )

    print(f"OK test_5_modelos_producen_timestamps_correctos ({len(modulos)}/5 modulos verificados en este entorno)")


# =========================================================
# 5. FCNN / Ensemble STL / LSTM Resid: demanda y exogenas
#    alineadas sobre el MISMO datetime
# =========================================================

def test_fcnn_ensemble_lstmresid_alinean_demanda_y_exogenas():
    if not MODELOS_PESADOS_DISPONIBLES:
        print(f"SALTADO test_fcnn_ensemble_lstmresid_alinean_demanda_y_exogenas (tensorflow/optuna no disponibles: {_IMPORT_ERROR_PESADOS})")
        return

    fecha_base = pd.Timestamp("2026-05-17")
    horas_raw = list(range(1, 25))

    # Fila anomala (Hora=0, fecha distinta) -- reproduce EXACTAMENTE el
    # disparador del bug real: antes del fix, esta sola fila en la demanda
    # hacia que TODA la columna Hora de la demanda quedara sin -1
    # (min()==0 rompe la condicion `>=1`), mientras Temperaturas_H (siempre
    # limpia, 1..24 por construccion via expandir_exogena()) seguia
    # restando 1 -- produciendo el desalineamiento demanda<->exogena
    # confirmado en la auditoria.
    demanda_df = pd.DataFrame({
        "fecha": [pd.Timestamp("2026-05-16")] + [fecha_base] * 24,
        "Hora": [0] + horas_raw,
        COL_DEMANDA: np.zeros(25),
    })

    # Temperaturas_H: SIEMPRE limpia (1..24), valor = hora cruda (para
    # poder predecir exactamente que valor deberia alinear con cada hora).
    temperaturas_df = pd.DataFrame({
        "fecha": [fecha_base] * 24, "hora": horas_raw, "valor": [float(h) for h in horas_raw],
    })
    exogenas_globales = {"Temperaturas_H": temperaturas_df, "IGAE_H": temperaturas_df.copy()}

    for modulo, nombre in [(fcnn_model, "fcnn_model"), (ensemble_stl, "ensemble_stl")]:
        _, fechas = modulo.extraer_serie_horaria(demanda_df, COL_DEMANDA)

        exog_interno = modulo.CANONICAL_TO_INTERNAL["Temperatura"]
        exog_region = modulo.construir_matriz_exogena_region("TEST", exogenas_globales, [exog_interno], data_dir=".")
        X = modulo.alinear_exogenas_con_fechas(fechas, exog_region, [exog_interno])

        # Posicion 0 = la fila anomala (2026-05-15 23:00, fuera del rango
        # de Temperaturas_H); posiciones 1..24 = las 24 horas de fecha_base,
        # en orden ascendente (hora_0_23 = 0..23).
        for k in range(24):
            esperado = float(k + 1)  # la fuente Temperaturas_H tiene valor=hora_cruda en esa hora_0_23
            obtenido = X[exog_interno].iloc[1 + k]
            assert obtenido == esperado, (
                f"{nombre}: demanda en hora_0_23={k} de {fecha_base.date()} deberia alinear con "
                f"Temperaturas={esperado}, obtuvo {obtenido} -- demanda y exogena DESALINEADAS"
            )

    # lstm_resid_model usa merge_exogenas/alinear_exogenas_a_fechas y nombres canonicos directos
    _, fechas_lr = lstm_resid_model.extraer_serie_horaria(demanda_df, COL_DEMANDA)
    exogenas_df_lr = lstm_resid_model.merge_exogenas("TEST", exogenas_globales, ["Temperatura"], data_dir=".")
    X_lr = lstm_resid_model.alinear_exogenas_a_fechas(fechas_lr, exogenas_df_lr, ["Temperatura"])

    for k in range(24):
        esperado = float(k + 1)
        obtenido = X_lr["Temperatura"].iloc[1 + k]
        assert obtenido == esperado, (
            f"lstm_resid_model: demanda en hora_0_23={k} deberia alinear con Temperatura={esperado}, "
            f"obtuvo {obtenido} -- demanda y exogena DESALINEADAS"
        )

    print("OK test_fcnn_ensemble_lstmresid_alinean_demanda_y_exogenas")


# =========================================================
# 6. Naive_Trend_Seasonal / AR_Resid_Trend_Seasonal: mismos
#    valores (y por lo tanto mismas metricas), solo cambia timestamp
# =========================================================

def test_naive_trend_seasonal_ar_resid_mismos_valores_solo_cambia_timestamp():
    """
    Estos 2 modelos son univariados y su train/test/metricas se calculan
    por POSICION sobre el array de valores extraido, nunca por fecha. Esta
    prueba demuestra que el array de VALORES (lo unico que importa para
    MAE/RMSE/MAPE/sMAPE) es identico sin importar la convencion horaria --
    el fix solo cambia la columna 'fecha' adjunta a esos mismos valores.
    """
    fecha_base = pd.Timestamp("2026-05-17")
    horas_raw = list(range(1, 25))
    valores = np.arange(24, dtype=float) * 10.0

    df = pd.DataFrame({"fecha": [fecha_base] * 24, "Hora": horas_raw, COL_DEMANDA: valores})

    for modulo, nombre in MODULOS_LIVIANOS:
        valores_extraidos, fechas_extraidas = modulo.extraer_serie_horaria(df, COL_DEMANDA)

        # Los VALORES no cambian -- son los unicos que entran a
        # calcular_metricas(test, pred) en evaluar_region().
        np.testing.assert_array_equal(valores_extraidos, valores)

        # El TIMESTAMP si cambia (ahora Hora-1 SIEMPRE, sin condicion):
        # primer valor -> 00:00 del mismo dia, no 01:00.
        assert pd.Timestamp(fechas_extraidas[0]) == fecha_base, (
            f"{nombre}: el timestamp deberia empezar en {fecha_base} 00:00"
        )
        assert pd.Timestamp(fechas_extraidas[-1]) == fecha_base + pd.Timedelta(hours=23)

    print("OK test_naive_trend_seasonal_ar_resid_mismos_valores_solo_cambia_timestamp")


def main():
    test_hora_1_a_0()
    test_hora_24_a_23()
    test_fila_anomala_no_rompe_la_columna_completa()
    test_5_modelos_producen_timestamps_correctos()
    test_fcnn_ensemble_lstmresid_alinean_demanda_y_exogenas()
    test_naive_trend_seasonal_ar_resid_mismos_valores_solo_cambia_timestamp()

    if not MODELOS_PESADOS_DISPONIBLES:
        print(
            "\nAVISO: fcnn_model/ensemble_stl/lstm_resid_model no se pudieron importar "
            f"en este entorno ({_IMPORT_ERROR_PESADOS}) -- las partes de los tests que los "
            "necesitan se saltaron. Correr este archivo en Colab (con tensorflow/optuna "
            "instalados) para la verificacion completa de los 5 modulos."
        )

    print("\nTODOS LOS TESTS DEL FIX HORARIO PASARON (en este entorno)")


if __name__ == "__main__":
    main()

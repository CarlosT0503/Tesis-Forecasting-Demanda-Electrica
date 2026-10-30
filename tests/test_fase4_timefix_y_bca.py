"""
Tests de la Fase 4 (rerun por fix horario, 21 configs con sufijo
'_timefix') y de BCA reconstruido con arquitectura ACTUAL sin exogenas
(cambio de criterio explicito -- ya NO se afirma equivalencia con el
pipeline legacy perdido, ver docstring de legacy_bca_reconstruido.py).

Solo depende de pandas/numpy + sklearn/statsmodels (para las 4 estructuras
BCA "livianas": ARIMA/SARIMA/AR_AIC/STL_AR_residuos). Las 4 pesadas
(FCNN univariada/STL+FCNN residuos/LSTM univariada/Ensemble univariado)
requieren tensorflow/optuna -- si no estan instalados en este entorno, las
partes que las necesitan se SALTAN explicitamente (mismo patron que
test_fase3_temp_igae.py con runner.py).

NO ejecuta ningun experimento real contra Pipeline_Resultados. Todos los
runs son sobre datos sinteticos en carpetas temporales.

Uso:
    python tests/test_fase4_timefix_y_bca.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tesis_forecast.config import build_run_name, ExperimentConfig
from tesis_forecast import legacy_bca_reconstruido as bca


# =========================================================
# FASE 4: 21 configs con sufijo _timefix
# =========================================================

MODELOS_FASE4 = ["fcnn", "ensemble_stl", "lstm_resid"]
CONFIGS_EXOG_FASE4 = [
    ["Temperatura"],
    ["IGAE"],
    ["Generacion"],
    ["Importacion"],
    ["Exportacion"],
    ["Temperatura", "IGAE"],
    ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"],
]


def _construir_configs_fase4():
    configs = []
    for modelo in MODELOS_FASE4:
        for exog in CONFIGS_EXOG_FASE4:
            configs.append(ExperimentConfig(
                modelo=modelo, exogenas=exog, train_hours=3600, forecast_horizon=168,
                sufijo_run_name="timefix",
                notas="Rerun tras fix de bug horario +1h (convertir_hora_0_23 incondicional).",
            ))
    return configs


def test_generacion_exacta_21_configs():
    configs = _construir_configs_fase4()
    assert len(configs) == 21, f"esperaba 21 configs, hubo {len(configs)}"
    assert len(configs) == len(MODELOS_FASE4) * len(CONFIGS_EXOG_FASE4) == 3 * 7

    for c in configs:
        assert c.modelo in MODELOS_FASE4
        assert c.train_hours == 3600
        assert c.forecast_horizon == 168
        assert c.sufijo_run_name == "timefix"

    print("OK test_generacion_exacta_21_configs")


def test_run_names_21_configs_sin_colision_y_con_sufijo():
    configs = _construir_configs_fase4()
    nombres = [
        build_run_name(c.modelo, c.train_hours, c.forecast_horizon, c.exogenas, sufijo=c.sufijo_run_name)
        for c in configs
    ]

    assert len(nombres) == len(set(nombres)) == 21, "los 21 run_names deben ser unicos entre si"
    assert all(n.endswith("_timefix") for n in nombres), "todos los run_names deben terminar en _timefix"

    print("OK test_run_names_21_configs_sin_colision_y_con_sufijo")


def test_run_names_timefix_no_colisionan_con_corridas_viejas():
    """
    Los run_names viejos (invalidados por el bug) NO llevan sufijo -- los
    21 nuevos SIEMPRE lo llevan. Estructuralmente no pueden coincidir.
    """
    configs = _construir_configs_fase4()
    nombres_nuevos = {
        build_run_name(c.modelo, c.train_hours, c.forecast_horizon, c.exogenas, sufijo=c.sufijo_run_name)
        for c in configs
    }
    nombres_viejos = {
        build_run_name(c.modelo, c.train_hours, c.forecast_horizon, c.exogenas)  # sin sufijo -- el invalidado
        for c in configs
    }

    assert not (nombres_nuevos & nombres_viejos), f"colision inesperada: {nombres_nuevos & nombres_viejos}"
    assert len(nombres_viejos) == 21  # las 21 carpetas viejas invalidadas, para referencia

    print("OK test_run_names_timefix_no_colisionan_con_corridas_viejas")


def test_build_run_name_sin_sufijo_identico_a_antes():
    """El sufijo es 100% opcional -- sin pasarlo, el comportamiento es EXACTAMENTE el de siempre (no-regresion)."""
    assert build_run_name("xgboost", 336, 168, ["Temperatura"]) == "XGBoost_train336h_fh168h_Temp"
    assert build_run_name("naive", "auto", "auto", []) == "Naive_trainautoh_fhautoh"
    print("OK test_build_run_name_sin_sufijo_identico_a_antes")


# =========================================================
# BCA reconstruido
# =========================================================

REGIONES_LEGACY_EXISTENTES = ["CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]


def test_exactamente_8_estructuras_bca_disponibles():
    assert len(bca.ESTRATEGIAS_BCA) == 8
    assert all(v["disponible_arquitectura_actual"] for v in bca.ESTRATEGIAS_BCA.values()), (
        "todas las 8 deben estar marcadas disponible_arquitectura_actual=True -- si alguna no lo "
        "fuera, no deberia inventarse una implementacion silenciosa"
    )
    esperadas = {
        "ARIMA_1_1_1", "SARIMA_1_1_1__1_0_1_168", "FCNN_Individual", "STL_FCNN_residuos",
        "AR_AIC", "STL_AR_residuos_AIC", "LSTM_Individual", "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid",
    }
    assert set(bca.ESTRATEGIAS_BCA) == esperadas
    print("OK test_exactamente_8_estructuras_bca_disponibles")


def test_ninguna_nota_bca_afirma_equivalencia_exacta_con_legacy():
    """
    Cambio de criterio explicito: ninguna 'nota' debe afirmar que la
    estructura ES la estrategia legacy -- deben describir la arquitectura
    ACTUAL usada, sin pretension de fidelidad historica.
    """
    frases_prohibidas = ["identico al legacy", "recupera el codigo legacy", "reproduce exactamente"]
    for nombre, info in bca.ESTRATEGIAS_BCA.items():
        nota_lower = info["nota"].lower()
        for frase in frases_prohibidas:
            assert frase not in nota_lower, f"{nombre}: la nota no deberia afirmar '{frase}'"
    print("OK test_ninguna_nota_bca_afirma_equivalencia_exacta_con_legacy")


def test_origen_y_metodologia_bca_marcados_explicitamente():
    assert bca.ORIGEN_BCA == "bca_univariado_reconstruido"
    assert bca.METODOLOGIA_BCA == "arquitectura_actual_sin_exogenas"
    assert bca.ORIGEN_BCA != "legacy_univariado", "el origen de BCA NUNCA debe coincidir con el origen legacy historico"
    print("OK test_origen_y_metodologia_bca_marcados_explicitamente")


def _escribir_bca_long_sintetico(data_dir, n_dias=25, fecha_inicio="2026-04-28"):
    """
    BCA_long.csv sintetico: n_dias completos, Hora 1..24 (convencion cruda
    real), demanda con algo de variacion horaria + tendencia leve (para que
    STL/AR tengan algo no trivial que ajustar).
    """
    filas = []
    base = pd.Timestamp(fecha_inicio)
    valor = 100.0
    for d in range(n_dias):
        fecha = base + pd.Timedelta(days=d)
        for h in range(1, 25):
            valor += np.sin(h / 24 * 2 * np.pi) * 2 + 0.05
            filas.append({
                "fecha": fecha.strftime("%Y-%m-%d"), "Hora": h,
                bca.COL_DEMANDA: valor,
            })
    df = pd.DataFrame(filas)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "BCA_long.csv"), index=False)
    return df


def test_bca_train_384_forecast_168_solo_bca():
    tmp_data = tempfile.mkdtemp(prefix="bca_data_")
    tmp_out = tempfile.mkdtemp(prefix="bca_out_")
    try:
        # 384+168=552h=23 dias exactos + 2 dias de margen
        _escribir_bca_long_sintetico(tmp_data, n_dias=25)

        # Solo estrategias livianas -- rapido, no requiere tensorflow/optuna
        metricas_df, series_df, metadata = bca.run_bca_reconstruido(
            data_dir=tmp_data, output_dir=tmp_out,
            train_hours=bca.TRAIN_HOURS_BCA, forecast_horizon=bca.FORECAST_HORIZON_BCA,
            estrategias=["ARIMA_1_1_1", "AR_AIC"],
        )

        assert metadata["train_hours"] == 384
        assert metadata["forecast_horizon"] == 168
        assert metadata["region"] == "BCA"
        assert metadata["origen"] == "bca_univariado_reconstruido"
        assert metadata["metodologia"] == "arquitectura_actual_sin_exogenas"

        # Cada fila de metricas queda marcada explicitamente -- nunca se puede confundir
        # con una fila de metricas_global.csv (legacy historico, sin estas columnas).
        assert (metricas_df["origen"] == "bca_univariado_reconstruido").all()
        assert (metricas_df["metodologia"] == "arquitectura_actual_sin_exogenas").all()
        assert (metricas_df["train_hours"] == 384).all()
        assert (metricas_df["forecast_horizon"] == 168).all()

        preds = series_df[series_df["tipo"] == "prediccion"]
        assert set(preds["serie"]) == {"BCA_DEMANDA"}  # solo BCA, ninguna otra region
        assert (preds.groupby("modelo").size() == 168).all(), "cada estrategia debe aportar 168 predicciones"

        print("OK test_bca_train_384_forecast_168_solo_bca")
    finally:
        shutil.rmtree(tmp_data, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_bca_fecha_test_esperada_se_deriva_del_final_de_la_serie():
    """
    No se hardcodea la fecha -- se deriva de fechas[-forecast_horizon:]. Esta
    prueba confirma el MECANISMO: si la serie sintetica termina en una fecha
    conocida, la ventana de test debe ser exactamente las ultimas 168 horas
    de esa serie (mismo criterio que se uso para las 7 regiones legacy:
    "misma ventana de test").
    """
    tmp_data = tempfile.mkdtemp(prefix="bca_data_fecha_")
    tmp_out = tempfile.mkdtemp(prefix="bca_out_fecha_")
    try:
        # 552 horas exactas = 23 dias, terminando el ultimo Hora=24 del dia 23
        df = _escribir_bca_long_sintetico(tmp_data, n_dias=23, fecha_inicio="2026-04-25")
        ultima_fecha_cruda = pd.Timestamp("2026-04-25") + pd.Timedelta(days=22)  # ultimo dia

        metricas_df, series_df, metadata = bca.run_bca_reconstruido(
            data_dir=tmp_data, output_dir=tmp_out,
            train_hours=bca.TRAIN_HOURS_BCA, forecast_horizon=bca.FORECAST_HORIZON_BCA,
            estrategias=["ARIMA_1_1_1"],
        )

        fin_esperado = ultima_fecha_cruda + pd.Timedelta(hours=23)  # Hora=24 -> hora_0_23=23 (fix aplicado)
        inicio_esperado = fin_esperado - pd.Timedelta(hours=167)

        assert pd.Timestamp(metadata["fecha_test_inicio"]) == inicio_esperado, (
            f"inicio de test esperado {inicio_esperado}, obtuvo {metadata['fecha_test_inicio']}"
        )
        assert pd.Timestamp(metadata["fecha_test_fin"]) == fin_esperado, (
            f"fin de test esperado {fin_esperado}, obtuvo {metadata['fecha_test_fin']}"
        )

        print("OK test_bca_fecha_test_esperada_se_deriva_del_final_de_la_serie")
    finally:
        shutil.rmtree(tmp_data, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_bca_no_modifica_legacy_original():
    """
    Escribe un Legacy_Univariados/ falso (7 regiones) al lado de donde se
    genera BCA reconstruido, y confirma que su contenido queda BYTE POR
    BYTE identico despues de correr run_bca_reconstruido() -- la salida de
    BCA va a una carpeta COMPLETAMENTE SEPARADA.
    """
    tmp_pipeline = tempfile.mkdtemp(prefix="bca_no_toca_legacy_")
    try:
        legacy_dir = os.path.join(tmp_pipeline, "Legacy_Univariados")
        os.makedirs(legacy_dir, exist_ok=True)

        metricas_legacy_path = os.path.join(legacy_dir, "metricas_global.csv")
        series_legacy_path = os.path.join(legacy_dir, "series_global.csv")

        contenido_metricas_original = "serie,modelo,MAE,RMSE,MAPE,sMAPE\nCEN_DEMANDA,AR_AIC,1.0,1.2,3.0,3.0\n"
        contenido_series_original = "serie,fecha,tipo,subset,modelo,valor\nCEN_DEMANDA,2026-05-17,real,completo,real,100.0\n"

        with open(metricas_legacy_path, "w", encoding="utf-8") as f:
            f.write(contenido_metricas_original)
        with open(series_legacy_path, "w", encoding="utf-8") as f:
            f.write(contenido_series_original)

        data_dir = tempfile.mkdtemp(prefix="bca_data_no_toca_")
        _escribir_bca_long_sintetico(data_dir, n_dias=25)

        bca_reconstruido_dir = os.path.join(tmp_pipeline, "Legacy_Univariados_BCA_Reconstruido")
        bca.run_bca_reconstruido(
            data_dir=data_dir, output_dir=bca_reconstruido_dir,
            train_hours=bca.TRAIN_HOURS_BCA, forecast_horizon=bca.FORECAST_HORIZON_BCA,
            estrategias=["ARIMA_1_1_1"],
        )

        with open(metricas_legacy_path, "r", encoding="utf-8") as f:
            assert f.read() == contenido_metricas_original, "metricas_global.csv fue modificado -- NO deberia tocarse"
        with open(series_legacy_path, "r", encoding="utf-8") as f:
            assert f.read() == contenido_series_original, "series_global.csv fue modificado -- NO deberia tocarse"

        # La salida de BCA quedo en una carpeta DISTINTA
        assert os.path.exists(os.path.join(bca_reconstruido_dir, "metricas_bca_reconstruido.csv"))
        assert os.path.exists(os.path.join(bca_reconstruido_dir, "series_bca_reconstruido.csv"))
        assert not os.path.exists(os.path.join(legacy_dir, "metricas_bca_reconstruido.csv")), (
            "los archivos de BCA NO deben aparecer dentro de Legacy_Univariados/"
        )

        shutil.rmtree(data_dir, ignore_errors=True)
        print("OK test_bca_no_modifica_legacy_original")
    finally:
        shutil.rmtree(tmp_pipeline, ignore_errors=True)


def test_run_names_bca_no_colisionan_con_timefix_ni_con_legacy_7_regiones():
    """
    Los nombres de estrategia BCA (ej. 'ARIMA_1_1_1') son las mismas 8
    estrategias que las 7 regiones legacy existentes -- eso es intencional
    (misma metodologia, otra region), NO una colision de RUN_NAME: BCA vive
    en su propio CSV/carpeta (Legacy_Univariados_BCA_Reconstruido/), nunca
    se mezcla como filas dentro de metricas_global.csv/series_global.csv
    originales. Esta prueba confirma que las estrategias BCA tampoco
    colisionan estructuralmente con los 21 RUN_NAME _timefix (namespaces
    completamente distintos: uno es un RUN_NAME de corrida moderna, el otro
    es un valor de columna 'modelo' dentro de un CSV legacy).
    """
    configs_timefix = _construir_configs_fase4()
    nombres_timefix = {
        build_run_name(c.modelo, c.train_hours, c.forecast_horizon, c.exogenas, sufijo=c.sufijo_run_name)
        for c in configs_timefix
    }
    estrategias_bca = set(bca.ESTRATEGIAS_BCA)

    assert not (nombres_timefix & estrategias_bca), "no deberian solaparse (son namespaces distintos)"

    print("OK test_run_names_bca_no_colisionan_con_timefix_ni_con_legacy_7_regiones")


def main():
    test_generacion_exacta_21_configs()
    test_run_names_21_configs_sin_colision_y_con_sufijo()
    test_run_names_timefix_no_colisionan_con_corridas_viejas()
    test_build_run_name_sin_sufijo_identico_a_antes()

    test_exactamente_8_estructuras_bca_disponibles()
    test_ninguna_nota_bca_afirma_equivalencia_exacta_con_legacy()
    test_origen_y_metodologia_bca_marcados_explicitamente()
    test_bca_train_384_forecast_168_solo_bca()
    test_bca_fecha_test_esperada_se_deriva_del_final_de_la_serie()
    test_bca_no_modifica_legacy_original()
    test_run_names_bca_no_colisionan_con_timefix_ni_con_legacy_7_regiones()

    print("\nTODOS LOS TESTS DE FASE 4 (TIMEFIX) Y BCA RECONSTRUIDO PASARON")


if __name__ == "__main__":
    main()

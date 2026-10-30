"""
Tests del CONSOLIDADO FINAL: reemplazos _timefix dentro de
consolidar_resultados(), integracion de BCA reconstruido, banderas de
calidad (BANDERAS_CALIDAD_CONOCIDAS/aplicar_banderas_calidad), y la nueva
auditoria de solo lectura auditar_consolidado().

Reutiliza los fixtures/helpers ya escritos en test_aggregator.py (carpetas
de corrida sinteticas, CSV legacy sintetico) en vez de duplicarlos -- ver
ese archivo para el detalle de por que cada fixture tiene la forma que
tiene.

Uso:
    python tests/test_consolidado_final.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from tesis_forecast import aggregator as agg

from test_aggregator import (
    REGIONES_TEST, REGIONES_LEGACY, ESTRATEGIAS_LEGACY_EJEMPLO,
    _escribir_run_1a_estilo_xgboost, _escribir_legacy_univariado, _pred_fechas,
)


# =========================================================
# FIXTURE: Legacy_Univariados_BCA_Reconstruido/ sintetico
# =========================================================

def _escribir_bca_reconstruido(
    pipeline_dir, estrategias=None, forecast_horizon=4, train_hours=384,
    incluir_real=True, offset_horas=0, valor_real_base=100.0,
    incluir_divergencia_stl_fcnn=False,
):
    """
    Escribe Pipeline_Resultados/Legacy_Univariados_BCA_Reconstruido/
    {metricas_bca_reconstruido.csv, series_bca_reconstruido.csv} sinteticos
    -- mismo esquema confirmado leyendo legacy_bca_reconstruido.py
    (metric_rows: serie/modelo/MAE/RMSE/MAPE/sMAPE/origen/metodologia/
    train_hours/forecast_horizon; series_rows: serie/fecha/tipo/subset/
    modelo/valor), solo la region BCA.

    `incluir_divergencia_stl_fcnn=True` agrega la fila 'STL_FCNN_residuos'
    con valores extremos (mismo orden de magnitud que el bug real
    documentado: MAE~2.85e18) -- para probar que se preserva intacta y
    queda marcada, nunca que se "arregla".
    """
    estrategias = estrategias if estrategias is not None else ESTRATEGIAS_LEGACY_EJEMPLO
    bca_dir = os.path.join(pipeline_dir, agg.NOMBRE_CARPETA_BCA_RECONSTRUIDO)
    os.makedirs(bca_dir, exist_ok=True)

    nombre_serie = "BCA_DEMANDA"
    metric_rows, series_rows = [], []

    if incluir_real:
        fechas_reales = _pred_fechas(20 + forecast_horizon, offset_horas=offset_horas - 20)
        for h, fecha in enumerate(fechas_reales):
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "real",
                "subset": "completo", "modelo": "real", "valor": valor_real_base + h,
            })

    fechas_test = _pred_fechas(forecast_horizon, offset_horas=offset_horas)
    for i, estrategia in enumerate(estrategias):
        divergente = incluir_divergencia_stl_fcnn and estrategia == "STL_FCNN_residuos"
        for h, fecha in enumerate(fechas_test):
            valor = 1e18 if divergente else (50.0 + i + h)
            series_rows.append({
                "serie": nombre_serie, "fecha": fecha, "tipo": "prediccion",
                "subset": "test", "modelo": estrategia, "valor": valor,
            })
        metric_rows.append({
            "serie": nombre_serie, "modelo": estrategia,
            "MAE": 2.85e18 if divergente else 1.0,
            "RMSE": 1.08e19 if divergente else 1.2,
            "MAPE": 1.38e17 if divergente else 3.0 + i,
            "sMAPE": 152.19 if divergente else 3.0 + i,
            "origen": agg.ORIGEN_BCA_RECONSTRUIDO,
            "metodologia": "arquitectura_actual_sin_exogenas",
            "train_hours": train_hours,
            "forecast_horizon": forecast_horizon,
        })

    metricas_path = os.path.join(bca_dir, "metricas_bca_reconstruido.csv")
    series_path = os.path.join(bca_dir, "series_bca_reconstruido.csv")
    pd.DataFrame(metric_rows).to_csv(metricas_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(series_rows).to_csv(series_path, index=False, encoding="utf-8-sig")
    return metricas_path, series_path


# =========================================================
# REEMPLAZOS _timefix
# =========================================================

def test_timefix_identifica_y_reemplaza_run_viejo():
    tmp = tempfile.mkdtemp(prefix="cf_test_timefix_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp, "FCNN_train336h_fh4h_Temp", "fcnn", ["Temperatura"], REGIONES_TEST)
        _escribir_run_1a_estilo_xgboost(tmp, "FCNN_train336h_fh4h_Temp_timefix", "fcnn", ["Temperatura"], REGIONES_TEST)
        # un run sin equivalente _timefix no debe verse afectado
        _escribir_run_1a_estilo_xgboost(tmp, "FCNN_train336h_fh4h_IGAE", "fcnn", ["IGAE"], REGIONES_TEST)

        descubrimiento = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)
        assert len(descubrimiento.runs) == 3

        reemplazos = agg.identificar_reemplazos_timefix(descubrimiento.runs)
        assert reemplazos == {"FCNN_train336h_fh4h_Temp": "FCNN_train336h_fh4h_Temp_timefix"}

        descubrimiento2 = agg.aplicar_reemplazos_timefix(descubrimiento)
        run_names = {r.run_name for r in descubrimiento2.runs}
        assert run_names == {"FCNN_train336h_fh4h_Temp_timefix", "FCNN_train336h_fh4h_IGAE"}
        assert descubrimiento2.n_reemplazados == 1

        # el descubrimiento original (crudo) no se muto -- aplicar_reemplazos_timefix es puro
        assert len(descubrimiento.runs) == 3

        reemplazado = [d for d in descubrimiento2.descartados if d.estado == "reemplazado"]
        assert len(reemplazado) == 1
        assert reemplazado[0].nombre == "FCNN_train336h_fh4h_Temp"
        assert "FCNN_train336h_fh4h_Temp_timefix" in reemplazado[0].razon

        reporte_df = agg.construir_reporte_consolidacion(descubrimiento2)
        fila_reemplazada = reporte_df[reporte_df["run_name"] == "FCNN_train336h_fh4h_Temp"].iloc[0]
        assert fila_reemplazada["estado"] == "reemplazado"
        assert fila_reemplazada["incluido"] == False

        print("OK test_timefix_identifica_y_reemplaza_run_viejo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_timefix_sin_pares_no_cambia_nada():
    tmp = tempfile.mkdtemp(prefix="cf_test_timefix_sin_pares_")
    try:
        _escribir_run_1a_estilo_xgboost(tmp, "FCNN_train336h_fh4h_Temp", "fcnn", ["Temperatura"], REGIONES_TEST)
        descubrimiento = agg.descubrir_runs_completos(tmp, regiones_esperadas=REGIONES_TEST)

        assert agg.identificar_reemplazos_timefix(descubrimiento.runs) == {}
        descubrimiento2 = agg.aplicar_reemplazos_timefix(descubrimiento)
        assert descubrimiento2 is descubrimiento, "sin reemplazos, debe devolver el MISMO objeto (no copiar innecesariamente)"

        print("OK test_timefix_sin_pares_no_cambia_nada")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# BCA reconstruido -- run_name, validacion, procesamiento
# =========================================================

def test_build_run_name_bca_no_colisiona_con_legacy():
    for estrategia in agg.MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO:
        rn_legacy = agg.build_run_name_legacy_univariado(estrategia)
        rn_bca = agg.build_run_name_bca_reconstruido(estrategia)
        assert rn_legacy != rn_bca
        assert rn_legacy.startswith("Legacy_Univariado__")
        assert rn_bca.startswith("BCA_Reconstruido__")

    print("OK test_build_run_name_bca_no_colisiona_con_legacy")


def test_procesar_bca_reconstruido_basico():
    tmp = tempfile.mkdtemp(prefix="cf_test_bca_basico_")
    try:
        metricas_path, series_path = _escribir_bca_reconstruido(tmp, ESTRATEGIAS_LEGACY_EJEMPLO, forecast_horizon=4)

        bca_metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        bca_series_df = pd.read_csv(series_path, encoding="utf-8-sig")

        resultado = agg.procesar_bca_reconstruido(bca_metricas_df, bca_series_df, metricas_path, series_path, verbose=False)

        assert len(resultado.metricas_df) == len(ESTRATEGIAS_LEGACY_EJEMPLO)
        assert set(resultado.metricas_df["region"]) == {"BCA"}
        assert set(resultado.metricas_df["origen"]) == {agg.ORIGEN_BCA_RECONSTRUIDO}
        assert set(resultado.metricas_df["metodologia"]) == {"arquitectura_actual_sin_exogenas"}
        assert set(resultado.metricas_df["train_hours"]) == {384}
        assert set(resultado.metricas_df["forecast_horizon"]) == {4}
        assert set(resultado.metricas_df["familia_experimento"]) == {agg.FAMILIA_UNIVARIADO}

        # run_name BCA nunca coincide con el run_name legacy de la misma estrategia
        run_names_bca = set(resultado.metricas_df["run_name"])
        run_names_legacy = {agg.build_run_name_legacy_univariado(e) for e in ESTRATEGIAS_LEGACY_EJEMPLO}
        assert not (run_names_bca & run_names_legacy)

        assert len(resultado.pred_df) == len(ESTRATEGIAS_LEGACY_EJEMPLO) * 4

        print("OK test_procesar_bca_reconstruido_basico")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validar_bca_reconstruido_detecta_train_hours_inconsistente():
    metricas_df = pd.DataFrame([
        {"serie": "BCA_DEMANDA", "modelo": "ARIMA_1_1_1", "MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0, "sMAPE": 1.0,
         "origen": agg.ORIGEN_BCA_RECONSTRUIDO, "metodologia": "arquitectura_actual_sin_exogenas",
         "train_hours": 384, "forecast_horizon": 4},
        {"serie": "BCA_DEMANDA", "modelo": "AR_AIC", "MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0, "sMAPE": 1.0,
         "origen": agg.ORIGEN_BCA_RECONSTRUIDO, "metodologia": "arquitectura_actual_sin_exogenas",
         "train_hours": 999, "forecast_horizon": 4},  # inconsistente a proposito
    ])
    fechas = _pred_fechas(4)
    series_rows = []
    for modelo in ["ARIMA_1_1_1", "AR_AIC"]:
        for h, fecha in enumerate(fechas):
            series_rows.append({"serie": "BCA_DEMANDA", "fecha": fecha, "tipo": "prediccion", "subset": "test", "modelo": modelo, "valor": 1.0 + h})
    series_df = pd.DataFrame(series_rows)

    validacion = agg.validar_bca_reconstruido(metricas_df, series_df)
    assert validacion.ok is False
    assert any("train_hours" in p for p in validacion.problemas)

    print("OK test_validar_bca_reconstruido_detecta_train_hours_inconsistente")


# =========================================================
# Banderas de calidad
# =========================================================

def test_aplicar_banderas_calidad_solo_marca_bca_stl_fcnn():
    df = pd.DataFrame({
        "run_name": [
            "BCA_Reconstruido__STL_FCNN_residuos",
            "Legacy_Univariado__STL_FCNN_residuos",  # MISMA estrategia, origen legacy -- NO debe marcarse
            "XGBoost_train336h_fh4h_baseline",
        ],
        "MAE": [2.85e18, 1.0, 1.0],
    })
    resultado = agg.aplicar_banderas_calidad(df)

    fila_bca = resultado[resultado["run_name"] == "BCA_Reconstruido__STL_FCNN_residuos"].iloc[0]
    assert fila_bca["calidad_resultado"] == "divergencia_numerica"
    assert fila_bca["valido_ranking"] == False

    fila_legacy = resultado[resultado["run_name"] == "Legacy_Univariado__STL_FCNN_residuos"].iloc[0]
    assert fila_legacy["calidad_resultado"] == agg.CALIDAD_DEFAULT
    assert fila_legacy["valido_ranking"] == True

    fila_otro = resultado[resultado["run_name"] == "XGBoost_train336h_fh4h_baseline"].iloc[0]
    assert fila_otro["calidad_resultado"] == agg.CALIDAD_DEFAULT
    assert fila_otro["valido_ranking"] == True

    # los valores originales de MAE NUNCA se tocan
    assert resultado.set_index("run_name")["MAE"].to_dict() == df.set_index("run_name")["MAE"].to_dict()

    print("OK test_aplicar_banderas_calidad_solo_marca_bca_stl_fcnn")


# =========================================================
# Integracion completa: moderno + legacy + BCA + timefix
# =========================================================

def test_consolidado_final_integra_todo():
    tmp = tempfile.mkdtemp(prefix="cf_test_integracion_")
    try:
        # 1 corrida moderna normal
        _escribir_run_1a_estilo_xgboost(
            tmp, "XGBoost_train336h_fh4h_baseline", "xgboost",
            ["Temperatura", "Primarias", "Secundarias", "Terciarias", "IGAE", "Generacion", "Importacion", "Exportacion"],
            REGIONES_TEST,
        )
        # par viejo + _timefix (el viejo debe quedar excluido del consolidado)
        _escribir_run_1a_estilo_xgboost(tmp, "FCNN_train336h_fh4h_Temp", "fcnn", ["Temperatura"], REGIONES_TEST)
        _escribir_run_1a_estilo_xgboost(tmp, "FCNN_train336h_fh4h_Temp_timefix", "fcnn", ["Temperatura"], REGIONES_TEST)

        legacy_metricas_path, legacy_series_path = _escribir_legacy_univariado(tmp, ESTRATEGIAS_LEGACY_EJEMPLO, forecast_horizon=4)
        bca_metricas_path, bca_series_path = _escribir_bca_reconstruido(
            tmp, ESTRATEGIAS_LEGACY_EJEMPLO, forecast_horizon=4, incluir_divergencia_stl_fcnn=True,
        )

        metricas_df, series_df, descubrimiento = agg.consolidar_resultados(
            tmp, regiones_esperadas=REGIONES_TEST,
            legacy_metricas_path=legacy_metricas_path, legacy_series_path=legacy_series_path,
            bca_metricas_path=bca_metricas_path, bca_series_path=bca_series_path,
        )

        # --- timefix: el viejo NO entra, el _timefix SI ---
        run_names = set(metricas_df["run_name"])
        assert "FCNN_train336h_fh4h_Temp" not in run_names
        assert "FCNN_train336h_fh4h_Temp_timefix" in run_names
        assert descubrimiento.n_reemplazados == 1

        fila_timefix = metricas_df[metricas_df["run_name"] == "FCNN_train336h_fh4h_Temp_timefix"].iloc[0]
        assert fila_timefix["es_timefix"] == True
        assert fila_timefix["reemplaza_a"] == "FCNN_train336h_fh4h_Temp"

        fila_baseline = metricas_df[metricas_df["run_name"] == "XGBoost_train336h_fh4h_baseline"].iloc[0]
        assert fila_baseline["es_timefix"] == False
        assert pd.isna(fila_baseline["reemplaza_a"])

        pred_timefix = series_df[(series_df["run_name"] == "FCNN_train336h_fh4h_Temp_timefix") & (series_df["serie_tipo"] == "prediccion")]
        assert len(pred_timefix) > 0
        assert (pred_timefix["es_timefix"] == True).all()
        assert "FCNN_train336h_fh4h_Temp" not in set(series_df.loc[series_df["serie_tipo"] == "prediccion", "run_name"])

        # --- origenes presentes ---
        origenes = set(metricas_df["origen"])
        assert origenes == {agg.ORIGEN_MODERNO, agg.ORIGEN_LEGACY_UNIVARIADO, agg.ORIGEN_BCA_RECONSTRUIDO}

        # --- legacy: 7 regiones x 8 estrategias ---
        legacy_m = metricas_df[metricas_df["origen"] == agg.ORIGEN_LEGACY_UNIVARIADO]
        assert len(legacy_m) == len(REGIONES_LEGACY) * len(ESTRATEGIAS_LEGACY_EJEMPLO)
        assert set(legacy_m["region"]) == set(REGIONES_LEGACY)

        # --- BCA: 1 region x 8 estrategias ---
        bca_m = metricas_df[metricas_df["origen"] == agg.ORIGEN_BCA_RECONSTRUIDO]
        assert len(bca_m) == len(ESTRATEGIAS_LEGACY_EJEMPLO)
        assert set(bca_m["region"]) == {"BCA"}

        # --- banderas de calidad: SOLO la fila BCA/STL_FCNN_residuos queda marcada ---
        fila_bca_divergente = bca_m[bca_m["modelo_estrategia"] == "STL_FCNN_residuos"].iloc[0]
        assert fila_bca_divergente["calidad_resultado"] == "divergencia_numerica"
        assert fila_bca_divergente["valido_ranking"] == False
        assert fila_bca_divergente["MAE"] == 2.85e18, "el valor divergente NUNCA se recalcula/capa"

        fila_legacy_stl_fcnn = legacy_m[legacy_m["modelo_estrategia"] == "STL_FCNN_residuos"].iloc[0]
        assert fila_legacy_stl_fcnn["calidad_resultado"] == agg.CALIDAD_DEFAULT, (
            "la fila legacy (origen distinto) NO debe marcarse solo por compartir el nombre de estrategia"
        )
        assert fila_legacy_stl_fcnn["valido_ranking"] == True

        resto_filas = metricas_df[metricas_df["run_name"] != "BCA_Reconstruido__STL_FCNN_residuos"]
        assert (resto_filas["valido_ranking"] == True).all()
        assert (resto_filas["calidad_resultado"] == agg.CALIDAD_DEFAULT).all()

        # --- auditoria de solo lectura sobre el consolidado recien construido ---
        resultado_auditoria = agg.auditar_consolidado(
            metricas_df, series_df,
            regiones_legacy_esperadas=REGIONES_LEGACY, region_bca="BCA",
            horizon_legacy_esperado=4,
            test_start_legacy_esperado=None, test_end_legacy_esperado=None,  # fechas sinteticas, no las reales de la tesis
            verbose=False,
        )
        # Check 6 (recalculo de metricas desde series_master) SI dispara aca a
        # proposito: las fixtures sinteticas escriben MAE/RMSE/MAPE/sMAPE como
        # constantes fabricadas (ver _escribir_legacy_univariado/_escribir_bca_reconstruido),
        # NUNCA derivadas de los 'valor'/'valor_real' sinteticos -- asi que el
        # recalculo real y legitimamente no coincide. Eso confirma que el
        # check funciona (detecta la discrepancia real), no que haya un bug.
        problemas_no_check6 = [p for p in resultado_auditoria.problemas if not p.startswith("Check 6")]
        assert problemas_no_check6 == [], f"auditoria encontro problemas inesperados: {problemas_no_check6}"
        assert any(p.startswith("Check 6") for p in resultado_auditoria.problemas), (
            "se esperaba que Check 6 SI reportara discrepancia (metricas fabricadas, no derivadas de series)"
        )
        assert resultado_auditoria.resumen["n_timefix_presentes"] == 1
        assert resultado_auditoria.resumen["legacy_combinaciones_observadas"] == len(REGIONES_LEGACY) * len(ESTRATEGIAS_LEGACY_EJEMPLO)
        assert resultado_auditoria.resumen["bca_n_filas_metricas"] == len(ESTRATEGIAS_LEGACY_EJEMPLO)

        print("OK test_consolidado_final_integra_todo")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# auditar_consolidado() -- deteccion de anomalias (fixtures directas, sin pipeline completo)
# =========================================================

def test_auditoria_detecta_corrida_vieja_no_reemplazada():
    metricas_df = pd.DataFrame({
        "run_name": ["Foo_train336h_fh4h", "Foo_train336h_fh4h_timefix"],
        "region": ["CEN", "CEN"],
        "modelo_estrategia": ["Foo", "Foo"],
        "origen": [agg.ORIGEN_MODERNO, agg.ORIGEN_MODERNO],
        "MAE": [1.0, 1.0], "RMSE": [1.0, 1.0], "MAPE": [1.0, 1.0], "sMAPE": [1.0, 1.0],
    })
    resultado = agg.auditar_consolidado(metricas_df, pd.DataFrame(), verbose=False)

    assert resultado.ok is False
    assert any("Check 1" in p for p in resultado.problemas)

    print("OK test_auditoria_detecta_corrida_vieja_no_reemplazada")


def test_auditoria_detecta_bca_stl_fcnn_sin_marcar():
    metricas_df = pd.DataFrame({
        "run_name": ["BCA_Reconstruido__STL_FCNN_residuos"],
        "region": ["BCA"],
        "modelo_estrategia": ["STL_FCNN_residuos"],
        "origen": [agg.ORIGEN_BCA_RECONSTRUIDO],
        "MAE": [2.85e18], "RMSE": [1.08e19], "MAPE": [1.38e17], "sMAPE": [152.19],
        "calidad_resultado": ["ok"],  # deberia ser 'divergencia_numerica' -- a proposito mal
        "valido_ranking": [True],  # deberia ser False -- a proposito mal
    })
    resultado = agg.auditar_consolidado(metricas_df, pd.DataFrame(), verbose=False)

    assert resultado.ok is False
    assert any("Check 8" in p for p in resultado.problemas)

    print("OK test_auditoria_detecta_bca_stl_fcnn_sin_marcar")


def test_auditoria_detecta_real_duplicado():
    ts = pd.Timestamp("2024-06-01")
    series_df = pd.DataFrame({
        "region": ["CEN", "CEN"],
        "timestamp": [ts, ts],
        "serie_tipo": ["real", "real"],
        "valor": [100.0, 105.0],  # mismo (region,timestamp), valores distintos -- no deberia poder pasar esto post-dedup
    })
    resultado = agg.auditar_consolidado(pd.DataFrame(), series_df, verbose=False)

    assert resultado.ok is False
    assert any("Check 9" in p for p in resultado.problemas)

    print("OK test_auditoria_detecta_real_duplicado")


def test_auditoria_sin_datos_no_revienta():
    """metricas_df/series_df vacios -- la auditoria no debe lanzar excepcion, solo reportar avisos."""
    resultado = agg.auditar_consolidado(pd.DataFrame(), pd.DataFrame(), verbose=False)
    assert isinstance(resultado, agg.ResultadoAuditoria)
    assert resultado.problemas == []

    print("OK test_auditoria_sin_datos_no_revienta")


def main():
    test_timefix_identifica_y_reemplaza_run_viejo()
    test_timefix_sin_pares_no_cambia_nada()

    test_build_run_name_bca_no_colisiona_con_legacy()
    test_procesar_bca_reconstruido_basico()
    test_validar_bca_reconstruido_detecta_train_hours_inconsistente()

    test_aplicar_banderas_calidad_solo_marca_bca_stl_fcnn()

    test_consolidado_final_integra_todo()

    test_auditoria_detecta_corrida_vieja_no_reemplazada()
    test_auditoria_detecta_bca_stl_fcnn_sin_marcar()
    test_auditoria_detecta_real_duplicado()
    test_auditoria_sin_datos_no_revienta()

    print("\nTODOS LOS TESTS DEL CONSOLIDADO FINAL PASARON")


if __name__ == "__main__":
    main()

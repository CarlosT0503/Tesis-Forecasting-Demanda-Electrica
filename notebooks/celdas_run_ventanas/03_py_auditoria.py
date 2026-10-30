import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def seleccionar(df, columnas):
    """Solo las columnas que existen (la auditoria agrega algunas solo cuando hay problemas)."""
    return df[[c for c in columnas if c in df.columns]]


dem, ex, gl = vt.auditar_datos_ventanas(DATA_DIR)
for nombre, df in [("auditoria_demanda", dem), ("auditoria_exog_regional", ex), ("auditoria_exog_globales", gl)]:
    df.to_csv(os.path.join(VENT_DIR, "_auditoria", nombre + ".csv"), index=False, encoding="utf-8-sig")

print("== Regiones NO aptas para alguna semana (vacio = todas aptas):")
apta = dem["apta_para_ventana"].fillna(False).astype(bool) if "apta_para_ventana" in dem.columns else pd.Series(False, index=dem.index)
print(seleccionar(dem[~apta], ["ventana", "region", "ultima_fecha", "filas_ventana", "problema"]).to_string(index=False))

print("== Cobertura de la demanda (S4):")
print(seleccionar(dem[dem["ventana"] == "S4"], ["region", "archivo_fecha_min", "archivo_fecha_max", "horas_totales", "nan_demanda_previas", "horas_previas_validas", "apta_para_ventana"]).to_string(index=False))

print("== IMP/EXP/GEN: cobertura con valor real (S4). Si IMP/EXP terminan en 2019-12-31 son constantes por ffill:")
print(seleccionar(ex[ex["ventana"] == "S4"], ["region", "exog", "primera_fecha_con_valor", "ultima_fecha_con_valor", "pct_filas_con_valor"]).to_string(index=False))

print("== Exogenas globales (S4):")
print(gl[gl["ventana"] == "S4"].to_string(index=False))

plan = vt.plan_dataframe()
plan.to_csv(os.path.join(VENT_DIR, "_auditoria", "plan_corridas_ventanas.csv"), index=False, encoding="utf-8-sig")
print("== Plan:", len(plan), "corridas;", plan.groupby("ejecuta_en").size().to_dict())

!cd "{DATA_DIR}" && md5sum *_long.csv *_GEN.csv *_IMP.csv *_EXP.csv IGAE_2.xlsx "Temperaturas promedio.csv"

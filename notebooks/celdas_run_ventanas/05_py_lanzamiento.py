SESION = "A"        # "A": S4 y S1 (56 corridas)  |  "B": S2 y S3 (56 corridas)  -- una sesion por pestana
LOTE = "TODO"       # "TODO", o un lote mas chico: "T2", "T3-fcnn", "T3-lstm_resid", "T3-ensemble_stl", "T3-lstm_direct"
MAX_INTENTOS = 3    # por configuracion; cada reintento reanuda por region (no repite regiones completas)
assert SESION in ("A", "B"), "SESION debe ser 'A' o 'B'"

resumen, motivo = ca.ejecutar_sesion(SESION, LOTE, vent_dir=VENT_DIR, data_dir=DATA_DIR, max_intentos=MAX_INTENTOS)
if motivo:
    print("")
    print("LA COLA SE DETUVO: " + motivo)
resumen[resumen["status"] == "failed"][["modelo", "ventana", "run_name", "error"]] if len(resumen) else resumen

cfg_s4 = vt.construir_configs("T1", "S4") + vt.construir_configs("T2", "S4")
detalle, resumen_s4 = vt.comparar_s4_con_original(cfg_s4, ORIG_DIR, VENT_DIR)
resumen_s4

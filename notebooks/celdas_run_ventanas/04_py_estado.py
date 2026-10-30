inf = ca.informe_progreso(VENT_DIR)
ruta_faltan = ca.guardar_faltantes(VENT_DIR, inf["faltan"])
print("")
print("Lista completa de lo que falta (112 - completas): " + ruta_faltan)

TRABAJADOR = "W2"   # "W2", "W3" o "PRINCIPAL"
MAX_INTENTOS = 3    # por configuracion; cada reintento reanuda por region
assert TRABAJADOR in ("PRINCIPAL", "W2", "W3"), "TRABAJADOR debe ser 'PRINCIPAL', 'W2' o 'W3'"

from tesis_forecast import distribuido as dist
assert hasattr(dist, "ejecutar_asignacion"), "Codigo desactualizado: Entorno de ejecucion > Reiniciar sesion y vuelve a ejecutar la celda 1"
ASIGNACION = os.path.join(REPO_DIR, "docs", "asignacion", TRABAJADOR + ".csv")
assert os.path.exists(ASIGNACION), "No existe %s: vuelve a ejecutar la celda 1 (actualiza el repositorio)" % ASIGNACION
assert os.path.isdir(DATA_DIR), "No se encuentra la carpeta de datos: %s (ver las instrucciones, paso de datos)" % DATA_DIR

# Registro del entorno (versiones, GPU, comprobacion de datos por MD5): util, pero NO bloquea el entrenamiento.
try:
    amb = dist.registrar_ambiente(VENT_DIR, TRABAJADOR, REPO_DIR, DATA_DIR)
    print("Entorno registrado | commit:", amb["commit"], "| GPU:", amb["gpu"])
    print("Versiones:", {k: v for k, v in amb["versiones"].items() if v})
    datos = amb["datos"]
    print("Datos de entrada: %s de %s archivos coinciden con la referencia" % (datos.get("coinciden"), datos.get("archivos_referencia")))
    if not datos.get("ok"):
        print("AVISO: los datos NO coinciden con la referencia (distintos: %s, faltan: %s). Se entrena igual, pero esas corridas NO se consolidaran hasta revisarlo." % (datos.get("distintos"), datos.get("faltan")))
except Exception as e:
    print("AVISO: no se pudo registrar el entorno (%s: %s). Se entrena igual; se puede registrar despues." % (type(e).__name__, e))

registros, motivo = dist.ejecutar_asignacion(ASIGNACION, trabajador=TRABAJADOR, vent_dir=VENT_DIR, data_dir=DATA_DIR, max_intentos=MAX_INTENTOS)
if motivo:
    print("")
    print("LA COLA SE DETUVO: " + motivo)

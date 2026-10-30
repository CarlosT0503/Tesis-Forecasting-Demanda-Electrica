"""
Registro de errores por region que los modelos "tragan" para seguir con la region siguiente.

Los modelos neuronales envuelven cada region (o cada estrategia) en un `try/except Exception` que solo imprimia una linea
y continuaba; si todas las regiones fallaban, `run()` devolvia DataFrames vacios y la corrida figuraba como completada.
Este modulo hace que cada error quede en `log.txt` con su traza completa y una marca facil de leer por maquina, SIN
cambiar ningun calculo ni dejar de tragar el error (una region mala sigue sin detener a las demas).

    registrar_error("CEN_DEMANDA", e)          # dentro del except: imprime marca + traza completa (stdout -> log.txt)
    extraer_errores(".../log.txt")             # errores de la ultima ejecucion registrada en ese log
"""

import traceback

MARCA = "@@ERROR_REGION@@"
FIN = "@@FIN_ERROR@@"
INICIO_EJECUCION = "Experimento: "  # lo imprime runner.run_experiment al empezar cada ejecucion (tambien las reanudaciones)


def _region_de(contexto):
    """`CEN_DEMANDA` -> `CEN`; `CEN` -> `CEN`."""
    return str(contexto).split("_")[0].strip()


def registrar_error(contexto, exc):
    """Imprime (stdout) la marca del error y su traza completa. Llamar dentro del `except` que ya existia."""
    print(f"{MARCA} {_region_de(contexto)} | {type(exc).__name__} | {str(exc)[:400].replace(chr(10), ' ')}")
    print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
    print(FIN)


def extraer_errores(ruta_log, solo_ultima_ejecucion=True):
    """Lee `log.txt` y devuelve [{region, tipo, mensaje, traza}] de los errores registrados.

    Con `solo_ultima_ejecucion=True` (por defecto) solo cuenta lo posterior a la ULTIMA linea `Experimento: ...`, es decir
    los errores de la ejecucion actual, no los de intentos anteriores que quedaron en el mismo archivo (append).
    """
    try:
        with open(ruta_log, encoding="utf-8", errors="replace") as f:
            lineas = f.read().splitlines()
    except OSError:
        return []
    if solo_ultima_ejecucion:
        inicios = [i for i, l in enumerate(lineas) if l.startswith(INICIO_EJECUCION)]
        if inicios:
            lineas = lineas[inicios[-1]:]
    errores, actual = [], None
    for l in lineas:
        if l.startswith(MARCA):
            partes = [p.strip() for p in l[len(MARCA):].split("|", 2)]
            partes += [""] * (3 - len(partes))
            actual = {"region": partes[0], "tipo": partes[1], "mensaje": partes[2], "traza": []}
        elif l.strip() == FIN and actual is not None:
            actual["traza"] = "\n".join(actual["traza"])
            errores.append(actual)
            actual = None
        elif actual is not None:
            actual["traza"].append(l)
    if actual is not None:  # el proceso murio a mitad de la traza
        actual["traza"] = "\n".join(actual["traza"])
        errores.append(actual)
    return errores


def resumir(errores, max_texto=300):
    """Resumen corto para mensajes y para el JSON del subproceso."""
    if not errores:
        return {"n": 0, "regiones": [], "mensajes_unicos": 0, "primero": None, "texto": ""}
    primero = errores[0]
    unicos = {(e["tipo"], e["mensaje"]) for e in errores}
    texto = "%d error(es) en regiones %s; primero: %s %s: %s" % (
        len(errores), sorted({e["region"] for e in errores}), primero["region"], primero["tipo"], primero["mensaje"])
    return {
        "n": len(errores),
        "regiones": sorted({e["region"] for e in errores}),
        "mensajes_unicos": len(unicos),
        "primero": {"region": primero["region"], "tipo": primero["tipo"], "mensaje": primero["mensaje"],
                    "traza_cola": "\n".join(primero["traza"].splitlines()[-14:]) if isinstance(primero["traza"], str) else ""},
        "texto": texto[:max_texto],
    }

"""
Soporte para el entrenamiento distribuido (varios Colab + la PC) sobre el pipeline y los checkpoints existentes.

No cambia ningun modelo ni el formato de resultados: solo anade (a) la lectura de una ASIGNACION EXCLUSIVA de
configuraciones por trabajador, (b) el registro del ENTORNO de cada trabajador (versiones, commit, datos) y (c) la
ejecucion de esa asignacion con la cola aislada de Colab (`colab_aislado.ejecutar_cola`: un subproceso por configuracion,
reanudacion por region, registro de errores). Cada trabajador escribe en SU carpeta de resultados.
"""

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

COLUMNAS_ASIGNACION = ("ventana", "nivel", "run_name")
LIBRERIAS = ("tensorflow", "keras", "numpy", "pandas", "scipy", "statsmodels", "optuna", "xgboost", "lightgbm", "sklearn")


# ---------------------------------------------------------------------------------------------------------
# asignacion exclusiva
# ---------------------------------------------------------------------------------------------------------

def cargar_asignacion(ruta):
    """Lee un CSV de asignacion (ventana, nivel, run_name [, trabajador]). Rechaza vacio, columnas faltantes y repetidos."""
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise ValueError("la asignacion esta vacia: %s" % ruta)
    faltan = [c for c in COLUMNAS_ASIGNACION if c not in filas[0]]
    if faltan:
        raise ValueError("la asignacion necesita las columnas %s; faltan %s" % (list(COLUMNAS_ASIGNACION), faltan))
    vistos = set()
    for f in filas:
        clave = (f["ventana"].strip(), f["run_name"].strip())
        if clave in vistos:
            raise ValueError("configuracion repetida en la asignacion: %s/%s" % clave)
        vistos.add(clave)
    return [{k: (v or "").strip() for k, v in f.items()} for f in filas]


def construir_cola(asignacion, entorno="colab"):
    """ExperimentConfig en el orden de la asignacion; falla si alguna no existe en el plan."""
    from . import ventanas_tools as vt
    from .runner import resolve_run
    por_grupo, cola, no_halladas = {}, [], []
    for f in asignacion:
        clave = (f["nivel"], f["ventana"])
        if clave not in por_grupo:
            configs = vt.filtrar_por_entorno(vt.construir_configs(*clave), entorno)
            por_grupo[clave] = {resolve_run(c).run_name: c for c in configs}
        c = por_grupo[clave].get(f["run_name"])
        (cola if c else no_halladas).append(c if c else f)
    if no_halladas:
        raise ValueError("%d configuraciones de la asignacion no existen en el plan (entorno %s): %s ..." % (
            len(no_halladas), entorno, [(f["ventana"], f["run_name"]) for f in no_halladas[:3]]))
    return cola


def agrupar_por_nivel_y_ventana(asignacion, cola):
    """Estructura que espera `colab_aislado.ejecutar_cola`: [(nivel, ventana, [configs])] conservando el orden."""
    grupos, orden = {}, []
    for fila, config in zip(asignacion, cola):
        clave = (fila["nivel"], fila["ventana"])
        if clave not in grupos:
            grupos[clave] = []
            orden.append(clave)
        grupos[clave].append(config)
    return [(n, v, grupos[(n, v)]) for n, v in orden]


# ---------------------------------------------------------------------------------------------------------
# huellas, datos y entorno
# ---------------------------------------------------------------------------------------------------------

def hash_archivo(ruta, algoritmo="sha256"):
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_md5(ruta_txt):
    """Lee un archivo estilo `md5sum` ('<md5> *<archivo>') y devuelve {archivo: md5}."""
    out = {}
    with open(ruta_txt, encoding="utf-8-sig") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            md5, _, nombre = linea.partition(" ")
            out[nombre.lstrip("*").strip()] = md5.lower()
    return out


def verificar_datos(data_dir, ruta_md5):
    """Compara los datos de entrada del trabajador con la lista de referencia (solo lectura)."""
    ref = leer_md5(ruta_md5)
    distintos, faltan, coinciden = [], [], 0
    for nombre, esperado in ref.items():
        ruta = os.path.join(data_dir, nombre)
        if not os.path.isfile(ruta):
            faltan.append(nombre)
        elif hash_archivo(ruta, "md5") != esperado:
            distintos.append(nombre)
        else:
            coinciden += 1
    return {"ok": not distintos and not faltan, "archivos_referencia": len(ref), "coinciden": coinciden,
            "distintos": distintos, "faltan": faltan}


def versiones():
    out = {"python": platform.python_version()}
    for nombre in LIBRERIAS:
        try:
            out[nombre] = getattr(__import__(nombre), "__version__", None)
        except Exception:
            out[nombre] = None
    return out


def _gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _commit(repo_dir):
    try:
        return subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def registrar_ambiente(vent_dir, trabajador, repo_dir, data_dir, ruta_md5=None):
    """Escribe `_auditoria/ambiente_<trabajador>.json` (versiones, commit, GPU, datos) y devuelve el diccionario.

    Sobrescribe solo el archivo de ESTE trabajador (el estado actual); cada llamada tambien se agrega a
    `ambiente_<trabajador>_historial.jsonl` para conservar la trazabilidad.
    """
    ruta_md5 = ruta_md5 or os.path.join(repo_dir, "docs", "datos_locales_MD5SUMS.txt")
    amb = {"trabajador": trabajador, "registrado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "commit": _commit(repo_dir), "plataforma": platform.platform(), "gpu": _gpu(), "versiones": versiones(),
           "datos": verificar_datos(data_dir, ruta_md5) if os.path.isfile(ruta_md5) else {"ok": None, "nota": "sin lista de referencia"}}
    aud = os.path.join(vent_dir, "_auditoria")
    os.makedirs(aud, exist_ok=True)
    tmp = os.path.join(aud, "ambiente_%s.json.tmp" % trabajador)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(amb, f, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(aud, "ambiente_%s.json" % trabajador))
    with open(os.path.join(aud, "ambiente_%s_historial.jsonl" % trabajador), "a", encoding="utf-8") as f:
        f.write(json.dumps(amb, ensure_ascii=False) + "\n")
    return amb


# ---------------------------------------------------------------------------------------------------------
# ejecucion de la asignacion de un trabajador
# ---------------------------------------------------------------------------------------------------------

def ejecutar_asignacion(ruta_csv, *, trabajador, vent_dir, data_dir, max_intentos=3, intervalo_telemetria_s=120, **kw):
    """Corre la asignacion de `trabajador` con la cola aislada de Colab. Lo completo se salta y lo parcial se reanuda."""
    from . import colab_aislado as ca
    from . import telemetria
    asignacion = cargar_asignacion(ruta_csv)
    ajenas = [f for f in asignacion if f.get("trabajador") and f["trabajador"] != trabajador]
    if ajenas:
        raise ValueError("%s tiene %d filas asignadas a otro trabajador (p. ej. %s): no se ejecuta" % (
            ruta_csv, len(ajenas), ajenas[0]["trabajador"]))
    cola = agrupar_por_nivel_y_ventana(asignacion, construir_cola(asignacion))
    ruta_tel = ca.ruta_telemetria(vent_dir, trabajador)
    print("Trabajador %s: %d configuraciones asignadas (las completas se saltan, las parciales continuan)" % (
        trabajador, len(asignacion)))
    monitor = telemetria.Monitor(ruta_tel, intervalo_s=intervalo_telemetria_s, etiqueta=lambda: ca.CONTEXTO_ACTUAL["txt"]).iniciar()
    try:
        registros, motivo = ca.ejecutar_cola(cola, data_dir=data_dir, vent_dir=vent_dir, sesion=trabajador,
                                             max_intentos=max_intentos, **kw)
    finally:
        monitor.detener()
    return registros, motivo


# ---------------------------------------------------------------------------------------------------------
# datos de entrada desde una carpeta de Drive COMPARTIDA (solo lectura), para los Colab adicionales
# ---------------------------------------------------------------------------------------------------------

DESTINO_DATOS_COLAB = "/content/bases_de_datos_tesis"
MIME_CARPETA = "application/vnd.google-apps.folder"


def _descargar_archivo_drive(servicio, archivo_id, ruta):
    import io
    from googleapiclient.http import MediaIoBaseDownload
    with open(ruta, "wb") as f:
        descarga = MediaIoBaseDownload(f, servicio.files().get_media(fileId=archivo_id, supportsAllDrives=True), chunksize=16 * 1024 * 1024)
        hecho = False
        while not hecho:
            _, hecho = descarga.next_chunk()


def descargar_carpeta_drive(folder_id, destino, nombres=None, servicio=None, descargar=None):
    """Descarga a `destino` (sin carpetas: se aplanan) los archivos de una carpeta de Drive compartida con la cuenta.

    Solo LEE el Drive de origen. Verifica el MD5 que informa Drive. `nombres`: si se da, solo se bajan esos archivos.
    `servicio` y `descargar` se inyectan solo en pruebas."""
    os.makedirs(destino, exist_ok=True)
    if servicio is None:
        from google.colab import auth
        auth.authenticate_user()
        from googleapiclient.discovery import build
        servicio = build("drive", "v3", cache_discovery=False)
    descargar = descargar or _descargar_archivo_drive
    pendientes, bajados, omitidos, md5_malos, repetidos = [folder_id], [], [], [], []
    while pendientes:
        carpeta, token = pendientes.pop(), None
        while True:
            r = servicio.files().list(q="'%s' in parents and trashed=false" % carpeta, pageSize=200, pageToken=token,
                                      fields="nextPageToken, files(id, name, mimeType, md5Checksum)",
                                      supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            for f in r.get("files", []):
                if f["mimeType"] == MIME_CARPETA:
                    pendientes.append(f["id"])
                elif f["mimeType"].startswith("application/vnd.google-apps."):
                    omitidos.append(f["name"])
                elif nombres is not None and f["name"] not in nombres:
                    omitidos.append(f["name"])
                elif f["name"] in bajados:
                    repetidos.append(f["name"])
                else:
                    ruta = os.path.join(destino, f["name"])
                    descargar(servicio, f["id"], ruta)
                    if f.get("md5Checksum") and hash_archivo(ruta, "md5") != f["md5Checksum"]:
                        md5_malos.append(f["name"])
                    bajados.append(f["name"])
            token = r.get("nextPageToken")
            if not token:
                break
    return {"bajados": bajados, "omitidos": len(omitidos), "repetidos": repetidos, "md5_drive_malos": md5_malos}


def preparar_datos(data_dir, repo_dir, destino=DESTINO_DATOS_COLAB, servicio=None, descargar=None):
    """Devuelve la carpeta de datos a usar. Si `data_dir` (en el Drive del trabajador) ya tiene archivos, se usa tal cual
    (Colab principal). Si no, se descargan los datos de la carpeta compartida (id en docs/datos_carpeta_id.txt) a `destino`."""
    if os.path.isdir(data_dir) and any(True for _ in os.scandir(data_dir)):
        print("Datos: se usa la carpeta del propio Drive (%s)" % data_dir)
        return data_dir
    ruta_id = os.path.join(repo_dir, "docs", "datos_carpeta_id.txt")
    folder_id = open(ruta_id, encoding="utf-8").read().strip() if os.path.isfile(ruta_id) else ""
    if not folder_id:
        raise RuntimeError("No hay datos en %s y falta el id de la carpeta compartida (docs/datos_carpeta_id.txt)" % data_dir)
    ruta_md5 = os.path.join(repo_dir, "docs", "datos_locales_MD5SUMS.txt")
    nombres = set(leer_md5(ruta_md5)) if os.path.isfile(ruta_md5) else None
    print("Datos: descargando (solo lectura) la carpeta compartida de Drive a %s ..." % destino)
    r = descargar_carpeta_drive(folder_id, destino, nombres, servicio, descargar)
    print("  descargados %d archivos (omitidos %d)" % (len(r["bajados"]), r["omitidos"]))
    if r["md5_drive_malos"]:
        print("  AVISO: el MD5 de Drive no coincide en", r["md5_drive_malos"])
    if os.path.isfile(ruta_md5):
        v = verificar_datos(destino, ruta_md5)
        print("  comprobacion contra la lista de referencia: %s de %s archivos coinciden" % (v["coinciden"], v["archivos_referencia"]))
        if not v["ok"]:
            print("  AVISO: distintos %s | faltan %s. Se continua; esas corridas no se consolidaran hasta revisarlo." % (v["distintos"], v["faltan"]))
    return destino

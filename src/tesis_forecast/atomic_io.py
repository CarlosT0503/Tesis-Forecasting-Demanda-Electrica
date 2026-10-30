"""
Escritura atomica de CSV.

Problema que resuelve: los modelos guardaban sus avances con `df.to_csv(ruta)`, que primero trunca el archivo
definitivo y luego lo escribe. Si la sesion de Colab muere en ese intervalo, queda un CSV truncado; el
checkpoint (`checkpoint._leer_csv_si_existe`) lo trata como vacio y la corrida se recalcula completa.

Aqui el CSV se escribe en un archivo temporal **dentro de la misma carpeta** que el definitivo (por tanto en el
mismo sistema de archivos, requisito para que `os.replace` sea atomico) y solo al terminar se reemplaza el
definitivo. Si algo falla antes del reemplazo, el archivo anterior queda intacto y el temporal se borra.

No cambia el contenido ni el formato de ningun CSV (mismos argumentos que `DataFrame.to_csv`).
"""

import os
import tempfile

PREFIJO_TEMPORAL = ".tmp_"


def escribir_csv_atomico(df, ruta, **kwargs_to_csv):
    """Escribe `df` en `ruta` de forma atomica. `kwargs_to_csv` se pasan tal cual a `DataFrame.to_csv`.

    Devuelve `ruta`. Si no se puede crear el temporal en la carpeta destino (permisos, sistema de archivos
    raro), se recurre a la escritura directa de siempre para no impedir el guardado.
    """
    ruta = os.fspath(ruta)
    carpeta = os.path.dirname(os.path.abspath(ruta))
    nombre = os.path.basename(ruta)

    try:
        fd, tmp = tempfile.mkstemp(prefix=PREFIJO_TEMPORAL + nombre + ".", suffix=".part", dir=carpeta)
    except OSError:
        df.to_csv(ruta, **kwargs_to_csv)
        return ruta

    try:
        with os.fdopen(fd, "w", encoding=kwargs_to_csv.pop("encoding", "utf-8"), newline="") as f:
            df.to_csv(f, **kwargs_to_csv)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # algunos sistemas de archivos de red no lo soportan
                pass
        os.replace(tmp, ruta)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return ruta


def limpiar_temporales(carpeta):
    """Borra los temporales `.tmp_*.part` que dejo una sesion que murio a mitad de una escritura.

    Solo toca archivos con el prefijo de este modulo, dentro de `carpeta`; nunca un CSV definitivo.
    Devuelve la lista de archivos borrados.
    """
    borrados = []
    if not os.path.isdir(carpeta):
        return borrados
    for nombre in os.listdir(carpeta):
        if nombre.startswith(PREFIJO_TEMPORAL) and nombre.endswith(".part"):
            ruta = os.path.join(carpeta, nombre)
            try:
                os.remove(ruta)
                borrados.append(ruta)
            except OSError:
                pass
    return borrados

"""
Descarga de los datos desde una carpeta de Drive compartida (`distribuido.preparar_datos`), con un servicio de Drive
SIMULADO: carpetas anidadas, archivos nativos de Google (se omiten), paginacion, MD5 de Drive, filtro por nombre,
nombres repetidos, reutilizacion de la carpeta del propio Drive y mensaje claro si falta el id.

    python tests/test_datos_compartidos.py
"""

import hashlib
import os
import shutil
import sys
import tempfile

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from tesis_forecast import distribuido as dd  # noqa: E402

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLO ") + msg)
    if not cond:
        FALLOS.append(msg)


class FakeDrive:
    """Misma interfaz minima que googleapiclient: files().list(...).execute()."""

    def __init__(self, arbol, contenido):
        self.arbol, self.contenido, self.consultas = arbol, contenido, []

    def files(self):
        return self

    def list(self, q, pageSize, pageToken, fields, supportsAllDrives, includeItemsFromAllDrives):
        padre = q.split("'")[1]
        self.consultas.append(padre)
        todos = self.arbol.get(padre, [])
        i = int(pageToken or 0)
        pagina, sig = todos[i:i + 2], (str(i + 2) if i + 2 < len(todos) else None)

        class R:
            def execute(_):
                r = {"files": pagina}
                if sig:
                    r["nextPageToken"] = sig
                return r
        return R()


def main():
    tmp = tempfile.mkdtemp(prefix="test_datos_")
    try:
        contenidos = {"id_a": b"region,demanda\nBCA,1\n", "id_b": b"x" * 5000, "id_c": b"otro", "id_d": b"dentro de subcarpeta"}
        md5 = lambda k: hashlib.md5(contenidos[k]).hexdigest()
        arbol = {
            "RAIZ": [{"id": "id_a", "name": "BCA_long.csv", "mimeType": "text/csv", "md5Checksum": md5("id_a")},
                     {"id": "id_b", "name": "IGAE_2.xlsx", "mimeType": "application/vnd.ms-excel", "md5Checksum": md5("id_b")},
                     {"id": "doc", "name": "Notas", "mimeType": "application/vnd.google-apps.document"},
                     {"id": "id_c", "name": "no_se_necesita.zip", "mimeType": "application/zip", "md5Checksum": md5("id_c")},
                     {"id": "SUB", "name": "Regiones", "mimeType": dd.MIME_CARPETA}],
            "SUB": [{"id": "id_d", "name": "CEN_long.csv", "mimeType": "text/csv", "md5Checksum": md5("id_d")}],
        }
        drive = FakeDrive(arbol, contenidos)

        def descargar(servicio, archivo_id, ruta):
            open(ruta, "wb").write(servicio.contenido[archivo_id])

        repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "docs"))
        open(os.path.join(repo, "docs", "datos_carpeta_id.txt"), "w").write("RAIZ\n")
        with open(os.path.join(repo, "docs", "datos_locales_MD5SUMS.txt"), "w") as f:
            f.write("%s *BCA_long.csv\n%s *IGAE_2.xlsx\n%s *CEN_long.csv\n%s *FALTA.csv\n" % (md5("id_a"), md5("id_b"), md5("id_d"), "0" * 32))

        print("1. descarga desde la carpeta compartida")
        dest = os.path.join(tmp, "content", "datos")
        sin_datos = os.path.join(tmp, "MyDrive", "Bases de datos Tesis")           # no existe: es un Colab adicional
        usado = dd.preparar_datos(sin_datos, repo, destino=dest, servicio=drive, descargar=descargar)
        check(usado == dest, "sin datos propios: se usa la descarga temporal en /content")
        check(sorted(os.listdir(dest)) == ["BCA_long.csv", "CEN_long.csv", "IGAE_2.xlsx"], "se bajan solo los archivos de la lista (incluida la subcarpeta, aplanada)")
        check(open(os.path.join(dest, "CEN_long.csv"), "rb").read() == contenidos["id_d"], "el contenido coincide")
        check(not os.path.exists(os.path.join(dest, "no_se_necesita.zip")) and not os.path.exists(os.path.join(dest, "Notas")),
              "se omiten archivos que no estan en la lista y los documentos nativos de Google")
        check(set(drive.consultas) == {"RAIZ", "SUB"}, "recorre la carpeta y su subcarpeta con paginacion (consultas: %s)" % drive.consultas)
        r = dd.descargar_carpeta_drive("RAIZ", os.path.join(tmp, "todo"), None, drive, descargar)
        check(len(r["bajados"]) == 4 and r["md5_drive_malos"] == [], "sin filtro baja todo lo no nativo y el MD5 de Drive coincide")

        print("2. integridad")
        arbol["RAIZ"][0]["md5Checksum"] = "0" * 32
        r = dd.descargar_carpeta_drive("RAIZ", os.path.join(tmp, "malo"), None, drive, descargar)
        check(r["md5_drive_malos"] == ["BCA_long.csv"], "un MD5 de Drive que no coincide se informa")

        print("3. Colab principal y errores")
        propio = os.path.join(tmp, "propio")
        os.makedirs(propio)
        open(os.path.join(propio, "BCA_long.csv"), "w").write("datos del principal")
        antes = drive.consultas[:]
        check(dd.preparar_datos(propio, repo, destino=dest, servicio=drive, descargar=descargar) == propio and drive.consultas == antes,
              "con datos en su propio Drive (principal) no descarga nada y los usa tal cual")
        os.remove(os.path.join(repo, "docs", "datos_carpeta_id.txt"))
        try:
            dd.preparar_datos(sin_datos, repo, destino=os.path.join(tmp, "otro"), servicio=drive, descargar=descargar)
            check(False, "sin id de carpeta debe fallar con un mensaje claro")
        except RuntimeError as e:
            check("datos_carpeta_id" in str(e), "sin id de carpeta falla con un mensaje claro")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("TODAS LAS PRUEBAS PASARON" if not FALLOS else "FALLARON %d: %s" % (len(FALLOS), FALLOS)))
    return 0 if not FALLOS else 1


if __name__ == "__main__":
    sys.exit(main())

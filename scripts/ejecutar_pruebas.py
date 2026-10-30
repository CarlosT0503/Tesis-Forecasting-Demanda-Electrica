"""Ejecuta las pruebas automatizadas de `tests/` (una por proceso) y resume el resultado.

Las pruebas son archivos autonomos (`python tests/test_<nombre>.py`): usan datos sinteticos y sustitutos (stubs) y NO entrenan modelos reales.
Uso:  python scripts/ejecutar_pruebas.py [--tiempo-max 900] [patron ...]
Codigo de salida: 0 si todas pasan; 1 si alguna falla o excede el tiempo.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiempo-max", type=int, default=900, help="segundos maximos por prueba")
    ap.add_argument("patron", nargs="*", help="subcadenas del nombre de archivo (por defecto, todas)")
    a = ap.parse_args()
    pruebas = sorted(glob.glob(os.path.join(RAIZ, "tests", "test_*.py")))
    if a.patron:
        pruebas = [p for p in pruebas if any(x in os.path.basename(p) for x in a.patron)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.pathsep.join([os.path.join(RAIZ, "src"), os.environ.get("PYTHONPATH", "")]))
    fallas = []
    for p in pruebas:
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, p], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=a.tiempo_max, env=env)
            ok, cola = r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            ok, cola = False, ["tiempo excedido"]
        print("%-6s %-42s %5.1f s  %s" % ("OK" if ok else "FALLA", os.path.basename(p), time.time() - t0, cola[0][:90]), flush=True)
        if not ok:
            fallas.append(os.path.basename(p))
    print("\n%d pruebas, %d con falla%s" % (len(pruebas), len(fallas), (": " + ", ".join(fallas)) if fallas else ""))
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())

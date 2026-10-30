"""
Construye notebooks/run_ventanas.ipynb a partir de notebooks/celdas_run_ventanas/.

Cada celda es un archivo: `NN_md_*.md` (markdown) o `NN_py_*.py` (codigo), en
orden alfabetico. Asi el codigo de las celdas nunca se genera dentro de cadenas
(que fue la causa de errores de comillas/escapes) y se puede validar con `ast`.

    python scripts/build_run_ventanas_notebook.py          # escribe el notebook
    python scripts/build_run_ventanas_notebook.py --check  # solo valida
"""

import ast
import json
import os
import sys

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CELDAS = os.path.join(RAIZ, "notebooks", "celdas_run_ventanas")
SALIDA = os.path.join(RAIZ, "notebooks", "run_ventanas.ipynb")


def sin_magics(codigo: str) -> str:
    """Reemplaza las lineas de magia de IPython (!cmd, %cmd) por `pass` para poder usar ast."""
    lineas = []
    for ln in codigo.split("\n"):
        t = ln.lstrip()
        if t.startswith("!") or t.startswith("%"):
            lineas.append(ln[: len(ln) - len(t)] + "pass")
        else:
            lineas.append(ln)
    return "\n".join(lineas)


def cargar_celdas():
    celdas = []
    for nombre in sorted(os.listdir(CELDAS)):
        ruta = os.path.join(CELDAS, nombre)
        if not os.path.isfile(ruta):
            continue
        with open(ruta, encoding="utf-8") as f:
            texto = f.read().rstrip("\n")
        if nombre.endswith(".md"):
            celdas.append(("markdown", nombre, texto))
        elif nombre.endswith(".py"):
            celdas.append(("code", nombre, texto))
    return celdas


def validar(celdas):
    for tipo, nombre, texto in celdas:
        if tipo == "code":
            ast.parse(sin_magics(texto), filename=nombre)


def construir(celdas) -> dict:
    cells = []
    for tipo, _, texto in celdas:
        base = {"cell_type": tipo, "metadata": {}, "source": texto.splitlines(True)}
        if tipo == "code":
            base.update({"execution_count": None, "outputs": []})
        cells.append(base)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    celdas = cargar_celdas()
    validar(celdas)
    nb = construir(celdas)
    if "--check" in sys.argv:
        with open(SALIDA, encoding="utf-8") as f:
            actual = json.load(f)
        if actual != nb:
            print("El notebook no coincide con las celdas: ejecuta el script sin --check.")
            return 1
        print("OK: notebook al dia y todas las celdas Python validas (%d celdas)." % len(celdas))
        return 0
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Escrito %s (%d celdas, %d de codigo)." % (SALIDA, len(celdas), sum(1 for c in celdas if c[0] == "code")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Auditoria de SOLO LECTURA de los trials de Optuna de la LSTM directa.

Antes de la correccion de reanudacion (commit 608ca6a), una region cortada despues de terminar su busqueda de
hiperparametros volvia a correr `n_trials` intentos MAS (10 + 10 = 20). Este script recorre las bases
`optuna_lstm_directa.db` de las ventanas y, por cada estudio (una region de una configuracion), informa:

  * cuantos intentos hay (completos, fallidos, otros) frente al presupuesto (10);
  * el mejor valor de validacion entre los primeros `presupuesto` intentos y entre todos;
  * si el mejor de todos cae FUERA de los primeros `presupuesto` (la seleccion de hiperparametros cambio por el
    exceso) o dentro (el exceso no cambio la seleccion).

No escribe en las bases (SQLite se abre en modo `ro`), no importa Optuna ni TensorFlow, no entrena nada y no toca
`metricas.csv`, `series.csv` ni los checkpoints. El unico archivo que puede escribir es el CSV de informe que se pida
con `--salida`, y rechaza escribirlo dentro de una carpeta de corrida.

    python scripts/auditar_trials_lstm_directa.py --base-ventanas <Pipeline_Resultados_Ventanas> \
        [--presupuesto 10] [--salida informe_trials_lstm_directa.csv]

Codigo de salida: 0 siempre que pueda leer; 2 si no encuentra ninguna base.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys

NOMBRE_DB = "optuna_lstm_directa.db"
COLUMNAS = [
    "ventana", "corrida", "estudio", "region",
    "n_total", "n_completos", "n_fallidos", "n_otros", "presupuesto", "excede_presupuesto",
    "mejor_valor_primeros", "trial_mejor_primeros", "mejor_valor_todos", "trial_mejor_todos",
    "mejor_fuera_del_presupuesto", "mejora_por_exceso", "params_mejor_primeros", "params_mejor_todos",
    "mismos_params", "error",
]


def _abrir_solo_lectura(ruta):
    uri = "file:" + ruta.replace("\\", "/").replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30)
    con.execute("PRAGMA query_only = ON")
    return con


def _columnas(con, tabla):
    return {fila[1] for fila in con.execute(f"PRAGMA table_info({tabla})")}


def _decodificar_param(valor, distribucion_json):
    """Optuna guarda los categoricos como indice de `choices`; se traduce al valor real si es posible."""
    try:
        dist = json.loads(distribucion_json)
        choices = (dist.get("attributes") or {}).get("choices")
        if choices is not None:
            return choices[int(valor)]
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        pass
    return valor


def _leer_estudio(con, study_id, presupuesto):
    trials = con.execute(
        "SELECT trial_id, number, state FROM trials WHERE study_id = ? ORDER BY number", (study_id,)
    ).fetchall()
    valores = {}
    if "trial_values" in {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
        cols_v = _columnas(con, "trial_values")
        col_valor = "value" if "value" in cols_v else None
        if col_valor:
            for trial_id, valor in con.execute(
                "SELECT tv.trial_id, tv.value FROM trial_values tv JOIN trials t ON t.trial_id = tv.trial_id "
                "WHERE t.study_id = ? AND tv.objective = 0", (study_id,)
            ):
                valores[trial_id] = valor
    params = {}
    for trial_id, nombre, valor, dist in con.execute(
        "SELECT tp.trial_id, tp.param_name, tp.param_value, tp.distribution_json FROM trial_params tp "
        "JOIN trials t ON t.trial_id = tp.trial_id WHERE t.study_id = ?", (study_id,)
    ):
        params.setdefault(trial_id, {})[nombre] = _decodificar_param(valor, dist)

    completos = [(tid, num) for tid, num, estado in trials if str(estado).upper().endswith("COMPLETE")]
    n_fallidos = sum(1 for _, _, estado in trials if str(estado).upper().endswith("FAIL"))
    n_otros = len(trials) - len(completos) - n_fallidos

    def mejor(lista):
        validos = [(valores[tid], num, tid) for tid, num in lista if tid in valores and valores[tid] is not None]
        # valor finito y menor; empate -> el de menor numero (igual criterio que `study.best_trial`)
        validos = [v for v in validos if v[0] == v[0] and abs(v[0]) != float("inf")]
        return min(validos, key=lambda v: (v[0], v[1])) if validos else None

    primeros = completos[:presupuesto]
    mp, mt = mejor(primeros), mejor(completos)
    fila = {
        "n_total": len(trials), "n_completos": len(completos), "n_fallidos": n_fallidos, "n_otros": n_otros,
        "presupuesto": presupuesto, "excede_presupuesto": len(completos) > presupuesto,
        "mejor_valor_primeros": None if mp is None else round(mp[0], 6),
        "trial_mejor_primeros": None if mp is None else mp[1],
        "mejor_valor_todos": None if mt is None else round(mt[0], 6),
        "trial_mejor_todos": None if mt is None else mt[1],
        "params_mejor_primeros": "" if mp is None else json.dumps(params.get(mp[2], {}), sort_keys=True),
        "params_mejor_todos": "" if mt is None else json.dumps(params.get(mt[2], {}), sort_keys=True),
    }
    if mp is not None and mt is not None:
        fila["mejor_fuera_del_presupuesto"] = mt[1] != mp[1]
        fila["mejora_por_exceso"] = round(mp[0] - mt[0], 6)
        fila["mismos_params"] = fila["params_mejor_primeros"] == fila["params_mejor_todos"]
    else:
        fila["mejor_fuera_del_presupuesto"] = ""
        fila["mejora_por_exceso"] = ""
        fila["mismos_params"] = ""
    return fila


def _region_de_estudio(nombre):
    # `<serie>_lstm_directa`; la serie termina en la region (p. ej. "demanda_NTE" o "NTE")
    base = nombre[: -len("_lstm_directa")] if nombre.endswith("_lstm_directa") else nombre
    return base.split("_")[-1]


def auditar(base_ventanas, presupuesto=10):
    filas = []
    for raiz, dirs, archivos in os.walk(base_ventanas):
        if NOMBRE_DB not in archivos:
            continue
        ruta = os.path.join(raiz, NOMBRE_DB)
        rel = os.path.relpath(raiz, base_ventanas).replace("\\", "/")
        partes = rel.split("/")
        ventana, corrida = (partes[0], "/".join(partes[1:])) if len(partes) > 1 else ("", partes[0])
        try:
            con = _abrir_solo_lectura(ruta)
        except sqlite3.Error as e:
            filas.append(_fila_error(ventana, corrida, f"no se pudo abrir: {e}"))
            continue
        try:
            for study_id, nombre in con.execute("SELECT study_id, study_name FROM studies ORDER BY study_id"):
                base = {"ventana": ventana, "corrida": corrida, "estudio": nombre, "region": _region_de_estudio(nombre)}
                try:
                    base.update(_leer_estudio(con, study_id, presupuesto))
                    base["error"] = ""
                except sqlite3.Error as e:
                    base["error"] = f"esquema inesperado: {e}"
                filas.append(base)
        except sqlite3.Error as e:
            filas.append(_fila_error(ventana, corrida, f"esquema inesperado: {e}"))
        finally:
            con.close()
    return sorted(filas, key=lambda f: (f["ventana"], f["corrida"], f["estudio"]))


def _fila_error(ventana, corrida, mensaje):
    fila = {c: "" for c in COLUMNAS}
    fila.update({"ventana": ventana, "corrida": corrida, "error": mensaje})
    return fila


def escribir_csv(filas, salida, base_ventanas):
    destino = os.path.abspath(salida)
    base = os.path.abspath(base_ventanas)
    try:
        dentro = os.path.commonpath([destino, base]) == base
    except ValueError:  # unidades distintas en Windows: no puede estar dentro
        dentro = False
    if dentro and os.path.basename(os.path.dirname(destino)) != "_auditoria":
        raise ValueError(
            "El informe no se escribe dentro de las carpetas de corrida; usa una ruta fuera de "
            f"{base} o la carpeta _auditoria."
        )
    with open(destino, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c, "") for c in COLUMNAS})
    return destino


def resumen(filas):
    leidos = [f for f in filas if not f.get("error")]
    excede = [f for f in leidos if f["excede_presupuesto"]]
    cambian = [f for f in excede if f["mejor_fuera_del_presupuesto"] is True]
    lineas = [
        f"Estudios leidos: {len(leidos)} (con error de lectura: {len(filas) - len(leidos)})",
        f"Estudios con mas intentos completos que el presupuesto: {len(excede)}",
        f"  de ellos, el mejor cae fuera de los primeros intentos (cambia la seleccion): {len(cambian)}",
        f"  de ellos, el mejor esta entre los primeros (el exceso no cambia la seleccion): {len(excede) - len(cambian)}",
    ]
    for f in excede:
        marca = "CAMBIA" if f["mejor_fuera_del_presupuesto"] is True else "igual"
        lineas.append(
            f"  - {f['ventana']}/{f['corrida']} [{f['region']}]: {f['n_completos']} completos, "
            f"mejor primeros={f['mejor_valor_primeros']} (trial {f['trial_mejor_primeros']}), "
            f"mejor todos={f['mejor_valor_todos']} (trial {f['trial_mejor_todos']}) -> {marca}"
        )
    return "\n".join(lineas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-ventanas", required=True, help="carpeta Pipeline_Resultados_Ventanas")
    ap.add_argument("--presupuesto", type=int, default=10, help="intentos de Optuna previstos por region (10)")
    ap.add_argument("--salida", help="CSV de informe (fuera de las carpetas de corrida)")
    args = ap.parse_args(argv)

    filas = auditar(args.base_ventanas, args.presupuesto)
    if not filas:
        print(f"No se encontro ninguna base {NOMBRE_DB} bajo {args.base_ventanas}")
        return 2
    print(resumen(filas))
    if args.salida:
        print("Informe:", escribir_csv(filas, args.salida, args.base_ventanas))
    return 0


if __name__ == "__main__":
    sys.exit(main())

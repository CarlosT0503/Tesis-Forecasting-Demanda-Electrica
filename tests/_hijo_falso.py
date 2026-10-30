"""
Subproceso simulado para probar colab_aislado sin entrenar nada.

    python _hijo_falso.py <run_dir> <secuencia> <contador> <resultado.json>

`secuencia` es una lista separada por comas con el comportamiento de cada intento (el contador, un archivo, dice en
cual va):
  ok         escribe una corrida completa (8 regiones) y sale con 0
  parcial3   escribe 3 regiones y muere con 137 (como el OOM-killer)
  parcial2   escribe 2 regiones y sale con 1 (error de codigo)
  kill       sale con 137 sin escribir nada
  mem        MemoryError (traceback) y sale con 1
  gpu        ResourceExhaustedError / OOM when allocating tensor y sale con 1
  err        ValueError y sale con 1
  dormir     no termina (para probar el timeout)
  incompleta sale con 0 pero deja solo 5 regiones
  rc3        sale con 3 (corrida incompleta); copia la plantilla JSON (5.o argumento) como resumen
"""

import json
import os
import sys
import time

REGIONES = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]


def escribir(run_dir, n):
    import pandas as pd  # import diferido: el hijo falso arranca al instante
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"fake": True}, f)
    pd.DataFrame({"serie": [r + "_DEMANDA" for r in REGIONES[:n]], "region": REGIONES[:n], "MAPE": [1.0] * n,
                  "sMAPE": [1.0] * n, "MAE": [1.0] * n, "RMSE": [1.0] * n}).to_csv(os.path.join(run_dir, "metricas.csv"), index=False)
    pd.DataFrame({"serie": ["x"]}).to_csv(os.path.join(run_dir, "series.csv"), index=False)


def main():
    run_dir, secuencia, contador, resultado = sys.argv[1:5]
    plantilla = sys.argv[5] if len(sys.argv) > 5 else None
    modos = secuencia.split(",")
    n = int(open(contador).read()) if os.path.exists(contador) else 0
    open(contador, "w").write(str(n + 1))
    modo = modos[min(n, len(modos) - 1)]
    print("hijo falso: intento %d modo %s" % (n + 1, modo), flush=True)
    if modo == "ok":
        escribir(run_dir, 8)
        json.dump({"status": "completed", "mape_promedio": 1.0, "pico_rss_mb": 123}, open(resultado, "w"))
        return 0
    if modo == "parcial3":
        escribir(run_dir, 3)
        os._exit(137)
    if modo == "parcial2":
        escribir(run_dir, 2)
        print("FALLO: ValueError: algo raro", flush=True)
        return 1
    if modo == "kill":
        os._exit(137)
    if modo == "mem":
        print("Traceback (most recent call last):\n  ...\nMemoryError: Unable to allocate 3.1 GiB for an array", file=sys.stderr, flush=True)
        return 1
    if modo == "gpu":
        print("tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[128,512]", file=sys.stderr, flush=True)
        return 1
    if modo == "err":
        print("FALLO: ValueError: columna inexistente", flush=True)
        return 1
    if modo == "dormir":
        with open(contador + ".pid", "a") as f:
            f.write(str(os.getpid()) + " ")
        time.sleep(600)
    if modo == "rc3":  # como el hijo real cuando termina sin error pero la corrida quedo incompleta
        if plantilla:
            with open(plantilla, encoding="utf-8") as f, open(resultado, "w", encoding="utf-8") as g:
                g.write(f.read())
        return 3
    if modo == "incompleta":
        escribir(run_dir, 5)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

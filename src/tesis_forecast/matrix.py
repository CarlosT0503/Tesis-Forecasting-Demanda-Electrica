"""
run_matrix(): cola minima de experimentos, ejecutados secuencialmente.
`build_individual_exog_matrix()`: generador de la matriz de exogenas
individuales (una config por modelo multivariado x exogena, sin
combinaciones acumulativas -- ver su docstring).

No es scheduler ni infraestructura de nube -- es un loop sobre
`run_experiment()` que:
  - salta configs ya completadas (`validar_resultado`);
  - reanuda automaticamente carpetas incompletas/fallidas de una corrida
    anterior (`overwrite=True`, seguro: `run_experiment` ya se niega a
    tocar una carpeta completa aunque se pida overwrite). Reanudar YA NO
    borra la carpeta ni recalcula las 8 regiones desde cero -- cada modelo
    salta, region por region, las que ya tienen un resultado completo y
    valido (ver `checkpoint.py` y docs/CHECKPOINT_RESUME.md);
  - continua con la siguiente config si una falla, registrando el error;
  - al final imprime un resumen de completadas/saltadas/fallidas y el MAPE
    promedio de cada una.

Disparo (cron, notebook programado, etc.) queda fuera de alcance por
ahora -- ver docs/AUTOMATIZACION_FUTURA.md.
"""

import os
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import io_drive
from .config import ExperimentConfig
from .runner import MODEL_DEFAULTS, resolve_run, run_experiment
from .validator import validar_resultado


def build_individual_exog_matrix(
    exogenas: list,
    modelos: Optional[list] = None,
) -> list:
    """
    Genera un `ExperimentConfig` por cada combinacion valida (modelo
    multivariado, UNA exogena individual) -- no acumulativas: cada config
    lleva `exogenas=[<una sola>]`, nunca una lista de varias.

    Elegibilidad de modelos (decidida leyendo `runner.MODEL_DEFAULTS`, no
    una lista fija a mano): un modelo participa solo si su `"catalogo"` (el
    mismo catalogo que ya usa `resolve_run()` para validar exogenas) no
    esta vacio. Los modelos univariados (`catalogo == []`: hoy `naive`,
    `naive_trend`, `ar`, `naive_trend_seasonal`, `ar_resid_trend_seasonal`)
    se excluyen enteros -- no generan una corrida por cada exogena de la
    lista, porque no aceptan ninguna.

    Si `modelos` es `None` (default), se consideran TODOS los modelos
    registrados en `MODEL_DEFAULTS`. Pasar una lista explicita restringe el
    universo (por ejemplo, solo los modelos nuevos de una tarea concreta).

    Para cada combinacion valida, se preservan TODOS los defaults
    cientificos vigentes del modelo (`train_hours`, `forecast_horizon`,
    `optuna_n_trials`) -- se dejan en `None` en el `ExperimentConfig`
    resultante, que es la señal para que `resolve_run()`/`run_experiment()`
    usen exactamente `MODEL_DEFAULTS[modelo]`, igual que en cualquier otra
    corrida. Lo UNICO que cambia respecto al default de cada modelo es
    `exogenas`.

    Si alguna de las `exogenas` pedidas no esta en el catalogo de un modelo
    dado (ej. Naive/AR nunca aceptan nada; o una exogena que ni siquiera
    exista en el catalogo global), esa combinacion puntual se excluye (no
    toda la matriz) y se imprime un aviso explicando por que -- el mismo
    criterio de catalogo por modelo que ya aplica `resolve_run()`, solo que
    aqui se filtra ANTES de construir el `ExperimentConfig` en vez de
    fallar en tiempo de ejecucion.

    El `RUN_NAME` de cada corrida sigue siendo el determinista de siempre
    (`build_run_name()`, ver config.py): al llevar una sola exogena en la
    lista, el slug de exogenas queda con un solo abreviado (ej. `_Temp` vs.
    `_IGAE`), asi que dos exogenas distintas para el mismo modelo nunca
    colisionan de carpeta, y el checkpoint por region
    (`checkpoint.cargar_checkpoint_regiones`, invocado dentro de cada
    `run()`) identifica cada una como una corrida completamente aparte,
    exactamente igual que con cualquier otra config.
    """
    modelos_a_considerar = list(modelos) if modelos is not None else list(MODEL_DEFAULTS.keys())

    configs = []
    for modelo in modelos_a_considerar:
        if modelo not in MODEL_DEFAULTS:
            raise ValueError(f"Modelo desconocido: {modelo}. Disponibles: {list(MODEL_DEFAULTS)}")

        catalogo = MODEL_DEFAULTS[modelo]["catalogo"]

        if not catalogo:
            # Univariado: no acepta ninguna exogena -- se excluye entero,
            # no genera una corrida (redundante) por cada exogena pedida.
            continue

        for exogena in exogenas:
            if exogena not in catalogo:
                print(
                    f"AVISO: '{modelo}' no reconoce la exogena '{exogena}' "
                    f"(catalogo: {catalogo}) -- se excluye esa combinacion."
                )
                continue

            configs.append(ExperimentConfig(
                modelo=modelo,
                exogenas=[exogena],
                notas=f"Exogena individual: {exogena}.",
            ))

    return configs


@dataclass
class MatrixRunRecord:
    config: ExperimentConfig
    run_name: Optional[str]
    status: str  # "completed" | "skipped_ya_completo" | "failed"
    mape_promedio: Optional[float]
    error: Optional[str]
    run_dir: Optional[str]
    duracion_s: Optional[float] = None


def run_matrix(
    configs: list,
    data_dir: str = "/content",
    mount_drive: bool = True,
) -> list:
    """Corre una lista de ExperimentConfig, una por una. Devuelve una lista
    de MatrixRunRecord (uno por config, en el mismo orden)."""

    if mount_drive:
        io_drive.mount_drive()

    registros = []

    for i, config in enumerate(configs):
        print("\n" + "#" * 80)
        print(f"# Experimento {i + 1}/{len(configs)}: modelo={config.modelo}")
        print("#" * 80)

        try:
            resolved = resolve_run(config)
        except Exception as e:
            print(f"Config invalida: {type(e).__name__}: {e}")
            registros.append(MatrixRunRecord(config, None, "failed", None, str(e), None))
            continue

        run_dir = resolved.run_dir
        overwrite = False

        if os.path.exists(run_dir):
            reporte = validar_resultado(run_dir)

            if reporte.es_completo:
                print(f"YA COMPLETO, se salta: {resolved.run_name}")
                registros.append(
                    MatrixRunRecord(config, resolved.run_name, "skipped_ya_completo", None, None, run_dir)
                )
                continue

            print(f"Carpeta incompleta detectada para {resolved.run_name}:")
            for problema in reporte.problemas:
                print(f"  - {problema}")
            print("Se reintenta (overwrite=True).")
            overwrite = True

        t0 = time.time()
        try:
            resultado = run_experiment(config, data_dir=data_dir, mount_drive=False, overwrite=overwrite)

            registros.append(MatrixRunRecord(
                config, resultado.run_name, "completed",
                resultado.mape_promedio, None, resultado.run_dir,
                duracion_s=time.time() - t0,
            ))

        except Exception as e:
            print(f"FALLO: {type(e).__name__}: {e}")
            print(traceback.format_exc())

            registros.append(MatrixRunRecord(
                config, resolved.run_name, "failed", None, f"{type(e).__name__}: {e}", run_dir,
                duracion_s=time.time() - t0,
            ))
            continue

    _imprimir_resumen(registros)

    return registros


def _imprimir_resumen(registros: list):
    completadas = [r for r in registros if r.status == "completed"]
    saltadas = [r for r in registros if r.status == "skipped_ya_completo"]
    fallidas = [r for r in registros if r.status == "failed"]

    print("\n" + "=" * 80)
    print("RESUMEN DE LA COLA")
    print("=" * 80)
    print(f"Total:        {len(registros)}")
    print(f"Completadas:  {len(completadas)}")
    print(f"Saltadas:     {len(saltadas)} (ya estaban completas)")
    print(f"Fallidas:     {len(fallidas)}")

    if completadas:
        print("\nCompletadas (MAPE promedio):")
        for r in completadas:
            mape_str = f"{r.mape_promedio:.2f}%" if r.mape_promedio is not None else "N/D"
            print(f"  - {r.run_name}: MAPE={mape_str}")

    if fallidas:
        print("\nFallidas:")
        for r in fallidas:
            nombre = r.run_name or f"modelo={r.config.modelo} (config invalida)"
            print(f"  - {nombre}: {r.error}")

    if saltadas:
        print("\nSaltadas (ya completas):")
        for r in saltadas:
            print(f"  - {r.run_name}")


def resumen_dataframe(registros: list) -> pd.DataFrame:
    """Version tabular del resumen, util para revisar en un notebook."""
    filas = []
    for r in registros:
        filas.append({
            "modelo": r.config.modelo,
            "run_name": r.run_name,
            "status": r.status,
            "mape_promedio": r.mape_promedio,
            "duracion_s": r.duracion_s,
            "ventana": getattr(r.config, "ventana", None),
            "error": r.error,
            "run_dir": r.run_dir,
        })
    return pd.DataFrame(filas)

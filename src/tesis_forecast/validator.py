"""
Validacion automatica de completitud de un experimento ya corrido.

Lee solo archivos (config.json/metricas.csv) de una carpeta de resultados
-- no ejecuta nada, no depende de ningun modelo en particular. Se usa en
dos lugares:

  1. `runner.run_experiment(..., overwrite=True)`: antes de borrar una
     carpeta existente para reintentarla, confirma que esa carpeta NO esta
     completa (nunca se sobrescribe una corrida validada como completa).
  2. `matrix.run_matrix()`: para decidir si una config ya corrida se salta
     (completa) o se reintenta (incompleta/fallida).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .regions import REGIONS_ALL


@dataclass
class ValidationReport:
    run_dir: str
    es_completo: bool
    problemas: list = field(default_factory=list)
    regiones_encontradas: list = field(default_factory=list)
    regiones_faltantes: list = field(default_factory=list)
    n_filas_metricas: int = 0
    n_filas_con_nan: int = 0


def validar_resultado(run_dir: str, regiones_esperadas: Optional[list] = None) -> ValidationReport:
    regiones_esperadas = list(regiones_esperadas) if regiones_esperadas is not None else list(REGIONS_ALL)

    problemas = []

    config_path = os.path.join(run_dir, "config.json")
    metricas_path = os.path.join(run_dir, "metricas.csv")
    series_path = os.path.join(run_dir, "series.csv")

    if not os.path.isdir(run_dir):
        return ValidationReport(run_dir=run_dir, es_completo=False, problemas=["La carpeta no existe."])

    if not os.path.exists(config_path):
        problemas.append("Falta config.json.")

    if not os.path.exists(series_path):
        problemas.append("Falta series.csv.")

    if not os.path.exists(metricas_path):
        problemas.append("Falta metricas.csv.")
        return ValidationReport(
            run_dir=run_dir, es_completo=False, problemas=problemas,
            regiones_faltantes=regiones_esperadas,
        )

    try:
        df_metricas = pd.read_csv(metricas_path)
    except Exception as e:
        problemas.append(f"No se pudo leer metricas.csv: {e}")
        return ValidationReport(
            run_dir=run_dir, es_completo=False, problemas=problemas,
            regiones_faltantes=regiones_esperadas,
        )

    if "region" not in df_metricas.columns:
        problemas.append("metricas.csv no tiene columna 'region'.")
        return ValidationReport(run_dir=run_dir, es_completo=False, problemas=problemas)

    regiones_encontradas = sorted(df_metricas["region"].dropna().unique().tolist())
    regiones_faltantes = [r for r in regiones_esperadas if r not in regiones_encontradas]

    if regiones_faltantes:
        problemas.append(f"Faltan regiones en metricas.csv: {regiones_faltantes}")

    columnas_metrica = [c for c in ["MAPE", "sMAPE", "MAE", "RMSE"] if c in df_metricas.columns]
    n_filas_con_nan = 0
    if columnas_metrica:
        filas_con_nan = df_metricas[columnas_metrica].isna().any(axis=1)
        n_filas_con_nan = int(filas_con_nan.sum())
        if n_filas_con_nan > 0:
            problemas.append(f"{n_filas_con_nan} fila(s) de metricas.csv tienen NaN en {columnas_metrica}.")

    es_completo = len(problemas) == 0

    return ValidationReport(
        run_dir=run_dir,
        es_completo=es_completo,
        problemas=problemas,
        regiones_encontradas=regiones_encontradas,
        regiones_faltantes=regiones_faltantes,
        n_filas_metricas=len(df_metricas),
        n_filas_con_nan=n_filas_con_nan,
    )

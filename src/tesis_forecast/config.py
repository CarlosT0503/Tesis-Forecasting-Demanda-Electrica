"""
Configuracion de un experimento y generacion deterministica del RUN_NAME.

RUN_NAME = <Modelo>_train<horas>h_fh<horizonte>h_<exogenas>
Ejemplo:  XGBoost_train336h_fh168h_Temp-Prim-Sec-Terc-IGAE-Gen-Imp-Exp

Sin timestamp ni hash: el mismo conjunto de (modelo, train_hours,
forecast_horizon, exogenas) siempre produce el mismo nombre, y el runner se
niega a sobrescribir un RUN_NAME que ya exista en disco.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Modelos disponibles.
#
# "lightgbm" es un caso especial: no existe un pipeline vigente dedicado
# para LightGBM en el notebook legacy (solo aparecia embebido en la celda
# 46, un prototipo). Es un modelo ADAPTADO/ESTANDARIZADO al marco vigente
# de XGBoost, no una extraccion exacta -- ver el docstring de
# models/lightgbm_model.py y docs/MODELOS_MIGRADOS.md.
#
# "naive" / "naive_trend" / "ar" son univariados (sin exogenas): "naive" y
# "naive_trend" se extraen de la celda 45 (bloques NAIVE / NAIVE TREND
# dentro de evaluar_serie()), con train_hours/forecast_horizon por defecto
# igual al string "auto" (split dinamico de la celda 45, no un numero fijo)
# -- ver models/naive_model.py. "ar" es una COMBINACION NUEVA (no existe en
# legacy un AR aplicado a la serie cruda): reutiliza
# seleccionar_ar_por_aic/forecast_ar_resid de la celda 60 aplicado
# directamente a la serie de Demanda, con el mismo split dinamico "auto" que
# naive/naive_trend -- ver models/ar_model.py.
#
# "naive_trend_seasonal" / "ar_resid_trend_seasonal" / "lstm_resid" son
# tambien COMBINACIONES NUEVAS confirmadas por el usuario el 2026-08-09: no
# extraen un pipeline completo de legacy, sino que combinan piezas de las
# celdas 60/64 (STL + tendencia lineal + estacionalidad repetida, mas AR o
# LSTM sobre el residuo segun el modelo). Los tres usan una ventana FIJA por
# defecto de 3600h/168h (no "auto"). "naive_trend_seasonal" y
# "ar_resid_trend_seasonal" son univariados; "lstm_resid" SI requiere
# exogenas (mismo catalogo/tratamiento que "lstm_direct") -- ver
# models/naive_trend_seasonal_model.py, models/ar_resid_trend_seasonal_model.py,
# models/lstm_resid_model.py y docs/MODELOS_MIGRADOS.md.
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm_direct": "LSTM_Directa",
    "sarimax": "SARIMAX",
    "fcnn": "FCNN",
    "ensemble_stl": "Ensemble_STL",
    "naive": "Naive",
    "naive_trend": "Naive_Trend",
    "ar": "AR",
    "naive_trend_seasonal": "Naive_Trend_Seasonal",
    "ar_resid_trend_seasonal": "AR_Resid_Trend_Seasonal",
    "lstm_resid": "LSTM_Resid_Trend_Seasonal",
}

# Catalogo de exogenas y su abreviatura para el RUN_NAME, en un orden fijo
# (independiente del orden en que el usuario las liste en ExperimentConfig).
EXOG_ABBR = {
    "Temperatura": "Temp",
    "Primarias": "Prim",
    "Secundarias": "Sec",
    "Terciarias": "Terc",
    "IGAE": "IGAE",
    "Generacion": "Gen",
    "Importacion": "Imp",
    "Exportacion": "Exp",
}


@dataclass
class ExperimentConfig:
    modelo: str
    exogenas: Optional[list] = None   # None -> usa el default vigente del modelo
    train_hours: Optional[int] = None
    forecast_horizon: Optional[int] = None
    optuna_n_trials: Optional[int] = None
    seed: int = 42
    notas: str = ""
    # Sufijo opcional para el RUN_NAME (ej. "timefix") -- pensado para
    # rerunear una config ya existente sin pisar ni mezclarse con la
    # carpeta anterior (que puede seguir siendo valida como evidencia
    # historica hasta que el rerun se valide). None (default) -> RUN_NAME
    # identico al de siempre, cero impacto en ninguna corrida existente.
    sufijo_run_name: Optional[str] = None
    # Ventana semanal de prueba ("S1".."S4", ver windows.py). None (default)
    # -> comportamiento identico al de siempre: la prueba son las ultimas
    # `forecast_horizon` filas de los datos y los resultados van a
    # Pipeline_Resultados/. Con una ventana, el runner materializa los datos
    # hasta el fin de esa semana y escribe en Pipeline_Resultados_Ventanas/<ventana>/.
    ventana: Optional[str] = None


def build_run_name(modelo: str, train_hours: int, forecast_horizon: int, exogenas: list, sufijo: Optional[str] = None) -> str:
    if modelo not in MODEL_LABELS:
        raise ValueError(f"Modelo desconocido: {modelo}")

    desconocidas = [e for e in exogenas if e not in EXOG_ABBR]
    if desconocidas:
        raise ValueError(f"Exogenas no reconocidas en el catalogo: {desconocidas}")

    slug = "-".join(abbr for nombre, abbr in EXOG_ABBR.items() if nombre in exogenas)

    base = f"{MODEL_LABELS[modelo]}_train{train_hours}h_fh{forecast_horizon}h"

    # Modelos univariados (exogenas=[], ej. naive/naive_trend/AR) no tienen
    # slug -- se omite el "_" final en vez de dejarlo colgando. No afecta a
    # ningun modelo existente: todos sus defaults tienen exogenas no vacias.
    nombre = f"{base}_{slug}" if slug else base

    # Sufijo opcional (ver ExperimentConfig.sufijo_run_name) -- se agrega
    # SIEMPRE al final, despues del slug de exogenas, para que nunca
    # colisione con un RUN_NAME sin sufijo.
    if sufijo:
        nombre = f"{nombre}_{sufijo}"

    return nombre


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def resolved_config_dict(
    config: ExperimentConfig,
    run_name: str,
    train_hours: int,
    forecast_horizon: int,
    optuna_n_trials: int,
    exogenas: list,
) -> dict:
    """Config completa a persistir en config.json, con todos los defaults ya resueltos."""
    extra_ventana = {}
    if config.ventana:
        from .windows import get_ventana

        v = get_ventana(config.ventana)
        extra_ventana = {
            "ventana": v.id,
            "test_start": str(v.test_start),
            "test_end": str(v.test_end),
        }

    return {
        **extra_ventana,
        "modelo": config.modelo,
        "exogenas": exogenas,
        "train_hours": train_hours,
        "forecast_horizon": forecast_horizon,
        "optuna_n_trials": optuna_n_trials,
        "seed": config.seed,
        "notas": config.notas,
        "sufijo_run_name": config.sufijo_run_name,
        "run_name": run_name,
        "git_commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

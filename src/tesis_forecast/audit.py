"""
Auditoria de solo lectura: compara los resultados que YA existen en
`Pipeline_Resultados/` (incluyendo carpetas con nombres/formato anteriores
al esquema `RUN_NAME` actual, sin `config.json`) contra las configs que
produce HOY `build_individual_exog_matrix()`, para decidir cuales son
reutilizables sin volver a correr el experimento -- sin alterar la
comparabilidad cientifica.

100% de solo lectura. Nunca escribe, mueve, renombra ni borra nada dentro
de `Pipeline_Resultados/`. No ejecuta ningun modelo.

Dos capas, a proposito:
  - El motor de comparacion (`auditar_matriz_individual`,
    `_identidad_desde_config_json`, `_identidad_desde_legacy`, etc.) solo
    depende de pandas + `checkpoint.py`/`regions.py` (livianos) -- no
    importa `matrix.py`/`runner.py` a nivel de modulo, para poder
    testearlo con fixtures sinteticas sin tensorflow/optuna/xgboost/
    lightgbm instalados.
  - `construir_objetivos_desde_matriz_actual()`/
    `auditar_matriz_individual_actual()` son atajos de conveniencia para
    Colab: SI importan `matrix.py`/`runner.py` (de forma perezosa, dentro
    de la funcion), porque necesitan llamar a la `build_individual_exog_matrix()`
    real para saber contra que 35 configs comparar.

Identidad de cada corrida (modelo, exogenas, train_hours,
forecast_horizon) se determina por CONTENIDO, nunca por el nombre de la
carpeta:
  1. Si existe `config.json`: se lee directo de ahi (la fuente mas
     confiable).
  2. Si NO existe (carpeta legacy, anterior a `runner.py`): se infiere
     best-effort a partir de la columna `modelo` de `metricas.csv`
     (mapeada a un identificador canonico via `MODELO_DISPLAY_A_CANONICO`/
     `PREFIJOS_MODELO_A_CANONICO`, tablas literales confirmadas leyendo
     cada modulo, no adivinadas) y de `config_usada.csv` (`train_horas`,
     `exogenas` -- traduciendo nombres internos a canonicos donde aplica,
     ver `EXOGENA_INTERNA_A_CANONICA_POR_MODELO`). Si algo no se puede
     recuperar con confianza (modelo ambiguo/desconocido, o mas de un
     valor distinto de `train_horas` entre regiones), se deja en `None`
     en vez de adivinar -- queda reflejado en `razon` para revision
     manual, nunca se fuerza un match.

`forecast_horizon` de una carpeta legacy NUNCA se toma de un campo de
texto: se MIDE contando las filas `tipo == "prediccion"` que realmente
tiene `series.csv` por region (la misma logica que ya usa
`checkpoint.region_es_completa` para decidir si una region esta completa).
"""

import ast
import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .checkpoint import _coerce_booleanos, _leer_csv_si_existe, region_es_completa
from .regions import REGIONS_ALL

# =========================================================
# REQUISITOS DE COMPLETITUD POR MODELO
#
# Identicos a los que ya usa cada modulo via
# checkpoint.cargar_checkpoint_regiones() dentro de su propio run() (ver
# docstring de cada modulo) -- se reutiliza la MISMA definicion de "region
# completa" que ya gobierna el resume, para que la auditoria sea
# consistente con lo que run_matrix() consideraria "ya hecho". Solo los 7
# modelos multivariados participan en la matrix de exogenas individuales.
# =========================================================

REQUISITOS_COMPLETITUD = {
    "xgboost": dict(n_modelos_esperados=1, requiere_trials=True, requiere_config_usada=True, trials_esperados=None),
    "lightgbm": dict(n_modelos_esperados=1, requiere_trials=True, requiere_config_usada=True, trials_esperados=None),
    "lstm_direct": dict(n_modelos_esperados=1, requiere_trials=True, requiere_config_usada=True, trials_esperados=None),
    "sarimax": dict(n_modelos_esperados=1, requiere_trials=False, requiere_config_usada=True, trials_esperados=None),
    "fcnn": dict(n_modelos_esperados=2, requiere_trials=True, requiere_config_usada=True, trials_esperados=None),
    "ensemble_stl": dict(n_modelos_esperados=1, requiere_trials=True, requiere_config_usada=True, trials_esperados=None),
    "lstm_resid": dict(n_modelos_esperados=1, requiere_trials=True, requiere_config_usada=True, trials_esperados=None),
}

_REQUISITOS_DEFAULT = dict(n_modelos_esperados=1, requiere_trials=False, requiere_config_usada=False, trials_esperados=None)


# =========================================================
# TABLAS LITERALES PARA CARPETAS LEGACY (sin config.json)
#
# Confirmadas leyendo el codigo fuente de cada modulo (NOMBRE_MODELO/
# MODELO_DIRECTA/MODELO_STL_RESIDUOS/NOMBRE_MODELO_FINAL, y sus
# CANONICAL_TO_INTERNAL), no adivinadas ni derivadas del nombre de
# carpeta.
# =========================================================

MODELO_DISPLAY_A_CANONICO = {
    "XGBoost_Tuned_2Semanas": "xgboost",
    "LightGBM_Tuned": "lightgbm",
    "LSTM_Directa": "lstm_direct",
    "FCNN_Multivariada_EXOG_Lag168": "fcnn",
    "STL_FCNN_Multivariada_Residuos_EXOG_Lag168": "fcnn",
    "LSTM_Resid_Trend_Seasonal": "lstm_resid",
}

# SARIMAX/Ensemble arman su NOMBRE_MODELO concatenando constantes (orden
# SARIMAX, nombre del ensamble) -- se matchean por PREFIJO, no por
# igualdad exacta, por si una version legacy (anterior a este refactor)
# armo el string con constantes ligeramente distintas.
PREFIJOS_MODELO_A_CANONICO = [
    ("SARIMAX", "sarimax"),
    ("ENSEMBLE_STL", "ensemble_stl"),
]


def _modelo_canonico_desde_display(display: str) -> Optional[str]:
    if display in MODELO_DISPLAY_A_CANONICO:
        return MODELO_DISPLAY_A_CANONICO[display]
    for prefijo, canonico in PREFIJOS_MODELO_A_CANONICO:
        if str(display).startswith(prefijo):
            return canonico
    return None


# xgboost/lightgbm/lstm_direct ya usan nombres canonicos en su propio
# config_usada.csv (celda 49/58 y adaptaciones ya usaban "Temperatura"
# singular) -- no necesitan traduccion. sarimax/fcnn/ensemble_stl SI
# traducen internamente (ver CANONICAL_TO_INTERNAL de cada modulo);
# copiados aqui literales para poder revertir la traduccion sin importar
# esos modulos (fcnn/ensemble_stl necesitan tensorflow, que este archivo
# deliberadamente no importa a nivel de modulo).
_INTERNO_A_CANONICO_SARIMAX = {"Temperaturas": "Temperatura", "IGAE": "IGAE", "Generacion": "Generacion", "Importacion": "Importacion", "Exportacion": "Exportacion"}
_INTERNO_A_CANONICO_FCNN_ENSEMBLE = {"Temperaturas": "Temperatura", "IGAE": "IGAE", "Generacion_lag168": "Generacion", "Importacion_lag168": "Importacion", "Exportacion_lag168": "Exportacion"}

EXOGENA_INTERNA_A_CANONICA_POR_MODELO = {
    "sarimax": _INTERNO_A_CANONICO_SARIMAX,
    "fcnn": _INTERNO_A_CANONICO_FCNN_ENSEMBLE,
    "ensemble_stl": _INTERNO_A_CANONICO_FCNN_ENSEMBLE,
}


def _parsear_exogenas_texto(texto, modelo):
    """
    `config_usada.csv["exogenas"]` se escribio con uno de dos formatos
    segun el modulo: `str(lista_python)` (xgboost/lightgbm/lstm_direct/
    sarimax) o `", ".join(...)`/`",".join(...)` (fcnn/ensemble_stl/
    lstm_resid). Se prueban ambos formatos, y si el modelo es uno de los
    que traduce a nombres internos, se revierte la traduccion.
    """
    texto = str(texto).strip()

    if texto.startswith("[") and texto.endswith("]"):
        try:
            items = list(ast.literal_eval(texto))
        except Exception:
            items = [t.strip(" '\"") for t in texto[1:-1].split(",") if t.strip(" '\"")]
    else:
        items = [t.strip() for t in texto.split(",") if t.strip()]

    mapa = EXOGENA_INTERNA_A_CANONICA_POR_MODELO.get(modelo)
    if mapa:
        items = [mapa.get(i, i) for i in items]
    return items


# =========================================================
# IDENTIDAD DE UNA CARPETA (config.json primero, legacy como fallback)
# =========================================================

def _identidad_desde_config_json(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None

    return {
        "modelo": cfg.get("modelo"),
        "exogenas": list(cfg.get("exogenas") or []),
        "train_hours": cfg.get("train_hours"),
        "forecast_horizon": cfg.get("forecast_horizon"),
        "fuente": "config.json",
    }


def _identidad_desde_legacy(run_dir: str) -> Optional[dict]:
    metricas_path = os.path.join(run_dir, "metricas.csv")
    if not os.path.exists(metricas_path):
        return None

    try:
        metricas_df = pd.read_csv(metricas_path, encoding="utf-8-sig")
    except Exception:
        return None

    if "modelo" not in metricas_df.columns or len(metricas_df) == 0:
        return None

    displays = metricas_df["modelo"].dropna().unique().tolist()
    canonicos = {_modelo_canonico_desde_display(d) for d in displays}
    canonicos.discard(None)
    if len(canonicos) != 1:
        return None  # ambiguo (mezcla de modelos) o ningun display reconocido -- no adivinar

    modelo = next(iter(canonicos))
    exogenas = None
    train_hours = None

    config_usada_path = os.path.join(run_dir, "config_usada.csv")
    if os.path.exists(config_usada_path):
        try:
            config_df = pd.read_csv(config_usada_path, encoding="utf-8-sig")
        except Exception:
            config_df = None

        if config_df is not None and len(config_df) > 0:
            if "train_horas" in config_df.columns:
                valores = pd.to_numeric(config_df["train_horas"], errors="coerce").dropna().unique()
                if len(valores) == 1:
                    train_hours = int(valores[0])

            if "exogenas" in config_df.columns:
                valores_exog = config_df["exogenas"].dropna().unique()
                if len(valores_exog) == 1:
                    exogenas = _parsear_exogenas_texto(valores_exog[0], modelo)

    return {
        "modelo": modelo,
        "exogenas": exogenas,
        "train_hours": train_hours,
        "forecast_horizon": None,  # se mide desde series.csv, nunca desde texto -- ver _medir_forecast_horizon
        "fuente": "legacy (sin config.json)",
    }


def _medir_forecast_horizon(series_df: Optional[pd.DataFrame], region: str) -> Optional[int]:
    if series_df is None or len(series_df) == 0 or "region" not in series_df.columns or "tipo" not in series_df.columns:
        return None
    sub = series_df[(series_df["region"] == region) & (series_df["tipo"] == "prediccion")]
    return int(len(sub)) if len(sub) > 0 else None


# =========================================================
# EVALUAR UN CANDIDATO CONTRA UN OBJETIVO
# =========================================================

@dataclass
class ConfigObjetivo:
    modelo: str
    exogena: str
    train_hours: int
    forecast_horizon: int
    run_name: str


def _evaluar_candidato(run_dir: str, identidad: dict, objetivo: ConfigObjetivo, regiones_esperadas: list, requisitos: dict):
    metricas_df = _coerce_booleanos(_leer_csv_si_existe(os.path.join(run_dir, "metricas.csv")))
    series_df = _leer_csv_si_existe(os.path.join(run_dir, "series.csv"))
    trials_df = _leer_csv_si_existe(os.path.join(run_dir, "trials.csv"))
    config_usada_df = _leer_csv_si_existe(os.path.join(run_dir, "config_usada.csv"))

    forecast_horizon_medido = identidad["forecast_horizon"]
    if forecast_horizon_medido is None:
        medidos = set()
        for region in regiones_esperadas:
            n = _medir_forecast_horizon(series_df, region)
            if n:
                medidos.add(n)
        forecast_horizon_medido = medidos.pop() if len(medidos) == 1 else None

    train_hours_ok = None if identidad["train_hours"] is None else (identidad["train_hours"] == objetivo.train_hours)
    forecast_horizon_ok = None if forecast_horizon_medido is None else (forecast_horizon_medido == objetivo.forecast_horizon)

    regiones_ok = []
    for region in regiones_esperadas:
        nombre_serie = f"{region}_DEMANDA"
        if region_es_completa(
            nombre_serie, metricas_df, series_df, trials_df, config_usada_df,
            forecast_horizon=objetivo.forecast_horizon, **requisitos,
        ):
            regiones_ok.append(region)

    if 0 < len(regiones_ok) < len(regiones_esperadas):
        regiones_validas_str = f"{len(regiones_ok)}/{len(regiones_esperadas)} ({','.join(regiones_ok)})"
    else:
        regiones_validas_str = f"{len(regiones_ok)}/{len(regiones_esperadas)}"

    # Solo se declara "distinto" cuando hay evidencia POSITIVA de que
    # difiere (False); si no se pudo determinar (None), no se descarta el
    # match, se marca como advertencia para revision manual.
    metodologia_equivalente = (train_hours_ok is not False) and (forecast_horizon_ok is not False)

    if not metodologia_equivalente:
        detalles = []
        if train_hours_ok is False:
            detalles.append(f"train_hours difiere ({identidad['train_hours']} en la carpeta vs {objetivo.train_hours} esperado)")
        if forecast_horizon_ok is False:
            detalles.append(f"forecast_horizon difiere ({forecast_horizon_medido} en la carpeta vs {objetivo.forecast_horizon} esperado)")
        return "legacy_not_equivalent", regiones_validas_str, "; ".join(detalles)

    avisos = []
    if train_hours_ok is None:
        avisos.append("train_hours no se pudo confirmar")
    if forecast_horizon_ok is None:
        avisos.append("forecast_horizon no se pudo confirmar")

    if len(regiones_ok) == len(regiones_esperadas):
        razon = f"Coincide en modelo+exogena+train_hours+forecast_horizon, {len(regiones_ok)}/{len(regiones_esperadas)} regiones validas (fuente: {identidad['fuente']})."
        if avisos:
            razon += " AVISO: " + "; ".join(avisos) + " -- revisar manualmente antes de reutilizar."
        return "exact_match_complete", regiones_validas_str, razon

    razon = f"Coincide en modelo+exogena+train_hours+forecast_horizon pero faltan regiones (fuente: {identidad['fuente']})."
    if avisos:
        razon += " AVISO: " + "; ".join(avisos) + "."
    return "exact_match_partial", regiones_validas_str, razon


# =========================================================
# MOTOR PRINCIPAL (liviano -- testeable sin tensorflow/optuna)
# =========================================================

_RANK_ESTADO = {"exact_match_complete": 0, "exact_match_partial": 1, "legacy_not_equivalent": 2}


def auditar_matriz_individual(objetivos: list, pipeline_resultados_dir: str, regiones_esperadas: Optional[list] = None) -> pd.DataFrame:
    """
    Compara `objetivos` (lista de `ConfigObjetivo`, tipicamente las 35 que
    produce hoy `build_individual_exog_matrix()`) contra TODAS las
    subcarpetas de `pipeline_resultados_dir` -- con o sin `config.json`,
    con cualquier nombre de carpeta.

    Solo lee archivos; nunca escribe, mueve, renombra ni borra nada.

    Devuelve un DataFrame con exactamente las columnas pedidas: `modelo`,
    `exogena`, `train_hours`, `forecast_horizon`, `ruta_legacy`, `estado`,
    `regiones_validas`, `razon`. `estado` es uno de: `exact_match_complete`,
    `exact_match_partial`, `legacy_not_equivalent`, `missing`.
    """
    regiones_esperadas = list(regiones_esperadas) if regiones_esperadas is not None else list(REGIONS_ALL)

    candidatos_por_identidad = {}
    n_carpetas = 0
    n_sin_identidad = 0

    if os.path.isdir(pipeline_resultados_dir):
        for nombre in sorted(os.listdir(pipeline_resultados_dir)):
            run_dir = os.path.join(pipeline_resultados_dir, nombre)
            if not os.path.isdir(run_dir):
                continue
            n_carpetas += 1

            identidad = _identidad_desde_config_json(run_dir)
            if identidad is None:
                identidad = _identidad_desde_legacy(run_dir)

            if identidad is None or identidad.get("modelo") is None:
                n_sin_identidad += 1
                continue

            exogenas = identidad["exogenas"]
            if exogenas is None or len(exogenas) != 1:
                # No es candidata para ESTA auditoria (matriz de UNA sola
                # exogena): puede ser una corrida baseline (1A/1B, varias
                # exogenas o ninguna) o una carpeta cuyas exogenas no se
                # pudieron recuperar -- no es un error, simplemente no
                # aplica aqui.
                continue

            clave = (identidad["modelo"], exogenas[0])
            candidatos_por_identidad.setdefault(clave, []).append((run_dir, identidad))
    else:
        print(f"AVISO: {pipeline_resultados_dir} no existe o no es una carpeta.")

    filas = []
    for objetivo in objetivos:
        requisitos = REQUISITOS_COMPLETITUD.get(objetivo.modelo, _REQUISITOS_DEFAULT)
        candidatos = candidatos_por_identidad.get((objetivo.modelo, objetivo.exogena), [])

        if not candidatos:
            filas.append({
                "modelo": objetivo.modelo, "exogena": objetivo.exogena,
                "train_hours": objetivo.train_hours, "forecast_horizon": objetivo.forecast_horizon,
                "ruta_legacy": None, "estado": "missing",
                "regiones_validas": f"0/{len(regiones_esperadas)}",
                "razon": "No se encontro ninguna carpeta (con o sin config.json) con este modelo+exogena en Pipeline_Resultados.",
            })
            continue

        mejor = None
        for run_dir, identidad in candidatos:
            estado, regiones_validas_str, razon = _evaluar_candidato(run_dir, identidad, objetivo, regiones_esperadas, requisitos)
            rank = _RANK_ESTADO[estado]
            if mejor is None or rank < mejor[0]:
                mejor = (rank, run_dir, estado, regiones_validas_str, razon)

        _, run_dir, estado, regiones_validas_str, razon = mejor
        filas.append({
            "modelo": objetivo.modelo, "exogena": objetivo.exogena,
            "train_hours": objetivo.train_hours, "forecast_horizon": objetivo.forecast_horizon,
            "ruta_legacy": run_dir, "estado": estado,
            "regiones_validas": regiones_validas_str, "razon": razon,
        })

    columnas = ["modelo", "exogena", "train_hours", "forecast_horizon", "ruta_legacy", "estado", "regiones_validas", "razon"]
    df = pd.DataFrame(filas, columns=columnas)

    print("=" * 80)
    print("AUDITORIA: matriz de exogenas individuales vs Pipeline_Resultados existente")
    print("=" * 80)
    print(f"Carpeta: {pipeline_resultados_dir}")
    print(f"Carpetas inspeccionadas: {n_carpetas}")
    print(f"Carpetas sin identidad reconocible (modelo ambiguo/desconocido, se ignoran): {n_sin_identidad}")
    print(f"\nConfigs objetivo evaluadas: {len(objetivos)}")
    if len(df) > 0:
        conteo = df["estado"].value_counts().reindex(
            ["exact_match_complete", "exact_match_partial", "legacy_not_equivalent", "missing"], fill_value=0,
        )
        print(conteo.to_string())

    return df


# =========================================================
# ATAJOS DE CONVENIENCIA PARA COLAB (import pesado, perezoso)
# =========================================================

EXOGENAS_INDIVIDUALES_DEFAULT = ["Temperatura", "IGAE", "Generacion", "Importacion", "Exportacion"]


def construir_objetivos_desde_matriz_actual(exogenas: Optional[list] = None) -> list:
    """
    SOLO funciona donde esten instalados optuna/lightgbm/tensorflow/
    xgboost (Colab) -- importa `matrix.py`/`runner.py` de forma perezosa
    (adentro de esta funcion) para construir la lista real de
    `ConfigObjetivo` a partir de `build_individual_exog_matrix()` +
    `resolve_run()`, la MISMA fuente de verdad que usaria `run_matrix()`
    hoy si se lanzara la Fase 2.
    """
    from .matrix import build_individual_exog_matrix
    from .runner import resolve_run

    configs = build_individual_exog_matrix(exogenas=list(exogenas) if exogenas is not None else list(EXOGENAS_INDIVIDUALES_DEFAULT))

    objetivos = []
    for c in configs:
        r = resolve_run(c)
        objetivos.append(ConfigObjetivo(
            modelo=c.modelo, exogena=c.exogenas[0],
            train_hours=r.train_hours, forecast_horizon=r.forecast_horizon,
            run_name=r.run_name,
        ))
    return objetivos


def auditar_matriz_individual_actual(pipeline_resultados_dir: str, exogenas: Optional[list] = None, regiones_esperadas: Optional[list] = None) -> pd.DataFrame:
    """Atajo de un solo llamado para Colab: construye las 35 configs objetivo actuales y las audita contra Pipeline_Resultados. Ver auditar_matriz_individual / construir_objetivos_desde_matriz_actual."""
    objetivos = construir_objetivos_desde_matriz_actual(exogenas=exogenas)
    return auditar_matriz_individual(objetivos, pipeline_resultados_dir, regiones_esperadas=regiones_esperadas)

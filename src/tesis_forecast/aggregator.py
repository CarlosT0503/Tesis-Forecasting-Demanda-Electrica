"""
Agregador de resultados: consolida las corridas ya producidas en
`Pipeline_Resultados/<RUN_NAME>/` en dos datasets planos pensados para
Looker Studio -- `metricas_master.csv` y `series_master.csv`.

Independiente del entrenamiento a proposito: no importa `runner.py` ni
ningun modulo de `models/` (esos arrastran tensorflow/optuna/lightgbm/
xgboost). Solo lee CSV/JSON ya escritos en disco -- pandas + `config.py`
(liviano) + `validator.py` (liviano) son las unicas dependencias del
proyecto. No entrena, no reejecuta, no modifica ni borra ninguna carpeta
de resultados: es de solo lectura sobre `Pipeline_Resultados/`, y solo
escribe los dos CSV maestros (por defecto, en la raiz de esa misma
carpeta, como archivos hermanos de las carpetas de corridas -- nunca dentro
de una carpeta de corrida).

Reejecutarlo es seguro e idempotente: cada llamada reconstruye los dos CSV
maestros DESDE CERO a partir del estado actual de `Pipeline_Resultados/`
(nunca lee/acumula su propia salida anterior), asi que no hay forma de que
una fila quede duplicada por correr esto varias veces -- el archivo de
salida simplemente se sobreescribe con el resultado fresco de releer todo.

Clasificacion de familia (1A/1B/individual/temp_igae/univariado/unknown): a
partir de `modelo` + `exogenas` de cada `config.json`, NUNCA del nombre de
la carpeta. Ver `clasificar_familia()`. La familia `univariado` es la unica
excepcion: no viene de un `config.json` (esas corridas no tienen uno --
son legacy, de un script/notebook anterior a `runner.py`), se asigna
directamente al importar `Legacy_Univariados/` -- ver
`procesar_legacy_univariado()`.

Salida: `Pipeline_Resultados/Consolidado/` (subcarpeta dedicada, separada
de las carpetas de corridas -- `descubrir_runs_completos()` la excluye
explicitamente del descubrimiento para que nunca se intente leer como si
fuera una corrida). `Pipeline_Resultados/Legacy_Univariados/` (los 2 CSV
legacy) tambien se excluye del descubrimiento por el mismo motivo: no es
una carpeta de corrida moderna, se integra por una ruta explicita via
`consolidar_resultados(legacy_metricas_path=..., legacy_series_path=...)`.
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import MODEL_LABELS
from .regions import REGIONS_ALL
from .validator import validar_resultado

# =========================================================
# CLASIFICACION DE FAMILIA (a partir de config, no de carpetas)
# =========================================================

# Mismos 12 modelos y el mismo agrupamiento 1A/1B que ya usa el resto del
# proyecto (ver runner.MODEL_RUNNERS, notebooks/run_matrix.ipynb). No se
# importa runner.py (pesado); en su lugar, el assert de abajo verifica en
# cada import que este agrupamiento sigue cubriendo exactamente los mismos
# modelos que config.MODEL_LABELS -- si alguien agrega un modelo 13 sin
# actualizar este archivo, falla aqui en vez de clasificar en silencio.
FAMILIA_1A_MODELOS = frozenset({"xgboost", "lightgbm", "lstm_direct", "sarimax", "fcnn", "ensemble_stl"})
FAMILIA_1B_MODELOS = frozenset({"naive", "naive_trend", "naive_trend_seasonal", "ar", "ar_resid_trend_seasonal", "lstm_resid"})

_modelos_clasificados = FAMILIA_1A_MODELOS | FAMILIA_1B_MODELOS
_modelos_registrados = set(MODEL_LABELS)
if _modelos_clasificados != _modelos_registrados:
    raise RuntimeError(
        "aggregator.py desactualizado respecto a config.MODEL_LABELS: "
        f"en MODEL_LABELS pero no clasificados aqui: {_modelos_registrados - _modelos_clasificados}; "
        f"clasificados aqui pero ya no en MODEL_LABELS: {_modelos_clasificados - _modelos_registrados}"
    )


FAMILIA_TEMP_IGAE_EXOGENAS = frozenset({"Temperatura", "IGAE"})


def clasificar_familia(modelo: str, exogenas: list):
    """
    Reglas evaluadas EN ESTE ORDEN (el orden importa -- ver el aviso mas
    abajo sobre por que Temp+IGAE se chequea ANTES que 1A/1B, no despues):

      1. **Individual**: `exogenas` tiene **exactamente 1** elemento -- el
         patron exacto que produce `build_individual_exog_matrix()`
         (Fase 2). `exogena_individual` = esa unica exogena.
      2. **temp_igae**: `set(exogenas) == {"Temperatura", "IGAE"}` (2
         elementos, exactamente esas dos, en cualquier orden) -- el patron
         exacto de la Fase 3. `exogena_individual = None`.
      3. **1A / 1B**: cualquier otro conjunto de exogenas (tipicamente el
         catalogo completo por defecto del modelo, 8 o 5 segun cual, o
         `[]` para univariados) -- "1A" si `modelo` esta en
         `FAMILIA_1A_MODELOS`, "1B" si esta en `FAMILIA_1B_MODELOS`.
      4. **unknown**: nada de lo anterior aplica (modelo no reconocido, o
         una combinacion de exogenas que no corresponde a ninguna fase
         registrada) -- NUNCA se inventa una familia; se devuelve
         "unknown" explicitamente para que quien llame decida (reportar,
         excluir, o incluir igual marcado como tal).

    IMPORTANTE (pedido explicito): el chequeo de `temp_igae` va ANTES del
    chequeo por modelo (1A/1B) a proposito. Si se invirtiera el orden,
    una corrida de `xgboost` con `exogenas=["Temperatura","IGAE"]` (Fase
    3) clasificaria como "1A" solo porque `xgboost` esta en
    `FAMILIA_1A_MODELOS` -- el modelo por si solo NO alcanza para decidir
    la familia, hace falta mirar las exogenas primero.

    Devuelve `(familia: str, exogena_individual: Optional[str])`.
    """
    exogenas = list(exogenas) if exogenas else []

    if len(exogenas) == 1:
        return "individual", exogenas[0]

    if set(exogenas) == FAMILIA_TEMP_IGAE_EXOGENAS:
        return "temp_igae", None

    if modelo in FAMILIA_1A_MODELOS:
        return "1A", None

    if modelo in FAMILIA_1B_MODELOS:
        return "1B", None

    return "unknown", None


# =========================================================
# DESCUBRIR CORRIDAS COMPLETAS
# =========================================================

# Nombre de la subcarpeta de salida (ver consolidar_resultados()). Excluida
# EXPLICITAMENTE del descubrimiento -- nunca se intenta leer como si fuera
# una carpeta de corrida, sin importar que contenga.
NOMBRE_CARPETA_CONSOLIDADO = "Consolidado"

# Nombre de la subcarpeta legacy (metricas_global.csv/series_global.csv).
# Excluida del descubrimiento por el MISMO mecanismo que Consolidado/: no
# tiene config.json (nunca lo tuvo, es de un script anterior a runner.py),
# asi que sin esta exclusion caeria en la rama "sin_config" y se reportaria
# como una carpeta descartada por error -- no es una corrida incompleta, es
# una fuente legacy valida que se integra por ruta explicita (ver
# procesar_legacy_univariado()/consolidar_resultados()).
NOMBRE_CARPETA_LEGACY_UNIVARIADO = "Legacy_Univariados"

# Carpeta de salida de legacy_bca_reconstruido.py (metricas_bca_reconstruido.csv/
# series_bca_reconstruido.csv/metadata_reconstruccion.json). Mismo motivo que
# las dos anteriores: no tiene config.json, no es una corrida moderna -- se
# integra por ruta explicita (ver procesar_bca_reconstruido()/
# consolidar_resultados(bca_metricas_path=, bca_series_path=)).
NOMBRE_CARPETA_BCA_RECONSTRUIDO = "Legacy_Univariados_BCA_Reconstruido"

_CARPETAS_NO_RUN = frozenset({
    NOMBRE_CARPETA_CONSOLIDADO, NOMBRE_CARPETA_LEGACY_UNIVARIADO, NOMBRE_CARPETA_BCA_RECONSTRUIDO,
})


@dataclass
class RunInfo:
    run_dir: str
    run_name: str
    modelo: str
    exogenas: list
    train_hours: object
    forecast_horizon: object
    optuna_n_trials: object
    seed: object
    notas: str
    git_commit: object
    generated_at: object
    familia_experimento: str
    exogena_individual: Optional[str]


@dataclass
class CandidatoDescartado:
    """Una carpeta que se considero pero NO entro a los masters -- para `reporte_consolidacion.csv`."""
    nombre: str
    estado: str  # "sin_config" | "incompleto"
    razon: str


@dataclass
class DescubrimientoResultado:
    runs: list = field(default_factory=list)
    n_incompletos: int = 0
    n_sin_config: int = 0
    n_familia_unknown: int = 0
    n_reemplazados: int = 0  # corridas viejas excluidas por tener una version _timefix equivalente (ver aplicar_reemplazos_timefix())
    incompletos: list = field(default_factory=list)  # [(run_name, problemas)]
    descartados: list = field(default_factory=list)  # list[CandidatoDescartado], sin_config + incompletos + reemplazados juntos


def descubrir_runs_completos(pipeline_resultados_dir: str, regiones_esperadas: Optional[list] = None) -> DescubrimientoResultado:
    """
    Recorre las subcarpetas INMEDIATAS de `pipeline_resultados_dir` (una
    por `RUN_NAME`), y para cada una:
      - si es `Consolidado/`, `Legacy_Univariados/` o
        `Legacy_Univariados_BCA_Reconstruido/` (ver `_CARPETAS_NO_RUN`):
        se ignora sin contarla como candidata de ningun tipo;
      - si falta `config.json`: se cuenta como "sin config" y se ignora
        (no es una corrida, o quedo interrumpida antes de escribirlo);
      - si `validar_resultado()` (el mismo validador que ya usan
        `runner.run_experiment`/`matrix.run_matrix`) dice que esta
        incompleta: se cuenta y se ignora, guardando el motivo;
      - si esta completa: se lee `config.json`, se clasifica la familia
        (incluyendo "unknown" si no se puede determinar con seguridad --
        se incluye igual en `runs`, nunca se inventa una familia, pero
        tampoco se descarta el resultado por eso), y se agrega a `runs`.

    Nunca escribe ni borra nada. Es seguro llamarla mientras otras corridas
    siguen en progreso en otra sesion de Colab -- las incompletas
    simplemente no entran en el resultado.
    """
    resultado = DescubrimientoResultado()

    if not os.path.isdir(pipeline_resultados_dir):
        print(f"AVISO: {pipeline_resultados_dir} no existe o no es una carpeta -- no hay nada que consolidar.")
        return resultado

    nombres = sorted(
        n for n in os.listdir(pipeline_resultados_dir)
        if os.path.isdir(os.path.join(pipeline_resultados_dir, n)) and n not in _CARPETAS_NO_RUN
    )

    for nombre in nombres:
        run_dir = os.path.join(pipeline_resultados_dir, nombre)
        config_path = os.path.join(run_dir, "config.json")

        if not os.path.exists(config_path):
            resultado.n_sin_config += 1
            resultado.descartados.append(CandidatoDescartado(nombre, "sin_config", "falta config.json"))
            continue

        reporte = validar_resultado(run_dir, regiones_esperadas=regiones_esperadas)
        if not reporte.es_completo:
            resultado.n_incompletos += 1
            resultado.incompletos.append((nombre, reporte.problemas))
            resultado.descartados.append(CandidatoDescartado(nombre, "incompleto", "; ".join(reporte.problemas)))
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            resultado.n_incompletos += 1
            motivo = f"config.json ilegible: {e}"
            resultado.incompletos.append((nombre, [motivo]))
            resultado.descartados.append(CandidatoDescartado(nombre, "incompleto", motivo))
            continue

        modelo = cfg.get("modelo")
        exogenas = cfg.get("exogenas") or []
        familia, exogena_individual = clasificar_familia(modelo, exogenas)
        if familia == "unknown":
            resultado.n_familia_unknown += 1
            print(f"AVISO: {nombre} -- familia no determinable con seguridad (modelo={modelo!r}, exogenas={exogenas!r}). Incluida en los masters con familia_experimento='unknown'.")

        resultado.runs.append(RunInfo(
            run_dir=run_dir,
            run_name=cfg.get("run_name", nombre),
            modelo=modelo,
            exogenas=exogenas,
            train_hours=cfg.get("train_hours"),
            forecast_horizon=cfg.get("forecast_horizon"),
            optuna_n_trials=cfg.get("optuna_n_trials"),
            seed=cfg.get("seed"),
            notas=cfg.get("notas"),
            git_commit=cfg.get("git_commit"),
            generated_at=cfg.get("generated_at"),
            familia_experimento=familia,
            exogena_individual=exogena_individual,
        ))

    return resultado


# =========================================================
# REEMPLAZOS _timefix (rerun por fix de bug, preserva la corrida vieja en
# disco pero la excluye del consolidado)
# =========================================================

SUFIJO_TIMEFIX = "_timefix"


def identificar_reemplazos_timefix(runs: list) -> dict:
    """
    Detecta, de forma GENERICA (por sufijo, no por una lista hardcodeada de
    RUN_NAMEs), que corridas `_timefix` tienen una corrida "vieja" (mismo
    RUN_NAME sin el sufijo) presente en el mismo `runs`. No asume cuantas
    hay ni de que modelo/exogenas son -- funciona para cualquier corrida
    futura que use `ExperimentConfig(sufijo_run_name="timefix")`
    (`config.py::build_run_name(..., sufijo=)`).

    Devuelve `{run_name_viejo: run_name_timefix}`. Si un `_timefix` no
    tiene corrida vieja equivalente en `runs` (ya se borro, o nunca
    existio), simplemente no aparece en el resultado -- no es un error.
    """
    nombres_presentes = {r.run_name for r in runs}
    reemplazos = {}

    for run in runs:
        if not run.run_name.endswith(SUFIJO_TIMEFIX):
            continue
        run_name_viejo = run.run_name[: -len(SUFIJO_TIMEFIX)]
        if run_name_viejo in nombres_presentes:
            reemplazos[run_name_viejo] = run.run_name

    return reemplazos


def aplicar_reemplazos_timefix(descubrimiento: DescubrimientoResultado) -> DescubrimientoResultado:
    """
    Post-procesa un `DescubrimientoResultado` YA construido por
    `descubrir_runs_completos()`: mueve cada corrida vieja que tiene una
    version `_timefix` equivalente de `.runs` a `.descartados` (estado
    "reemplazado"), sin tocar nada en disco. La corrida `_timefix`
    correspondiente permanece en `.runs` sin cambios.

    Puramente aditivo/no destructivo sobre el objeto: devuelve un
    `DescubrimientoResultado` NUEVO (no muta el que recibio), para que
    `descubrir_runs_completos()` siga siendo pura y testeable por separado
    de esta logica de reemplazo.
    """
    reemplazos = identificar_reemplazos_timefix(descubrimiento.runs)
    if not reemplazos:
        return descubrimiento

    runs_incluidos = []
    descartados_nuevos = list(descubrimiento.descartados)
    n_reemplazados = 0

    for run in descubrimiento.runs:
        if run.run_name in reemplazos:
            run_name_timefix = reemplazos[run.run_name]
            descartados_nuevos.append(CandidatoDescartado(
                nombre=run.run_name,
                estado="reemplazado",
                razon=(
                    f"reemplazado por '{run_name_timefix}' -- fix de bug de alineacion "
                    "horaria +1h, ver auditoria. Carpeta preservada en disco como evidencia "
                    "historica, excluida del consolidado."
                ),
            ))
            n_reemplazados += 1
        else:
            runs_incluidos.append(run)

    return DescubrimientoResultado(
        runs=runs_incluidos,
        n_incompletos=descubrimiento.n_incompletos,
        n_sin_config=descubrimiento.n_sin_config,
        n_familia_unknown=descubrimiento.n_familia_unknown,
        n_reemplazados=n_reemplazados,
        incompletos=descubrimiento.incompletos,
        descartados=descartados_nuevos,
    )


def construir_reporte_consolidacion(
    descubrimiento: DescubrimientoResultado,
    ventanas_por_run: Optional[dict] = None,
    entradas_legacy: Optional[list] = None,
    entradas_bca: Optional[list] = None,
) -> pd.DataFrame:
    """
    Inventario de TODAS las carpetas/fuentes consideradas (incluidas y
    excluidas -- moderno, legacy y BCA reconstruido), con columnas
    `run_name`, `familia`, `origen` ("moderno" | "legacy_univariado" |
    "bca_univariado_reconstruido"), `estado` ("completo" | "incompleto" |
    "sin_config" | "reemplazado"), `incluido`, `razon`, `regiones`,
    `test_start`, `test_end`, `n_predicciones`. Pensado para guardarse como
    `reporte_consolidacion.csv` -- la trazabilidad de que se incluyo/
    excluyo y por que, separada de los datos mismos de `metricas_master`/
    `series_master`.

    Las corridas modernas "reemplazadas" (ver `aplicar_reemplazos_timefix()`)
    aparecen aca con `estado="reemplazado"`, `incluido=False` -- son
    corridas COMPLETAS que existen en disco (no incompletas, no sin
    config), simplemente excluidas del consolidado porque existe una
    version `_timefix` equivalente; `descubrimiento.descartados` ya las
    trae con ese `estado`, esta funcion no necesita saber nada especifico
    de _timefix, solo itera `descartados` genericamente.

    `ventanas_por_run`: dict opcional `run_name -> {"regiones", "test_start",
    "test_end", "n_predicciones"}` (ver `calcular_ventanas_por_run()`
    agregado por run_name) -- si se pasa, rellena esas 4 columnas para
    CUALQUIER run_name presente ahi, derivado unicamente de lo que ya quedo
    en `series_master` (nunca recalculado aca).

    `entradas_legacy`/`entradas_bca`: listas opcionales de dicts (una por
    estrategia, ver `procesar_legacy_univariado()`/`procesar_bca_reconstruido()`)
    con `run_name`/`familia`/`estado`/`incluido`/`razon` -- se agregan tal
    cual, con `origen` fijo segun corresponda, sin tocar la logica de
    arriba (que sigue siendo exclusivamente sobre `descubrimiento`,
    corridas modernas).
    """
    filas = []
    ventanas_por_run = ventanas_por_run or {}

    def _fila_ventana(run_name):
        v = ventanas_por_run.get(run_name)
        if not v:
            return {"regiones": None, "test_start": None, "test_end": None, "n_predicciones": None}
        return {
            "regiones": "|".join(v.get("regiones") or []),
            "test_start": v.get("test_start"),
            "test_end": v.get("test_end"),
            "n_predicciones": v.get("n_predicciones"),
        }

    for run in descubrimiento.runs:
        razon = "" if run.familia_experimento != "unknown" else (
            f"familia no determinable con seguridad (modelo={run.modelo!r}, exogenas={run.exogenas!r})"
        )
        filas.append({
            "run_name": run.run_name,
            "familia": run.familia_experimento,
            "origen": ORIGEN_MODERNO,
            "estado": "completo",
            "incluido": True,
            "razon": razon,
            **_fila_ventana(run.run_name),
        })

    for descartado in descubrimiento.descartados:
        filas.append({
            "run_name": descartado.nombre,
            "familia": None,
            "origen": ORIGEN_MODERNO,
            "estado": descartado.estado,
            "incluido": False,
            "razon": descartado.razon,
            **_fila_ventana(descartado.nombre),
        })

    for entrada in (entradas_legacy or []):
        filas.append({
            "run_name": entrada["run_name"],
            "familia": entrada.get("familia", FAMILIA_UNIVARIADO),
            "origen": ORIGEN_LEGACY_UNIVARIADO,
            "estado": entrada.get("estado", "completo"),
            "incluido": entrada.get("incluido", True),
            "razon": entrada.get("razon", ""),
            **_fila_ventana(entrada["run_name"]),
        })

    for entrada in (entradas_bca or []):
        filas.append({
            "run_name": entrada["run_name"],
            "familia": entrada.get("familia", FAMILIA_UNIVARIADO),
            "origen": ORIGEN_BCA_RECONSTRUIDO,
            "estado": entrada.get("estado", "completo"),
            "incluido": entrada.get("incluido", True),
            "razon": entrada.get("razon", ""),
            **_fila_ventana(entrada["run_name"]),
        })

    columnas = [
        "run_name", "familia", "origen", "estado", "incluido", "razon",
        "regiones", "test_start", "test_end", "n_predicciones",
    ]
    if not filas:
        return pd.DataFrame(columns=columnas)

    df = pd.DataFrame(filas, columns=columnas)
    return df.sort_values("run_name").reset_index(drop=True)


# =========================================================
# UTILIDADES DE COLUMNAS (evitar colisiones con metadata nueva)
# =========================================================

_COLUMNAS_METADATA = [
    "run_name", "modelo", "familia_experimento", "exogenas", "exogena_individual",
    "train_hours", "forecast_horizon", "optuna_n_trials", "seed", "notas",
    "git_commit", "generated_at", "run_dir", "origen", "metodologia",
]


def _renombrar_modelo_original(df: pd.DataFrame) -> pd.DataFrame:
    """
    `series.csv`/`metricas.csv` YA traen una columna `modelo` propia (el
    nombre interno de cada pipeline, ej. "XGBoost_Tuned_2Semanas",
    "STL_FCNN_Multivariada_Residuos_EXOG_Lag168", o "real" para las filas
    de la serie real) -- distinta del `modelo` canonico que pide esta
    tarea (ej. "xgboost", "fcnn", el mismo identificador de
    `ExperimentConfig.modelo`/`runner.MODEL_RUNNERS`). Se preserva esa
    columna original bajo `modelo_estrategia` (no se descarta informacion:
    es lo unico que distingue, por ejemplo, las 2 estrategias de FCNN
    dentro de una misma corrida) y se deja `modelo` libre para la
    metadata nueva.
    """
    df = df.copy()
    if "modelo" in df.columns:
        df = df.rename(columns={"modelo": "modelo_estrategia"})
    return df


def _evitar_colision_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ademas de `modelo` (manejado aparte, ver `_renombrar_modelo_original`),
    al menos un modulo (`sarimax_model.py`) ya escribe su propia columna
    `exogenas` en `metricas.csv` (los nombres INTERNOS que usa esa celda,
    ej. "['Temperaturas', 'IGAE', ...]") -- si se dejara tal cual,
    quedaria pisada por la columna `exogenas` nueva (el catalogo CANONICO
    de la config). Cualquier columna original cuyo nombre coincida con una
    columna de metadata nueva se renombra con sufijo `_original`, sin
    tocar su contenido.
    """
    df = df.copy()
    renombres = {
        col: f"{col}_original"
        for col in _COLUMNAS_METADATA
        if col in df.columns and col != "modelo"
    }
    return df.rename(columns=renombres) if renombres else df


def _agregar_metadata(df: pd.DataFrame, run: RunInfo) -> pd.DataFrame:
    df = df.copy()
    df["run_name"] = run.run_name
    df["modelo"] = run.modelo
    df["familia_experimento"] = run.familia_experimento
    df["exogenas"] = "|".join(run.exogenas)
    df["exogena_individual"] = run.exogena_individual
    df["train_hours"] = run.train_hours
    df["forecast_horizon"] = run.forecast_horizon
    df["optuna_n_trials"] = run.optuna_n_trials
    df["seed"] = run.seed
    df["notas"] = run.notas
    df["git_commit"] = run.git_commit
    df["generated_at"] = run.generated_at
    df["run_dir"] = run.run_dir
    # Esta funcion SOLO se usa para corridas modernas (runs de
    # descubrir_runs_completos()) -- origen es siempre ORIGEN_MODERNO aca;
    # legacy y BCA reconstruido tienen su propia construccion de metadata
    # (ver procesar_legacy_univariado()/procesar_bca_reconstruido()).
    df["origen"] = ORIGEN_MODERNO
    df["metodologia"] = None
    return df


# =========================================================
# metricas_master.csv
# =========================================================

def construir_metricas_master(runs: list) -> pd.DataFrame:
    """
    Une `metricas.csv` de cada corrida COMPLETA (`runs`, ya filtradas por
    `descubrir_runs_completos`), preservando la granularidad original
    (una fila = una fila de metricas por region/modelo -- FCNN sigue
    aportando 2 filas por region, una por estrategia). No se recalcula
    ni se toca ningun valor de MAE/RMSE/MAPE/sMAPE; solo se le agregan
    columnas de metadata.

    Columnas heterogeneas entre modelos (ej. `AIC`/`BIC`/`order` solo en
    SARIMAX, `tuneado`/`horizonte_usado` solo en XGBoost/LightGBM/LSTM
    directa/Naive-family) se preservan tal cual via union de columnas
    (`pd.concat` con relleno `NaN` donde no aplique) -- ninguna se
    descarta ni se fuerza a un esquema comun.
    """
    frames = []

    for run in runs:
        metricas_path = os.path.join(run.run_dir, "metricas.csv")
        if not os.path.exists(metricas_path):
            print(f"AVISO: {run.run_name} paso validar_resultado() pero no tiene metricas.csv -- se omite.")
            continue

        try:
            df = pd.read_csv(metricas_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"AVISO: no se pudo leer metricas.csv de {run.run_name}: {e} -- se omite.")
            continue

        df = _renombrar_modelo_original(df)
        df = _evitar_colision_metadata(df)
        df = _agregar_metadata(df, run)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


# =========================================================
# series_master.csv
# =========================================================

_TIPOS_COMPARABLES = {"real", "prediccion"}


def construir_series_master(
    runs: list,
    verbose: bool = True,
    extra_pred_frames: Optional[list] = None,
    extra_real_frames: Optional[list] = None,
) -> pd.DataFrame:
    """
    Consolida `series.csv` de cada corrida completa en un unico dataset
    "largo" (una fila por timestamp/region/modelo), pensado para comparar
    predicciones de distintos modelos en Looker Studio SIN repetir
    innecesariamente la serie real:

      1. Se descartan filas `tipo == "componente_pred"` (desglose interno
         de Ensemble STL -- trend/seasonal/resid por separado, no es la
         prediccion final comparable).
      2. Las filas `tipo == "prediccion"` se conservan TODAS, una por cada
         corrida/modelo (eso es exactamente lo que hay que comparar).
      3. El "horizonte comparable" por region se determina a partir de los
         RESULTADOS -- la union de todos los timestamps que aparecen como
         `prediccion` en CUALQUIER corrida para esa region -- no de
         `train_hours`/`forecast_horizon` de la config (que ademas puede
         ser el string "auto" para Naive/Naive_Trend/AR, sin un numero
         fijo de horas que asumir).
      4. Las filas `tipo == "real"` (que cada modulo guarda con la serie
         COMPLETA, no solo la ventana de test -- ver docstring de
         `fcnn_model.py`/`sarimax_model.py`/etc.) se recortan a solo esos
         timestamps comparables, y se deduplican por `(region, timestamp)`
         -- el mismo dato real no se repite una vez por cada corrida que
         lo tenia guardado.

    Si dos corridas traen un valor "real" distinto para el mismo
    `(region, timestamp)` (evidencia de que corrieron contra snapshots de
    datos distintos), se imprime un aviso y se conserva el primer valor
    visto -- no se promedia ni se descarta silenciosamente.

    Las filas `real` no pertenecen a ninguna corrida en particular, asi que
    sus columnas de metadata por-corrida (`run_name`, `modelo`,
    `familia_experimento`, `exogenas`, `exogena_individual`) quedan en
    `NaN`; `modelo_estrategia` conserva el valor original ("real") para
    que `serie_tipo`/`modelo_estrategia` sigan siendo suficientes para
    filtrar sin depender de que las demas columnas esten vacias.

    `extra_pred_frames`/`extra_real_frames` (opcionales, default `None` ->
    comportamiento IDENTICO al de antes): fuentes adicionales YA taggeadas
    con su propio `run_name`/metadata (ej. `Legacy_Univariados/`, ver
    `procesar_legacy_univariado()`), que se suman al MISMO pool de
    predicciones/reales de las corridas modernas antes del merge y la
    deduplicacion -- reutilizan exactamente la misma logica de union de
    horizonte comparable, dedup de `real` y deteccion de conflictos que ya
    corre para corridas modernas, sin duplicar esa logica.
    """
    pred_frames = []
    real_candidatos = []  # lista de DataFrames crudos (region, timestamp, valor), 1 por corrida que trajo "real"

    ventanas_test_por_region = {}  # region -> lista de (run_name, min_ts, max_ts, n_timestamps)

    for run in runs:
        series_path = os.path.join(run.run_dir, "series.csv")
        if not os.path.exists(series_path):
            print(f"AVISO: {run.run_name} paso validar_resultado() pero no tiene series.csv -- se omite.")
            continue

        try:
            df = pd.read_csv(series_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"AVISO: no se pudo leer series.csv de {run.run_name}: {e} -- se omite.")
            continue

        if "tipo" not in df.columns:
            print(f"AVISO: series.csv de {run.run_name} no tiene columna 'tipo' -- se omite.")
            continue

        df = df[df["tipo"].isin(_TIPOS_COMPARABLES)].copy()
        if len(df) == 0:
            continue

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = _renombrar_modelo_original(df)
        df = df.rename(columns={"fecha": "timestamp", "tipo": "serie_tipo"})

        pred = df[df["serie_tipo"] == "prediccion"]
        real = df[df["serie_tipo"] == "real"]

        if len(pred) > 0:
            pred_tagged = _evitar_colision_metadata(pred)
            pred_tagged = _agregar_metadata(pred_tagged, run)
            pred_frames.append(pred_tagged)

            for region, grupo in pred.groupby("region"):
                ts = grupo["timestamp"]
                ventanas_test_por_region.setdefault(region, []).append(
                    (run.run_name, ts.min(), ts.max(), ts.nunique())
                )

        if len(real) > 0:
            real_candidatos.append(real[["region", "timestamp", "valor", "modelo_estrategia"]])

    # Fuentes adicionales (ej. legacy univariado) -- ya vienen taggeadas con
    # su propio run_name/modelo_estrategia; se suman tal cual al mismo pool,
    # SIN tocar nada de la logica de arriba (corridas modernas).
    for pred_extra in (extra_pred_frames or []):
        if pred_extra is None or len(pred_extra) == 0:
            continue
        pred_frames.append(pred_extra)
        for (region, run_name_extra), grupo in pred_extra.groupby(["region", "run_name"]):
            ts = grupo["timestamp"]
            ventanas_test_por_region.setdefault(region, []).append(
                (run_name_extra, ts.min(), ts.max(), ts.nunique())
            )

    for real_extra in (extra_real_frames or []):
        if real_extra is None or len(real_extra) == 0:
            continue
        real_candidatos.append(real_extra[["region", "timestamp", "valor", "modelo_estrategia"]])

    if verbose:
        _reportar_ventanas_test(ventanas_test_por_region)

    pred_master = pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()

    real_master = _construir_real_deduplicado(real_candidatos, pred_master, verbose=verbose)

    columnas_metadata_real = [c for c in _COLUMNAS_METADATA if c != "modelo"]
    for col in columnas_metadata_real:
        if col not in real_master.columns:
            real_master[col] = np.nan
    real_master["modelo"] = np.nan

    columnas_finales = ["timestamp", "region", "valor", "serie_tipo", "modelo_estrategia"] + _COLUMNAS_METADATA
    for df in (real_master, pred_master):
        for col in columnas_finales:
            if col not in df.columns:
                df[col] = np.nan

    frames_finales = [d for d in (real_master, pred_master) if len(d) > 0]
    if not frames_finales:
        return pd.DataFrame(columns=columnas_finales)

    resultado = pd.concat([d[columnas_finales] for d in frames_finales], ignore_index=True, sort=False)
    return resultado.sort_values(["region", "timestamp", "serie_tipo", "run_name"], na_position="first").reset_index(drop=True)


def _construir_real_deduplicado(real_candidatos: list, pred_master: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if not real_candidatos or len(pred_master) == 0:
        return pd.DataFrame(columns=["timestamp", "region", "valor", "serie_tipo", "modelo_estrategia"])

    timestamps_comparables = (
        pred_master[["region", "timestamp"]].drop_duplicates()
    )

    real_todo = pd.concat(real_candidatos, ignore_index=True, sort=False)
    real_todo = real_todo.dropna(subset=["timestamp"])

    real_recortado = real_todo.merge(timestamps_comparables, on=["region", "timestamp"], how="inner")

    if len(real_recortado) == 0:
        return pd.DataFrame(columns=["timestamp", "region", "valor", "serie_tipo", "modelo_estrategia"])

    conflictos = (
        real_recortado.groupby(["region", "timestamp"])["valor"]
        .agg(lambda s: s.max() - s.min() if len(s) > 1 else 0.0)
    )
    n_conflictos = int((conflictos.abs() > 1e-6).sum())
    if verbose and n_conflictos > 0:
        print(
            f"AVISO: {n_conflictos} (region, timestamp) tienen valores 'real' distintos entre corridas "
            "-- probablemente corrieron contra snapshots de datos distintos. Se conserva el primer valor visto."
        )

    real_dedup = real_recortado.drop_duplicates(subset=["region", "timestamp"], keep="first").copy()
    real_dedup["serie_tipo"] = "real"
    return real_dedup


def _reportar_ventanas_test(ventanas_test_por_region: dict):
    """
    Imprime, por region, si todas las corridas comparten exactamente el
    mismo rango de timestamps de test o no -- pedido explicito de la
    tarea ("verifica especialmente si todos los runs usan exactamente los
    mismos timestamps de test por region"). No fuerza nada: el join real
    ya esta disenado como union (`_construir_real_deduplicado`), asi que
    esto es puramente informativo.
    """
    if not ventanas_test_por_region:
        return

    print("\nVentanas de prediccion (horizonte de test) por region:")
    for region in sorted(ventanas_test_por_region):
        entradas = ventanas_test_por_region[region]
        rangos = {(str(min_ts), str(max_ts), n) for _, min_ts, max_ts, n in entradas}

        if len(rangos) == 1:
            (min_ts, max_ts, n) = next(iter(rangos))
            print(f"  {region}: {len(entradas)} corrida(s), TODAS con el mismo rango [{min_ts} .. {max_ts}] ({n} horas)")
        else:
            print(f"  {region}: {len(entradas)} corrida(s) con rangos DISTINTOS de test -- no se asume alineacion:")
            for run_name, min_ts, max_ts, n in sorted(entradas, key=lambda t: t[0]):
                print(f"      {run_name}: [{min_ts} .. {max_ts}] ({n} horas)")


# =========================================================
# VENTANAS TEMPORALES POR RUN (test_start/test_end/n_predicciones)
# =========================================================

def calcular_ventanas_por_run(series_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de series_master YA construido (moderno, o moderno+legacy),
    deriva por (run_name, region) la ventana de prediccion comparable
    (test_start/test_end) y el numero de predicciones -- unicamente de lo
    que ya quedo en series_master, sin recalcular nada ni asumir un
    horizonte fijo. No fuerza alineacion entre corridas: cada (run_name,
    region) conserva su propia ventana tal cual salio de sus predicciones.

    Pensada para analisis posterior (ej. distinguir en Looker Studio "todos
    los resultados" vs "comparacion estrictamente por misma ventana
    temporal") -- NUNCA se usa para recalcular metricas ni para forzar
    ninguna interseccion aca.
    """
    columnas = ["run_name", "region", "test_start", "test_end", "n_predicciones"]
    if len(series_master_df) == 0:
        return pd.DataFrame(columns=columnas)

    pred = series_master_df[series_master_df["serie_tipo"] == "prediccion"]
    pred = pred.dropna(subset=["run_name"])
    if len(pred) == 0:
        return pd.DataFrame(columns=columnas)

    agg = (
        pred.groupby(["run_name", "region"])["timestamp"]
        .agg(test_start="min", test_end="max", n_predicciones="count")
        .reset_index()
    )
    return agg[columnas]


def calcular_ventanas_agregadas_por_run(series_master_df: pd.DataFrame) -> dict:
    """
    Igual que `calcular_ventanas_por_run()` pero agregado por `run_name`
    solamente (union de todas sus regiones): `{run_name: {"regiones": [...],
    "test_start", "test_end", "n_predicciones"}}` -- el formato que espera
    `construir_reporte_consolidacion(ventanas_por_run=...)` (una fila por
    run_name, no por region).
    """
    por_region = calcular_ventanas_por_run(series_master_df)
    if len(por_region) == 0:
        return {}

    resultado = {}
    for run_name, grupo in por_region.groupby("run_name"):
        resultado[run_name] = {
            "regiones": sorted(grupo["region"].unique().tolist()),
            "test_start": grupo["test_start"].min(),
            "test_end": grupo["test_end"].max(),
            "n_predicciones": int(grupo["n_predicciones"].sum()),
        }
    return resultado


def agregar_ventanas_temporales(metricas_df: pd.DataFrame, series_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece metricas_master (SIN recalcular ningun MAE/RMSE/MAPE/sMAPE)
    con `test_start`/`test_end`/`n_predicciones` por (run_name, region),
    derivados de las predicciones ya presentes en series_master. Puramente
    aditivo (merge `how="left"`): una corrida sin predicciones comparables
    simplemente queda con esas 3 columnas en NaN/NaT.
    """
    columnas_nuevas = ["test_start", "test_end", "n_predicciones"]

    if len(metricas_df) == 0:
        out = metricas_df.copy()
        for col in columnas_nuevas:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    ventanas = calcular_ventanas_por_run(series_master_df)
    if len(ventanas) == 0:
        out = metricas_df.copy()
        for col in columnas_nuevas:
            out[col] = np.nan
        return out

    return metricas_df.merge(ventanas, on=["run_name", "region"], how="left")


# =========================================================
# LEGACY UNIVARIADO (Legacy_Univariados/metricas_global.csv +
# series_global.csv) -- integracion como quinta familia experimental
# =========================================================
#
# Estos 2 CSV NO vienen de runner.py (no tienen config.json, no siguen la
# convencion RUN_NAME=<Modelo>_train<h>h_fh<h>h_<exog>): son resultados de
# un script/notebook ANTERIOR, ya movidos a Drive tal cual. Esta seccion es
# la UNICA via de entrada para ellos -- nunca se descubren como carpeta
# (ver NOMBRE_CARPETA_LEGACY_UNIVARIADO en descubrir_runs_completos()), se
# integran por ruta EXPLICITA (ver consolidar_resultados(legacy_metricas_path=,
# legacy_series_path=)). Solo lectura sobre los 2 CSV: nunca se escriben,
# mueven ni sobreescriben.

FAMILIA_UNIVARIADO = "univariado"

# Valores canonicos de la columna "origen" -- las 3 fuentes que puede tener
# una fila de metricas_master/series_master. Usados en TODO el modulo (no
# solo en esta seccion) para que nunca se escriban a mano en mas de un
# lugar.
ORIGEN_MODERNO = "moderno"
ORIGEN_LEGACY_UNIVARIADO = "legacy_univariado"
ORIGEN_BCA_RECONSTRUIDO = "bca_univariado_reconstruido"

# Tabla explicita estrategia_legacy -> modelo_canonico. Cada entrada esta
# documentada con la evidencia de codigo que la respalda -- NINGUNA es una
# adivinanza. Evidencia completa (ver tambien el reporte de auditoria
# entregado por separado):
#
#   - ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid:
#       models/ensemble_stl.py:124 -> NOMBRE_MODELO_FINAL =
#       "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid_EXOG_ALL_Lag168" -- el
#       modulo MODERNO (multivariado) reutiliza EXACTAMENTE este nombre
#       como raiz y le agrega el sufijo de exogenas. La version legacy (sin
#       sufijo) es el equivalente UNIVARIADO -- distinto canonico porque es
#       una corrida distinta (script legacy, no ensemble_stl.py).
#   - ARIMA_1_1_1:
#       orden (1,1,1), sin estacionalidad ni exogenas -- no existe un
#       modulo moderno equivalente (sarimax_model.py ya es estacional Y
#       multivariado). Precursor simple de SARIMAX.
#   - SARIMA_1_1_1__1_0_1_168:
#       models/sarimax_model.py:427 -> NOMBRE_MODELO =
#       "SARIMAX_1_1_1__1_0_1_168_EXOG_2M" -- MISMO order=(1,1,1) y
#       seasonal_order=(1,0,1,168) (ver sarimax_model.py:63-64), version SIN
#       exogenas (SARIMA en vez de SARIMAX).
#   - FCNN_Individual:
#       analogo univariado de "FCNN_Multivariada_EXOG_Lag168"
#       (fcnn_model.py) -- sin modulo moderno univariado dedicado.
#   - STL_FCNN_residuos:
#       analogo univariado de "STL_FCNN_Multivariada_Residuos_EXOG_Lag168"
#       (fcnn_model.py) -- sin modulo moderno univariado dedicado.
#   - AR_AIC:
#       metodologia "orden AR elegido por AIC" -- coincide con
#       models/ar_model.py (docstring: "Seleccionando orden AR por AIC") Y
#       con models/ensemble_stl.py:905 ("resid_model": "AR_AIC", el AR
#       sobre el residuo dentro del ensemble). PERO ar_model.py moderno
#       guarda su metrica como "AR" (no "AR_AIC") -- no son la misma
#       corrida/modulo, por eso NO se reusa el canonico "ar" (reservado
#       para corridas modernas de ar_model.py).
#   - STL_AR_residuos_AIC:
#       analogo simplificado (STL + AR(AIC) sobre el residuo, SIN
#       tendencia lineal ni estacionalidad repetida reconstruidas aparte).
#       IMPORTANTE: el modulo moderno ar_resid_trend_seasonal_model.py es
#       EXPLICITAMENTE una "combinacion nueva" (ver config.py, docstring de
#       MODEL_LABELS: "no extraen un pipeline completo de legacy"), NO una
#       extraccion de esta estrategia legacy -- por eso NO comparten
#       canonico: metodologicamente son distintos, no serian comparables
#       como si fueran el mismo modelo.
#   - LSTM_Individual:
#       sin modulo moderno equivalente -- lstm_direct.py/lstm_resid_model.py
#       modernos son ambos multivariados (requieren exogenas, ver
#       config.py). Precursor univariado independiente.
MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO = {
    "ENSEMBLE_STL_LSTMtrend_FCNNseason_ARresid": "ensemble_stl_univariado",
    "ARIMA_1_1_1": "arima",
    "SARIMA_1_1_1__1_0_1_168": "sarima",
    "FCNN_Individual": "fcnn_univariada",
    "STL_FCNN_residuos": "fcnn_residuos_stl",
    "AR_AIC": "ar_aic",
    "STL_AR_residuos_AIC": "ar_residuos_stl",
    "LSTM_Individual": "lstm_univariada",
}


def build_run_name_legacy_univariado(estrategia_legacy: str) -> str:
    """
    RUN_NAME deterministico para una estrategia legacy univariada: estable
    entre ejecuciones (depende solo del nombre de la estrategia, nunca del
    orden de lectura del CSV ni de un timestamp/hash), y estructuralmente
    no puede colisionar con ningun RUN_NAME moderno -- `build_run_name()`
    (config.py) siempre arranca con `MODEL_LABELS[modelo]` (ej. "XGBoost_",
    "SARIMAX_", "Naive_", ...), nunca con el prefijo fijo
    "Legacy_Univariado__" que se usa aca.
    """
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(estrategia_legacy).strip())
    return f"Legacy_Univariado__{slug}"


def _derivar_region_desde_serie(nombre_serie) -> str:
    """Igual que `_region_de_serie()` en cada modulo de modelo/checkpoint.py: la region es el primer token de 'serie' (ej. 'BCA_DEMANDA' -> 'BCA')."""
    return str(nombre_serie).split("_")[0]


def _resolver_columna_region(df: pd.DataFrame, columna_serie: str = "serie") -> pd.Series:
    """
    Devuelve la columna 'region': si el CSV ya trae una columna 'region' la
    usa tal cual; si no, la deriva de 'serie' -- mismo criterio que usa el
    resto del proyecto (ver `_derivar_region_desde_serie`). Lanza un error
    claro (no adivina en silencio) si no hay forma segura de obtenerla.
    """
    if "region" in df.columns:
        return df["region"]
    if columna_serie not in df.columns:
        raise ValueError(
            f"No se pudo derivar 'region': el CSV no tiene columna 'region' ni '{columna_serie}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return df[columna_serie].map(_derivar_region_desde_serie)


@dataclass
class ValidacionLegacyUnivariado:
    """Resultado de `validar_legacy_univariado()` -- solo lectura, nunca modifica los CSV legacy."""
    ok: bool
    problemas: list
    regiones_metricas: list
    regiones_series: list
    estrategias_metricas: list
    estrategias_series_prediccion: list
    columnas_metricas: list
    columnas_series: list
    n_filas_metricas: int
    n_filas_series: int
    tipos_series: list
    tiene_real: bool
    tiene_componente_pred: bool
    n_predicciones_por_region_estrategia: dict
    ventana_prediccion_por_estrategia: dict
    estrategias_sin_mapeo: list


def validar_legacy_univariado(metricas_df: pd.DataFrame, series_df: pd.DataFrame) -> ValidacionLegacyUnivariado:
    """
    Validacion PREVIA, de solo lectura, de `metricas_global.csv`/
    `series_global.csv` YA LEIDOS en memoria (esta funcion no abre ni
    escribe ningun archivo). No hardcodea numeros esperados -- todo lo que
    reporta se deriva del contenido real de los DataFrames recibidos.
    """
    problemas = []

    columnas_metricas = list(metricas_df.columns)
    columnas_series = list(series_df.columns)

    for requerida in ["modelo", "MAE", "RMSE", "MAPE", "sMAPE"]:
        if requerida not in metricas_df.columns:
            problemas.append(f"metricas_global.csv: falta columna requerida '{requerida}'")

    for requerida in ["modelo", "fecha", "tipo", "valor"]:
        if requerida not in series_df.columns:
            problemas.append(f"series_global.csv: falta columna requerida '{requerida}'")

    if "region" not in metricas_df.columns and "serie" not in metricas_df.columns:
        problemas.append("metricas_global.csv: no tiene 'region' ni 'serie' -- no se puede derivar la region")
    if "region" not in series_df.columns and "serie" not in series_df.columns:
        problemas.append("series_global.csv: no tiene 'region' ni 'serie' -- no se puede derivar la region")

    if problemas:
        return ValidacionLegacyUnivariado(
            ok=False, problemas=problemas,
            regiones_metricas=[], regiones_series=[],
            estrategias_metricas=[], estrategias_series_prediccion=[],
            columnas_metricas=columnas_metricas, columnas_series=columnas_series,
            n_filas_metricas=len(metricas_df), n_filas_series=len(series_df),
            tipos_series=[], tiene_real=False, tiene_componente_pred=False,
            n_predicciones_por_region_estrategia={}, ventana_prediccion_por_estrategia={},
            estrategias_sin_mapeo=[],
        )

    region_metricas = _resolver_columna_region(metricas_df)
    region_series = _resolver_columna_region(series_df)

    regiones_metricas = sorted(region_metricas.dropna().unique().tolist())
    regiones_series = sorted(region_series.dropna().unique().tolist())

    estrategias_metricas = sorted(metricas_df["modelo"].dropna().unique().tolist())

    for col in ["MAE", "RMSE", "MAPE", "sMAPE"]:
        n_nan = int(metricas_df[col].isna().sum())
        if n_nan:
            problemas.append(f"metricas_global.csv: columna '{col}' tiene {n_nan} valor(es) NaN")

    tipos_series = sorted(series_df["tipo"].dropna().unique().tolist())
    tiene_real = "real" in tipos_series
    tiene_componente_pred = "componente_pred" in tipos_series

    pred_df = series_df[series_df["tipo"] == "prediccion"].copy()
    pred_df["region"] = _resolver_columna_region(pred_df)
    estrategias_series_prediccion = sorted(pred_df["modelo"].dropna().unique().tolist())

    n_predicciones_por_region_estrategia = pred_df.groupby(["region", "modelo"]).size().to_dict()

    fechas_pred = pd.to_datetime(pred_df["fecha"], errors="coerce")
    ventana_prediccion_por_estrategia = {}
    for estrategia, grupo in pred_df.assign(_fecha=fechas_pred).groupby("modelo"):
        ventana_prediccion_por_estrategia[estrategia] = (grupo["_fecha"].min(), grupo["_fecha"].max())

    estrategias_sin_mapeo = sorted(
        (set(estrategias_metricas) | set(estrategias_series_prediccion)) - set(MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO)
    )
    if estrategias_sin_mapeo:
        problemas.append(
            f"Estrategias legacy SIN mapeo en MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO -- se incluiran "
            f"con modelo=None, no se inventa un canonico: {estrategias_sin_mapeo}"
        )

    return ValidacionLegacyUnivariado(
        ok=True, problemas=problemas,
        regiones_metricas=regiones_metricas, regiones_series=regiones_series,
        estrategias_metricas=estrategias_metricas,
        estrategias_series_prediccion=estrategias_series_prediccion,
        columnas_metricas=columnas_metricas, columnas_series=columnas_series,
        n_filas_metricas=len(metricas_df), n_filas_series=len(series_df),
        tipos_series=tipos_series, tiene_real=tiene_real, tiene_componente_pred=tiene_componente_pred,
        n_predicciones_por_region_estrategia=n_predicciones_por_region_estrategia,
        ventana_prediccion_por_estrategia=ventana_prediccion_por_estrategia,
        estrategias_sin_mapeo=estrategias_sin_mapeo,
    )


def imprimir_validacion_legacy(validacion: ValidacionLegacyUnivariado):
    """Imprime el reporte de `validar_legacy_univariado()` en un formato legible -- pensado para la celda de diagnostico read-only en Colab."""
    print("=" * 80)
    print("VALIDACION LEGACY_UNIVARIADOS (solo lectura)")
    print("=" * 80)

    if not validacion.ok:
        print("NO SE PUEDE INTEGRAR -- problemas bloqueantes:")
        for p in validacion.problemas:
            print(f"  - {p}")
        return

    print(f"metricas_global.csv: {validacion.n_filas_metricas:,} filas, columnas: {validacion.columnas_metricas}")
    print(f"series_global.csv:   {validacion.n_filas_series:,} filas, columnas: {validacion.columnas_series}")

    print(f"\nRegiones en metricas_global.csv ({len(validacion.regiones_metricas)}): {validacion.regiones_metricas}")
    print(f"Regiones en series_global.csv   ({len(validacion.regiones_series)}): {validacion.regiones_series}")

    print(f"\nEstrategias en metricas_global.csv ({len(validacion.estrategias_metricas)}): {validacion.estrategias_metricas}")
    print(f"Estrategias 'prediccion' en series_global.csv ({len(validacion.estrategias_series_prediccion)}): {validacion.estrategias_series_prediccion}")

    print(f"\nTipos de serie encontrados: {validacion.tipos_series}")
    print(f"  Existe 'real': {validacion.tiene_real}")
    print(f"  Existe 'componente_pred': {validacion.tiene_componente_pred}")

    print("\nN predicciones por (region, estrategia):")
    for (region, estrategia), n in sorted(validacion.n_predicciones_por_region_estrategia.items()):
        print(f"  {region:6s} | {estrategia:45s} | {n}")

    print("\nVentana de prediccion por estrategia (min/max de 'fecha' en filas tipo=prediccion):")
    for estrategia, (min_ts, max_ts) in sorted(validacion.ventana_prediccion_por_estrategia.items()):
        print(f"  {estrategia:45s} | {min_ts} .. {max_ts}")

    if validacion.estrategias_sin_mapeo:
        print(f"\nAVISO: estrategias SIN mapeo a modelo canonico: {validacion.estrategias_sin_mapeo}")

    if validacion.problemas:
        print("\nProblemas/avisos encontrados:")
        for p in validacion.problemas:
            print(f"  - {p}")
    else:
        print("\nSin problemas encontrados.")


@dataclass
class LegacyUnivariadoResultado:
    metricas_df: pd.DataFrame
    pred_df: pd.DataFrame
    real_df: pd.DataFrame
    entradas_reporte: list
    validacion: ValidacionLegacyUnivariado
    estrategias_sin_mapeo: list


def procesar_legacy_univariado(
    legacy_metricas_df: pd.DataFrame,
    legacy_series_df: pd.DataFrame,
    legacy_metricas_path: str,
    legacy_series_path: str,
    verbose: bool = True,
) -> LegacyUnivariadoResultado:
    """
    Transforma los 2 CSV legacy YA LEIDOS (nunca se modifican) al mismo
    esquema que usan metricas_master/series_master:

      - `familia_experimento` se fija DIRECTO a "univariado"
        (FAMILIA_UNIVARIADO) -- NUNCA pasa por clasificar_familia(), que es
        exclusiva de corridas modernas con config.json (modelo+exogenas).
      - `modelo` (canonico) sale de MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO; una
        estrategia SIN mapeo se incluye igual (nunca se pierden resultados
        reales por esto) con modelo=None y un AVISO explicito -- nunca se
        inventa un id canonico.
      - `run_name` sale de build_run_name_legacy_univariado(estrategia).
      - `forecast_horizon` se confirma con series_global.csv: si TODAS las
        regiones de una estrategia coinciden en el mismo numero de
        predicciones, ese numero se usa; si no coinciden (o no hay
        predicciones para esa estrategia), queda None -- nunca se asume
        168 sin confirmarlo.
      - `train_hours`/`optuna_n_trials`/`seed`/`git_commit`/`generated_at`
        quedan en None (sin evidencia en los CSV legacy para derivarlos).
      - `exogenas`/`exogena_individual`: vacio/None (univariado, sin
        exogenas por definicion).
      - `run_dir` apunta a la ruta legacy original (metricas o series,
        segun corresponda).

    Lanza `ValueError` si `validar_legacy_univariado()` encuentra columnas
    requeridas ausentes (no se integra a medias).
    """
    validacion = validar_legacy_univariado(legacy_metricas_df, legacy_series_df)

    if not validacion.ok:
        raise ValueError(
            "Los CSV legacy no tienen el esquema minimo requerido -- no se integra a medias. "
            f"Problemas: {validacion.problemas}"
        )

    if verbose:
        imprimir_validacion_legacy(validacion)

    # ---------------------------------------------------------------
    # forecast_horizon confirmado por estrategia (ver docstring)
    # ---------------------------------------------------------------
    conteos_por_estrategia = {}
    for (region, estrategia), n in validacion.n_predicciones_por_region_estrategia.items():
        conteos_por_estrategia.setdefault(estrategia, set()).add(n)

    horizonte_por_estrategia = {}
    for estrategia, valores in conteos_por_estrategia.items():
        if len(valores) == 1:
            horizonte_por_estrategia[estrategia] = next(iter(valores))
        elif verbose:
            print(
                f"AVISO: '{estrategia}' tiene distinto numero de predicciones segun la region "
                "-- forecast_horizon queda sin confirmar (None)."
            )

    estrategias_presentes = sorted(set(validacion.estrategias_metricas) | set(validacion.estrategias_series_prediccion))

    filas_metadata = []
    for estrategia in estrategias_presentes:
        filas_metadata.append({
            "modelo_estrategia": estrategia,
            "run_name": build_run_name_legacy_univariado(estrategia),
            "modelo": MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO.get(estrategia),
            "familia_experimento": FAMILIA_UNIVARIADO,
            "exogenas": "",
            "exogena_individual": None,
            "train_hours": None,
            "forecast_horizon": horizonte_por_estrategia.get(estrategia),
            "optuna_n_trials": None,
            "seed": None,
            "notas": "Importado de Legacy_Univariados -- NO es una corrida moderna (sin config.json, sin runner.py).",
            "git_commit": None,
            "generated_at": None,
            "origen": ORIGEN_LEGACY_UNIVARIADO,
            "metodologia": None,  # sin evidencia en el CSV legacy para derivarla -- no se inventa
        })
    metadata_df = pd.DataFrame(filas_metadata)

    # ---------------------------------------------------------------
    # metricas_global.csv -> filas listas para pd.concat con metricas_master
    # ---------------------------------------------------------------
    metricas_legacy = legacy_metricas_df.copy()
    metricas_legacy["region"] = _resolver_columna_region(metricas_legacy)
    metricas_legacy = _renombrar_modelo_original(metricas_legacy)
    metricas_legacy = _evitar_colision_metadata(metricas_legacy)
    metricas_legacy = metricas_legacy.merge(metadata_df, on="modelo_estrategia", how="left")
    metricas_legacy["run_dir"] = legacy_metricas_path

    # ---------------------------------------------------------------
    # series_global.csv -> prediccion (extra_pred_frames) + real (extra_real_frames)
    # componente_pred (y cualquier otro tipo que no sea real/prediccion) se
    # excluye aca, igual que en construir_series_master() para corridas
    # modernas.
    # ---------------------------------------------------------------
    series_legacy = legacy_series_df.copy()
    series_legacy["region"] = _resolver_columna_region(series_legacy)
    series_legacy["fecha"] = pd.to_datetime(series_legacy["fecha"], errors="coerce")
    series_legacy = _renombrar_modelo_original(series_legacy)
    series_legacy = series_legacy.rename(columns={"fecha": "timestamp", "tipo": "serie_tipo"})

    pred_legacy = series_legacy[series_legacy["serie_tipo"] == "prediccion"].copy()
    real_legacy = series_legacy[series_legacy["serie_tipo"] == "real"].copy()

    pred_legacy = _evitar_colision_metadata(pred_legacy)
    pred_legacy = pred_legacy.merge(metadata_df, on="modelo_estrategia", how="left")
    pred_legacy["run_dir"] = legacy_series_path

    real_legacy = real_legacy[["region", "timestamp", "valor", "modelo_estrategia"]].copy()

    # ---------------------------------------------------------------
    # reporte_consolidacion: una fila por estrategia legacy
    # ---------------------------------------------------------------
    entradas_reporte = []
    for _, fila in metadata_df.iterrows():
        razon = (
            "" if fila["modelo"] is not None
            else f"estrategia legacy sin mapeo en MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO -- modelo=None"
        )
        entradas_reporte.append({
            "run_name": fila["run_name"],
            "familia": FAMILIA_UNIVARIADO,
            "estado": "completo",
            "incluido": True,
            "razon": razon,
        })

    return LegacyUnivariadoResultado(
        metricas_df=metricas_legacy,
        pred_df=pred_legacy,
        real_df=real_legacy,
        entradas_reporte=entradas_reporte,
        validacion=validacion,
        estrategias_sin_mapeo=validacion.estrategias_sin_mapeo,
    )


# =========================================================
# BCA RECONSTRUIDO (Legacy_Univariados_BCA_Reconstruido/
# metricas_bca_reconstruido.csv + series_bca_reconstruido.csv) --
# arquitectura ACTUAL sin exogenas, NO legacy historico
# =========================================================
#
# A diferencia de Legacy_Univariados/, estos 2 CSV YA TRAEN su propio
# origen/metodologia/train_hours/forecast_horizon por fila (ver
# legacy_bca_reconstruido.py::run_bca_reconstruido()) -- no hay que
# inferirlos ni dejarlos en None. Por eso esta seccion NO reutiliza
# procesar_legacy_univariado() tal cual (esa funcion siempre fuerza
# train_hours=None, lo cual seria descartar informacion que BCA si tiene) --
# reutiliza los MISMOS helpers de bajo nivel (_resolver_columna_region,
# _renombrar_modelo_original, _evitar_colision_metadata,
# MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO) con su propia orquestacion.
#
# IMPORTANTE: usa un prefijo de RUN_NAME DISTINTO
# (build_run_name_bca_reconstruido, "BCA_Reconstruido__") -- las 8
# estrategias tienen el MISMO nombre que las de Legacy_Univariados
# (ARIMA_1_1_1, AR_AIC, etc.); si se reutilizara
# build_run_name_legacy_univariado() aca, "ARIMA_1_1_1" de BCA y
# "ARIMA_1_1_1" del legacy de 7 regiones producirian el MISMO run_name,
# borrando la distincion legacy-real vs arquitectura-actual-reconstruida
# que este modulo existe para preservar.

# Este modulo NO importa legacy_bca_reconstruido.py (aggregator.py se
# mantiene independiente de models/ a proposito, ver docstring del modulo)
# -- por eso NO se asume/hardcodea el valor esperado de origen/metodologia,
# se LEE lo que el propio CSV trae y se reporta si no es consistente.


def build_run_name_bca_reconstruido(estrategia: str) -> str:
    """
    RUN_NAME deterministico para una estrategia BCA reconstruida --
    mismo criterio que build_run_name_legacy_univariado() (estable,
    estructuralmente sin colision con RUN_NAMEs modernos), pero con un
    prefijo DISTINTO para no colisionar con las mismas 8 estrategias de
    Legacy_Univariados/ (ver nota de la seccion).
    """
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(estrategia).strip())
    return f"BCA_Reconstruido__{slug}"


@dataclass
class ValidacionBCAReconstruido:
    """Resultado de `validar_bca_reconstruido()` -- solo lectura, nunca modifica los CSV de BCA."""
    ok: bool
    problemas: list
    regiones_metricas: list
    regiones_series: list
    estrategias_metricas: list
    estrategias_series_prediccion: list
    columnas_metricas: list
    columnas_series: list
    n_filas_metricas: int
    n_filas_series: int
    tipos_series: list
    tiene_real: bool
    tiene_componente_pred: bool
    n_predicciones_por_region_estrategia: dict
    ventana_prediccion_por_estrategia: dict
    estrategias_sin_mapeo: list
    origenes_encontrados: list
    metodologias_encontradas: list
    train_hours_encontrados: list
    forecast_horizons_encontrados: list


def validar_bca_reconstruido(metricas_df: pd.DataFrame, series_df: pd.DataFrame) -> ValidacionBCAReconstruido:
    """
    Validacion PREVIA, de solo lectura, de `metricas_bca_reconstruido.csv`/
    `series_bca_reconstruido.csv` YA LEIDOS en memoria. A diferencia de
    `validar_legacy_univariado()`, ademas confirma que `origen`/
    `metodologia`/`train_hours`/`forecast_horizon` (que este CSV SI trae)
    tengan un unico valor consistente -- si hay mas de uno, es un problema
    real a reportar, no algo para promediar/elegir en silencio.
    """
    problemas = []

    columnas_metricas = list(metricas_df.columns)
    columnas_series = list(series_df.columns)

    for requerida in ["modelo", "MAE", "RMSE", "MAPE", "sMAPE"]:
        if requerida not in metricas_df.columns:
            problemas.append(f"metricas_bca_reconstruido.csv: falta columna requerida '{requerida}'")

    for requerida in ["modelo", "fecha", "tipo", "valor"]:
        if requerida not in series_df.columns:
            problemas.append(f"series_bca_reconstruido.csv: falta columna requerida '{requerida}'")

    if "region" not in metricas_df.columns and "serie" not in metricas_df.columns:
        problemas.append("metricas_bca_reconstruido.csv: no tiene 'region' ni 'serie' -- no se puede derivar la region")
    if "region" not in series_df.columns and "serie" not in series_df.columns:
        problemas.append("series_bca_reconstruido.csv: no tiene 'region' ni 'serie' -- no se puede derivar la region")

    for requerida in ["origen", "metodologia", "train_hours", "forecast_horizon"]:
        if requerida not in metricas_df.columns:
            problemas.append(
                f"metricas_bca_reconstruido.csv: falta columna '{requerida}' -- se esperaba que "
                "legacy_bca_reconstruido.py la escribiera (ver ORIGEN_BCA/METODOLOGIA_BCA)"
            )

    if problemas:
        return ValidacionBCAReconstruido(
            ok=False, problemas=problemas,
            regiones_metricas=[], regiones_series=[],
            estrategias_metricas=[], estrategias_series_prediccion=[],
            columnas_metricas=columnas_metricas, columnas_series=columnas_series,
            n_filas_metricas=len(metricas_df), n_filas_series=len(series_df),
            tipos_series=[], tiene_real=False, tiene_componente_pred=False,
            n_predicciones_por_region_estrategia={}, ventana_prediccion_por_estrategia={},
            estrategias_sin_mapeo=[], origenes_encontrados=[], metodologias_encontradas=[],
            train_hours_encontrados=[], forecast_horizons_encontrados=[],
        )

    region_metricas = _resolver_columna_region(metricas_df)
    region_series = _resolver_columna_region(series_df)

    regiones_metricas = sorted(region_metricas.dropna().unique().tolist())
    regiones_series = sorted(region_series.dropna().unique().tolist())

    estrategias_metricas = sorted(metricas_df["modelo"].dropna().unique().tolist())

    for col in ["MAE", "RMSE", "MAPE", "sMAPE"]:
        n_nan = int(metricas_df[col].isna().sum())
        if n_nan:
            problemas.append(f"metricas_bca_reconstruido.csv: columna '{col}' tiene {n_nan} valor(es) NaN")

    origenes_encontrados = sorted(metricas_df["origen"].dropna().unique().tolist())
    metodologias_encontradas = sorted(metricas_df["metodologia"].dropna().unique().tolist())
    train_hours_encontrados = sorted(metricas_df["train_hours"].dropna().unique().tolist())
    forecast_horizons_encontrados = sorted(metricas_df["forecast_horizon"].dropna().unique().tolist())

    # Estos 4 campos, a diferencia de NaN sueltos en metricas o estrategias sin
    # mapeo (avisos, no bloquean), SI bloquean `ok`: procesar_bca_reconstruido()
    # toma un UNICO valor de cada uno (origenes_encontrados[0], etc.) y lo
    # aplica a TODAS las filas via metadata_df -- si no hay exactamente un
    # valor, seguir adelante corrompería el origen/metodologia/train_hours/
    # forecast_horizon de TODO el resultado (o reventaria con IndexError si
    # la lista queda vacia), asi que se marca como bloqueante en vez de aviso.
    problemas_campos_inconsistentes = []
    for nombre, valores in [
        ("origen", origenes_encontrados), ("metodologia", metodologias_encontradas),
        ("train_hours", train_hours_encontrados), ("forecast_horizon", forecast_horizons_encontrados),
    ]:
        if len(valores) > 1:
            msg = f"metricas_bca_reconstruido.csv: columna '{nombre}' tiene MAS de un valor: {valores} (se esperaba uno solo)"
            problemas.append(msg)
            problemas_campos_inconsistentes.append(msg)
        elif len(valores) == 0:
            msg = f"metricas_bca_reconstruido.csv: columna '{nombre}' esta vacia"
            problemas.append(msg)
            problemas_campos_inconsistentes.append(msg)

    tipos_series = sorted(series_df["tipo"].dropna().unique().tolist())
    tiene_real = "real" in tipos_series
    tiene_componente_pred = "componente_pred" in tipos_series

    pred_df = series_df[series_df["tipo"] == "prediccion"].copy()
    pred_df["region"] = _resolver_columna_region(pred_df)
    estrategias_series_prediccion = sorted(pred_df["modelo"].dropna().unique().tolist())

    n_predicciones_por_region_estrategia = pred_df.groupby(["region", "modelo"]).size().to_dict()

    fechas_pred = pd.to_datetime(pred_df["fecha"], errors="coerce")
    ventana_prediccion_por_estrategia = {}
    for estrategia, grupo in pred_df.assign(_fecha=fechas_pred).groupby("modelo"):
        ventana_prediccion_por_estrategia[estrategia] = (grupo["_fecha"].min(), grupo["_fecha"].max())

    estrategias_sin_mapeo = sorted(
        (set(estrategias_metricas) | set(estrategias_series_prediccion)) - set(MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO)
    )
    if estrategias_sin_mapeo:
        problemas.append(
            f"Estrategias BCA SIN mapeo en MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO -- se incluiran "
            f"con modelo=None, no se inventa un canonico: {estrategias_sin_mapeo}"
        )

    # Cruce forecast_horizon declarado (columna) vs confirmado por series --
    # mismo espiritu que procesar_legacy_univariado(), pero aca es una
    # comparacion (ambas fuentes existen), no una derivacion.
    if len(forecast_horizons_encontrados) == 1:
        fh_declarado = forecast_horizons_encontrados[0]
        for (region, estrategia), n in n_predicciones_por_region_estrategia.items():
            if n != fh_declarado:
                problemas.append(
                    f"'{estrategia}' en {region}: forecast_horizon declarado en metricas ({fh_declarado}) "
                    f"no coincide con el numero de predicciones en series ({n})"
                )

    return ValidacionBCAReconstruido(
        ok=(len(problemas_campos_inconsistentes) == 0), problemas=problemas,
        regiones_metricas=regiones_metricas, regiones_series=regiones_series,
        estrategias_metricas=estrategias_metricas,
        estrategias_series_prediccion=estrategias_series_prediccion,
        columnas_metricas=columnas_metricas, columnas_series=columnas_series,
        n_filas_metricas=len(metricas_df), n_filas_series=len(series_df),
        tipos_series=tipos_series, tiene_real=tiene_real, tiene_componente_pred=tiene_componente_pred,
        n_predicciones_por_region_estrategia=n_predicciones_por_region_estrategia,
        ventana_prediccion_por_estrategia=ventana_prediccion_por_estrategia,
        estrategias_sin_mapeo=estrategias_sin_mapeo,
        origenes_encontrados=origenes_encontrados, metodologias_encontradas=metodologias_encontradas,
        train_hours_encontrados=train_hours_encontrados, forecast_horizons_encontrados=forecast_horizons_encontrados,
    )


def imprimir_validacion_bca(validacion: ValidacionBCAReconstruido):
    """Imprime el reporte de `validar_bca_reconstruido()` -- pensado para la celda de diagnostico read-only en Colab."""
    print("=" * 80)
    print("VALIDACION BCA RECONSTRUIDO (solo lectura)")
    print("=" * 80)

    if not validacion.ok:
        print("NO SE PUEDE INTEGRAR -- problemas bloqueantes:")
        for p in validacion.problemas:
            print(f"  - {p}")
        return

    print(f"metricas_bca_reconstruido.csv: {validacion.n_filas_metricas:,} filas, columnas: {validacion.columnas_metricas}")
    print(f"series_bca_reconstruido.csv:   {validacion.n_filas_series:,} filas, columnas: {validacion.columnas_series}")
    print(f"\norigen: {validacion.origenes_encontrados}")
    print(f"metodologia: {validacion.metodologias_encontradas}")
    print(f"train_hours: {validacion.train_hours_encontrados}")
    print(f"forecast_horizon: {validacion.forecast_horizons_encontrados}")
    print(f"\nRegiones ({len(validacion.regiones_metricas)}): {validacion.regiones_metricas}")
    print(f"Estrategias ({len(validacion.estrategias_metricas)}): {validacion.estrategias_metricas}")
    print(f"\nTipos de serie: {validacion.tipos_series} (real={validacion.tiene_real}, componente_pred={validacion.tiene_componente_pred})")

    print("\nN predicciones por (region, estrategia):")
    for (region, estrategia), n in sorted(validacion.n_predicciones_por_region_estrategia.items()):
        print(f"  {region:6s} | {estrategia:45s} | {n}")

    if validacion.estrategias_sin_mapeo:
        print(f"\nAVISO: estrategias SIN mapeo a modelo canonico: {validacion.estrategias_sin_mapeo}")

    if validacion.problemas:
        print("\nProblemas/avisos encontrados:")
        for p in validacion.problemas:
            print(f"  - {p}")
    else:
        print("\nSin problemas encontrados.")


@dataclass
class BCAReconstruidoResultado:
    metricas_df: pd.DataFrame
    pred_df: pd.DataFrame
    real_df: pd.DataFrame
    entradas_reporte: list
    validacion: ValidacionBCAReconstruido
    estrategias_sin_mapeo: list


def procesar_bca_reconstruido(
    bca_metricas_df: pd.DataFrame,
    bca_series_df: pd.DataFrame,
    bca_metricas_path: str,
    bca_series_path: str,
    verbose: bool = True,
) -> BCAReconstruidoResultado:
    """
    Transforma los 2 CSV de BCA reconstruido (NUNCA se modifican) al mismo
    esquema que usan metricas_master/series_master:

      - `familia_experimento` = FAMILIA_UNIVARIADO ("univariado") -- misma
        familia metodologica que Legacy_Univariados (son estructuras
        univariadas), la distincion legacy-real vs reconstruido queda en
        `origen`/`metodologia`, NUNCA en `familia_experimento`.
      - `origen`/`metodologia`/`train_hours`/`forecast_horizon` se LEEN del
        propio CSV (confirmados unicos por `validar_bca_reconstruido()`),
        nunca se ponen en None como en el legacy de 7 regiones -- BCA SI
        trae esta informacion.
      - `modelo` (canonico) sale de MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO
        (reutilizado tal cual -- son las MISMAS 8 estrategias que
        Legacy_Univariados, el mapeo estrategia->modelo-canonico es
        legitimo compartirlo; lo que NO se comparte es el run_name ni el
        origen).
      - `run_name` sale de build_run_name_bca_reconstruido(estrategia) --
        prefijo DISTINTO al legacy de 7 regiones, ver docstring de la
        seccion.
      - `notas` deja explicito que es una reconstruccion con arquitectura
        actual, NO codigo legacy recuperado.

    Lanza `ValueError` si `validar_bca_reconstruido()` encuentra columnas
    requeridas ausentes o valores de origen/metodologia/train_hours/
    forecast_horizon inconsistentes (no se integra a medias).
    """
    validacion = validar_bca_reconstruido(bca_metricas_df, bca_series_df)

    if not validacion.ok:
        raise ValueError(
            "Los CSV de BCA reconstruido no tienen el esquema minimo requerido -- no se integra "
            f"a medias. Problemas: {validacion.problemas}"
        )

    if verbose:
        imprimir_validacion_bca(validacion)

    origen_bca = validacion.origenes_encontrados[0]
    metodologia_bca = validacion.metodologias_encontradas[0]
    train_hours_bca = validacion.train_hours_encontrados[0]
    forecast_horizon_bca = validacion.forecast_horizons_encontrados[0]

    estrategias_presentes = sorted(set(validacion.estrategias_metricas) | set(validacion.estrategias_series_prediccion))

    filas_metadata = []
    for estrategia in estrategias_presentes:
        filas_metadata.append({
            "modelo_estrategia": estrategia,
            "run_name": build_run_name_bca_reconstruido(estrategia),
            "modelo": MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO.get(estrategia),
            "familia_experimento": FAMILIA_UNIVARIADO,
            "exogenas": "",
            "exogena_individual": None,
            "train_hours": train_hours_bca,
            "forecast_horizon": forecast_horizon_bca,
            "optuna_n_trials": None,
            "seed": None,
            "notas": (
                f"Reconstruido con {metodologia_bca} (BCA) -- NO es codigo legacy historico "
                "recuperado, ver legacy_bca_reconstruido.py."
            ),
            "git_commit": None,
            "generated_at": None,
            "origen": origen_bca,
            "metodologia": metodologia_bca,
        })
    metadata_df = pd.DataFrame(filas_metadata)

    # ---------------------------------------------------------------
    # metricas_bca_reconstruido.csv -> filas listas para pd.concat con metricas_master
    # ---------------------------------------------------------------
    metricas_bca = bca_metricas_df.copy()
    metricas_bca["region"] = _resolver_columna_region(metricas_bca)
    metricas_bca = _renombrar_modelo_original(metricas_bca)
    metricas_bca = _evitar_colision_metadata(metricas_bca)  # origen/metodologia/train_hours/forecast_horizon propios -> "_original"
    metricas_bca = metricas_bca.merge(metadata_df, on="modelo_estrategia", how="left")
    metricas_bca["run_dir"] = bca_metricas_path

    # ---------------------------------------------------------------
    # series_bca_reconstruido.csv -> prediccion (extra_pred_frames) + real (extra_real_frames)
    # ---------------------------------------------------------------
    series_bca = bca_series_df.copy()
    series_bca["region"] = _resolver_columna_region(series_bca)
    series_bca["fecha"] = pd.to_datetime(series_bca["fecha"], errors="coerce")
    series_bca = _renombrar_modelo_original(series_bca)
    series_bca = series_bca.rename(columns={"fecha": "timestamp", "tipo": "serie_tipo"})

    pred_bca = series_bca[series_bca["serie_tipo"] == "prediccion"].copy()
    real_bca = series_bca[series_bca["serie_tipo"] == "real"].copy()

    pred_bca = _evitar_colision_metadata(pred_bca)
    pred_bca = pred_bca.merge(metadata_df, on="modelo_estrategia", how="left")
    pred_bca["run_dir"] = bca_series_path

    real_bca = real_bca[["region", "timestamp", "valor", "modelo_estrategia"]].copy()

    # ---------------------------------------------------------------
    # reporte_consolidacion: una fila por estrategia BCA
    # ---------------------------------------------------------------
    entradas_reporte = []
    for _, fila in metadata_df.iterrows():
        razon = (
            "" if fila["modelo"] is not None
            else "estrategia BCA sin mapeo en MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO -- modelo=None"
        )
        entradas_reporte.append({
            "run_name": fila["run_name"],
            "familia": FAMILIA_UNIVARIADO,
            "estado": "completo",
            "incluido": True,
            "razon": razon,
        })

    return BCAReconstruidoResultado(
        metricas_df=metricas_bca,
        pred_df=pred_bca,
        real_df=real_bca,
        entradas_reporte=entradas_reporte,
        validacion=validacion,
        estrategias_sin_mapeo=validacion.estrategias_sin_mapeo,
    )


# =========================================================
# BANDERAS DE CALIDAD (calidad_resultado / valido_ranking)
# =========================================================
#
# Tabla explicita y documentada (mismo patron que
# MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO) de resultados con un problema de
# CALIDAD NUMERICA ya diagnosticado -- NO de completitud (eso ya lo maneja
# descubrir_runs_completos()/validar_resultado()). Se conserva la fila y
# sus series intactas; esto solo agrega metadata para poder excluir el
# resultado de un RANKING sin borrar nada.
#
# Cada entrada esta keyed por `run_name` (identificador unico e inequivoco
# -- ver build_run_name_bca_reconstruido()/build_run_name_legacy_univariado()/
# build_run_name(), estructuralmente sin colision entre origenes). Agregar
# un caso nuevo es agregar una entrada aca, no escribir logica nueva.
BANDERAS_CALIDAD_CONOCIDAS = {
    "BCA_Reconstruido__STL_FCNN_residuos": {
        "calidad_resultado": "divergencia_numerica",
        "valido_ranking": False,
        "razon_calidad": (
            "Forecast recursivo diverge (MAE~2.8e18, prediccion recursiva realimentada con "
            "train_hours=384/WINDOW=168 deja solo ~48 filas de entrenamiento para el FCNN del "
            "residuo STL -- ver auditoria especifica). Resultado real, NO recalculado ni alterado; "
            "se conserva para poder estudiar el fallo, marcado como no valido para ranking."
        ),
    },
}

CALIDAD_DEFAULT = "ok"
VALIDO_RANKING_DEFAULT = True


def aplicar_banderas_calidad(df: pd.DataFrame, columna_run_name: str = "run_name") -> pd.DataFrame:
    """
    Agrega `calidad_resultado`/`valido_ranking`/`razon_calidad` a `df` (
    `metricas_master` o `series_master`) por `run_name`, usando
    `BANDERAS_CALIDAD_CONOCIDAS`. Filas cuyo `run_name` no esta en la tabla
    quedan con el default ("ok"/`True`/None) -- puramente aditivo, nunca
    quita ni modifica ninguna fila existente.
    """
    df = df.copy()
    if columna_run_name not in df.columns:
        df["calidad_resultado"] = CALIDAD_DEFAULT
        df["valido_ranking"] = VALIDO_RANKING_DEFAULT
        df["razon_calidad"] = None
        return df

    banderas_df = pd.DataFrame([
        {"__run_name__": run_name, **flags}
        for run_name, flags in BANDERAS_CALIDAD_CONOCIDAS.items()
    ]) if BANDERAS_CALIDAD_CONOCIDAS else pd.DataFrame(columns=["__run_name__", "calidad_resultado", "valido_ranking", "razon_calidad"])

    df = df.merge(banderas_df, left_on=columna_run_name, right_on="__run_name__", how="left")
    df = df.drop(columns=["__run_name__"])

    df["calidad_resultado"] = df["calidad_resultado"].fillna(CALIDAD_DEFAULT)
    # El merge left con banderas_df deja 'valido_ranking' en dtype object
    # (mezcla True/False/NaN antes del fillna) -- sin este cast explicito a
    # bool, quedaria en dtype object con bool de Python puros, y "~serie"
    # sobre eso hace inversion BITWISE (~True == -2), no negacion logica,
    # arruinando cualquier conteo tipo (~df["valido_ranking"]).sum().
    df["valido_ranking"] = df["valido_ranking"].where(df["valido_ranking"].notna(), VALIDO_RANKING_DEFAULT).astype(bool)

    return df


# =========================================================
# API DE ALTO NIVEL
# =========================================================

def consolidar_resultados(
    pipeline_resultados_dir: str,
    output_dir: Optional[str] = None,
    regiones_esperadas: Optional[list] = None,
    escribir_csv: bool = True,
    escribir_reporte: bool = True,
    legacy_metricas_path: Optional[str] = None,
    legacy_series_path: Optional[str] = None,
    bca_metricas_path: Optional[str] = None,
    bca_series_path: Optional[str] = None,
):
    """
    Punto de entrada unico: descubre las corridas MODERNAS completas, arma
    `metricas_master.csv`/`series_master.csv`/`reporte_consolidacion.csv`,
    y (si `escribir_csv=True`, default) los escribe en `output_dir` --
    por defecto, `pipeline_resultados_dir/Consolidado/` (subcarpeta
    dedicada, nunca mezclada con las carpetas de corridas: el propio
    `descubrir_runs_completos()` la excluye del descubrimiento). Solo
    lectura sobre las carpetas de corridas existentes; nunca las modifica
    ni las borra.

    `legacy_metricas_path`/`legacy_series_path` (opcionales, default
    `None` -> comportamiento IDENTICO al de antes): si se pasan AMBOS, se
    integra `Legacy_Univariados/` como `origen="legacy_univariado"` (ver
    `procesar_legacy_univariado()`). Solo uno de los dos -> `ValueError`
    inmediato, nunca se integra a medias.

    `bca_metricas_path`/`bca_series_path` (opcionales, default `None`):
    mismo patron, integra `Legacy_Univariados_BCA_Reconstruido/` como
    `origen="bca_univariado_reconstruido"` (ver `procesar_bca_reconstruido()`)
    -- run_name con prefijo DISTINTO al legacy, nunca se presenta como
    legacy historico.

    REEMPLAZOS `_timefix`: ANTES de construir los masters, se identifican
    (por sufijo `_timefix`, ver `identificar_reemplazos_timefix()`) las
    corridas modernas viejas que tienen una version `_timefix` equivalente
    -- esas viejas se EXCLUYEN del consolidado (nunca se borran de disco;
    quedan documentadas en `reporte_consolidacion.csv` con
    `estado="reemplazado"`). Las filas de `metricas_master` llevan
    `es_timefix` (bool) y `reemplaza_a` (run_name viejo, solo en filas
    `_timefix`) para trazabilidad.

    BANDERAS DE CALIDAD: `metricas_master`/`series_master` llevan
    `calidad_resultado`/`valido_ranking`/`razon_calidad` (ver
    `BANDERAS_CALIDAD_CONOCIDAS`) -- default `"ok"`/`True`/`None`; ningun
    resultado se excluye ni se recalcula por esto, solo se marca.

    Idempotente: cada llamada reconstruye todo desde `Pipeline_Resultados/`
    (y, si se pidio, desde los CSV legacy/BCA) -- nunca lee su propia
    salida anterior, asi que correrlo varias veces seguidas sobre los
    mismos resultados produce el mismo contenido, sin acumular ni duplicar
    nada.

    Devuelve `(metricas_df, series_df, descubrimiento)`.
    """
    if bool(legacy_metricas_path) != bool(legacy_series_path):
        raise ValueError(
            "Se paso solo uno de legacy_metricas_path/legacy_series_path -- se requieren "
            "AMBOS para integrar Legacy_Univariados (o ninguno de los dos para omitirlo). "
            f"legacy_metricas_path={legacy_metricas_path!r}, legacy_series_path={legacy_series_path!r}"
        )
    if bool(bca_metricas_path) != bool(bca_series_path):
        raise ValueError(
            "Se paso solo uno de bca_metricas_path/bca_series_path -- se requieren "
            "AMBOS para integrar BCA reconstruido (o ninguno de los dos para omitirlo). "
            f"bca_metricas_path={bca_metricas_path!r}, bca_series_path={bca_series_path!r}"
        )

    integrar_legacy = bool(legacy_metricas_path and legacy_series_path)
    integrar_bca = bool(bca_metricas_path and bca_series_path)

    if integrar_legacy:
        if not os.path.exists(legacy_metricas_path):
            raise FileNotFoundError(f"legacy_metricas_path no existe: {legacy_metricas_path}")
        if not os.path.exists(legacy_series_path):
            raise FileNotFoundError(f"legacy_series_path no existe: {legacy_series_path}")

    if integrar_bca:
        if not os.path.exists(bca_metricas_path):
            raise FileNotFoundError(f"bca_metricas_path no existe: {bca_metricas_path}")
        if not os.path.exists(bca_series_path):
            raise FileNotFoundError(f"bca_series_path no existe: {bca_series_path}")

    output_dir = output_dir or os.path.join(pipeline_resultados_dir, NOMBRE_CARPETA_CONSOLIDADO)
    regiones_esperadas = list(regiones_esperadas) if regiones_esperadas is not None else list(REGIONS_ALL)

    descubrimiento_bruto = descubrir_runs_completos(pipeline_resultados_dir, regiones_esperadas=regiones_esperadas)

    # Reemplazos _timefix -- se calculan sobre el descubrimiento CRUDO
    # (antes de excluir nada) para poder mapear timefix -> viejo mas abajo.
    reemplazos_timefix = identificar_reemplazos_timefix(descubrimiento_bruto.runs)
    descubrimiento = aplicar_reemplazos_timefix(descubrimiento_bruto)

    print("=" * 80)
    print("CONSOLIDACION DE RESULTADOS")
    print("=" * 80)
    print(f"Carpeta: {pipeline_resultados_dir}")
    print(f"Salida:  {output_dir}")
    print(f"Corridas completas incluidas: {len(descubrimiento.runs)}")
    print(f"Corridas incompletas excluidas: {descubrimiento.n_incompletos}")
    print(f"Carpetas sin config.json excluidas: {descubrimiento.n_sin_config}")
    print(f"Corridas viejas reemplazadas por version _timefix (excluidas): {descubrimiento.n_reemplazados}")
    if descubrimiento.n_familia_unknown:
        print(f"AVISO: {descubrimiento.n_familia_unknown} corrida(s) con familia 'unknown' (incluidas igual, revisar modelo/exogenas)")

    if descubrimiento.incompletos:
        print("\nExcluidas por incompletas (motivo):")
        for run_name, problemas in descubrimiento.incompletos:
            print(f"  - {run_name}: {'; '.join(problemas)}")

    if reemplazos_timefix:
        print("\nReemplazos _timefix aplicados (viejo -> nuevo, viejo excluido del consolidado):")
        for viejo, nuevo in sorted(reemplazos_timefix.items()):
            print(f"  {viejo} -> {nuevo}")

    metricas_df = construir_metricas_master(descubrimiento.runs)

    legacy_resultado = None
    bca_resultado = None
    extra_pred_frames = []
    extra_real_frames = []

    if integrar_legacy:
        print("\n" + "=" * 80)
        print("INTEGRANDO LEGACY_UNIVARIADOS")
        print("=" * 80)
        print(f"Metricas: {legacy_metricas_path}")
        print(f"Series:   {legacy_series_path}")

        legacy_metricas_df = pd.read_csv(legacy_metricas_path, encoding="utf-8-sig")
        legacy_series_df = pd.read_csv(legacy_series_path, encoding="utf-8-sig")

        legacy_resultado = procesar_legacy_univariado(
            legacy_metricas_df, legacy_series_df, legacy_metricas_path, legacy_series_path, verbose=True,
        )

        metricas_df = pd.concat([metricas_df, legacy_resultado.metricas_df], ignore_index=True, sort=False)
        extra_pred_frames.append(legacy_resultado.pred_df)
        extra_real_frames.append(legacy_resultado.real_df)

    if integrar_bca:
        print("\n" + "=" * 80)
        print("INTEGRANDO BCA RECONSTRUIDO")
        print("=" * 80)
        print(f"Metricas: {bca_metricas_path}")
        print(f"Series:   {bca_series_path}")

        bca_metricas_df = pd.read_csv(bca_metricas_path, encoding="utf-8-sig")
        bca_series_df = pd.read_csv(bca_series_path, encoding="utf-8-sig")

        bca_resultado = procesar_bca_reconstruido(
            bca_metricas_df, bca_series_df, bca_metricas_path, bca_series_path, verbose=True,
        )

        metricas_df = pd.concat([metricas_df, bca_resultado.metricas_df], ignore_index=True, sort=False)
        extra_pred_frames.append(bca_resultado.pred_df)
        extra_real_frames.append(bca_resultado.real_df)

    # UNA sola construccion de series_master (moderno, mas legacy/BCA si se
    # pasaron) -- evita releer series.csv de cada corrida dos veces.
    series_df = construir_series_master(
        descubrimiento.runs, verbose=True,
        extra_pred_frames=extra_pred_frames or None, extra_real_frames=extra_real_frames or None,
    )

    ventanas_por_run = calcular_ventanas_agregadas_por_run(series_df)
    metricas_df = agregar_ventanas_temporales(metricas_df, series_df)

    # Trazabilidad _timefix a nivel de fila (ver docstring)
    mapa_reemplaza_a = {timefix: viejo for viejo, timefix in reemplazos_timefix.items()}
    for df in (metricas_df, series_df):
        if "run_name" in df.columns and len(df) > 0:
            df["es_timefix"] = df["run_name"].astype("string").str.endswith(SUFIJO_TIMEFIX).fillna(False)
            df["reemplaza_a"] = df["run_name"].map(mapa_reemplaza_a)
        else:
            df["es_timefix"] = False
            df["reemplaza_a"] = None

    # Banderas de calidad (ver BANDERAS_CALIDAD_CONOCIDAS) -- puramente
    # aditivo, nunca excluye ni recalcula nada.
    metricas_df = aplicar_banderas_calidad(metricas_df)
    series_df = aplicar_banderas_calidad(series_df)

    reporte_df = construir_reporte_consolidacion(
        descubrimiento,
        ventanas_por_run=ventanas_por_run,
        entradas_legacy=legacy_resultado.entradas_reporte if legacy_resultado else None,
        entradas_bca=bca_resultado.entradas_reporte if bca_resultado else None,
    )

    if descubrimiento.runs or legacy_resultado or bca_resultado:
        por_familia = pd.Series(
            [r.familia_experimento for r in descubrimiento.runs]
            + ([FAMILIA_UNIVARIADO] * len(legacy_resultado.entradas_reporte) if legacy_resultado else [])
            + ([FAMILIA_UNIVARIADO] * len(bca_resultado.entradas_reporte) if bca_resultado else [])
        ).value_counts()
        print("\nCorridas completas por familia (moderno + legacy + BCA):")
        for familia, n in por_familia.items():
            print(f"  {familia}: {n}")

        por_origen = metricas_df.drop_duplicates("run_name")["origen"].value_counts(dropna=False)
        print("\nCorridas completas por origen:")
        for origen, n in por_origen.items():
            print(f"  {origen}: {n}")

        por_modelo = pd.Series(
            [r.modelo for r in descubrimiento.runs]
            + (legacy_resultado.metricas_df.drop_duplicates("run_name")["modelo"].tolist() if legacy_resultado else [])
            + (bca_resultado.metricas_df.drop_duplicates("run_name")["modelo"].tolist() if bca_resultado else [])
        ).value_counts(dropna=False)
        print("\nCorridas completas por modelo (moderno + legacy + BCA):")
        for modelo, n in por_modelo.items():
            print(f"  {modelo}: {n}")

    if legacy_resultado:
        print(f"\nRuns modernos incluidos: {len(descubrimiento.runs)}")
        print(f"Runs legacy incluidos:   {len(legacy_resultado.entradas_reporte)}")
        if legacy_resultado.estrategias_sin_mapeo:
            print(f"AVISO: {len(legacy_resultado.estrategias_sin_mapeo)} estrategia(s) legacy sin mapeo canonico: {legacy_resultado.estrategias_sin_mapeo}")

    if bca_resultado:
        print(f"Runs BCA reconstruido incluidos: {len(bca_resultado.entradas_reporte)}")
        if bca_resultado.estrategias_sin_mapeo:
            print(f"AVISO: {len(bca_resultado.estrategias_sin_mapeo)} estrategia(s) BCA sin mapeo canonico: {bca_resultado.estrategias_sin_mapeo}")

    n_no_validas_ranking = int((~metricas_df["valido_ranking"]).sum()) if "valido_ranking" in metricas_df.columns and len(metricas_df) else 0
    if n_no_validas_ranking:
        print(f"\nAVISO: {n_no_validas_ranking} fila(s) de metricas_master con valido_ranking=False (ver calidad_resultado/razon_calidad).")

    print(f"\nmetricas_master: {len(metricas_df):,} filas")
    print(f"series_master:   {len(series_df):,} filas "
          f"({(series_df['serie_tipo'] == 'real').sum():,} real, {(series_df['serie_tipo'] == 'prediccion').sum():,} prediccion)"
          if len(series_df) else "")

    if escribir_csv:
        os.makedirs(output_dir, exist_ok=True)
        metricas_path = os.path.join(output_dir, "metricas_master.csv")
        series_path = os.path.join(output_dir, "series_master.csv")

        metricas_df.to_csv(metricas_path, index=False, encoding="utf-8-sig")
        series_df.to_csv(series_path, index=False, encoding="utf-8-sig")

        print(f"\nEscrito: {metricas_path}")
        print(f"Escrito: {series_path}")

        if escribir_reporte:
            reporte_path = os.path.join(output_dir, "reporte_consolidacion.csv")
            reporte_df.to_csv(reporte_path, index=False, encoding="utf-8-sig")
            print(f"Escrito: {reporte_path}")

    return metricas_df, series_df, descubrimiento


# =========================================================
# AUDITORIA DE SOLO LECTURA DEL CONSOLIDADO
# =========================================================

@dataclass
class ResultadoAuditoria:
    """Resultado de `auditar_consolidado()` -- ver esa funcion para el detalle de cada chequeo. `ok=False` si `problemas` tiene al menos un elemento; `avisos` nunca bloquea `ok`."""
    ok: bool
    problemas: list
    avisos: list
    resumen: dict


def auditar_consolidado(
    metricas_df: pd.DataFrame,
    series_df: pd.DataFrame,
    reporte_df: Optional[pd.DataFrame] = None,
    regiones_legacy_esperadas: Optional[list] = None,
    region_bca: str = "BCA",
    estrategias_univariado_esperadas: Optional[list] = None,
    horizon_legacy_esperado: int = 168,
    test_start_legacy_esperado: Optional[str] = "2026-05-17 00:00",
    test_end_legacy_esperado: Optional[str] = "2026-05-23 23:00",
    tolerancia_metricas: float = 1e-2,
    verbose: bool = True,
) -> ResultadoAuditoria:
    """
    Auditoria de SOLO LECTURA sobre un `metricas_df`/`series_df` ya
    construidos por `consolidar_resultados()` -- nunca los modifica, nunca
    corrige nada en silencio, solo reporta. Implementa los 10 chequeos:

      1. Ninguna corrida moderna vieja invalidada (con equivalente
         `_timefix` presente) sigue en `metricas_master` -- se detecta
         SOLO por el patron de sufijo `_timefix` sobre los `run_name` ya
         presentes (no requiere `reporte_df`); si se pasa `reporte_df`,
         ademas verifica que ningun `run_name` marcado `estado="reemplazado"`
         se haya colado.
      2. Reporta (en `resumen["timefix_presentes"]`) todas las corridas
         `_timefix` que SI quedaron incluidas.
      3. Duplicados `(run_name, region, timestamp)` en filas de prediccion
         de `series_master`, y consistencia cruzada de que todo `run_name`
         con metricas tenga tambien predicciones en `series_master` (y
         viceversa).
      4. Legacy univariado (`origen=legacy_univariado`): regiones y
         estrategias esperadas presentes, combinaciones region x
         estrategia completas, `horizon_legacy_esperado` predicciones por
         combinacion, ventana de test observada == esperada.
      5. BCA reconstruido (`origen=bca_univariado_reconstruido`): 1 sola
         region (`region_bca`), 8 estrategias esperadas, `metodologia`
         unica, misma ventana/horizonte que el legacy.
      6. Recalcula MAE/RMSE/MAPE/sMAPE por (run_name, region,
         modelo_estrategia) a partir de `series_master` (formula estandar;
         puede no coincidir exactamente con la formula de cada modulo
         individual, ej. manejo de division por cero en MAPE/sMAPE -- por
         eso se usa TOLERANCIA relativa, no igualdad exacta) y compara
         contra `metricas_master`; las discrepancias fuera de
         `tolerancia_metricas` se reportan como aviso/problema, NUNCA se
         corrigen aca.
      7. Busca NaN/inf en MAE/RMSE/MAPE/sMAPE/valor, y filas duplicadas
         (run_name, region, modelo_estrategia) en metricas_master.
      8. Confirma que la fila BCA/`STL_FCNN_residuos` sigue presente en
         metricas_master (nunca eliminada) y que quedo marcada
         `calidad_resultado="divergencia_numerica"` / `valido_ranking=False`.
      9. Reporta si quedan filas `real` duplicadas por (region, timestamp)
         en series_master (deberian haber sido deduplicadas por
         `construir_series_master()`/`_construir_real_deduplicado()`;
         esta funcion NO deduplica de nuevo, solo documenta la politica
         vigente y verifica que se haya aplicado).
      10. Cobertura observada por familia/origen/modelo/region (conteos,
          sin comparar contra un total "esperado" generico -- eso ya lo
          hacen los chequeos 4/5 especificamente para legacy/BCA).

    `regiones_legacy_esperadas` (default: `REGIONS_ALL` sin `region_bca`,
    la unica fuente de verdad de nombres de region del proyecto) /
    `estrategias_univariado_esperadas` (default: las 8 keys de
    `MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO`, la unica fuente de verdad de
    estrategias univariadas de este modulo) -- overrideables, nunca
    inventados sin evidencia de codigo.

    Si `metricas_df`/`series_df` no traen filas `origen=legacy_univariado`
    o `origen=bca_univariado_reconstruido` (por ejemplo, se llamo a
    `consolidar_resultados()` sin esos paths), los chequeos 4/5/8
    correspondientes se omiten con un aviso -- no se reportan como fallo.
    """
    problemas = []
    avisos = []
    resumen = {}

    regiones_legacy_esperadas = (
        list(regiones_legacy_esperadas) if regiones_legacy_esperadas is not None
        else [r for r in REGIONS_ALL if r != region_bca]
    )
    estrategias_univariado_esperadas = (
        list(estrategias_univariado_esperadas) if estrategias_univariado_esperadas is not None
        else sorted(MAPEO_ESTRATEGIA_LEGACY_UNIVARIADO)
    )

    tiene_metricas = metricas_df is not None and len(metricas_df) > 0
    tiene_series = series_df is not None and len(series_df) > 0

    # ------------------------------------------------------------
    # 1 y 2: reemplazos _timefix
    # ------------------------------------------------------------
    if tiene_metricas and "run_name" in metricas_df.columns:
        run_names_presentes = set(metricas_df["run_name"].dropna().unique())
        timefix_presentes = sorted(rn for rn in run_names_presentes if str(rn).endswith(SUFIJO_TIMEFIX))
        viejos_invalidados_presentes = sorted(
            rn[: -len(SUFIJO_TIMEFIX)] for rn in timefix_presentes
            if rn[: -len(SUFIJO_TIMEFIX)] in run_names_presentes
        )
        if viejos_invalidados_presentes:
            problemas.append(
                f"Check 1 FALLO: {len(viejos_invalidados_presentes)} corrida(s) vieja(s) invalidada(s) "
                f"presente(s) en metricas_master junto con su equivalente _timefix: {viejos_invalidados_presentes}"
            )
        resumen["timefix_presentes"] = timefix_presentes
        resumen["n_timefix_presentes"] = len(timefix_presentes)

        if reporte_df is not None and len(reporte_df) and "estado" in reporte_df.columns:
            reemplazados_reporte = set(reporte_df.loc[reporte_df["estado"] == "reemplazado", "run_name"])
            colados = sorted(reemplazados_reporte & run_names_presentes)
            if colados:
                problemas.append(
                    f"Check 1 FALLO: {len(colados)} corrida(s) marcada(s) 'reemplazado' en reporte_consolidacion "
                    f"pero IGUAL presentes en metricas_master: {colados}"
                )
            resumen["n_reemplazados_en_reporte"] = len(reemplazados_reporte)
    else:
        avisos.append("Check 1/2: metricas_master vacio o sin columna run_name -- nada que verificar.")

    # ------------------------------------------------------------
    # 3: duplicados de prediccion + consistencia metricas<->series por run_name
    # ------------------------------------------------------------
    if tiene_series and {"run_name", "region", "timestamp", "serie_tipo"}.issubset(series_df.columns):
        pred_series = series_df[series_df["serie_tipo"] == "prediccion"]
        dup = pred_series.groupby(["run_name", "region", "timestamp"]).size()
        dup = dup[dup > 1]
        resumen["n_duplicados_prediccion"] = int(len(dup))
        if len(dup):
            problemas.append(
                f"Check 3 FALLO: {len(dup)} combinacion(es) (run_name, region, timestamp) duplicadas "
                f"en series_master (tipo=prediccion). Ejemplos: {dup.head(10).reset_index().to_dict('records')}"
            )

    if tiene_metricas and "run_name" in metricas_df.columns and tiene_series and "run_name" in series_df.columns:
        runs_metricas = set(metricas_df["run_name"].dropna().unique())
        runs_series = set(series_df.loc[series_df["serie_tipo"] == "prediccion", "run_name"].dropna().unique())
        sin_series = sorted(runs_metricas - runs_series)
        sin_metricas = sorted(runs_series - runs_metricas)
        if sin_series:
            avisos.append(f"Check 3: {len(sin_series)} run_name(s) con metricas pero SIN filas de prediccion en series_master: {sin_series}")
        if sin_metricas:
            avisos.append(f"Check 3: {len(sin_metricas)} run_name(s) con predicciones en series_master pero SIN fila en metricas_master: {sin_metricas}")

    # ------------------------------------------------------------
    # 4: legacy univariado
    # ------------------------------------------------------------
    if tiene_metricas and "origen" in metricas_df.columns:
        legacy_m = metricas_df[metricas_df["origen"] == ORIGEN_LEGACY_UNIVARIADO]
        resumen["legacy_n_filas_metricas"] = int(len(legacy_m))
        if len(legacy_m) == 0:
            avisos.append("Check 4: no hay filas origen=legacy_univariado en metricas_master (no se integro legacy en esta llamada).")
        else:
            regiones_obs = sorted(legacy_m["region"].dropna().unique().tolist()) if "region" in legacy_m.columns else []
            estrategias_obs = sorted(legacy_m["modelo_estrategia"].dropna().unique().tolist()) if "modelo_estrategia" in legacy_m.columns else []
            faltan_regiones = sorted(set(regiones_legacy_esperadas) - set(regiones_obs))
            sobran_regiones = sorted(set(regiones_obs) - set(regiones_legacy_esperadas))
            faltan_estrategias = sorted(set(estrategias_univariado_esperadas) - set(estrategias_obs))
            if faltan_regiones:
                problemas.append(f"Check 4 FALLO: legacy univariado sin region(es) esperada(s): {faltan_regiones}")
            if sobran_regiones:
                avisos.append(f"Check 4: legacy univariado con region(es) NO esperada(s): {sobran_regiones}")
            if faltan_estrategias:
                problemas.append(f"Check 4 FALLO: legacy univariado sin estrategia(s) esperada(s): {faltan_estrategias}")

            n_combos_esperadas = len(regiones_legacy_esperadas) * len(estrategias_univariado_esperadas)
            n_combos_obs = (
                legacy_m.drop_duplicates(["region", "modelo_estrategia"]).shape[0]
                if {"region", "modelo_estrategia"}.issubset(legacy_m.columns) else None
            )
            resumen["legacy_combinaciones_esperadas"] = n_combos_esperadas
            resumen["legacy_combinaciones_observadas"] = n_combos_obs
            if n_combos_obs is not None and n_combos_obs != n_combos_esperadas:
                problemas.append(f"Check 4 FALLO: legacy univariado tiene {n_combos_obs} combinaciones region-estrategia, se esperaban {n_combos_esperadas}")

            if tiene_series and "origen" in series_df.columns:
                legacy_pred = series_df[(series_df["origen"] == ORIGEN_LEGACY_UNIVARIADO) & (series_df["serie_tipo"] == "prediccion")]
                if len(legacy_pred):
                    n_pred_por_combo = legacy_pred.groupby(["region", "modelo_estrategia"]).size()
                    mal_horizon = n_pred_por_combo[n_pred_por_combo != horizon_legacy_esperado]
                    if len(mal_horizon):
                        problemas.append(
                            f"Check 4 FALLO: {len(mal_horizon)} combinacion(es) legacy con numero de predicciones "
                            f"distinto de {horizon_legacy_esperado}: {mal_horizon.to_dict()}"
                        )
                    ts = pd.to_datetime(legacy_pred["timestamp"], errors="coerce")
                    resumen["legacy_test_start_observado"] = str(ts.min())
                    resumen["legacy_test_end_observado"] = str(ts.max())
                    if test_start_legacy_esperado and pd.Timestamp(ts.min()) != pd.Timestamp(test_start_legacy_esperado):
                        problemas.append(f"Check 4 FALLO: legacy test_start observado {ts.min()} != esperado {test_start_legacy_esperado}")
                    if test_end_legacy_esperado and pd.Timestamp(ts.max()) != pd.Timestamp(test_end_legacy_esperado):
                        problemas.append(f"Check 4 FALLO: legacy test_end observado {ts.max()} != esperado {test_end_legacy_esperado}")

    # ------------------------------------------------------------
    # 5: BCA reconstruido
    # ------------------------------------------------------------
    if tiene_metricas and "origen" in metricas_df.columns:
        bca_m = metricas_df[metricas_df["origen"] == ORIGEN_BCA_RECONSTRUIDO]
        resumen["bca_n_filas_metricas"] = int(len(bca_m))
        if len(bca_m) == 0:
            avisos.append("Check 5: no hay filas origen=bca_univariado_reconstruido en metricas_master (no se integro BCA en esta llamada).")
        else:
            regiones_bca_obs = sorted(bca_m["region"].dropna().unique().tolist()) if "region" in bca_m.columns else []
            estrategias_bca_obs = sorted(bca_m["modelo_estrategia"].dropna().unique().tolist()) if "modelo_estrategia" in bca_m.columns else []
            if regiones_bca_obs != [region_bca]:
                problemas.append(f"Check 5 FALLO: BCA reconstruido con region(es) {regiones_bca_obs}, se esperaba solo ['{region_bca}']")
            faltan_estrategias_bca = sorted(set(estrategias_univariado_esperadas) - set(estrategias_bca_obs))
            if faltan_estrategias_bca:
                problemas.append(f"Check 5 FALLO: BCA reconstruido sin estrategia(s) esperada(s): {faltan_estrategias_bca}")
            if "metodologia" in bca_m.columns:
                metodologias = bca_m["metodologia"].dropna().unique().tolist()
                if len(metodologias) != 1:
                    problemas.append(f"Check 5 FALLO: BCA reconstruido con metodologia inconsistente/ausente: {metodologias}")
            resumen["bca_estrategias_observadas"] = estrategias_bca_obs

            if tiene_series and "origen" in series_df.columns:
                bca_pred = series_df[(series_df["origen"] == ORIGEN_BCA_RECONSTRUIDO) & (series_df["serie_tipo"] == "prediccion")]
                if len(bca_pred):
                    n_pred_por_estrategia = bca_pred.groupby("modelo_estrategia").size()
                    mal_horizon_bca = n_pred_por_estrategia[n_pred_por_estrategia != horizon_legacy_esperado]
                    if len(mal_horizon_bca):
                        problemas.append(
                            f"Check 5 FALLO: {len(mal_horizon_bca)} estrategia(s) BCA con numero de predicciones "
                            f"distinto de {horizon_legacy_esperado}: {mal_horizon_bca.to_dict()}"
                        )
                    ts_bca = pd.to_datetime(bca_pred["timestamp"], errors="coerce")
                    resumen["bca_test_start_observado"] = str(ts_bca.min())
                    resumen["bca_test_end_observado"] = str(ts_bca.max())
                    if test_start_legacy_esperado and pd.Timestamp(ts_bca.min()) != pd.Timestamp(test_start_legacy_esperado):
                        problemas.append(f"Check 5 FALLO: BCA test_start observado {ts_bca.min()} != esperado {test_start_legacy_esperado}")
                    if test_end_legacy_esperado and pd.Timestamp(ts_bca.max()) != pd.Timestamp(test_end_legacy_esperado):
                        problemas.append(f"Check 5 FALLO: BCA test_end observado {ts_bca.max()} != esperado {test_end_legacy_esperado}")

    # ------------------------------------------------------------
    # 6: recalculo de MAE/RMSE/MAPE/sMAPE desde series_master
    # ------------------------------------------------------------
    discrepancias_por_columna = {}
    if tiene_metricas and tiene_series and {"run_name", "region", "modelo_estrategia"}.issubset(metricas_df.columns):
        pred = series_df[series_df["serie_tipo"] == "prediccion"].copy()
        real = series_df[series_df["serie_tipo"] == "real"][["region", "timestamp", "valor"]].rename(columns={"valor": "valor_real"})
        if len(pred) and len(real):
            comparado = pred.merge(real, on=["region", "timestamp"], how="inner")
            if len(comparado):
                comparado["error"] = comparado["valor"] - comparado["valor_real"]
                comparado["error_abs"] = comparado["error"].abs()
                comparado["error_pct"] = np.where(comparado["valor_real"] != 0, comparado["error_abs"] / comparado["valor_real"].abs() * 100, np.nan)
                denom_smape = comparado["valor"].abs() + comparado["valor_real"].abs()
                comparado["error_smape"] = np.where(denom_smape != 0, comparado["error_abs"] / denom_smape * 200, np.nan)

                agregados = comparado.groupby(["run_name", "region", "modelo_estrategia"]).agg(
                    MAE_recalc=("error_abs", "mean"),
                    RMSE_recalc=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                    MAPE_recalc=("error_pct", lambda s: float(np.nanmean(s))),
                    sMAPE_recalc=("error_smape", lambda s: float(np.nanmean(s))),
                ).reset_index()

                cols_orig = [c for c in ["MAE", "RMSE", "MAPE", "sMAPE"] if c in metricas_df.columns]
                comparacion = metricas_df[["run_name", "region", "modelo_estrategia"] + cols_orig].merge(
                    agregados, on=["run_name", "region", "modelo_estrategia"], how="inner"
                )
                resumen["n_comparaciones_metricas_recalculadas"] = int(len(comparacion))

                for col_orig in cols_orig:
                    col_recalc = f"{col_orig}_recalc"
                    diff = (comparacion[col_orig] - comparacion[col_recalc]).abs()
                    rel = diff / comparacion[col_orig].abs().replace(0, np.nan)
                    fuera_tol = comparacion[(rel > tolerancia_metricas) & comparacion[col_orig].notna()]
                    if len(fuera_tol):
                        discrepancias_por_columna[col_orig] = int(len(fuera_tol))
                        for _, fila in fuera_tol.head(20).iterrows():
                            avisos.append(
                                f"Check 6: {fila['run_name']} / {fila['region']} / {fila['modelo_estrategia']}: "
                                f"{col_orig} master={fila[col_orig]:.6g} vs recalculado={fila[col_recalc]:.6g}"
                            )
    resumen["discrepancias_metricas_por_columna"] = discrepancias_por_columna
    if discrepancias_por_columna:
        problemas.append(
            f"Check 6: discrepancias fuera de tolerancia ({tolerancia_metricas:.1%}) entre metricas_master y "
            f"recalculo desde series_master: {discrepancias_por_columna} -- NO se corrigen automaticamente, "
            "ver avisos para el detalle por fila (la formula de recalculo es estandar y puede no coincidir "
            "exactamente con la de cada modulo, ej. manejo de division por cero)."
        )

    # ------------------------------------------------------------
    # 7: NaN / inf / duplicados
    # ------------------------------------------------------------
    if tiene_metricas:
        for col in ["MAE", "RMSE", "MAPE", "sMAPE"]:
            if col in metricas_df.columns:
                n_nan = int(metricas_df[col].isna().sum())
                n_inf = int(np.isinf(pd.to_numeric(metricas_df[col], errors="coerce")).sum())
                if n_nan:
                    avisos.append(f"Check 7: metricas_master columna '{col}' tiene {n_nan} NaN.")
                if n_inf:
                    avisos.append(f"Check 7: metricas_master columna '{col}' tiene {n_inf} valor(es) infinito(s).")
        cols_dup = [c for c in ["run_name", "region", "modelo_estrategia"] if c in metricas_df.columns]
        if cols_dup:
            dup_metricas = metricas_df.duplicated(subset=cols_dup, keep=False)
            if dup_metricas.any():
                avisos.append(f"Check 7: {int(dup_metricas.sum())} fila(s) duplicadas ({', '.join(cols_dup)}) en metricas_master.")
    if tiene_series and "valor" in series_df.columns:
        n_nan_series = int(series_df["valor"].isna().sum())
        n_inf_series = int(np.isinf(pd.to_numeric(series_df["valor"], errors="coerce")).sum())
        if n_nan_series:
            avisos.append(f"Check 7: series_master columna 'valor' tiene {n_nan_series} NaN.")
        if n_inf_series:
            avisos.append(f"Check 7: series_master columna 'valor' tiene {n_inf_series} valor(es) infinito(s).")

    # ------------------------------------------------------------
    # 8: BCA / STL_FCNN_residuos preservado y marcado
    # ------------------------------------------------------------
    if tiene_metricas and {"modelo_estrategia", "origen"}.issubset(metricas_df.columns):
        fila_divergente = metricas_df[
            (metricas_df["origen"] == ORIGEN_BCA_RECONSTRUIDO) & (metricas_df["modelo_estrategia"] == "STL_FCNN_residuos")
        ]
        if len(fila_divergente) == 0:
            avisos.append("Check 8: no hay fila BCA/STL_FCNN_residuos en metricas_master (BCA no integrado en esta llamada, o estrategia ausente).")
        else:
            for _, fila in fila_divergente.iterrows():
                if fila.get("calidad_resultado") != "divergencia_numerica":
                    problemas.append(f"Check 8 FALLO: BCA/STL_FCNN_residuos con calidad_resultado={fila.get('calidad_resultado')!r}, se esperaba 'divergencia_numerica'.")
                if fila.get("valido_ranking") is not False:
                    problemas.append(f"Check 8 FALLO: BCA/STL_FCNN_residuos con valido_ranking={fila.get('valido_ranking')!r}, se esperaba False.")
            resumen["bca_stl_fcnn_residuos_MAE"] = fila_divergente["MAE"].tolist() if "MAE" in fila_divergente.columns else None

    # ------------------------------------------------------------
    # 9: consistencia de "real" compartido entre corridas
    # ------------------------------------------------------------
    if tiene_series and {"region", "timestamp", "serie_tipo"}.issubset(series_df.columns):
        real_rows = series_df[series_df["serie_tipo"] == "real"]
        dup_real = real_rows.duplicated(subset=["region", "timestamp"], keep=False)
        n_dup_real = int(dup_real.sum())
        resumen["n_filas_real_duplicadas_region_timestamp"] = n_dup_real
        if n_dup_real:
            problemas.append(
                f"Check 9 FALLO: {n_dup_real} fila(s) 'real' con (region, timestamp) repetido en series_master -- "
                "construir_series_master() deberia haber deduplicado esto; revisar _construir_real_deduplicado()."
            )
        avisos.append(
            "Check 9: politica actual de 'real' compartido documentada en construir_series_master()/"
            "_construir_real_deduplicado(): si dos corridas traen un valor distinto para el mismo "
            "(region, timestamp), se conserva el PRIMER valor visto y se imprime un aviso al construir "
            "el consolidado (no se promedia ni se descarta silenciosamente). Esta auditoria no reaplica "
            "deduplicacion, solo confirma que no quedaron duplicados sin resolver."
        )

    # ------------------------------------------------------------
    # 10: cobertura observada
    # ------------------------------------------------------------
    cobertura = {}
    if tiene_metricas:
        base = metricas_df.drop_duplicates("run_name") if "run_name" in metricas_df.columns else metricas_df
        if "familia_experimento" in base.columns:
            cobertura["por_familia"] = base["familia_experimento"].value_counts(dropna=False).to_dict()
        if "origen" in base.columns:
            cobertura["por_origen"] = base["origen"].value_counts(dropna=False).to_dict()
        if "modelo" in base.columns:
            cobertura["por_modelo"] = base["modelo"].value_counts(dropna=False).to_dict()
        if "region" in metricas_df.columns:
            cobertura["por_region"] = metricas_df["region"].value_counts(dropna=False).to_dict()
        cobertura["n_runs_unicos"] = int(metricas_df["run_name"].nunique()) if "run_name" in metricas_df.columns else None
    resumen["cobertura"] = cobertura

    resultado = ResultadoAuditoria(ok=(len(problemas) == 0), problemas=problemas, avisos=avisos, resumen=resumen)
    if verbose:
        imprimir_auditoria(resultado)
    return resultado


def imprimir_auditoria(resultado: ResultadoAuditoria):
    """Imprime el reporte de `auditar_consolidado()` en formato legible -- pensado para la celda de auditoria en Colab."""
    print("=" * 80)
    print("AUDITORIA DEL CONSOLIDADO (solo lectura)")
    print("=" * 80)
    print(f"Resultado: {'OK' if resultado.ok else 'CON PROBLEMAS'}")

    if resultado.problemas:
        print(f"\nPROBLEMAS ({len(resultado.problemas)}):")
        for p in resultado.problemas:
            print(f"  [X] {p}")
    else:
        print("\nSin problemas bloqueantes.")

    if resultado.avisos:
        print(f"\nAVISOS ({len(resultado.avisos)}):")
        for a in resultado.avisos:
            print(f"  [!] {a}")

    print("\nResumen:")
    for k, v in resultado.resumen.items():
        print(f"  {k}: {v}")

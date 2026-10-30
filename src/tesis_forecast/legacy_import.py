"""
Importacion ADITIVA de resultados legacy verificados hacia el esquema
moderno de `Pipeline_Resultados/<RUN_NAME>/`, para que `checkpoint.py` los
reconozca como regiones completas sin tener que re-entrenarlas.

Dos funciones, deliberadamente separadas:

  - `verificar_fcnn_legacy_temp_igae()`: SOLO LECTURA. Lee los CSV legacy
    y verifica, por region, si cumplen EXACTAMENTE lo que
    `fcnn_model.py`/`checkpoint.py` exigirian de una corrida moderna.
    Nunca escribe nada. Nunca "asume" un valor que no pudo confirmar --
    si algo no se puede verificar con evidencia real, esa region queda
    bloqueada (no reutilizable), nunca se fuerza a pasar.

  - `importar_fcnn_legacy_a_moderno()`: escribe la carpeta moderna, pero
    SOLO para las regiones que ya pasaron `verificar_fcnn_legacy_temp_igae`
    (nunca fabrica una region que no verifico). Es aditiva: se niega a
    tocar una carpeta moderna que ya exista, y nunca lee/mueve/borra nada
    de la carpeta legacy (abre sus CSV de solo lectura).

Requisito duro que este modulo NUNCA fabrica: `checkpoint.py` exige, para
FCNN, `requiere_trials=True` (ver `fcnn_model.py::run()`) -- un archivo de
trials con registros individuales de Optuna por region. Si la carpeta
legacy no tiene eso (solo tenia, por ejemplo, un contador `trials=N`
dentro de `metricas.csv`, que NO son registros reales), la region queda
bloqueada explicitamente en vez de inventar un `trials.csv` vacio o falso
-- eso violaria la regla de "no falsificar metadata" y ademas
`region_es_completa()` seguiria rechazandola de todos modos (una tabla de
trials vacia no pasa el chequeo `len(t) == 0: return False`).
"""

import ast
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# Requisitos EXACTOS que fcnn_model.py exige via checkpoint (ver
# fcnn_model.py::run() -> cargar_checkpoint_regiones(...,
# n_modelos_esperados=2, requiere_trials=True, requiere_config_usada=True)).
FCNN_N_MODELOS_ESPERADOS = 2
FCNN_REQUIERE_TRIALS = True
FCNN_REQUIERE_CONFIG_USADA = True

_EXOG_INTERNO_A_CANONICO = {"Temperaturas": "Temperatura", "IGAE": "IGAE"}


@dataclass
class VerificacionRegion:
    region: str
    ok: bool = False
    problemas: list = field(default_factory=list)
    metricas_ok: bool = False
    n_predicciones: dict = field(default_factory=dict)
    exogenas_detectadas: Optional[list] = None
    train_horas_detectado: Optional[int] = None
    test_horas_detectado: Optional[int] = None
    window_detectado: Optional[int] = None
    trials_disponibles: bool = False
    archivos_usados: list = field(default_factory=list)
    metricas_df: Optional[pd.DataFrame] = None
    series_df: Optional[pd.DataFrame] = None
    trials_df: Optional[pd.DataFrame] = None


@dataclass
class VerificacionFCNNLegacy:
    legacy_dir: str
    regiones: dict = field(default_factory=dict)
    regiones_reutilizables: list = field(default_factory=list)
    regiones_bloqueadas: list = field(default_factory=list)
    bloqueo_global: Optional[str] = None


def _leer_csv(ruta):
    if not ruta or not os.path.exists(ruta):
        return None
    try:
        return pd.read_csv(ruta, encoding="utf-8-sig")
    except Exception:
        return None


def _parsear_exogenas_texto(texto):
    texto = str(texto).strip()
    if texto.startswith("[") and texto.endswith("]"):
        try:
            items = [str(x).strip() for x in ast.literal_eval(texto)]
        except Exception:
            items = [x.strip(" '\"") for x in texto[1:-1].split(",") if x.strip(" '\"")]
    else:
        items = [x.strip(" '\"") for x in texto.split(",") if x.strip(" '\"")]
    return sorted({_EXOG_INTERNO_A_CANONICO.get(x, x) for x in items})


def _archivos_de_la_region(carpeta_por_serie, region):
    if not os.path.isdir(carpeta_por_serie):
        return []
    return [f for f in os.listdir(carpeta_por_serie) if f.startswith(f"{region}_") and f.endswith(".csv")]


def verificar_fcnn_legacy_temp_igae(
    legacy_dir: str,
    regiones_a_verificar: list,
    train_hours_esperado: int = 3600,
    forecast_horizon_esperado: int = 168,
    window_esperado: int = 168,
    exogenas_esperadas=("Temperatura", "IGAE"),
) -> VerificacionFCNNLegacy:
    """
    Verifica, region por region, si `legacy_dir` (se espera una
    subcarpeta `por_serie/` con archivos `{region}_..._series.csv` /
    `{region}_..._metricas.csv`, y opcionalmente archivos de trials)
    cumple EXACTAMENTE el esquema que `fcnn_model.py`/`checkpoint.py`
    exigirian de una corrida moderna. Solo lectura -- nunca escribe nada.
    """
    resultado = VerificacionFCNNLegacy(legacy_dir=legacy_dir)

    if not os.path.isdir(legacy_dir):
        resultado.bloqueo_global = f"La carpeta legacy no existe (o no es accesible): {legacy_dir}"
        for region in regiones_a_verificar:
            resultado.regiones[region] = VerificacionRegion(region=region, ok=False, problemas=[resultado.bloqueo_global])
            resultado.regiones_bloqueadas.append(region)
        return resultado

    por_serie_dir = os.path.join(legacy_dir, "por_serie")

    def _hay_archivo_trials(carpeta):
        if not os.path.isdir(carpeta):
            return False
        return any("trial" in f.lower() and f.endswith(".csv") for f in os.listdir(carpeta))

    def _hay_optuna_db(carpeta):
        if not os.path.isdir(carpeta):
            return False
        return any(f.endswith(".db") for f in os.listdir(carpeta))

    hay_trials_global = _hay_archivo_trials(por_serie_dir) or _hay_archivo_trials(legacy_dir)
    hay_optuna_db = _hay_optuna_db(por_serie_dir) or _hay_optuna_db(legacy_dir)

    for region in regiones_a_verificar:
        v = VerificacionRegion(region=region)

        archivos_region = _archivos_de_la_region(por_serie_dir, region)
        archivos_metricas = [f for f in archivos_region if "metricas" in f.lower()]
        archivos_series = [f for f in archivos_region if "series" in f.lower()]
        archivos_trials_region = [f for f in archivos_region if "trial" in f.lower()]

        if not archivos_metricas:
            v.problemas.append(f"No se encontro ningun archivo de metricas para '{region}' en {por_serie_dir}.")
            resultado.regiones[region] = v
            resultado.regiones_bloqueadas.append(region)
            continue

        frames_metricas = []
        for f in archivos_metricas:
            df = _leer_csv(os.path.join(por_serie_dir, f))
            if df is not None:
                frames_metricas.append(df)
                v.archivos_usados.append(f"por_serie/{f}")
        if not frames_metricas:
            v.problemas.append(f"Los archivos de metricas de '{region}' no se pudieron leer.")
            resultado.regiones[region] = v
            resultado.regiones_bloqueadas.append(region)
            continue
        metricas_df = pd.concat(frames_metricas, ignore_index=True, sort=False)

        if "modelo" not in metricas_df.columns:
            v.problemas.append("metricas.csv no tiene columna 'modelo' -- no se pueden distinguir las 2 estrategias.")
        else:
            estrategias = sorted(metricas_df["modelo"].dropna().unique().tolist())
            if len(estrategias) != FCNN_N_MODELOS_ESPERADOS:
                v.problemas.append(f"Se esperaban {FCNN_N_MODELOS_ESPERADOS} estrategias, se encontraron {len(estrategias)}: {estrategias}.")

        columnas_metrica = [c for c in ["MAE", "RMSE", "MAPE", "sMAPE"] if c in metricas_df.columns]
        if len(columnas_metrica) < 4:
            v.problemas.append(f"Faltan columnas de metricas (se esperaban MAE/RMSE/MAPE/sMAPE): hay {columnas_metrica}.")
        else:
            numericas = metricas_df[columnas_metrica].apply(pd.to_numeric, errors="coerce")
            invalidas = numericas.isna().any(axis=1)
            if invalidas.any():
                v.problemas.append(f"{int(invalidas.sum())} fila(s) de metricas con NaN/no-numerico en MAE/RMSE/MAPE/sMAPE.")
            else:
                v.metricas_ok = True

        for col, atributo in [("train_horas", "train_horas_detectado"), ("test_horas", "test_horas_detectado"),
                               ("window", "window_detectado")]:
            if col in metricas_df.columns:
                valores = pd.to_numeric(metricas_df[col], errors="coerce").dropna().unique()
                if len(valores) == 1:
                    setattr(v, atributo, int(valores[0]))
                elif len(valores) > 1:
                    v.problemas.append(f"Columna '{col}' tiene valores INCONSISTENTES entre filas: {list(valores)}.")
            else:
                v.problemas.append(f"metricas.csv no tiene columna '{col}' -- no se puede confirmar.")

        if "exogenas" in metricas_df.columns:
            valores_exog = metricas_df["exogenas"].dropna().unique()
            if len(valores_exog) == 1:
                v.exogenas_detectadas = _parsear_exogenas_texto(valores_exog[0])
                if set(v.exogenas_detectadas) != set(exogenas_esperadas):
                    v.problemas.append(
                        f"Exogenas detectadas {v.exogenas_detectadas} != esperadas {list(exogenas_esperadas)} "
                        f"(valor literal encontrado en el CSV: {valores_exog[0]!r})."
                    )
            elif len(valores_exog) > 1:
                v.problemas.append(f"Columna 'exogenas' tiene valores INCONSISTENTES entre filas: {list(valores_exog)}.")
        else:
            v.problemas.append("metricas.csv no tiene columna 'exogenas' -- no se puede confirmar cuales se usaron.")

        if v.train_horas_detectado is not None and v.train_horas_detectado != train_hours_esperado:
            v.problemas.append(f"train_horas detectado={v.train_horas_detectado} != esperado={train_hours_esperado}.")
        if v.test_horas_detectado is not None and v.test_horas_detectado != forecast_horizon_esperado:
            v.problemas.append(f"test_horas detectado={v.test_horas_detectado} != esperado={forecast_horizon_esperado}.")
        if v.window_detectado is not None and v.window_detectado != window_esperado:
            v.problemas.append(f"window detectado={v.window_detectado} != esperado={window_esperado}.")

        if not archivos_series:
            v.problemas.append(f"No se encontro ningun archivo de series/predicciones para '{region}'.")
            series_df = None
        else:
            frames_series = []
            for f in archivos_series:
                df = _leer_csv(os.path.join(por_serie_dir, f))
                if df is not None:
                    frames_series.append(df)
                    v.archivos_usados.append(f"por_serie/{f}")
            series_df = pd.concat(frames_series, ignore_index=True, sort=False) if frames_series else None

            if series_df is not None:
                if "tipo" not in series_df.columns or "modelo" not in series_df.columns or "valor" not in series_df.columns:
                    v.problemas.append(f"series.csv de '{region}' no tiene columnas esperadas (tipo/modelo/valor). Columnas: {list(series_df.columns)}.")
                else:
                    pred = series_df[series_df["tipo"] == "prediccion"]
                    conteos = pred.groupby("modelo").size().to_dict()
                    v.n_predicciones = conteos
                    for estrategia, n in conteos.items():
                        if n != forecast_horizon_esperado:
                            v.problemas.append(f"Estrategia '{estrategia}': {n} predicciones encontradas, se esperaban {forecast_horizon_esperado}.")
                    if "modelo" in metricas_df.columns:
                        estrategias_metricas = set(metricas_df["modelo"].dropna().unique())
                        estrategias_series = set(conteos.keys())
                        if estrategias_metricas != estrategias_series:
                            v.problemas.append(f"Estrategias en metricas.csv {estrategias_metricas} != estrategias en series.csv (prediccion) {estrategias_series}.")
                    if pred["valor"].isna().any():
                        v.problemas.append(f"{int(pred['valor'].isna().sum())} valor(es) NaN entre las predicciones.")

        v.trials_disponibles = bool(archivos_trials_region) or hay_trials_global
        if v.trials_disponibles and archivos_trials_region:
            frames_trials = []
            for f in archivos_trials_region:
                df = _leer_csv(os.path.join(por_serie_dir, f))
                if df is not None:
                    frames_trials.append(df)
                    v.archivos_usados.append(f"por_serie/{f}")
            v.trials_df = pd.concat(frames_trials, ignore_index=True, sort=False) if frames_trials else None
            if v.trials_df is None or len(v.trials_df) == 0:
                v.trials_disponibles = False

        if not v.trials_disponibles:
            nota_db = (
                " Se detecto un archivo .db de Optuna en la carpeta, pero extraer registros de trials desde ahi "
                "requiere abrir el study con la libreria optuna (paso aparte, no se hace automaticamente aqui)."
                if hay_optuna_db else " Tampoco se encontro un archivo .db de Optuna."
            )
            v.problemas.append(
                "No se encontro un archivo de trials (registros individuales de Optuna) para esta region -- "
                "checkpoint.region_es_completa() exige requiere_trials=True para fcnn (ver fcnn_model.py::run()). "
                "Sin esto, la region NO seria reconocida como completa por el sistema de checkpoint actual, "
                "sin importar que tan buenas sean las metricas/predicciones. No se fabrica un trials.csv vacio "
                "o falso -- eso ademas seguiria sin pasar el chequeo (una tabla vacia tampoco cuenta)." + nota_db
            )

        v.metricas_df = metricas_df
        v.series_df = series_df
        v.ok = (len(v.problemas) == 0)
        resultado.regiones[region] = v
        (resultado.regiones_reutilizables if v.ok else resultado.regiones_bloqueadas).append(region)

    return resultado


def imprimir_reporte_verificacion(verificacion: VerificacionFCNNLegacy):
    print("=" * 80)
    print(f"VERIFICACION FCNN LEGACY TEMP+IGAE -- {verificacion.legacy_dir}")
    print("=" * 80)
    if verificacion.bloqueo_global:
        print(f"BLOQUEO GLOBAL: {verificacion.bloqueo_global}")
        return
    for region, v in verificacion.regiones.items():
        estado = "OK -- reutilizable" if v.ok else "BLOQUEADA"
        print(f"\n{region}: {estado}")
        print(f"  archivos usados: {v.archivos_usados}")
        print(f"  exogenas detectadas: {v.exogenas_detectadas}")
        print(f"  train_horas={v.train_horas_detectado}  test_horas={v.test_horas_detectado}  window={v.window_detectado}")
        print(f"  predicciones por estrategia: {v.n_predicciones}")
        print(f"  trials disponibles: {v.trials_disponibles}")
        if v.problemas:
            print("  problemas:")
            for p in v.problemas:
                print(f"    - {p}")

    print("\n" + "-" * 80)
    print(f"Reutilizables ({len(verificacion.regiones_reutilizables)}): {sorted(verificacion.regiones_reutilizables)}")
    print(f"Bloqueadas ({len(verificacion.regiones_bloqueadas)}): {sorted(verificacion.regiones_bloqueadas)}")


# =========================================================
# IMPORTAR (aditivo, solo con lo ya verificado)
# =========================================================

def importar_fcnn_legacy_a_moderno(
    pipeline_resultados_dir: str,
    verificacion: VerificacionFCNNLegacy,
    run_name: str = "FCNN_train3600h_fh168h_Temp-IGAE",
    dry_run: bool = True,
) -> dict:
    """
    Construye ADITIVAMENTE `pipeline_resultados_dir/<run_name>/` a partir
    UNICAMENTE de `verificacion.regiones_reutilizables` (nunca de una
    region que no paso los checks). Nunca toca `verificacion.legacy_dir`.
    Se niega a escribir si `run_name` ya existe (no sobreescribe).

    `dry_run=True` (default): no escribe nada en disco, solo devuelve un
    resumen de que haria. Pasar `dry_run=False` para escribir de verdad.
    """
    run_dir = os.path.join(pipeline_resultados_dir, run_name)

    if os.path.exists(run_dir):
        return {"escrito": False, "motivo": f"{run_dir} ya existe -- import aditivo se niega a tocarla.", "run_dir": run_dir}

    if not verificacion.regiones_reutilizables:
        return {
            "escrito": False,
            "motivo": "Ninguna region paso la verificacion -- no hay nada legitimo para importar.",
            "run_dir": run_dir,
            "regiones_bloqueadas": verificacion.regiones_bloqueadas,
        }

    regiones = sorted(verificacion.regiones_reutilizables)

    series_frames, metricas_frames, config_usada_rows = [], [], []
    for region in regiones:
        v = verificacion.regiones[region]
        series_frames.append(v.series_df)
        metricas_frames.append(v.metricas_df)
        for estrategia in v.metricas_df["modelo"].dropna().unique():
            config_usada_rows.append({
                "serie": f"{region}_DEMANDA",
                "modelo": estrategia,
                "train_horas": v.train_horas_detectado,
                "test_horas": v.test_horas_detectado,
                "window": v.window_detectado,
                "exogenas": ", ".join(v.exogenas_detectadas or []),
                "origen": "IMPORTADO_DE_LEGACY",
                "region": region,
            })

    series_df = pd.concat(series_frames, ignore_index=True, sort=False)
    series_df["region"] = series_df["serie"].map(lambda s: s.split("_")[0])

    metricas_df = pd.concat(metricas_frames, ignore_index=True, sort=False)
    metricas_df["region"] = metricas_df["serie"].map(lambda s: s.split("_")[0])

    config_usada_df = pd.DataFrame(config_usada_rows)

    trials_frames = [verificacion.regiones[r].trials_df for r in regiones if verificacion.regiones[r].trials_df is not None]
    trials_df = pd.concat(trials_frames, ignore_index=True, sort=False) if trials_frames else None

    resumen = {
        "escrito": False,
        "run_dir": run_dir,
        "regiones_importadas": regiones,
        "n_regiones_importadas": len(regiones),
        "filas_series": len(series_df),
        "filas_metricas": len(metricas_df),
        "filas_config_usada": len(config_usada_df),
        "filas_trials": len(trials_df) if trials_df is not None else 0,
    }

    if dry_run:
        resumen["motivo"] = "dry_run=True: no se escribio nada. Volver a llamar con dry_run=False para escribir de verdad."
        return resumen

    os.makedirs(run_dir)

    config_json = {
        "modelo": "fcnn",
        "exogenas": ["Temperatura", "IGAE"],
        "train_hours": 3600,
        "forecast_horizon": 168,
        "optuna_n_trials": None,
        "seed": None,
        "notas": (
            f"IMPORTADO DE RESULTADOS LEGACY ({verificacion.legacy_dir}) el "
            f"{datetime.now(timezone.utc).isoformat()}. Regiones importadas: {regiones}. "
            f"Region(es) pendiente(s) de correr con el pipeline actual: las que no esten en esta lista. "
            f"Ver config_usada.csv columna 'origen'=IMPORTADO_DE_LEGACY para el detalle por fila."
        ),
        "run_name": run_name,
        "git_commit": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "importado_de_legacy": True,
        "legacy_dir_origen": verificacion.legacy_dir,
        "regiones_importadas": regiones,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)

    series_df.to_csv(os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
    metricas_df.to_csv(os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
    config_usada_df.to_csv(os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
    if trials_df is not None:
        trials_df.to_csv(os.path.join(run_dir, "trials.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Carpeta creada por importacion de resultados legacy verificados.\n")
        f.write(f"Origen: {verificacion.legacy_dir}\n")
        f.write(f"Fecha de importacion: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Regiones importadas ({len(regiones)}): {regiones}\n")
        f.write(f"Regiones bloqueadas (no importadas, requieren correrse con el pipeline actual): {sorted(verificacion.regiones_bloqueadas)}\n")

    resumen["escrito"] = True
    resumen["motivo"] = "Import completado."
    return resumen

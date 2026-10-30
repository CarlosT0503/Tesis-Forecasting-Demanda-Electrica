"""
Orquestador de un experimento: resuelve defaults, arma el RUN_NAME, crea la
carpeta (o la reintenta con overwrite=True si esta incompleta, o aborta si
ya esta completa), monta Drive, corre el modelo, y guarda config.json /
series.csv / metricas.csv / trials.csv / config_usada.csv.

`matrix.py` (cola de experimentos) y `validator.py` (validacion de
completitud) se apoyan en las funciones de este archivo sin tener que
tocarlas. Scheduling por hora e infraestructura de nube siguen sin existir.
Ver docs/AUTOMATIZACION_FUTURA.md para el detalle de que capacidad futura
quedo resuelta por cual pieza.
"""

import contextlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

from . import errores_log, io_drive, windows
from .atomic_io import escribir_csv_atomico
from .config import ExperimentConfig, build_run_name, resolved_config_dict
from .data.exogenas import cargar_exogenas_horarias
from .models import (
    xgboost_model, lightgbm_model, lstm_direct, sarimax_model, fcnn_model, ensemble_stl,
    naive_model, naive_trend_model, ar_model, naive_trend_seasonal_model,
    ar_resid_trend_seasonal_model, lstm_resid_model,
)
from .regions import REGIONS_ALL
from .validator import validar_resultado

MODEL_DEFAULTS = {
    "xgboost": {
        "train_hours": xgboost_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": xgboost_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": xgboost_model.N_TRIALS_OPTUNA_DEFAULT,
        "exogenas": xgboost_model.EXOG_COLS_DEFAULT,
        "catalogo": xgboost_model.EXOG_CATALOGO,
    },
    "lightgbm": {
        "train_hours": lightgbm_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": lightgbm_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": lightgbm_model.N_TRIALS_OPTUNA_DEFAULT,
        "exogenas": lightgbm_model.EXOG_COLS_DEFAULT,
        "catalogo": lightgbm_model.EXOG_CATALOGO,
    },
    "lstm_direct": {
        "train_hours": lstm_direct.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": lstm_direct.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": lstm_direct.N_TRIALS_LSTM_DEFAULT,
        "exogenas": lstm_direct.EXOG_COLS_DEFAULT,
        "catalogo": lstm_direct.EXOG_CATALOGO,
    },
    "sarimax": {
        "train_hours": sarimax_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": sarimax_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": None,  # SARIMAX no tunea, orden fijo
        "exogenas": sarimax_model.EXOG_COLS_DEFAULT,
        "catalogo": sarimax_model.EXOG_CATALOGO,
    },
    "fcnn": {
        "train_hours": fcnn_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": fcnn_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": fcnn_model.N_TRIALS_DEFAULT,
        "exogenas": fcnn_model.EXOG_COLS_DEFAULT,
        "catalogo": fcnn_model.EXOG_CATALOGO,
    },
    "ensemble_stl": {
        "train_hours": ensemble_stl.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": ensemble_stl.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": ensemble_stl.N_TRIALS_DEFAULT,
        "exogenas": ensemble_stl.EXOG_COLS_DEFAULT,
        "catalogo": ensemble_stl.EXOG_CATALOGO,
    },
    "naive": {
        "train_hours": naive_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": naive_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": None,  # sin tuning
        "exogenas": naive_model.EXOG_COLS_DEFAULT,
        "catalogo": naive_model.EXOG_CATALOGO,
    },
    "naive_trend": {
        "train_hours": naive_trend_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": naive_trend_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": None,  # sin tuning
        "exogenas": naive_trend_model.EXOG_COLS_DEFAULT,
        "catalogo": naive_trend_model.EXOG_CATALOGO,
    },
    "ar": {
        "train_hours": ar_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": ar_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": None,  # sin Optuna, orden por AIC
        "exogenas": ar_model.EXOG_COLS_DEFAULT,
        "catalogo": ar_model.EXOG_CATALOGO,
    },
    "naive_trend_seasonal": {
        "train_hours": naive_trend_seasonal_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": naive_trend_seasonal_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": None,  # sin tuning
        "exogenas": naive_trend_seasonal_model.EXOG_COLS_DEFAULT,
        "catalogo": naive_trend_seasonal_model.EXOG_CATALOGO,
    },
    "ar_resid_trend_seasonal": {
        "train_hours": ar_resid_trend_seasonal_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": ar_resid_trend_seasonal_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": None,  # sin Optuna, orden por AIC
        "exogenas": ar_resid_trend_seasonal_model.EXOG_COLS_DEFAULT,
        "catalogo": ar_resid_trend_seasonal_model.EXOG_CATALOGO,
    },
    "lstm_resid": {
        "train_hours": lstm_resid_model.TRAIN_LAST_HOURS_DEFAULT,
        "forecast_horizon": lstm_resid_model.FORECAST_HORIZON_DEFAULT,
        "optuna_n_trials": lstm_resid_model.N_TRIALS_DEFAULT,
        "exogenas": lstm_resid_model.EXOG_COLS_DEFAULT,
        "catalogo": lstm_resid_model.EXOG_CATALOGO,
    },
}

MODEL_RUNNERS = {
    "xgboost": xgboost_model.run,
    "lightgbm": lightgbm_model.run,
    "lstm_direct": lstm_direct.run,
    "sarimax": sarimax_model.run,
    "fcnn": fcnn_model.run,
    "ensemble_stl": ensemble_stl.run,
    "naive": naive_model.run,
    "naive_trend": naive_trend_model.run,
    "ar": ar_model.run,
    "naive_trend_seasonal": naive_trend_seasonal_model.run,
    "ar_resid_trend_seasonal": ar_resid_trend_seasonal_model.run,
    "lstm_resid": lstm_resid_model.run,
}


@dataclass
class ResolvedRun:
    """Config de un experimento con todos los defaults ya resueltos, y el
    RUN_NAME/ruta que le corresponden -- sin ejecutar nada ni tocar disco.

    Pensado para que una capa futura (deteccion de duplicados, matriz de
    experimentos) pueda calcular "que archivo le tocaria a esta config" sin
    tener que duplicar la logica de defaults que ya vive en run_experiment().
    """

    modelo: str
    run_name: str
    run_dir: str
    train_hours: int
    forecast_horizon: int
    optuna_n_trials: int
    exogenas: list
    ventana: Optional[str] = None


@dataclass
class ExperimentResult:
    """Resultado de una corrida completada con exito.

    run_experiment() sigue levantando una excepcion si algo falla (no
    devuelve un status="failed"); este objeto es el "caso feliz". Una capa
    futura que agregue resultados de varios experimentos en un resumen
    puede construir su propio registro con status="failed" a partir de la
    excepcion capturada, usando los mismos nombres de campo para que un
    resumen combinado sea consistente.
    """

    run_name: str
    run_dir: str
    status: str  # siempre "completed" cuando lo devuelve run_experiment()
    regiones_esperadas: int
    regiones_con_metricas: int
    mape_promedio: Optional[float]
    archivos: dict = field(default_factory=dict)


def resolve_run(config: ExperimentConfig) -> ResolvedRun:
    """
    Resuelve defaults + valida exogenas + calcula RUN_NAME y run_dir, sin
    efectos secundarios (no monta Drive, no crea carpetas, no entrena nada).

    Util para que un futuro matrix runner pueda decidir "esta config ya
    existe, la salto" ANTES de invertir tiempo de computo, simplemente
    llamando experiment_exists(config).
    """
    if config.modelo not in MODEL_RUNNERS:
        raise ValueError(
            f"Modelo '{config.modelo}' todavia no esta extraido al pipeline nuevo. "
            f"Disponibles: {list(MODEL_RUNNERS)}"
        )

    defaults = MODEL_DEFAULTS[config.modelo]

    train_hours = config.train_hours if config.train_hours is not None else defaults["train_hours"]
    forecast_horizon = (
        config.forecast_horizon if config.forecast_horizon is not None else defaults["forecast_horizon"]
    )
    optuna_n_trials = (
        config.optuna_n_trials if config.optuna_n_trials is not None else defaults["optuna_n_trials"]
    )
    exogenas = config.exogenas if config.exogenas is not None else list(defaults["exogenas"])

    desconocidas = [e for e in exogenas if e not in defaults["catalogo"]]
    if desconocidas:
        raise ValueError(
            f"El modelo '{config.modelo}' no sabe tratar estas exogenas: {desconocidas}. "
            f"Catalogo permitido: {defaults['catalogo']}"
        )

    run_name = build_run_name(config.modelo, train_hours, forecast_horizon, exogenas, sufijo=config.sufijo_run_name)

    if config.ventana:
        windows.get_ventana(config.ventana)  # valida el id
        if forecast_horizon != windows.HORAS_VENTANA:
            raise ValueError(
                f"Con ventana={config.ventana!r} el horizonte debe ser {windows.HORAS_VENTANA}h "
                f"(recibido {forecast_horizon})."
            )
        run_dir = os.path.join(io_drive.resolve_ventanas_base_dir(config.ventana), run_name)
    else:
        run_dir = os.path.join(io_drive.resolve_base_dir(), run_name)

    return ResolvedRun(
        modelo=config.modelo,
        run_name=run_name,
        run_dir=run_dir,
        train_hours=train_hours,
        forecast_horizon=forecast_horizon,
        optuna_n_trials=optuna_n_trials,
        exogenas=exogenas,
        ventana=config.ventana,
    )


def experiment_exists(config: ExperimentConfig) -> bool:
    """
    True si ya existe una carpeta de resultados para este RUN_NAME.

    Importante: requiere que Google Drive ya este montado en esta sesion de
    Colab (drive.mount ya corrido, por ejemplo en la celda de setup del
    notebook, o pasando mount_drive=True en una llamada previa a
    run_experiment). Si Drive no esta montado, esta funcion no puede ver el
    contenido real de Drive y siempre devuelve False -- un falso negativo,
    no una confirmacion de que el experimento no existe.
    """
    resolved = resolve_run(config)
    return os.path.exists(resolved.run_dir)


class _Tee:
    """Duplica escrituras de texto a varios streams a la vez (usado para que
    todo lo impreso durante el pipeline quede tambien en log.txt, ademas de
    en la salida normal de la celda de Colab)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def _tee_stdout_to_file(path):
    log_file = open(path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        log_file.close()


def _preparar_datos_ventana(data_dir: str, ventana_id: str, train_hours: int, run_dir: str) -> str:
    """
    Materializa los datos hasta el fin de la ventana (ver windows.py) y
    devuelve el directorio que los modelos deben usar como `data_dir`. Deja
    una copia del manifiesto en la carpeta de la corrida.
    """
    destino = os.path.join(windows.carpeta_datos_ventana(ventana_id), f"train{train_hours}h")
    manifest = windows.materializar_datos_ventana(
        data_dir, ventana_id, destino, regiones=REGIONS_ALL, train_hours=train_hours,
    )

    v = windows.get_ventana(ventana_id)
    print(f"Ventana {ventana_id}: prueba {v.test_start} .. {v.test_end} ({windows.HORAS_VENTANA} h)")
    print(f"Datos de la ventana materializados en: {destino}")
    for region, d in manifest["regiones"].items():
        if region in manifest["regiones_omitidas"]:
            continue
        print(
            f"   {region}: ultima fecha {d['ultima_fecha']}, "
            f"{d['horas_previas_validas']:,} h previas validas, "
            f"{d['nan_demanda_ventana']} NaN en la semana de prueba"
        )
    for region, motivo in manifest["regiones_omitidas"].items():
        print(f"   *** REGION OMITIDA {region}: {motivo}")

    with open(os.path.join(run_dir, windows.MANIFEST_NOMBRE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return destino


def run_experiment(
    config: ExperimentConfig,
    data_dir: str = "/content",
    mount_drive: bool = True,
    overwrite: bool = False,
) -> ExperimentResult:
    """
    `overwrite=False` (default): si el RUN_NAME ya existe, aborta con
    FileExistsError -- comportamiento identico al de siempre.

    `overwrite=True`: permite REANUDAR una carpeta EXISTENTE, pero solo si
    `validar_resultado()` confirma que esa carpeta esta incompleta o
    fallida. Si la carpeta ya representa una corrida completa (8 regiones,
    sin NaN en las metricas), se sigue rechazando el sobrescribir aunque
    `overwrite=True` -- este flag es para reanudar corridas interrumpidas,
    nunca para pisar un resultado valido.

    IMPORTANTE (cambio 2026-08-09): reanudar YA NO borra la carpeta ni
    reinicia las 8 regiones desde cero. `run_dir` se conserva tal cual
    (series.csv/metricas.csv/trials.csv/config_usada.csv/log.txt), y es
    cada `MODEL_RUNNERS[modelo]` quien decide, region por region, cuales
    ya estan completas y las salta (`checkpoint.cargar_checkpoint_regiones`,
    invocado dentro de cada `run()`). Antes de este cambio, `overwrite=True`
    hacia `shutil.rmtree(run_dir)` y volvia a correr las 8 regiones
    completas -- en Colab, donde una sesion puede caerse a medio pipeline,
    eso desperdiciaba todo el computo de las regiones que ya habian
    terminado. Ver docs/CHECKPOINT_RESUME.md.
    """
    resolved = resolve_run(config)
    run_name = resolved.run_name
    run_dir = resolved.run_dir
    train_hours = resolved.train_hours
    forecast_horizon = resolved.forecast_horizon
    optuna_n_trials = resolved.optuna_n_trials
    exogenas = resolved.exogenas

    if mount_drive:
        io_drive.mount_drive()

    if os.path.exists(run_dir):
        if not overwrite:
            raise FileExistsError(
                f"Ya existe un experimento con este RUN_NAME:\n{run_dir}\n"
                "No se sobrescribe. Borralo manualmente si de verdad quieres repetirlo, "
                "o llama run_experiment(..., overwrite=True) para reintentar "
                "SOLO si esta carpeta quedo incompleta."
            )

        reporte = validar_resultado(run_dir)
        if reporte.es_completo:
            raise FileExistsError(
                f"El experimento en {run_dir} ya esta completo (todas las regiones, "
                "sin NaN en las metricas). No se sobrescribe ni con overwrite=True. "
                "Borralo manualmente si de verdad quieres repetirlo."
            )

        print(f"Carpeta existente incompleta detectada en {run_dir}:")
        for problema in reporte.problemas:
            print(f"  - {problema}")
        print(
            "overwrite=True: se REANUDA en la misma carpeta (no se borra nada). "
            "El modelo saltara las regiones que ya tengan un resultado completo y "
            "valido, y solo recalculara las que falten o esten incompletas."
        )

    os.makedirs(run_dir, exist_ok=True)

    resolved_cfg = resolved_config_dict(config, run_name, train_hours, forecast_horizon, optuna_n_trials, exogenas)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, ensure_ascii=False)

    # log.txt: todo lo que el pipeline imprime durante esta corrida (mismos
    # prints que ya existian en xgboost_model.py, no se agrego ningun print
    # nuevo alli) mas el traceback completo si la corrida falla. Es lo que
    # permite revisar despues -- o que un agente inspeccione -- que paso en
    # una corrida nocturna sin que nadie haya estado mirando la pantalla.
    log_path = os.path.join(run_dir, "log.txt")

    try:
        with _tee_stdout_to_file(log_path):
            print(f"Experimento: {run_name}")
            print(f"Resultados en: {run_dir}")

            data_dir_modelos = data_dir
            if config.ventana:
                data_dir_modelos = _preparar_datos_ventana(data_dir, config.ventana, train_hours, run_dir)

            exogenas_globales = cargar_exogenas_horarias(data_dir=data_dir_modelos)

            modelo_run = MODEL_RUNNERS[config.modelo]
            series_df, metricas_df, trials_df, config_usada_df = modelo_run(
                exogenas_globales=exogenas_globales,
                regions_all=REGIONS_ALL,
                train_hours=train_hours,
                forecast_horizon=forecast_horizon,
                exog_cols=exogenas,
                optuna_n_trials=optuna_n_trials,
                data_dir=data_dir_modelos,
                output_dir=run_dir,
            )

            # Guardado final (el pipeline ya guardo avances incrementales por
            # region; esto asegura que el csv final refleje exactamente lo
            # que se devolvio). metricas.csv (desempeno) y config_usada.csv
            # (mejores hiperparametros por region) se guardan por separado a
            # proposito: uno describe que tan bien predijo el modelo, el
            # otro con que hiperparametros se logro.
            archivos = {"config": os.path.join(run_dir, "config.json")}

            escribir_csv_atomico(series_df, os.path.join(run_dir, "series.csv"), index=False, encoding="utf-8-sig")
            archivos["series"] = os.path.join(run_dir, "series.csv")

            escribir_csv_atomico(metricas_df, os.path.join(run_dir, "metricas.csv"), index=False, encoding="utf-8-sig")
            archivos["metricas"] = os.path.join(run_dir, "metricas.csv")

            if trials_df is not None and len(trials_df) > 0:
                escribir_csv_atomico(trials_df, os.path.join(run_dir, "trials.csv"), index=False, encoding="utf-8-sig")
                archivos["trials"] = os.path.join(run_dir, "trials.csv")

            if config_usada_df is not None and len(config_usada_df) > 0:
                escribir_csv_atomico(config_usada_df, os.path.join(run_dir, "config_usada.csv"), index=False, encoding="utf-8-sig")
                archivos["config_usada"] = os.path.join(run_dir, "config_usada.csv")

            archivos["log"] = log_path

            # Los modelos tragan el error de cada region para seguir con la siguiente; si TODAS fallan, `run()` devuelve
            # tablas vacias y antes la corrida figuraba como completada. Una corrida sin ninguna region con metricas
            # (ni nuevas ni heredadas del checkpoint) es una corrida fallida. Los CSV ya quedaron guardados.
            if len(metricas_df) == 0:
                sys.stdout.flush()
                resumen = errores_log.resumir(errores_log.extraer_errores(log_path))
                raise RuntimeError(
                    "Ninguna region produjo metricas (0/%d). %s (ver %s)" % (
                        len(REGIONS_ALL), resumen["texto"] or "Sin errores registrados: revisar regiones omitidas o series insuficientes", log_path))

            print("\nExperimento completado.")
            for nombre_archivo, ruta in archivos.items():
                print(f"  {nombre_archivo}: {ruta}")

    except Exception:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("ERROR: la corrida no completo\n")
            f.write("=" * 80 + "\n")
            f.write(traceback.format_exc())
        raise

    mape_promedio = None
    if len(metricas_df) > 0 and "MAPE" in metricas_df.columns:
        mape_promedio = float(metricas_df["MAPE"].mean())

    return ExperimentResult(
        run_name=run_name,
        run_dir=run_dir,
        status="completed",
        regiones_esperadas=len(REGIONS_ALL),
        regiones_con_metricas=len(metricas_df),
        mape_promedio=mape_promedio,
        archivos=archivos,
    )

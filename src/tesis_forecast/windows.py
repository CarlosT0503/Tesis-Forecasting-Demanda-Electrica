"""
Ventanas semanales de prueba (extension experimental a cuatro semanas).

Todos los modelos del proyecto toman como ventana de prueba las ULTIMAS
`forecast_horizon` filas de la serie de demanda de `{REGION}_long.csv` y como
entrenamiento las `train_hours` filas inmediatamente anteriores. La ventana
de prueba, por tanto, la fija el FINAL de los datos, no una fecha dentro del
codigo del modelo.

Para evaluar otra semana sin tocar ningun modelo, este modulo materializa un
directorio de datos "visto desde" el final de la ventana: copia de cada
`{REGION}_long.csv` y `{REGION}_{GEN,IMP,EXP}.csv` con solo las filas con
`fecha <= ultimo dia de la ventana`. Asi ninguna fila posterior al fin de la
prueba existe para el modelo, y la ausencia de fuga de demanda/exogenas
regionales es estructural y no depende de que cada modelo se porte bien.

Las dos exogenas globales (`IGAE_2.xlsx`, `Temperaturas promedio.csv`) se
copian sin cambios: son series mensuales que `data/exogenas.py` ya recorta a
`FECHA_CORTE_FIN = 2026-03-01`, anterior al entrenamiento de las cuatro
ventanas.

No se importa ninguna libreria pesada (sin tensorflow/optuna/xgboost).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .regions import REGIONS_ALL

HORAS_VENTANA = 168

COL_FECHA = "fecha"
COL_HORA = "Hora"
COL_DEMANDA = "Estimacion de Demanda por Balance (MWh)"

ARCHIVOS_GLOBALES = ["IGAE_2.xlsx", "Temperaturas promedio.csv"]
SUFIJOS_EXOG_REGIONAL = ["GEN", "IMP", "EXP"]

MANIFEST_NOMBRE = "ventana_manifest.json"


@dataclass(frozen=True)
class Ventana:
    id: str
    inicio: str  # primer dia de la prueba, 00:00
    fin: str     # ultimo dia de la prueba, 23:00

    @property
    def test_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.inicio)

    @property
    def test_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.fin) + pd.Timedelta(hours=23)


VENTANAS = {
    "S1": Ventana("S1", "2026-04-26", "2026-05-02"),
    "S2": Ventana("S2", "2026-05-03", "2026-05-09"),
    "S3": Ventana("S3", "2026-05-10", "2026-05-16"),
    "S4": Ventana("S4", "2026-05-17", "2026-05-23"),  # semana original de la tesis
}


def _validar_ventanas():
    for v in VENTANAS.values():
        horas = int((v.test_end - v.test_start) / pd.Timedelta(hours=1)) + 1
        if horas != HORAS_VENTANA:
            raise ValueError(f"La ventana {v.id} tiene {horas} horas, se esperaban {HORAS_VENTANA}")


_validar_ventanas()


def get_ventana(ventana_id: str) -> Ventana:
    if ventana_id not in VENTANAS:
        raise ValueError(f"Ventana desconocida: {ventana_id!r}. Disponibles: {sorted(VENTANAS)}")
    return VENTANAS[ventana_id]


def _leer_csv_texto(path: str):
    """Lee un CSV tal cual (todo como texto) para reescribir las filas
    conservadas sin alterar su formato. Devuelve (df, encoding_usado)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            return df, enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {path}")


def _columna(df: pd.DataFrame, nombre: str) -> Optional[str]:
    """Busca una columna sin distinguir mayusculas ni espacios laterales."""
    for c in df.columns:
        if str(c).strip().lower() == nombre.lower():
            return c
    return None


def _truncar_csv(src: str, dst: str, ultimo_dia: pd.Timestamp) -> dict:
    df, enc = _leer_csv_texto(src)
    col_f = _columna(df, COL_FECHA)
    if col_f is None:
        raise ValueError(f"{src} no tiene columna 'fecha'")

    fechas = pd.to_datetime(df[col_f], errors="coerce")
    mask = fechas <= ultimo_dia  # NaT -> False: esas filas los modelos igual las descartan
    salida = df[mask]
    salida.to_csv(dst, index=False, encoding="utf-8")

    fechas_out = fechas[mask]
    return {
        "filas_origen": int(len(df)),
        "filas_destino": int(len(salida)),
        "filas_fecha_invalida": int(fechas.isna().sum()),
        "fecha_min": None if fechas_out.empty else str(fechas_out.min().date()),
        "fecha_max": None if fechas_out.empty else str(fechas_out.max().date()),
    }


def diagnosticar_demanda(df_truncado: pd.DataFrame, ventana: Ventana) -> dict:
    """Cobertura de una region (ya truncada al fin de la ventana)."""
    d = df_truncado.copy()
    d.columns = [str(c).strip() for c in d.columns]
    fecha = pd.to_datetime(d[COL_FECHA], errors="coerce")
    hora = pd.to_numeric(d[COL_HORA], errors="coerce")
    demanda = pd.to_numeric(d[COL_DEMANDA], errors="coerce")

    inicio = ventana.test_start.normalize()
    fin = pd.Timestamp(ventana.fin)

    en_ventana = (fecha >= inicio) & (fecha <= fin)
    previas = fecha < inicio

    pares_ventana = pd.DataFrame({"f": fecha[en_ventana], "h": hora[en_ventana]}).dropna()
    horas_unicas = int(len(pares_ventana.drop_duplicates()))
    dias_completos = bool((pares_ventana.groupby("f")["h"].nunique() == 24).all()) if len(pares_ventana) else False

    ultima = fecha.max()
    return {
        "ultima_fecha": None if pd.isna(ultima) else str(ultima.date()),
        "termina_en_fin_de_ventana": bool((not pd.isna(ultima)) and ultima.normalize() == fin),
        "filas_ventana": int(en_ventana.sum()),
        "horas_unicas_ventana": horas_unicas,
        "ventana_completa": bool(
            int(en_ventana.sum()) == HORAS_VENTANA and horas_unicas == HORAS_VENTANA and dias_completos
        ),
        "nan_demanda_ventana": int(demanda[en_ventana].isna().sum()),
        "filas_previas": int(previas.sum()),
        "horas_previas_validas": int(demanda[previas].notna().sum()),
        "nan_demanda_previas": int(demanda[previas].isna().sum()),
        "filas_duplicadas_fecha_hora": int(pd.DataFrame({"f": fecha, "h": hora}).dropna().duplicated().sum()),
    }


def materializar_datos_ventana(
    data_dir: str,
    ventana_id: str,
    out_dir: str,
    regiones: Optional[list] = None,
    train_hours: Optional[int] = None,
) -> dict:
    """
    Escribe en `out_dir` la vista de los datos hasta el fin de la ventana.

    Una region se OMITE (no se escribe ninguno de sus archivos, y los modelos
    la saltan con el aviso "No encontre ... salto") si:
      - su demanda no termina el ultimo dia de la ventana, o
      - la semana de prueba no tiene sus 168 horas completas, o
      - se pasa `train_hours` y tiene menos horas de historia previa con
        demanda valida que `train_hours`.
    Es preferible omitir a que el modelo evalue en silencio otra semana.

    `out_dir` es un directorio DERIVADO: nunca se escribe en `data_dir`, y de
    una materializacion anterior solo se retiran los archivos que ese mismo
    manifiesto registro.
    """
    ventana = get_ventana(ventana_id)
    data_dir_abs = os.path.abspath(data_dir)
    out_dir_abs = os.path.abspath(out_dir)
    if data_dir_abs == out_dir_abs:
        raise ValueError("out_dir no puede ser igual a data_dir: nunca se escribe sobre los datos originales")

    os.makedirs(out_dir_abs, exist_ok=True)

    manifest_prev = os.path.join(out_dir_abs, MANIFEST_NOMBRE)
    if os.path.exists(manifest_prev):
        with open(manifest_prev, encoding="utf-8") as f:
            previo = json.load(f)
        for nombre in previo.get("archivos_escritos", []):
            ruta = os.path.join(out_dir_abs, nombre)
            if os.path.isfile(ruta):
                os.remove(ruta)

    regiones = list(regiones) if regiones is not None else list(REGIONS_ALL)
    ultimo_dia = pd.Timestamp(ventana.fin)

    manifest = {
        "ventana": ventana.id,
        "test_start": str(ventana.test_start),
        "test_end": str(ventana.test_end),
        "horizonte_horas": HORAS_VENTANA,
        "train_hours_requerido": train_hours,
        "data_dir_origen": data_dir_abs,
        "regiones": {},
        "regiones_omitidas": {},
        "globales": {},
        "archivos_escritos": [],
    }

    for region in regiones:
        src_long = os.path.join(data_dir_abs, f"{region}_long.csv")
        if not os.path.exists(src_long):
            manifest["regiones_omitidas"][region] = "no existe {region}_long.csv".format(region=region)
            continue

        df_long, _ = _leer_csv_texto(src_long)
        col_f = _columna(df_long, COL_FECHA)
        if col_f is None:
            manifest["regiones_omitidas"][region] = "sin columna fecha"
            continue
        trunc = df_long[pd.to_datetime(df_long[col_f], errors="coerce") <= ultimo_dia]
        diag = diagnosticar_demanda(trunc, ventana)

        motivo = None
        if not diag["termina_en_fin_de_ventana"]:
            motivo = f"la demanda termina en {diag['ultima_fecha']}, no en {ventana.fin}"
        elif not diag["ventana_completa"]:
            motivo = (
                f"la semana de prueba no esta completa ({diag['filas_ventana']} filas, "
                f"{diag['horas_unicas_ventana']} horas unicas de {HORAS_VENTANA})"
            )
        elif train_hours is not None and diag["horas_previas_validas"] < train_hours:
            motivo = (
                f"historia insuficiente: {diag['horas_previas_validas']} horas previas validas "
                f"< train_hours={train_hours}"
            )

        manifest["regiones"][region] = diag
        if motivo is not None:
            manifest["regiones_omitidas"][region] = motivo
            continue

        nombre_long = f"{region}_long.csv"
        trunc.to_csv(os.path.join(out_dir_abs, nombre_long), index=False, encoding="utf-8")
        manifest["archivos_escritos"].append(nombre_long)

        for suf in SUFIJOS_EXOG_REGIONAL:
            nombre = f"{region}_{suf}.csv"
            src = os.path.join(data_dir_abs, nombre)
            if not os.path.exists(src):
                continue  # el modelo falla solo si esa exogena esta activa (comportamiento original)
            stats = _truncar_csv(src, os.path.join(out_dir_abs, nombre), ultimo_dia)
            manifest["archivos_escritos"].append(nombre)
            manifest["regiones"][region][f"exog_{suf}"] = stats

    for nombre in ARCHIVOS_GLOBALES:
        src = os.path.join(data_dir_abs, nombre)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Falta el archivo global {src}")
        shutil.copy2(src, os.path.join(out_dir_abs, nombre))
        manifest["archivos_escritos"].append(nombre)
        manifest["globales"][nombre] = "copiado sin cambios"

    with open(manifest_prev, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def carpeta_datos_ventana(ventana_id: str) -> str:
    """Directorio local (rapido) donde se materializan los datos de una ventana."""
    raiz = "/content/ventanas_data" if os.path.isdir("/content") else os.path.join(tempfile.gettempdir(), "ventanas_data")
    return os.path.join(raiz, ventana_id)

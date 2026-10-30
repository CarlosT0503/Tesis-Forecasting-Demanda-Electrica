> **Nota.** Este documento procede del desarrollo del estudio. Menciona un cuaderno anterior («legacy») y `notebooks/run_experiments.ipynb`, que no forman parte de este repositorio; el cuaderno vigente es `notebooks/run_ventanas.ipynb`. Los archivos y los esquemas que describe sí son los que exige la Ruta B. Los datos no se distribuyen (véase `DATOS.md`).

# Datos que el pipeline XGBoost espera

Estos son exactamente los mismos archivos que esperaba la celda 49 del
notebook legacy — el pipeline nuevo no cambia ningun requisito de datos,
solo centraliza donde se buscan (parametro `data_dir`).

Viven permanentemente en Google Drive, en `MyDrive/Bases de datos Tesis`.
En `notebooks/run_experiments.ipynb` esa ruta esta centralizada en la
constante `DATA_DIR = "/content/drive/MyDrive/Bases de datos Tesis"`, que
se pasa a `run_experiment(config, data_dir=DATA_DIR, ...)`. No hace falta
subir nada a mano en cada sesion de Colab.

## 1. Exogenas globales (Temperatura, IGAE, Primarias, Secundarias, Terciarias)

### `IGAE_2.xlsx`
- Se lee la **primera hoja** (`sheet_name=0`).
- Se usan 4 filas por posicion (0-indexado), cada una como una serie de
  valores mensuales en columnas:
  - fila `6` → IGAE nacional
  - fila `7` → Primarias
  - fila `10` → Secundarias
  - fila `15` → Terciarias
- Cada una de esas filas debe tener **al menos 443 valores** (columnas) para
  que los recortes `iloc[118:]` / `iloc[:-10]` no fallen o corten mal. Si el
  archivo cambia de estructura, estos indices posicionales quedan
  desalineados silenciosamente — es un riesgo heredado del notebook legacy,
  no algo que este pipeline corrija.

### `Temperaturas promedio.csv`
- Se lee con `skiprows=2` (las primeras 2 filas del CSV se ignoran).
- Se usa la fila `0` (tras el skip) como serie de valores mensuales,
  descartando la primera columna (`[1:]`, se asume que es una etiqueta, no
  un valor).

## 2. Datos por region (BCA, CEN, NES, NOR, NTE, OCC, ORI, PEN)

Para cada una de las 8 regiones se esperan 4 archivos:

### `{REGION}_long.csv` (ej. `BCA_long.csv`)
Columnas requeridas, **con estos nombres exactos** (sensible a mayusculas):
- `fecha` — parseable como fecha.
- `Hora` — numerica, 1 a 24.
- `Estimacion de Demanda por Balance (MWh)` — la variable objetivo (demanda).

### `{REGION}_GEN.csv`, `{REGION}_IMP.csv`, `{REGION}_EXP.csv`
Generacion, Importacion y Exportacion de esa region. Columnas requeridas
(**no sensible a mayusculas/minusculas**, se normalizan internamente):
- `fecha` (o `Fecha`, `FECHA`, ...) — parseable como fecha.
- `hora` (o `Hora`, `HORA`, ...) — numerica, aceptando 1-24 o 0-23
  (se detecta automaticamente segun el rango de valores presentes).
- `valor` (o `Valor`, `VALOR`, ...) — numerico.

## Resumen — archivos esperados en `DATA_DIR`

```
IGAE_2.xlsx
Temperaturas promedio.csv

BCA_long.csv   BCA_GEN.csv   BCA_IMP.csv   BCA_EXP.csv
CEN_long.csv   CEN_GEN.csv   CEN_IMP.csv   CEN_EXP.csv
NES_long.csv   NES_GEN.csv   NES_IMP.csv   NES_EXP.csv
NOR_long.csv   NOR_GEN.csv   NOR_IMP.csv   NOR_EXP.csv
NTE_long.csv   NTE_GEN.csv   NTE_IMP.csv   NTE_EXP.csv
OCC_long.csv   OCC_GEN.csv   OCC_IMP.csv   OCC_EXP.csv
ORI_long.csv   ORI_GEN.csv   ORI_IMP.csv   ORI_EXP.csv
PEN_long.csv   PEN_GEN.csv   PEN_IMP.csv   PEN_EXP.csv
```

34 archivos en total. Si falta alguno para una region, esa region se salta
con un aviso (`cargar_regiones`) o falla con `FileNotFoundError` si el
archivo `_long.csv` existe pero falta su `_GEN`/`_IMP`/`_EXP` correspondiente
y esa exogena esta activa en la config del experimento.

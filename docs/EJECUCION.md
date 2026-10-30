# Guía de ejecución

Hay dos rutas. La **Ruta A** verifica los resultados guardados sin entrenar; la **Ruta B** repite los experimentos. Antes de usar la Ruta B,
lea [`REPRODUCIBILIDAD.md`](../REPRODUCIBILIDAD.md): explica qué resultados se repetirán de forma exacta y cuáles solo de forma aproximada.

## Convenciones del plan de corridas

`docs/plan_corridas_ventanas.csv` lista las 216 corridas (una por modelo, configuración de variables exógenas y semana). Columnas principales:

- `ventana`: semana de prueba. S1: 26 abr–2 may; S2: 3–9 may; S3: 10–16 may; S4: 17–23 may de 2026 (168 horas cada una).
- `nivel`: **T1** y **T2** son la configuración de la comparación principal de cada modelo (sin variables exógenas, o cinco variables, u ocho en XGBoost y LightGBM);
  **T3** son las seis configuraciones con menos variables (cada variable por separado y temperatura + IGAE).
- `ejecuta_en`: `local` (PC) o `colab`.
- `run_name`: nombre determinista de la corrida (modelo, horas de entrenamiento, horizonte y variables).

## Ruta A: verificación sin entrenar

```bash
python -m venv .venv && source .venv/bin/activate         # en Windows: .venv\Scripts\activate
pip install -r requirements-verificacion.txt
python scripts/verificar_resultados.py
```

Sin más datos es una **verificación parcial**: los valores reales de demanda son datos de CENACE y no se distribuyen. El script comprueba la cobertura, regenera las tablas y las
figuras del Capítulo 6 desde las métricas publicadas y las compara con la referencia. No recalcula las métricas desde las predicciones, y las tres tablas de error que
requieren los valores reales (`error_por_dia`, `error_por_hora`, `error_sesgo`) las toma de la referencia (las tablas y figuras que dependen de ellas sí se regeneran desde esa referencia).
La última línea lo declara: «Verificacion PARCIAL».

Para la **verificación completa** necesita los valores reales (véase la sección 6 de [`DATOS.md`](DATOS.md)):

```bash
python scripts/extraer_valores_reales.py --data-dir datos_locales --salida valores_reales_prueba.csv.gz
python scripts/verificar_resultados.py --reales valores_reales_prueba.csv.gz
```

El script compara primero los valores con las huellas SHA-256 de `docs/huellas_valores_reales.csv` y se detiene si no coinciden. Si coinciden, recalcula las 1 952 métricas desde
las predicciones y regenera todas las tablas y figuras. Los resultados obtenidos al probar ambos modos se registran en `REPRODUCIBILIDAD.md`.

Las figuras dependen de la versión de matplotlib: las de la tesis se generaron con matplotlib 3.11.2 (siete figuras) y 3.10.1 (la figura F6_11).
El script compara pixel a pixel solo las figuras generadas con la misma versión y avisa de las demás; para comprobar las ocho, ejecútelo una vez con cada versión.

## Ruta B: repetir los experimentos

### Requisitos

- Los 34 archivos de datos descritos en [`DATOS.md`](DATOS.md) en `datos_locales/` (o `--data-dir`), con las sumas MD5 verificadas. **No se distribuyen con este repositorio y el procedimiento original para construirlos no está disponible como código**: sin ellos, la Ruta B no puede ejecutarse.
- Dos entornos, porque el estudio usó dos: `requirements-pc.txt` (PC) y `requirements-colab.txt` (Colab, GPU T4). Las versiones y su origen se explican en `REPRODUCIBILIDAD.md`.
- Tiempo de cómputo. Solo hay mediciones de las corridas originales: SARIMAX tardó entre 30 y 45 minutos por región con cuatro procesos en paralelo en la PC del autor;
  el supervisor local no lanza trabajos nuevos con menos de 1.2 GB de memoria libre. No se dispone de una estimación total más precisa.

### PC (modelos de referencia, AR, AR sobre residuo STL, SARIMAX, XGBoost y LightGBM; 104 corridas)

```bash
python -m venv .venv_local && source .venv_local/bin/activate     # en Windows: .venv_local\Scripts\activate
pip install -r requirements-pc.txt
python scripts/run_local_estadisticos.py --data-dir datos_locales --ventanas S4 S1 S2 S3 --niveles T1 T3
```

`--modelos` permite elegir un subconjunto; por omisión se ejecutan los ocho modelos locales. Los resultados se escriben en
`resultados_locales/Pipeline_Resultados_Ventanas/<semana>/<run_name>/`. Cada corrida es reanudable: si se interrumpe, al repetir el comando continúa desde la última región guardada y
no repite lo completo. `scripts/cola_local.py` es un supervisor que lanza y vigila las corridas pendientes (prioridades, límite de procesos y de memoria descritos en su cabecera).

### Colab (FCNN, LSTM directa, LSTM sobre residuo y Ensemble STL; 112 corridas)

1. Coloque los datos en Google Drive en `MyDrive/Bases de datos Tesis` (ruta definida en el cuaderno).
2. Abra `notebooks/run_ventanas.ipynb` en Colab (Archivo → Abrir cuaderno → GitHub, o súbalo) y, en la celda 1, ajuste `REPO_URL` a la dirección de su copia del repositorio.
3. Entorno de ejecución → **GPU (T4)**.
4. La celda 1 instala `optuna` y `lightgbm` **sin fijar versión** (`!pip install -q optuna==5.0.0 lightgbm==4.6.0`). Para usar las versiones que registró el estudio, sustituya esa línea por
   `!pip install -q optuna==5.0.0 lightgbm==4.6.0`.
5. Ejecute la celda 1 (preparación; monta Drive, clona el repositorio, registra versiones y comprueba los datos), la celda 3 (progreso, solo lectura) y la celda 4 (lanzamiento) con `SESION = "A"` (semanas S4 y S1);
   una segunda pestaña puede usar `SESION = "B"` (semanas S2 y S3), nunca la misma sesión en dos pestañas.
   `LOTE` permite lanzar subconjuntos (`"T2"`, `"T3-fcnn"`, `"T3-lstm_resid"`, `"T3-ensemble_stl"`, `"T3-lstm_direct"`).
6. Si la sesión se corta, reinicie la sesión y repita las celdas 1 y 4 con la misma `SESION` y `LOTE`: lo completo se salta y lo parcial continúa desde la última región guardada.

Los resultados quedan en `MyDrive/Pipeline_Resultados_Ventanas/<semana>/<run_name>/`.

### Qué esperar al comparar con `results/`

Cada carpeta de corrida contiene `metricas.csv`, `series.csv`, `trials.csv`, `config_usada.csv` y `config.json`. Compare sus métricas con las filas correspondientes de
`results/consolidado/metricas_largas.csv` (columnas `ventana`, `run_name`, `region`, `modelo_metrica`).
Para los modelos deterministas se espera coincidencia exacta o prácticamente exacta; para XGBoost, LightGBM y, sobre todo, las redes neuronales, **se esperan diferencias**, de magnitud
documentada en `REPRODUCIBILIDAD.md`. Una diferencia dentro de esos órdenes de magnitud no indica un error en la repetición.

## Pruebas automatizadas

```bash
python scripts/ejecutar_pruebas.py
```

Ejecuta cada archivo de `tests/test_*.py` en su propio proceso. Las pruebas no entrenan modelos reales. Las que necesitan TensorFlow, XGBoost, LightGBM u Optuna requieren el entorno de `requirements-colab.txt`
o `requirements-pc.txt`; el resultado obtenido en cada entorno figura en `REPRODUCIBILIDAD.md`.

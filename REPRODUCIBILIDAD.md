# Reproducibilidad

Este documento distingue tres niveles y declara las limitaciones. Todo lo afirmado como verificado fue comprobado entre el 29 y el 30 de octubre de 2026 con los archivos de este repositorio;
lo que no pudo comprobarse se dice expresamente en la sección 7.

## 1. Niveles

| Nivel | Significado |
|---|---|
| **Exacta** | Repetir el procedimiento con las mismas entradas y el mismo entorno debe dar los mismos números (hasta redondeo de máquina). |
| **Con variabilidad** | El procedimiento tiene semilla, pero el resultado depende del entorno o del hardware en una cantidad medida o acotada. |
| **Solo documentable** | No puede repetirse con los números idénticos; se conservan la configuración, los hiperparámetros elegidos y los resultados, y se declara la variabilidad esperada. |

## 2. Ruta A: resultados guardados → tablas, figuras y métricas (sin entrenar)

Los valores reales de demanda de las semanas de prueba son datos de CENACE y **no se incluyen** en este repositorio (véase [`docs/DATOS.md`](docs/DATOS.md)). Por eso la Ruta A tiene dos modos:

| Comprobación | Parcial (por omisión, sin datos externos) | Completa (con `--reales`, ver `docs/DATOS.md`, sección 6) |
|---|---|---|
| Cobertura: 216 corridas y 1 952 conjuntos de métricas | sí: 216 corridas y 1 952 conjuntos de métricas | sí |
| Los valores reales aportados coinciden con las huellas SHA-256 publicadas | no aplica | sí: 40 de 40 series |
| Recálculo de las 1 952 métricas (MAE, RMSE, MAPE, sMAPE) desde `predicciones_prueba.csv.gz` | **no puede hacerse** | sí: 1 952 de 1 952 coinciden; diferencia relativa máxima 2.2 × 10⁻⁷ (tolerancia 10⁻⁶) |
| Tablas del Capítulo 6 regeneradas con los scripts originales | sí: 22 de 22 coinciden (3 de ellas tomadas de la referencia, sin regenerar) | sí: 22 de 22 coinciden, todas regeneradas |
| Figuras del Capítulo 6 | 8 de 8 idénticas pixel a pixel (7 con matplotlib 3.11.2 y 1 con 3.10.1); F6_7 y F6_8 se regeneran desde las tablas de error de la referencia | 8 de 8 idénticas pixel a pixel (7 con matplotlib 3.11.2 y 1 con 3.10.1) |

En el modo parcial, las tablas `error_por_dia`, `error_por_hora` y `error_sesgo` (calculadas con los valores reales) se toman de `results/referencia_tesis/` y no se regeneran; las tablas y figuras que dependen de
ellas se regeneran desde esa referencia. La verificación parcial **no** comprueba que las predicciones publicadas produzcan las métricas publicadas. Las predicciones del modelo Naive no se publican (son una constante igual a un dato de CENACE); en el modo completo se reconstruyen exactamente desde los valores reales aportados y sus 32 conjuntos de métricas se recalculan como los demás. Sin esos valores, las métricas del Naive solo se comparan con la tabla publicada.

Las figuras se generaron con dos versiones de matplotlib (3.11.2 y 3.10.1); el script solo compara pixel a pixel las figuras producidas con la misma versión que la referencia.

## 3. Ruta B: repetir los experimentos, por modelo

| Modelo | Nivel esperado | Razón y evidencia |
|---|---|---|
| Naive, Naive con tendencia, Naive con tendencia y estacionalidad | **Exacta** | Deterministas, sin aleatoriedad. Naive y Naive con tendencia coincidieron exactamente, en las siete regiones de la semana S4, con resultados obtenidos antes con un procedimiento previo. |
| AR, AR sobre residuo STL | **Exacta**, con una restricción de versión | Deterministas (el orden se elige por AIC). Coincidieron exactamente con resultados anteriores, obtenidos con un procedimiento previo, en las siete regiones de S4. **Requieren statsmodels 0.14.x**: usan `AutoReg(..., old_names=False)`, argumento eliminado en statsmodels 0.15, con el que fallan («No se pudo ajustar ningún AR válido»; comprobado con `tests/test_checkpoint.py` y `tests/test_windows.py` en el entorno de Colab). Es un problema de compatibilidad de versiones, no de reproducción: estos dos modelos se ejecutaron en la PC con 0.14.5 (`requirements-pc.txt`), y el AR de Ensemble STL, que sí se ejecutó en Colab con 0.15.0, no pasa ese argumento. |
| SARIMAX | **Exacta** (esperada) | Determinista, orden fijo (1,1,1)(1,0,1)₁₆₈. No se verificó entre equipos distintos; la optimización numérica podría diferir en los últimos decimales. |
| LightGBM | **Con variabilidad** | Semilla 42 en la búsqueda de Optuna y en el modelo. Sus resultados de la semana S4 coincidieron entre dos entornos. Depende de las versiones de las bibliotecas. |
| XGBoost | **Con variabilidad** | Semilla 42, pero con 336 horas de entrenamiento y 168 rezagos los diez intentos de la búsqueda devolvieron un valor infinito en todas las regiones: se usaron los hiperparámetros del primer intento (idénticos en las ocho regiones). Con los mismos datos, configuración e hiperparámetros, dos ejecuciones en entornos distintos difirieron hasta 1.0 pp de MAPE en una región (0.2 pp en el promedio de ocho, configuración de solo generación). No se identificó qué componente del entorno lo causa. |
| LSTM directa | **Con variabilidad** | La búsqueda de Optuna tiene semilla 42, pero el entrenamiento de TensorFlow no es determinista entre ejecuciones ni entre hardware. Auditoría de las búsquedas (`results/trials_optuna.csv.gz`): 223 de 224 completaron los 10 intentos del presupuesto; en una se completaron 19 por una reanudación. |
| FCNN directa, FCNN sobre residuo STL, LSTM sobre residuo STL, Ensemble STL | **Solo documentable** | La búsqueda de hiperparámetros de estas redes **no tiene semilla** (el estudio de Optuna se crea sin muestreador con semilla), y el entrenamiento de TensorFlow no es determinista. Las corridas se ejecutaron en hardware distinto (GPU T4 en un trabajador; CPU en los otros dos). Al recalcular una misma configuración de la FCNN, el MAPE cambió en una mediana de 2.34 pp y un máximo de 18.09 pp (14 comparaciones, 8 con más de 2 pp; `results/variacion_reentrenamiento_FCNN_S3_Temp.csv`). Una corrida reanudada cambia además el estado del generador aleatorio. |

Las diferencias esperadas entre redes son del orden de las medidas anteriores. Una diferencia dentro de ese orden al repetir una red no indica un error.

## 4. Entornos

| | PC | Colab |
|---|---|---|
| Modelos | referencia, AR, AR sobre residuo STL, SARIMAX, XGBoost, LightGBM (104 corridas) | FCNN, LSTM directa, LSTM sobre residuo STL, Ensemble STL (112 corridas) |
| Python | 3.13.1 | 3.13.16 |
| numpy / pandas / scipy | 2.2.3 / 2.2.3 / 1.15.2 | 2.1.3 / 2.2.3 / 1.16.3 |
| statsmodels | **0.14.5** | **0.15.0** |
| scikit-learn / optuna / xgboost | 1.6.1 / 5.0.0 / 3.4.1 | 1.6.1 / 5.0.0 / 3.4.1 |
| lightgbm | 4.7.0 | 4.6.0 |
| tensorflow / keras | — | 2.21.0 / 3.13.2 |
| Hardware | CPU | GPU T4 (un trabajador); CPU (dos trabajadores) |
| Origen del dato | versiones instaladas el 29/10/2026 en el entorno local usado; **no se registraron versiones por corrida** | registradas por el cuaderno en `results/ambiente/ambiente_*.json` (idénticas en los tres trabajadores) |

Los archivos `requirements-*.txt` reflejan estas versiones. La celda de preparación del cuaderno instala `optuna` y `lightgbm` sin fijar versión: para repetir con las registradas,
fije `optuna==5.0.0` y `lightgbm==4.6.0` (véase `docs/EJECUCION.md`).

## 5. Procedencia del código y de los datos

- Cada corrida guarda su semilla, su configuración y la fecha en `results/configuracion_corridas.csv`. Las corridas de la PC guardan además el identificador del código que las produjo (`git_commit`); corresponde al
  repositorio de desarrollo, que no se publica. El código de los modelos de este repositorio es idéntico al de esos identificadores para los modelos no neuronales (comprobado comparando archivos).
- Las corridas de Colab no guardan el identificador por corrida (`git_commit` vacío); el cuaderno registró el del trabajador (`e506e4b`) en `results/ambiente/`, y el código neuronal de este repositorio es idéntico al de esa versión.
- Los tres trabajadores de Colab verificaron los 34 archivos de datos contra las sumas MD5 de `docs/datos_locales_MD5SUMS.txt` antes de ejecutar (34 de 34 coinciden).
- **Los datos no se incluyen**, ni siquiera los valores reales de las semanas de prueba (solo sus huellas SHA-256 en `docs/huellas_valores_reales.csv`). El procedimiento original que transformó los reportes de CENACE en los archivos por región no se conserva como código; véase [`docs/DATOS.md`](docs/DATOS.md).

## 6. Limitaciones que debe conocer quien repita el estudio

1. Dos entornos con versiones distintas (sección 4); no se probó repetir todo en uno solo.
2. Las redes no son reproducibles exactamente (sección 3); la búsqueda de hiperparámetros de tres de ellas carece de semilla.
3. XGBoost: la búsqueda de hiperparámetros no operó con 336 horas de entrenamiento y sus resultados varían con el entorno.
4. El AR y el AR sobre residuo STL no funcionan con statsmodels ≥ 0.15.
5. La construcción de los archivos de datos desde CENACE no es reproducible con código; los datos de CENACE de las semanas de prueba son de liquidación preliminar (versión 0).
6. Importación y exportación solo tienen valor real en 2019 (constantes después); temperatura e IGAE terminan en marzo de 2026 (véase `docs/DATOS.md`).
7. El tiempo total de cómputo de la Ruta B no se midió de forma sistemática.

## 7. Verificaciones realizadas y no realizadas

**Realizadas (29/10/2026).**

- Ruta A, modos parcial y completo (sección 2), con Python 3.13 en Windows, con el entorno de la PC (matplotlib 3.11.2) y con el entorno general (matplotlib 3.10.1). El modo completo se probó con valores reales extraídos con `scripts/extraer_valores_reales.py` de los archivos procesados de la persona autora; no se probó con valores descargados de nuevo desde CENACE.
- Pruebas automatizadas (`python scripts/ejecutar_pruebas.py`), sin entrenar modelos reales: Entorno de la PC (`requirements-pc.txt`, statsmodels 0.14.5): **20 de 20 archivos de prueba pasan**. En `test_distribuido.py` se omite la comparación con resultados locales de referencia, que este repositorio no contiene. Entorno de Colab (`requirements-colab.txt`, statsmodels 0.15.0): **18 de 20 pasan**; fallan `test_checkpoint.py` y `test_windows.py`. Ambos fallos tienen una sola causa, comprobada: ejercitan `ar_model.py`, que llama a `AutoReg(..., old_names=False)`, y statsmodels 0.15.0 lanza `TypeError: unexpected keyword argument 'old_names'` (con 0.14.5 el mismo ajuste funciona y da el mismo AIC en una serie de prueba, y esas dos pruebas pasan). Es un problema de compatibilidad de versiones y no afecta a la reproducción de los resultados publicados: el AR y el AR sobre residuo STL se ejecutaron en la PC con 0.14.5, y el AR de Ensemble STL, que se ejecutó en Colab con 0.15.0, no pasa ese argumento (`test_ensemble_ar_compat.py` pasa en ambos entornos).
- Auditoría de privacidad del contenido de este repositorio (sin correos, rutas privadas, credenciales, nombres de revisores ni identificadores de carpetas).

**No realizadas.**

- La Ruta B no se ejecutó de principio a fin desde este repositorio limpio; no se repitió ninguna de las 216 corridas. Solo se comprobó sin entrenar que los puntos de entrada existen y aceptan los argumentos documentados (`--help`), y las pruebas automatizadas de la lógica de ventanas, reanudación y consolidación.
- No se comprobó el modo completo de la Ruta A con valores reales descargados directamente de CENACE.
- No se comprobó el determinismo de SARIMAX entre equipos distintos ni el de LightGBM más allá de la semana S4.
- No se probó el cuaderno de Colab en una sesión nueva con la versión publicada del repositorio.
- No se reconstruyeron los archivos de datos desde los reportes originales de CENACE.
- No se verificaron las condiciones de redistribución de CENACE ni de CONAGUA-SMN (véase `docs/DATOS.md`).

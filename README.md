# Pronóstico de la demanda eléctrica horaria por región de México: código y resultados verificables

Material complementario de la tesis de maestría **«Series de Tiempo para la predicción de Consumo Eléctrico»**
(Carlos Augusto Torres Ramos, Maestría en Investigación en Ciencia de Datos, Facultad de Ciencias Físico Matemáticas, BUAP).

El estudio compara **13 modelos** (referencia, estadísticos, aprendizaje automático, aprendizaje profundo e híbridos con STL) para
pronosticar **168 horas** de demanda eléctrica en **ocho regiones** del Sistema Eléctrico Nacional, en **cuatro semanas de prueba
consecutivas** (26 de abril al 23 de mayo de 2026). En total son **216 corridas**, **61 configuraciones por semana** y
**1 952 conjuntos de métricas** (MAE, RMSE, MAPE, sMAPE por región, semana y configuración).

Este repositorio contiene el código de los modelos, las configuraciones y los hiperparámetros registrados, las métricas
validadas, las predicciones de las semanas de prueba y las pruebas automatizadas. **No redistribuye datos de terceros** (CENACE, INEGI, SENER, CONAGUA-SMN),
ni siquiera los valores reales de demanda de las semanas de prueba: [`docs/DATOS.md`](docs/DATOS.md) indica de dónde obtenerlos. Tampoco se publican las predicciones del modelo Naive, que son una constante igual a un dato de CENACE. **Lea primero [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md):
indica qué puede reproducirse exactamente, qué presenta variabilidad y qué solo puede documentarse.**

## Dos rutas de reproducción

| Ruta | Qué reproduce | Requiere | Entrena modelos |
|---|---|---|---|
| **A. Verificación desde resultados guardados** | Las tablas y figuras del Capítulo 6 a partir de las métricas publicadas (parcial); además, el recálculo de las 1 952 métricas desde las predicciones si se aportan los valores reales (completa) | Python y bibliotecas básicas (`requirements-verificacion.txt`) | **No** |
| **B. Repetición de los experimentos** | Las 216 corridas, con las configuraciones originales | Datos procesados (ver [`DATOS.md`](docs/DATOS.md)), entorno de la PC o de Colab y horas de cómputo | **Sí** |

**Estado de las pruebas.** La Ruta A (parcial y completa) se probó; la Ruta B **no se ha ejecutado de principio a fin** desde este repositorio. El detalle está en [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md).

### Ruta A (minutos, sin entrenar)

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements-verificacion.txt
python scripts/verificar_resultados.py
```

Sin más datos, el script hace una **verificación parcial**: comprueba la cobertura (216 corridas, 1 952 conjuntos de métricas), regenera con los scripts
originales las tablas y las figuras del Capítulo 6 a partir de las métricas publicadas y las compara con la referencia de la tesis en `results/referencia_tesis/`.
No puede recalcular las métricas desde las predicciones, porque los valores reales de demanda no se distribuyen aquí; tampoco regenera las tablas de error por día, por hora y
de sesgo (las toma de la referencia). Si usted obtiene esos valores de CENACE ([`docs/DATOS.md`](docs/DATOS.md)), `python scripts/verificar_resultados.py --reales <archivo>`
comprueba primero que coinciden con las huellas publicadas y hace la **verificación completa**. Detalle en [`docs/EJECUCION.md`](docs/EJECUCION.md).

### Ruta B (horas de cómputo)

Describe los dos entornos, los comandos y los requisitos en [`docs/EJECUCION.md`](docs/EJECUCION.md). Resumen:

- **PC** (modelos de referencia, AR, SARIMAX, XGBoost, LightGBM; 104 corridas): `scripts/run_local_estadisticos.py` y el supervisor `scripts/cola_local.py`.
- **Colab con GPU** (FCNN, LSTM directa, LSTM sobre residuo y Ensemble STL; 112 corridas): `notebooks/run_ventanas.ipynb`.
- La lista exacta de las 216 corridas está en `docs/plan_corridas_ventanas.csv`; la columna `ejecuta_en` indica dónde se ejecutó cada una.

Los datos procesados **no se incluyen** (ver [`DATOS.md`](docs/DATOS.md)): el procedimiento original que convirtió los reportes de CENACE en los archivos
por región no está disponible como código; se documentan su procedencia y sus sumas de verificación.

## Pruebas automatizadas

```bash
pip install -r requirements-verificacion.txt pytest
python -m pytest tests -q
```

Las pruebas no entrenan modelos reales: usan datos sintéticos y sustitutos (*stubs*) para comprobar la lógica de ventanas, la prevención de fuga
de información, la reanudación de corridas, la consolidación y la generación de tablas. Algunas requieren las bibliotecas de la Ruta B; ver
[`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md) para el resultado obtenido en cada entorno.

## Estructura

```
src/tesis_forecast/    código: modelos (models/), datos, ventanas, métricas, ejecución y consolidación
scripts/               ejecución (PC y aislada), consolidación, auditoría y generación de tablas y figuras del Capítulo 6
notebooks/             cuaderno de Colab (run_ventanas.ipynb) y las celdas que lo generan
tests/                 pruebas automatizadas
docs/                  plan de las 216 corridas, datos requeridos y guía de ejecución
results/               configuraciones, hiperparámetros, métricas, predicciones y referencia de la tesis (sin datos de terceros)
```

## Datos, licencia y cita

- Código bajo licencia MIT (`LICENSE`). Los resultados originales (`results/`) y la documentación se comparten bajo CC BY 4.0 (`LICENSE-RESULTADOS.md`), que no se aplica a datos de terceros.
- Las series originales provienen de CENACE, INEGI, la Secretaría de Energía y CONAGUA-SMN; no se redistribuyen aquí. Véase [`DATOS.md`](docs/DATOS.md).
- Para citar este material use [`CITATION.cff`](CITATION.cff).

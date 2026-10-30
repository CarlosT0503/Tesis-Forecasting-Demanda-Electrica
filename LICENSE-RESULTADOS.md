# Licencia de los resultados originales y de la documentación

Los resultados originales de la autoría de Carlos Augusto Torres Ramos y la documentación de este repositorio se comparten bajo
**Creative Commons Atribución 4.0 Internacional (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/deed.es

## Qué cubre esta licencia

- La documentación: `README.md`, `REPRODUCIBILIDAD.md`, `docs/*.md`, `LICENSE-RESULTADOS.md`.
- Los resultados calculados por la persona autora con sus modelos, que son salidas propias del estudio:
  `results/configuracion_corridas.csv`, `results/hiperparametros_por_corrida.csv`, `results/trials_optuna.csv.gz`,
  `results/predicciones_prueba.csv.gz` (pronósticos de los modelos, sin los del modelo Naive, que repiten un dato de CENACE), `results/variacion_reentrenamiento_FCNN_S3_Temp.csv`,
  `results/consolidado/` (métricas calculadas) y `results/referencia_tesis/` (tablas y figuras de la tesis).
- Los registros de entorno `results/ambiente/`.

## Qué NO cubre

- El código fuente (`src/`, `scripts/`, `tests/`, `notebooks/`) se rige por la licencia MIT (`LICENSE`).
- **Datos de terceros.** Este repositorio no contiene ni redistribuye las series de CENACE, INEGI, SENER ni CONAGUA-SMN, incluidos los valores
  reales de demanda de las semanas de prueba. Esa información pertenece a sus fuentes y se rige por los términos de ellas
  (véase `docs/DATOS.md`). Esta licencia no se aplica a ella. Las huellas SHA-256 de `docs/huellas_valores_reales.csv` y las sumas MD5 de
  `docs/datos_locales_MD5SUMS.txt` son solo códigos de verificación y no permiten reconstruir los datos.
- Las marcas, los nombres de las instituciones y las publicaciones citadas.

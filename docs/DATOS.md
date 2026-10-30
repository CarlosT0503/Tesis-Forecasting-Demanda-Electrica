# Datos: fuentes, preparación, verificación y condiciones de uso

Este repositorio **no redistribuye las series originales**. Documenta de dónde provienen, cómo deben estar organizadas para la Ruta B y cómo
comprobar que se dispone de los mismos archivos que usó el estudio.

## 1. Fuentes

| Variable | Fuente | Resolución |
|---|---|---|
| Demanda eléctrica (variable pronosticada): *Estimación de Demanda por Balance* | CENACE, Sistema de Información del Mercado, reportes diarios «Estimación de la Demanda Real» (https://www.cenace.gob.mx/Paginas/SIM/Reportes/EstimacionDemandaReal.aspx) | horaria, por región |
| Generación, importación y exportación | CENACE, los mismos reportes | horaria, por región |
| Índice Global de Actividad Económica (IGAE) y sus componentes primario, secundario y terciario | INEGI, Sistema de Cuentas Nacionales de México, año base 2018 (https://www.inegi.org.mx/app/tabulados/default.aspx?cno=1&idrt=3251&in=2&opc=p&pr=10&tp=20&vr=1&wr=1) | mensual, nacional |
| Temperatura media | CONAGUA, Servicio Meteorológico Nacional, «Temperatura Media por Entidad Federativa y Nacional» (fila «Nacional») | mensual, nacional |

Regiones: Baja California (BCA), Central (CEN), Noreste (NES), Noroeste (NOR), Norte (NTE), Occidental (OCC), Oriental (ORI) y Peninsular (PEN).
Cada serie horaria abarca del 1 de enero de 2019 al 23 de mayo de 2026 (64 800 registros; 64 799 en BCA). Las semanas de prueba usan la
versión de liquidación 0 de CENACE, de carácter preliminar.

## 2. Archivos esperados por la Ruta B

Los 34 archivos siguientes deben estar en una carpeta `datos_locales/` en la raíz del repositorio (o indicarse con `--data-dir`):

```
IGAE_2.xlsx
Temperaturas promedio.csv
{REGION}_long.csv   {REGION}_GEN.csv   {REGION}_IMP.csv   {REGION}_EXP.csv      (para cada una de las 8 regiones)
```

El esquema columna por columna (nombres exactos, cómo se leen la hoja del IGAE y el archivo de temperatura) está en
[`DATOS_REQUERIDOS.md`](DATOS_REQUERIDOS.md). En resumen, `{REGION}_long.csv` trae `fecha`, `Hora` (1 a 24) y
`Estimacion de Demanda por Balance (MWh)`, y los archivos `_GEN`, `_IMP` y `_EXP` traen `fecha`, `hora` y `valor`.

**Comprobar que los archivos son los del estudio.** `datos_locales_MD5SUMS.txt` contiene la suma MD5 de cada archivo. Desde la carpeta de datos:

```bash
cd datos_locales
md5sum -c ../docs/datos_locales_MD5SUMS.txt        # en Windows: certutil -hashfile <archivo> MD5
```

Los tres trabajadores de Colab que ejecutaron las redes neuronales verificaron los 34 archivos contra esa lista antes de ejecutar
(resultado registrado en `results/ambiente/ambiente_*.json`, campo `datos`: 34 de 34 coinciden).

## 3. Preparación de los datos y lo que NO está disponible

- **Disponible:** el código que lee los archivos por región, alinea las variables exógenas con la demanda, construye las ventanas de
  entrenamiento y de prueba y evita el uso de información posterior a la prueba (`src/tesis_forecast/data/`, `windows.py`, `ventanas_tools.py`).
- **No disponible:** el procedimiento original que convirtió los reportes diarios de CENACE en los archivos `{REGION}_long.csv`,
  `_GEN`, `_IMP` y `_EXP`. No se conserva como código. En su lugar, `scripts/verificar_procedencia_series.py` comprueba, sobre una muestra
  (un día de cada cinco), que las series procesadas coinciden con la versión de mayor liquidación que contiene datos para cada día y que
  se cumple la ecuación `Estimación = Generación + Importación − Exportación + Intercambio neto`. **No es el script original de construcción.**
  Quien desee partir de los reportes originales deberá rehacer esa transformación y comprobar sus resultados contra las sumas de verificación.

## 4. Características de los datos que afectan a los resultados

- Las series nacionales mensuales (temperatura e IGAE) terminan en marzo de 2026 y se mantienen constantes (relleno hacia adelante) en las
  semanas de prueba. Los archivos de importación y exportación solo traen valor real en 2019 y permanecen constantes de 2020 en adelante.
- La hoja del IGAE se lee por posición de fila y columna (véase `DATOS_REQUERIDOS.md`); un cambio de estructura del archivo desalinearía los índices sin avisar.
- La variable pronosticada es una estimación de demanda por balance, no una medición directa.

## 5. Condiciones de redistribución (revisión del 29/10/2026)

| Fuente | Condición encontrada | Decisión |
|---|---|---|
| INEGI | «Términos de Libre Uso de la Información del INEGI»: uso libre citando la fuente; no presentar el producto como oficial del INEGI (https://www.inegi.org.mx/inegi/terminos.html) | No se incluyen los archivos; podrían redistribuirse citando la fuente |
| CENACE | No se encontró una licencia o términos de reutilización del portal del Sistema de Información del Mercado | No se incluyen las series ni los valores reales de demanda de las semanas de prueba (ni siquiera como extracto). Se publican solo huellas SHA-256 (véase la sección 6) |
| CONAGUA-SMN | No se encontró una licencia oficial de reutilización para la temperatura media | No se incluyen los archivos |
| SENER (SIE) | No verificada (solo se usó en el contexto descriptivo de la tesis) | No se incluye |

Quien necesite los datos debe obtenerlos de las fuentes oficiales citadas arriba y respetar sus términos de uso.

## 6. Valores reales de las semanas de prueba (para la verificación completa)

La verificación completa de la Ruta A recalcula las 1 952 métricas desde las predicciones publicadas y los valores reales de demanda de las cuatro semanas de prueba
(8 regiones × 4 semanas × 168 horas = 5 376 valores). Esos valores son datos de CENACE y **no se incluyen**. Para obtenerlos:

1. Descargue de CENACE los reportes «Estimación de la Demanda Real» (enlace de la sección 1) y organícelos en los archivos `{REGION}_long.csv` descritos en
   [`DATOS_REQUERIDOS.md`](DATOS_REQUERIDOS.md). **El procedimiento original que hizo esa transformación no está disponible como código** (sección 3); deberá rehacerlo y
   comprobar el resultado. Las semanas de prueba usan la versión de liquidación 0 (preliminar), que puede haber sido sustituida por versiones posteriores en el portal.
2. Genere el archivo de valores reales:
   `python scripts/extraer_valores_reales.py --data-dir datos_locales --salida valores_reales_prueba.csv.gz`
   (columnas `ventana`, `region`, `fecha`, `real`; la demanda es la columna «Estimacion de Demanda por Balance (MWh)» y la hora `Hora` corresponde a `fecha + (Hora − 1)` horas).
3. Ejecute `python scripts/verificar_resultados.py --reales valores_reales_prueba.csv.gz`. El script compara primero cada una de las 40 series (32 de semana × región y 8 de la hora previa a S1) con las huellas
   SHA-256 de [`huellas_valores_reales.csv`](huellas_valores_reales.csv) y se detiene si alguna difiere, por ejemplo porque el portal ya publicó una versión de liquidación distinta.

Las huellas son códigos de verificación de 64 caracteres: permiten confirmar que dos series son idénticas, pero no reconstruir los valores.

**Predicciones del modelo Naive.** El pronóstico del modelo Naive es una línea constante: repite la última hora observada antes de cada semana de prueba, es decir, un dato de CENACE.
Por eso **las predicciones del modelo Naive no se publican** en `results/predicciones_prueba.csv.gz` (se retiraron las 32 series: ocho regiones por cuatro semanas); el resto de las predicciones sí se publica.
Con los valores reales de este apartado (que incluyen la hora previa a S1, ventana `PREVIA`) el script de verificación **reconstruye exactamente** esas predicciones y recalcula sus métricas.
Sin esos valores, las métricas del Naive solo pueden compararse con la tabla publicada, no recalcularse.

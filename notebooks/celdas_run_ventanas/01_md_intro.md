# Extension a cuatro semanas (S1-S4) - ejecucion en Colab

Ventanas de 168 h: **S1** 26 abr-2 may, **S2** 3-9 may, **S3** 10-16 may, **S4** 17-23 may 2026 (semana original). Cada semana entrena y tunea solo con datos anteriores a su inicio (`windows.py` recorta los datos al fin de la ventana).
Plan completo: **216 corridas (54 por semana)**, ver `docs/plan_corridas_ventanas.csv`.

## Reparto PC / Colab (sin duplicados)
- **PC** (`scripts/run_local_estadisticos.py`): naive, naive_trend, naive_trend_seasonal, ar, ar_resid_trend_seasonal, sarimax, **xgboost y lightgbm** (T1 y T3) = 26 corridas por semana (104).
- **Colab (este cuaderno)**: las 4 redes (fcnn, lstm_resid, lstm_direct, ensemble_stl) en T2 y T3 = 28 corridas por semana (**112**). Esta celda solo lanza lo marcado `ejecuta_en = colab` en el plan.
- Resultados: `Pipeline_Resultados_Ventanas/<S#>/<RUN_NAME>/` (nunca `Pipeline_Resultados/`). `run_matrix` **salta** lo completo y **reanuda** lo incompleto region por region.

## Que ejecutar (las celdas de codigo se numeran 1 a 7, en orden)
1. Entorno de ejecucion: **GPU (T4)**.
2. **Celda 1 - Setup** (autoriza Drive). Si pide reiniciar, hazlo y vuelve a ejecutarla.
3. **Celda 3 - Progreso** (solo lectura): cuantas de las 112 estan completas y cuales faltan. **Celda 2 - Auditoria** es opcional (1 minuto).
4. **Celda 4 - Lanzamiento** con `SESION = "A"` y dejarla corriendo. En una segunda pestana, `SESION = "B"`. **Nunca la misma sesion en dos pestanas.** Si la RAM falla, usa lotes mas chicos (`LOTE`, ver la celda).
5. Mas tarde: **Celdas 5 y 6** (comparar S4 con los originales) y **Celda 7** (consolidado).

**Para reanudar** (corte, desconexion o falta de RAM): *Entorno de ejecucion > Reiniciar sesion*, celda 1 y celda 4 con la misma `SESION`. No hay que borrar nada ni repetir regiones: lo completo se salta y lo parcial continua.

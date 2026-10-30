## Celda 4 - LANZAMIENTO (dejar corriendo)

Cada configuracion se ejecuta en **su propio subproceso**: al terminar, el sistema operativo recupera toda su memoria; si un subproceso muere (RAM, GPU o error) se registra y la cola sigue con la siguiente. Orden: **T2 -> T3**, una semana tras otra.

- **Sesion A**: S4 y S1. **Sesion B**: S2 y S3 (otra pestana). **Nunca la misma sesion en dos pestanas.**
- **Lo completo se salta y lo parcial continua desde la ultima region guardada**: ningun resultado ya obtenido se repite ni se sobrescribe.
- **Reintentos acotados**: hasta `MAX_INTENTOS` por configuracion y se abandona si 2 intentos seguidos no agregan ninguna region. Si 3 configuraciones seguidas fallan por RAM, la cola se detiene (el entorno no alcanza: reinicia la sesion, usa RAM alta o un lote mas chico).
- **Lotes mas chicos** (`LOTE`): `T2` (8 corridas) -> `T3-fcnn`, `T3-lstm_resid`, `T3-ensemble_stl`, `T3-lstm_direct` (12 cada uno). Terminado un lote, *Entorno de ejecucion > Reiniciar sesion*, ejecuta la celda 1 y cambia `LOTE`.
- **Que ver despues**: celda 3 (progreso). Archivos en `Pipeline_Resultados_Ventanas/_auditoria/`: `subprocesos_colab_<A|B>.csv` (cada intento: clase de fallo, regiones antes/despues, pico de RAM), `telemetria_colab_<A|B>.csv` (RAM y GPU cada 2 min), `tiempos_colab_<A|B>.csv` y `faltan_colab.csv`.
- **Si se corta la sesion** (desconexion, 12 h): *Entorno de ejecucion > Reiniciar sesion*, celda 1 y celda 4 con el mismo `SESION` y `LOTE`.

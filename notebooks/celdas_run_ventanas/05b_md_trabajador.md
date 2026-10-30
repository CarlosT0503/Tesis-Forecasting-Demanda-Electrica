## Celda 4-W - LANZAMIENTO DE UN TRABAJADOR (W2, W3 o PRINCIPAL) - en lugar de la celda 4

Corre **solo** las configuraciones de `docs/asignacion/<TRABAJADOR>.csv` (lote exclusivo: nunca las de otro). Cada configuración va en su propio subproceso; **lo completo se salta y lo parcial continúa desde la última región guardada**. Guarda en **tu** Drive (`Pipeline_Resultados_Ventanas/`). No necesita ninguna otra celda adicional: antes de entrenar registra tu entorno (versiones y datos) y, si algo de eso falla, **avisa pero no bloquea**.

1. Cambia `TRABAJADOR` por el tuyo y ejecuta. Deja la pestaña abierta.
2. **No ejecutes la celda 4 (SESION/LOTE)** en un Colab adicional: duplicaría trabajo.
3. **Si se corta la sesión** (desconexión, 12 h): *Entorno de ejecución > Reiniciar sesión*, celda 1 y esta celda otra vez. No repite lo ya hecho.

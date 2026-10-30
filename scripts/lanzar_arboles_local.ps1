# Lanza en la PC XGBoost y LightGBM (T1 y luego T3, semanas S4 S1 S2 S3) con prioridad normal
# (un hilo por proceso). Usa el entorno aislado .venv_local (ver docs/EJECUCION.md).
# Reanudable: si se interrumpe, volver a ejecutarlo continua desde la ultima region guardada.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv_local\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

foreach ($nivel in @("T1", "T3")) {
    $args = "scripts/run_local_estadisticos.py --ventanas S4 S1 S2 S3 --niveles $nivel --modelos xgboost lightgbm"
    $out = Join-Path $repo "resultados_locales\run_arboles_$nivel.log"
    $err = Join-Path $repo "resultados_locales\run_arboles_$nivel.err"
    Start-Process -FilePath $py -ArgumentList $args -Wait -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
}

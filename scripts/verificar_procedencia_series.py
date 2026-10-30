"""
Verificacion (por muestra: 1 de cada 5 dias) de la procedencia de las series regionales a partir de los
reportes diarios de CENACE. Solo lectura. Uso:
    python scripts/verificar_procedencia_series.py <Archivozip.zip> <carpeta_con_long_csv> <prefijo_salida>
Comprueba: (1) que la serie procesada coincide con la version de liquidacion mas alta que contiene datos,
filtrando por (Sistema, Area); (2) la ecuacion Estimacion = Generacion + Importacion - Exportacion + Intercambio neto;
(3) la diferencia entre la liquidacion 0 y la version mas alta. No es el script original de construccion.
"""
import zipfile, re, csv, collections, sys
import numpy as np, pandas as pd
Z = sys.argv[1]  # Archivozip.zip con los reportes diarios de CENACE
DATA = sys.argv[2]  # carpeta con {REGION}_long.csv
z = zipfile.ZipFile(Z)
pat = re.compile(r"Demanda Real Balance_(\d)(?:_v(\d))? Dia Operacion (\d{4}-\d{2}-\d{2}) ")
files = collections.defaultdict(dict)
for n in z.namelist():
    m = pat.match(n); files[m.group(3)][int(m.group(1))] = n
DEM = "Estimacion de Demanda por Balance (MWh)"
def leer(n):
    t = z.read(n).decode("utf-8-sig", "replace").splitlines()
    i = next((k for k, l in enumerate(t) if "Sistema" in l and "Hora" in l), None)
    if i is None: return None
    cols = [c.strip() for c in next(csv.reader([t[i]]))]
    f = [r for r in csv.reader(t[i+1:]) if len(r) == len(cols)]
    d = pd.DataFrame(f, columns=cols)
    d["Sistema"] = d["Sistema"].str.strip(); d["Area"] = d["Area"].str.strip()
    for c in d.columns[2:]: d[c] = pd.to_numeric(d[c].str.strip().replace("---", np.nan), errors="coerce")
    return d
reg = ["BCA", "CEN", "NES", "NOR", "NTE", "OCC", "ORI", "PEN"]
largo = {}
for r in reg:
    d = pd.read_csv(f"{DATA}/{r}_long.csv"); d.columns = [c.strip() for c in d.columns]
    d["fecha"] = pd.to_datetime(d["fecha"]).dt.strftime("%Y-%m-%d")
    largo[r] = {k: g.sort_values("Hora") for k, g in d.groupby("fecha")}
dias = sorted(files)[::5]
filas, bal, difL = [], [], []
for dia in dias:
    L = {v: leer(n) for v, n in files[dia].items()}
    ok = {v: d for v, d in L.items() if d is not None}
    if not ok: continue
    vmax = max(ok)
    for r in reg:
        sis = "BCA" if r == "BCA" else "SIN"
        lg = largo[r][dia]
        coinc = []
        for v, d in ok.items():
            g = d[(d["Sistema"] == sis) & (d["Area"] == r)].sort_values("Hora")
            if len(g) == len(lg) and np.allclose(g[DEM].values, lg[DEM].values, rtol=0, atol=1e-6):
                coinc.append(v)
        filas.append(dict(dia=dia, region=r, nver=len(files[dia]), nlegibles=len(ok), vmax=vmax, coincide_max=vmax in coinc, coinc=",".join(map(str, coinc))))
        # balance sobre la version mas alta
        g = ok[vmax]; g = g[(g["Sistema"] == sis) & (g["Area"] == r)]
        if len(g) == 0:
            bal.append(dict(dia=dia, region=r, formato="vacio", tiene_neto=False, max_abs=np.nan, media_abs=np.nan)); continue
        imp = [c for c in g.columns if c.startswith("Importacion Total") or c.startswith("Flujo Real de Importacion")][0]
        exp = [c for c in g.columns if c.startswith("Exportacion Total") or c.startswith("Flujo Real de Exportacion")][0]
        neto = [c for c in g.columns if c.startswith("Intercambio neto")]
        n = g[neto[0]].fillna(0).values if neto else 0.0
        calc = g["Generacion (MWh)"].values + g[imp].values - g[exp].values + n
        bal.append(dict(dia=dia, region=r, formato="2019" if "Flujo" in imp else "2020+", tiene_neto=bool(neto), max_abs=float(np.nanmax(np.abs(calc - g[DEM].values))), media_abs=float(np.nanmean(np.abs(calc - g[DEM].values)))))
        # L0 vs ultima
        if 0 in ok and vmax != 0:
            a = ok[0]; a = a[(a["Sistema"] == sis) & (a["Area"] == r)].sort_values("Hora")[DEM].values
            b = g.sort_values("Hora")[DEM].values
            if len(a) == len(b): difL.append(dict(dia=dia, region=r, vmax=vmax, mape=float(np.mean(np.abs(a - b) / np.abs(b)) * 100), dif_media_mwh=float(np.mean(np.abs(a - b)))))
pd.DataFrame(filas).to_csv(sys.argv[3] + "_match.csv", index=False)
pd.DataFrame(bal).to_csv(sys.argv[3] + "_balance.csv", index=False)
pd.DataFrame(difL).to_csv(sys.argv[3] + "_L0vsmax.csv", index=False)
print("listo", len(dias), "dias")

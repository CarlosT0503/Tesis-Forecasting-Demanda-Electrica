res = vt.consolidar_ventanas(VENT_DIR, out_dir=os.path.join(VENT_DIR, "Consolidado"))
print(res["inventario"].shape)
res["global_"].sort_values("MAPE_media").head(30)

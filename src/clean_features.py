from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
input_path = processed_dir / "laps_clean.parquet"
output_path = processed_dir / "laps_clean_final.parquet"

# -------------------------
# 2️⃣ Caricamento dati
# -------------------------
df = pd.read_parquet(input_path)
print(f"ℹ️ Caricato {len(df)} righe da {input_path}")

# -------------------------
# 3️⃣ Ridenominazione colonne duplicate o ambigue
# -------------------------
rename_map = {
    "url": "url_generic",
    "url_drv": "url_driver",
    "url_circuit": "url_circuit",
    "number": "driver_number",
    "number_drv": "driver_number_orig",
    "position": "position_lap",
    "position_res": "position_race",
    "time": "time_raw",
    "time_res": "time_race",
    "lap": "lap_raw",
}

# applica solo le colonne esistenti
rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
df.rename(columns=rename_map, inplace=True)

# -------------------------
# 4️⃣ Salvataggio finale
# -------------------------
df.to_parquet(output_path, index=False)
print(f"✅ Dataset finale salvato in {output_path} ({len(df)} righe, {len(df.columns)} colonne)")

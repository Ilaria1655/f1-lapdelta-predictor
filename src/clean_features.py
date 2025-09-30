from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
input_path = processed_dir / "laps_features.parquet"
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

# applica solo se la colonna esiste
rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
df.rename(columns=rename_map, inplace=True)

# -------------------------
# 4️⃣ Mantieni solo colonne necessarie al training
# -------------------------
required_cols = [
    # numeriche
    "LapNumber", "RollingAvgLap", "TyreAge", "DegradationRate",
    # categoriche
    "Driver", "CircuitId", "IsOutLap", "Compound",
    # target
    "LapDelta",
    # per il circuito
    "name"
]

# tiene solo quelle presenti
cols_to_keep = [c for c in required_cols if c in df.columns]
df = df[cols_to_keep]

# -------------------------
# 5️⃣ Salvataggio finale
# -------------------------
df.to_parquet(output_path, index=False)
print(f"✅ Dataset finale salvato in {output_path} "
      f"({len(df)} righe, {len(df.columns)} colonne)")

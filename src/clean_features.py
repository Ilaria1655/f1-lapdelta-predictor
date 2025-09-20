# src/clean_features.py

from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle e file
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / 'data' / 'processed'

input_path = processed_dir / 'laps_features.parquet'
output_path = processed_dir / 'laps_clean.parquet'

# -------------------------
# 2️⃣ Carica dataset con feature
# -------------------------
df = pd.read_parquet(input_path)

# -------------------------
# 3️⃣ Colonne utili per il modello
# -------------------------
columns_keep = [
    'Driver',          # sigla pilota
    'LapNumber',       # numero giro
    'LapTimeSeconds',  # tempo giro
    'RollingAvgLap',   # media mobile ultimi 3 giri
    'TyreAge',         # giri nello stint
    'DegradationRate', # variazione giro precedente
    'IsOutLap',        # True se giro out lap
    'Compound'         # tipo di gomma
]

df_clean = df[columns_keep].copy()

# -------------------------
# 4️⃣ Gestione valori NaN e tipi
# -------------------------
df_clean = df_clean.dropna(subset=['LapTimeSeconds'])  # giri senza tempo

# Riempie NaN con valori sensati
df_clean['RollingAvgLap'] = df_clean['RollingAvgLap'].fillna(df_clean['LapTimeSeconds'])
df_clean['TyreAge'] = df_clean['TyreAge'].fillna(0).astype(int)
df_clean['DegradationRate'] = df_clean['DegradationRate'].fillna(0)
df_clean['IsOutLap'] = df_clean['IsOutLap'].fillna(False).astype(bool)
df_clean['Compound'] = df_clean['Compound'].fillna('Unknown')

# -------------------------
# 5️⃣ Salva dataset pulito
# -------------------------
df_clean.to_parquet(output_path, index=False)
print(f"✅ Dataset pulito salvato in {output_path}")

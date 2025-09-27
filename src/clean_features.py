from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

input_path = processed_dir / 'laps_features.parquet'
output_path = processed_dir / 'laps_clean.parquet'

# -------------------------
# 2️⃣ Caricamento dati
# -------------------------
if not input_path.exists():
    print(f"❌ File {input_path} non trovato, impossibile creare laps_clean.parquet")
    exit(1)

df = pd.read_parquet(input_path)
if df.empty:
    print("❌ laps_features.parquet vuoto, niente da pulire")
    exit(1)

print(f"ℹ️ Caricato {len(df)} righe da laps_features.parquet")

# -------------------------
# 3️⃣ Colonne da mantenere
# -------------------------
columns_keep = [
    'Driver',
    'driverId',         # utile per join/analisi
    'RaceId',           # fondamentale per GroupKFold
    'CircuitId',        # aggiunta circuito
    'LapNumber',
    'LapTimeSeconds',
    'RollingAvgLap',
    'TyreAge',
    'DegradationRate',
    'IsOutLap',
    'Compound',
    'LapDelta'          # target
]

# filtra solo colonne esistenti
columns_keep = [c for c in columns_keep if c in df.columns]
df_clean = df[columns_keep].copy()
print(f"ℹ️ Colonne mantenute: {columns_keep}")

# -------------------------
# 4️⃣ Pulizia righe e tipizzazione
# -------------------------
# rimuovi righe senza target
df_clean = df_clean.dropna(subset=['LapTimeSeconds', 'LapDelta'])
print(f"ℹ️ Righe rimaste dopo rimozione NaN su LapTimeSeconds/LapDelta: {len(df_clean)}")

# fillna coerente e tipizzazione
if 'RollingAvgLap' in df_clean.columns:
    df_clean['RollingAvgLap'] = df_clean['RollingAvgLap'].fillna(df_clean['LapTimeSeconds'])
if 'TyreAge' in df_clean.columns:
    df_clean['TyreAge'] = df_clean['TyreAge'].fillna(0).astype(int)
if 'DegradationRate' in df_clean.columns:
    df_clean['DegradationRate'] = df_clean['DegradationRate'].fillna(0)
if 'IsOutLap' in df_clean.columns:
    df_clean['IsOutLap'] = df_clean['IsOutLap'].fillna(False).astype(bool)
if 'Compound' in df_clean.columns:
    df_clean['Compound'] = df_clean['Compound'].fillna('Unknown')

# -------------------------
# 5️⃣ Salvataggio dataset pulito
# -------------------------
df_clean.to_parquet(output_path, index=False)
print(f"✅ Dataset pulito salvato in {output_path} ({len(df_clean)} righe)")

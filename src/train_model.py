# src/train_model_lap_based.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
import joblib

# -------------------------
# 1️⃣ Setup cartelle e file
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"

laps_path = processed_dir / "laps_processed.parquet"
pit_path = processed_dir / "pit_stops.parquet"
races_path = processed_dir / "races.parquet"
circuits_path = processed_dir / "circuits.parquet"

# -------------------------
# 2️⃣ Carica dati parquet
# -------------------------
laps_df = pd.read_parquet(laps_path)
pit_df = pd.read_parquet(pit_path)
races_df = pd.read_parquet(races_path)
circuits_df = pd.read_parquet(circuits_path)

# -------------------------
# 3️⃣ Costruzione dataset lap-based
# -------------------------
# Aggiungiamo pit count cumulativo per ogni pilota e gara
pit_count = pit_df.groupby(['raceId', 'driverId'])['lap'].count().reset_index()
pit_count.rename(columns={'lap': 'TotalPits'}, inplace=True)

laps_df = laps_df.merge(pit_count, on=['raceId', 'driverId'], how='left')
laps_df['TotalPits'] = laps_df['TotalPits'].fillna(0)

# Filtra righe con valori target mancanti
laps_df = laps_df.dropna(subset=['LapTimeSeconds'])

# -------------------------
# 4️⃣ Definizione feature e target
# -------------------------
NUM_COLS = ['LapNumber', 'TotalPits']
if 'Speed' in laps_df.columns:
    NUM_COLS.append('Speed')

CAT_COLS = ['Driver', 'Circuit', 'Compound']

# 🔹 Assicuriamoci che tutte le colonne categoriche esistano
for col in CAT_COLS:
    if col not in laps_df.columns:
        laps_df[col] = 'Unknown'

X = laps_df[NUM_COLS + CAT_COLS]
y = laps_df['LapTimeSeconds']

# -------------------------
# 5️⃣ Preprocessing
# -------------------------
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preproc = ColumnTransformer([
    ('num', num_pipe, NUM_COLS),
    ('cat', cat_pipe, CAT_COLS)
])

# -------------------------
# 6️⃣ Modello
# -------------------------
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

pipe = Pipeline([
    ('preproc', preproc),
    ('model', model)
])

# -------------------------
# 7️⃣ Addestramento
# -------------------------
pipe.fit(X, y)
print("✅ Modello lap-based addestrato con successo!")

# -------------------------
# 8️⃣ Salvataggio modello
# -------------------------
joblib.dump(pipe, processed_dir / 'lap_model.joblib')
print(f"✅ Modello salvato in {processed_dir / 'lap_model.joblib'}")

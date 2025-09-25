from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
import joblib
import numpy as np

# -------------------------
# Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Caricamento dati
# -------------------------
laps_path = processed_dir / "laps_clean.parquet"
df = pd.read_parquet(laps_path)
if df.empty:
    print(f"⚠️ File vuoto: {laps_path}")
    exit(1)

# -------------------------
# Features & target
# -------------------------
NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate']
CAT_COLS = ['Driver', 'Compound']
TARGET = 'LapDelta'

# Assicurati che tutte le colonne esistano
for col in NUM_COLS + CAT_COLS:
    if col not in df.columns:
        df[col] = 0 if col in NUM_COLS else 'Unknown'

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]

# -------------------------
# Funzione build_pipeline
# -------------------------
def build_pipeline():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preproc = ColumnTransformer([
        ('num', num_pipe, NUM_COLS),
        ('cat', cat_pipe, CAT_COLS)
    ])
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
    return pipe

pipe = build_pipeline()

# -------------------------
# Cross-validation robusta
# -------------------------
groups = df['RaceId'] if 'RaceId' in df.columns else df['Driver']
n_unique_groups = groups.nunique()
if n_unique_groups < 2:
    print(f"⚠️ Troppo pochi gruppi distinti ({n_unique_groups}) per GroupKFold. Salto CV e faccio solo fit.")
    pipe.fit(X, y)
    print("✅ Modello LapDelta addestrato con successo!")
else:
    n_splits = min(5, n_unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error')
    print(f"📊 CV MAE medio: {-np.mean(scores):.4f} ± {np.std(scores):.4f}")
    pipe.fit(X, y)
    print("✅ Modello LapDelta addestrato con successo!")

# -------------------------
# Salvataggio modello
# -------------------------
joblib.dump(pipe, processed_dir / 'lapdelta_model.joblib')
print(f"✅ Modello salvato in {processed_dir / 'lapdelta_model.joblib'}")

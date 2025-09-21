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

root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"

laps_path = processed_dir / "laps_clean.parquet"
df = pd.read_parquet(laps_path)

# Features & target
NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate']
CAT_COLS = ['Driver', 'Compound']
TARGET = 'LapDelta'

# assicurati che colonne esistano
for col in NUM_COLS + CAT_COLS:
    if col not in df.columns:
        if col in NUM_COLS:
            df[col] = 0
        else:
            df[col] = 'Unknown'

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

# -------------------------
# Cross-validation
# -------------------------
pipe = build_pipeline()
if 'RaceId' in df.columns:
    groups = df['RaceId']
    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error')
    print("📊 CV MAE medio (GroupKFold su RaceId):", -np.mean(scores))
else:
    groups = df['Driver']
    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error')
    print("📊 CV MAE medio (GroupKFold su Driver):", -np.mean(scores))

# -------------------------
# Fit finale
# -------------------------
pipe.fit(X, y)
print("✅ Modello LapDelta addestrato con successo!")

# salva
joblib.dump(pipe, processed_dir / 'lapdelta_model.joblib')
print(f"✅ Modello salvato in {processed_dir / 'lapdelta_model.joblib'}")

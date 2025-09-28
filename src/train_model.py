# train_model_realistic_fast.py
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import datetime
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# ------------------------- Setup cartelle
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"

processed_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

laps_path = processed_dir / "laps_clean_final.parquet"
if not laps_path.exists():
    raise FileNotFoundError(f"{laps_path} non trovato!")

df = pd.read_parquet(laps_path)
if df.empty:
    raise ValueError("laps_clean_final.parquet è vuoto")

# ------------------------- Feature engineering
NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate']
CAT_COLS = ['Driver', 'CircuitId', 'IsOutLap', 'Compound']
TARGET = 'LapDelta'

# Feature derivate
df['LapDiffFromRollingAvg'] = df['LapNumber'] - df['RollingAvgLap']
df['TyreEff'] = df['TyreAge'] * df['DegradationRate']
df['LapAgeFactor'] = df['LapNumber'] / (df['TyreAge'] + 1)
df['PrevLapDelta'] = df.groupby(['Driver', 'CircuitId'])[TARGET].shift(1).bfill().fillna(0.0)

NUM_COLS += ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']

# Converti categoriche in tipo 'category'
for col in CAT_COLS:
    df[col] = df[col].astype('category')

# Filtraggio outlier
df = df[(df[TARGET] > 0) & (df[TARGET] < 5)]

# Target scalato
df['LapDeltaScaled'] = df[TARGET] * 10
TARGET_SCALED = 'LapDeltaScaled'

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET_SCALED]

# ------------------------- Split train/validation realistico
train_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42)
X_train, X_val = X.loc[train_idx], X.loc[val_idx]
y_train, y_val = y.loc[train_idx], y.loc[val_idx]

# ------------------------- Funzione obiettivo Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 50, 120),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'subsample': trial.suggest_float('subsample', 0.8, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 40),
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 5.0),
        'random_state': 42,
        'verbose': -1
    }

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CAT_COLS)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, preds)
    return mae

# ------------------------- Ottimizzazione Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15, show_progress_bar=True)
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'l1',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1
})

print(f"✅ Migliori parametri trovati: {best_params}")

# ------------------------- Training finale
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CAT_COLS)
lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS, reference=lgb_train)

model = lgb.train(
    best_params,
    lgb_train,
    num_boost_round=study.best_trial.user_attrs.get("best_iteration", 500),
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)]
)

# ------------------------- Valutazione
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
mae = mean_absolute_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2 = r2_score(y_val, y_pred_val)

print(f"📊 Metriche sul validation set:")
print(f"MAE: {mae/10:.3f} sec")
print(f"RMSE: {rmse/10:.3f} sec")
print(f"R²: {r2:.3f}")

# ------------------------- Salvataggio
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_folder = models_dir / timestamp
model_folder.mkdir(parents=True, exist_ok=True)

model.save_model(model_folder / "lgb_model.txt")

feature_info = {
    "NUM_COLS": NUM_COLS,
    "CAT_COLS": CAT_COLS,
    "TARGET": TARGET,
    "TARGET_SCALED": TARGET_SCALED
}
joblib.dump(feature_info, model_folder / "feature_info.joblib")

print(f"✅ Modello e feature info salvati in {model_folder}")

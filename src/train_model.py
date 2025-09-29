# train_model_realistic_cv.py
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import datetime
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.integration import LightGBMPruningCallback

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

df['LapDiffFromRollingAvg'] = df['LapNumber'] - df['RollingAvgLap']
df['TyreEff'] = df['TyreAge'] * df['DegradationRate']
df['LapAgeFactor'] = df['LapNumber'] / (df['TyreAge'] + 1)

# PrevLapDelta corretto: calcolato prima del fold per evitare leakage
df['PrevLapDelta'] = df.groupby(['Driver', 'CircuitId'])[TARGET].shift(1)

NUM_COLS += ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']

for col in CAT_COLS:
    df[col] = df[col].astype('category')

df = df[(df[TARGET] > 0) & (df[TARGET] < 5)]

df['LapDeltaScaled'] = df[TARGET] * 10
TARGET_SCALED = 'LapDeltaScaled'

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET_SCALED]

df['group'] = df['Driver'].astype(str) + "___" + df['CircuitId'].astype(str)
groups = df['group']

# ------------------------- Funzione obiettivo Optuna (solo su fold 0 per velocità)
def objective(trial, X_train, X_val, y_train, y_val):
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 24, 80),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'min_child_samples': trial.suggest_int('min_child_samples', 40, 120),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'max_bin': trial.suggest_int('max_bin', 255, 511),
        'num_threads': os.cpu_count(),
        'random_state': 42,
        'verbosity': -1
    }

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CAT_COLS, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS, reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            LightGBMPruningCallback(trial, "l1"),
            lgb.log_evaluation(period=0)
        ]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, preds)
    return mae

# ------------------------- Ottimizzazione Optuna (solo sul primo fold)
gkf = GroupKFold(n_splits=3)
folds = list(gkf.split(X, y, groups=groups))

train_idx, val_idx = folds[0]
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Riempi PrevLapDelta con mediana di training per fold usando .loc per evitare warning
X_train.loc[:, 'PrevLapDelta'] = X_train['PrevLapDelta'].fillna(0.0)
X_val.loc[:, 'PrevLapDelta'] = X_val['PrevLapDelta'].fillna(X_train['PrevLapDelta'].median())

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val),
               n_trials=25, show_progress_bar=True)
best_params = study.best_params

best_params.update({
    'objective': 'regression',
    'metric': 'l1',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbosity': -1,
    'num_threads': os.cpu_count()
})

print(f"✅ Migliori parametri trovati (fold 0): {best_params}")

# ------------------------- Training su 3 fold con i migliori parametri
metrics = []
models = []

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_folder = models_dir / timestamp
model_folder.mkdir(parents=True, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"\n🔄 Training fold {fold+1}/3...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Riempi PrevLapDelta per fold usando .loc
    X_train.loc[:, 'PrevLapDelta'] = X_train['PrevLapDelta'].fillna(0.0)
    X_val.loc[:, 'PrevLapDelta'] = X_val['PrevLapDelta'].fillna(X_train['PrevLapDelta'].median())

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CAT_COLS, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS, reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    )

    models.append(model)
    model.save_model(model_folder / f"lgb_model_fold{fold}.txt")

    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)

    metrics.append((mae, rmse, r2))
    print(f"📊 Fold {fold} - MAE: {mae/10:.3f} sec | RMSE: {rmse/10:.3f} sec | R²: {r2:.3f}")

# ------------------------- Media metriche sui fold
mae_mean = np.mean([m[0] for m in metrics])
rmse_mean = np.mean([m[1] for m in metrics])
r2_mean = np.mean([m[2] for m in metrics])

print("\n📈 Metriche medie su 3 fold:")
print(f"MAE: {mae_mean/10:.3f} sec")
print(f"RMSE: {rmse_mean/10:.3f} sec")
print(f"R²: {r2_mean:.3f}")

# ------------------------- Salvataggio info
feature_info = {
    "NUM_COLS": NUM_COLS,
    "CAT_COLS": CAT_COLS,
    "TARGET": TARGET,
    "TARGET_SCALED": TARGET_SCALED,
    "n_folds": 3,
    "model_paths": [str(model_folder / f"lgb_model_fold{i}.txt") for i in range(3)]
}
joblib.dump(feature_info, model_folder / "feature_info.joblib")

print(f"\n✅ Modelli e feature info salvati in {model_folder}")

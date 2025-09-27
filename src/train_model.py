# train_model.py
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import optuna

# ------------------------- 1️⃣ Setup cartelle professionale
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"
outputs_dir = data_dir / "outputs"

processed_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

laps_path = processed_dir / "laps_clean.parquet"
df = pd.read_parquet(laps_path)
if df.empty:
    raise ValueError("laps_clean.parquet vuoto")

# ------------------------- 2️⃣ Feature ingegnerizzate
NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate']
CAT_COLS = ['Driver', 'CircuitId', 'IsOutLap']
TARGET = 'LapDelta'

df['LapDiffFromRollingAvg'] = df['LapNumber'] - df['RollingAvgLap']
df['TyreEff'] = df['TyreAge'] * df['DegradationRate']
df['LapAgeFactor'] = df['LapNumber'] / (df['TyreAge'] + 1)
df['PrevLapDelta'] = df.groupby(['Driver', 'CircuitId'])[TARGET].shift(1).fillna(0.0)
NUM_COLS += ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']

for col in CAT_COLS:
    df[col] = df[col].astype('category')

# Compound
compound_dummies = pd.get_dummies(df['Compound'], prefix='Compound')
df = pd.concat([df, compound_dummies], axis=1)
NUM_COLS += list(compound_dummies.columns)
df['Compound'] = df['Compound'].astype('category')

# Filtraggio outlier
df = df[(df[TARGET] > 0) & (df[TARGET] < 5)]
df['LapDeltaScaled'] = df[TARGET] * 10
TARGET_SCALED = 'LapDeltaScaled'

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET_SCALED]

# ------------------------- 3️⃣ Ottimizzazione iperparametri
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 64, 512),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'random_state': 42,
        'verbose': -1
    }

    df_sample = df.sample(frac=0.05, random_state=42)
    X_sample = df_sample[NUM_COLS + CAT_COLS]
    y_sample = df_sample[TARGET_SCALED]

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X_sample, y_sample, groups=df_sample['Driver']):
        X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]
        break

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CAT_COLS)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
    )

    trial.set_user_attr("best_iteration", model.best_iteration)
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    mae = np.mean(np.abs(preds - y_val))
    return mae

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'l1',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1
})
print(f"✅ Migliori parametri trovati: {best_params}")

# ------------------------- 4️⃣ Training finale K-Fold
gkf = GroupKFold(n_splits=5)
models = []
fold_maes = []

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_folder = models_dir / timestamp
model_folder.mkdir(parents=True, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df['Driver'])):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=CAT_COLS)
    lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS, reference=lgb_train)

    model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=study.best_trial.user_attrs.get("best_iteration", 2000),
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=500)]
    )

    models.append(model)
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    mae_fold = mean_absolute_error(y_val, y_pred_val)
    fold_maes.append(mae_fold)
    print(f"Fold {fold+1} MAE: {mae_fold/10:.3f} sec")

    model_path = model_folder / f"fold{fold+1}.joblib"
    joblib.dump(model, model_path)

# ------------------------- 5️⃣ Salvataggio feature_info + dummy values
feature_info = {
    "NUM_COLS": NUM_COLS,
    "CAT_COLS": CAT_COLS,
    "TARGET": TARGET,
    "TARGET_SCALED": TARGET_SCALED,
    "trained_compound_columns": list(compound_dummies.columns)
}
joblib.dump(feature_info, model_folder / "feature_info.joblib")

# Salva i dummy già calcolati
compound_values = df[list(compound_dummies.columns)]
compound_values.to_parquet(model_folder / "compound_dummies.parquet", index=False)  # per analyze_results.py
np.save(model_folder / "compound_values.npy", compound_values.to_numpy(dtype=np.float32))  # per calcoli veloci

# ------------------------- 6️⃣ Stampa veloce metriche MAE medie sui fold
mean_mae = np.mean(fold_maes)/10  # rimappa in secondi
print(f"\n📊 MAE medio sui {len(fold_maes)} fold: {mean_mae:.3f} sec")

print(f"✅ Tutti i modelli, feature_info e compound values salvati in {model_folder}")
print(f"✅ Cartelle dedicate create: processed/, models/{timestamp}/, outputs/")

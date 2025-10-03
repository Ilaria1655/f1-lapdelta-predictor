import os
import datetime
import joblib
import warnings
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import early_stopping, log_evaluation
from tqdm import tqdm

# ------------------------- Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

# ------------------------- Silenzia warning inutili
warnings.filterwarnings("ignore", category=UserWarning)

class SilentLogger:
    def info(self, msg):
        pass
    def warning(self, msg):
        pass

lgb.register_logger(SilentLogger())

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

# ------------------------- Caricamento dati
logger.info("Caricamento e preparazione dati...")
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

# PrevLapDelta calcolato per evitare leakage
df['PrevLapDelta'] = df.groupby(['Driver', 'CircuitId'])[TARGET].shift(1)

NUM_COLS += ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']

for col in CAT_COLS:
    df[col] = df[col].astype('category')

# Filtro outlier
df = df[(df[TARGET] > 0) & (df[TARGET] < 5)]

# Target scalato per maggiore stabilità
df['LapDeltaScaled'] = df[TARGET] * 10
TARGET_SCALED = 'LapDeltaScaled'

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET_SCALED]

df['group'] = df['Driver'].astype(str) + "___" + df['CircuitId'].astype(str)
groups = df['group']

# ------------------------- Creazione test set separato (30%)
X_trainval, X_test, y_trainval, y_test, groups_trainval, groups_test = train_test_split(
    X, y, groups, test_size=0.3, random_state=42, stratify=groups
)

# Salvataggio test set per il testing finale
joblib.dump(X_test, processed_dir / "X_test.joblib")
joblib.dump(y_test, processed_dir / "y_test.joblib")

logger.info(f"Test set salvato: {X_test.shape[0]} esempi")

# Sostituisco X, y, groups con la parte train/validation
X, y, groups = X_trainval, y_trainval, groups_trainval

# ------------------------- Parametri fissi LightGBM
BEST_PARAMS = {
    'objective': 'regression',
    'metric': 'l1',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 48,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'min_child_samples': 60,
    'reg_alpha': 3.0,
    'reg_lambda': 3.0,
    'max_bin': 400,
    'random_state': 42,
    'verbosity': -1,
    'num_threads': os.cpu_count()
}

# ------------------------- Main training
def main():
    gkf = GroupKFold(n_splits=3)
    folds = list(gkf.split(X, y, groups=groups))

    metrics = []
    models = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = models_dir / timestamp
    model_folder.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="Training Folds", ncols=100)):
        logger.info(f"Training fold {fold+1}/3...")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fill PrevLapDelta
        X_train.loc[:, 'PrevLapDelta'] = X_train['PrevLapDelta'].fillna(0.0)
        X_val.loc[:, 'PrevLapDelta'] = X_val['PrevLapDelta'].fillna(X_train['PrevLapDelta'].median())

        model = lgb.LGBMRegressor(**BEST_PARAMS, n_estimators=2000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(period=0)
            ]
        )

        models.append(model)
        model.booster_.save_model(str(model_folder / f"lgb_model_fold{fold}.txt"))

        # 🔹 Salvataggio fold di validazione
        joblib.dump(X_val, model_folder / f"X_val_fold{fold}.joblib")
        joblib.dump(y_val, model_folder / f"y_val_fold{fold}.joblib")

        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration_)
        mae = mean_absolute_error(y_val, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        r2 = r2_score(y_val, y_pred_val)

        metrics.append((mae, rmse, r2))
        logger.info(f"Fold {fold+1} - MAE: {mae/10:.3f} sec | RMSE: {rmse/10:.3f} sec | R²: {r2:.3f}")

    # ------------------------- Media metriche
    mae_mean = np.mean([m[0] for m in metrics])
    rmse_mean = np.mean([m[1] for m in metrics])
    r2_mean = np.mean([m[2] for m in metrics])

    logger.info("\nMetriche medie su 3 fold:")
    logger.info(f"MAE: {mae_mean/10:.3f} sec")
    logger.info(f"RMSE: {rmse_mean/10:.3f} sec")
    logger.info(f"R²: {r2_mean:.3f}")

    # 🔹 Debug per verificare stabilità MAE
    logger.info("Controllo stabilità MAE (run ripetibili):")
    for i, (mae, _, _) in enumerate(metrics):
        logger.info(f"Fold {i+1} MAE (scaled): {mae:.5f}")

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

    logger.info(f"Modelli e feature info salvati in {model_folder}")


if __name__ == "__main__":
    main()

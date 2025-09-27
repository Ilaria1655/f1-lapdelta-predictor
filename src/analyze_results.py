# analyze_results.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------------------- 1️⃣ Setup cartelle
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"
outputs_dir = data_dir / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

# ------------------------- 2️⃣ Carica dataset
laps_path = processed_dir / "laps_clean.parquet"
df = pd.read_parquet(laps_path)
if df.empty:
    raise ValueError("laps_clean.parquet vuoto")

# ------------------------- 3️⃣ Trova ultima cartella modelli
model_folders = sorted([f for f in models_dir.iterdir() if f.is_dir()])
if not model_folders:
    raise FileNotFoundError("Nessun modello trovato in data/models/")
latest_model_folder = model_folders[-1]
print(f"📁 Ultima cartella modelli: {latest_model_folder}")

# ------------------------- 4️⃣ Carica feature_info e dummy compound
feature_info = joblib.load(latest_model_folder / "feature_info.joblib")
NUM_COLS = feature_info["NUM_COLS"]
CAT_COLS = feature_info["CAT_COLS"]
TARGET = feature_info["TARGET"]
TARGET_SCALED = feature_info["TARGET_SCALED"]

compound_dummies = pd.read_parquet(latest_model_folder / "compound_dummies.parquet")
df = pd.concat([df.reset_index(drop=True), compound_dummies.reset_index(drop=True)], axis=1)
NUM_COLS += list(compound_dummies.columns)

# ------------------------- 5️⃣ Prepara X e y
X = df[NUM_COLS + CAT_COLS].copy()
y_true = df[TARGET].copy()

# Converti categoriche in int per predizioni LightGBM
for col in CAT_COLS:
    X[col] = X[col].astype('category').cat.codes

# ------------------------- 6️⃣ Carica modelli K-Fold
model_files = sorted(latest_model_folder.glob("fold*.joblib"))
if not model_files:
    raise FileNotFoundError(f"Nessun modello K-Fold trovato in {latest_model_folder}")
models = [joblib.load(f) for f in model_files]
print(f"📦 Caricati {len(models)} modelli K-Fold")

# ------------------------- 7️⃣ Predizioni medie K-Fold (parallelo)
def predict_model(m):
    return m.predict(X, num_iteration=m.best_iteration)

print("⏳ Calcolo predizioni medie K-Fold...")
preds_list = Parallel(n_jobs=-1)(
    delayed(predict_model)(m) for m in tqdm(models, desc="Predizioni fold", unit="modello")
)
df['LapDeltaPred'] = np.mean(preds_list, axis=0) / 10  # rimappa in secondi

# ------------------------- 8️⃣ Metriche globali
mae = mean_absolute_error(y_true, df['LapDeltaPred'])
rmse = np.sqrt(mean_squared_error(y_true, df['LapDeltaPred']))
r2 = r2_score(y_true, df['LapDeltaPred'])

metrics_text = (
    f"📊 Metriche sul dataset completo (modello: {latest_model_folder.name}):\n"
    f"   MAE  : {mae:.3f} sec\n"
    f"   RMSE : {rmse:.3f} sec\n"
    f"   R²   : {r2:.3f}"
)
print(metrics_text)
(metrics_path := outputs_dir / f"metrics_{latest_model_folder.name}.txt").write_text(metrics_text, encoding="utf-8")
print(f"✅ Metriche salvate in {metrics_path}")

# ------------------------- 9️⃣ Salva predizioni per eventuale analisi
preds_path = outputs_dir / f"predictions_{latest_model_folder.name}.parquet"
df.to_parquet(preds_path, index=False)
print(f"✅ Predizioni salvate in {preds_path}")

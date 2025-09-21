from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from train_model import build_pipeline  # usa la pipeline del modello
from features import add_features
from data_prep import save_all_csvs, save_pit_stops

# -----------------------
# Setup cartelle
# -----------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
output_dir = root_dir / "outputs"
output_dir.mkdir(exist_ok=True)

laps_path = processed_dir / "laps_clean.parquet"
pit_path = processed_dir / "pit_stops.parquet"
model_path = processed_dir / "lapdelta_model.joblib"

# -----------------------
# Step 1: Prepara CSV e pit
# -----------------------
save_all_csvs()
save_pit_stops()

# -----------------------
# Step 2: Carica dataset
# -----------------------
df = pd.read_parquet(laps_path)
pit_df = pd.read_parquet(pit_path)

# -----------------------
# Step 3: Addestra modello
# -----------------------
X = df[['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate', 'Driver', 'Compound']]
y = df['LapDelta']

pipe = build_pipeline()
pipe.fit(X, y)

joblib.dump(pipe, model_path)
print(f"✅ Modello addestrato e salvato in {model_path}")

# -----------------------
# Step 4: Predizioni
# -----------------------
y_pred = pipe.predict(X)
errors = y_pred - y

# -----------------------
# Step 5: Metriche
# -----------------------
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\n📊 Metriche modello LapDelta:")
print(f"   MAE  : {mae:.3f} sec")
print(f"   RMSE : {rmse:.3f} sec")
print(f"   R²   : {r2:.3f}")

# -----------------------
# Step 6: Grafici
# -----------------------

# Pred vs Real
plt.figure(figsize=(6,6))
sns.scatterplot(x=y, y=y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Ideale")
plt.xlabel("LapDelta Reale [s]")
plt.ylabel("LapDelta Predetto [s]")
plt.title("Predetto vs Reale")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "pred_vs_real.png", dpi=150)

# Distribuzione errori
plt.figure(figsize=(6,4))
sns.histplot(errors, bins=50, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Errore (Pred - Real) [s]")
plt.ylabel("Frequenza")
plt.title("Distribuzione degli errori")
plt.tight_layout()
plt.savefig(output_dir / "error_distribution.png", dpi=150)

# Boxplot per driver
df_eval = df.copy()
df_eval["Error"] = errors
plt.figure(figsize=(10,5))
sns.boxplot(x="Driver", y="Error", data=df_eval)
plt.axhline(0, color="red", linestyle="--")
plt.title("Distribuzione errore per pilota")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(output_dir / "error_by_driver.png", dpi=150)

print(f"\n📈 Grafici salvati in {output_dir}")

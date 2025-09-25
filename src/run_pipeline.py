from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------
# Setup cartelle
# -----------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
output_dir = root_dir / "outputs"
output_dir.mkdir(exist_ok=True)

laps_path = processed_dir / "laps_clean.parquet"
model_path = processed_dir / "lapdelta_model.joblib"
report_path = output_dir / "lapdelta_report.pdf"

# -----------------------
# Caricamento dati e modello
# -----------------------
df = pd.read_parquet(laps_path)
model = joblib.load(model_path)
print(f"✅ Modello caricato da {model_path}")

# -----------------------
# Colonne coerenti con il modello
# -----------------------
NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate', 'Stint']
CAT_COLS = ['Driver', 'Compound', 'CircuitId', 'IsOutLap']
TARGET = 'LapDelta'

# Assicurati che tutte le colonne esistano
for col in NUM_COLS + CAT_COLS:
    if col not in df.columns:
        df[col] = 0 if col in NUM_COLS else 'Unknown'

X = df[NUM_COLS + CAT_COLS]
y_true = df[TARGET]

# -----------------------
# Predizioni
# -----------------------
y_pred = model.predict(X)
errors = y_pred - y_true

# -----------------------
# Metriche
# -----------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📊 Metriche modello LapDelta:")
print(f"   MAE  : {mae:.3f} sec")
print(f"   RMSE : {rmse:.3f} sec")
print(f"   R²   : {r2:.3f}")

# -----------------------
# Configurazioni grafici
# -----------------------
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

with PdfPages(report_path) as pdf:

    # 1️⃣ Pred vs Real
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Ideale")
    ax.set_xlabel("LapDelta Reale [s]")
    ax.set_ylabel("LapDelta Predetto [s]")
    ax.set_title("Predetto vs Reale")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    fig.savefig(output_dir / "pred_vs_real.png", dpi=150)

    # 2️⃣ Distribuzione errori
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(errors, bins=50, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Errore (Pred - Real) [s]")
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione degli errori")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    fig.savefig(output_dir / "error_distribution.png", dpi=150)

    # 3️⃣ Boxplot per driver (top 20)
    df_eval = df.copy()
    df_eval["Error"] = errors
    top_drivers = df['Driver'].value_counts().index[:20].tolist()
    fig, ax = plt.subplots(figsize=(12,5))
    sns.boxplot(x="Driver", y="Error", data=df_eval[df_eval['Driver'].isin(top_drivers)], ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Distribuzione errore per pilota (top 20)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    fig.savefig(output_dir / "error_by_driver.png", dpi=150)

    # 4️⃣ Feature importance (se LGBM presente nella pipeline)
    try:
        model_lgbm = model.named_steps['model']
        importances = model_lgbm.feature_importances_
        preproc = model.named_steps['preproc']
        num_features = NUM_COLS
        cat_pipe = preproc.named_transformers_['cat']
        ohe = cat_pipe.named_steps['ohe']
        cat_features = list(ohe.get_feature_names_out(CAT_COLS))
        feature_names = num_features + cat_features
        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances[:len(feature_names)]})
        feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp.head(20), ax=ax)
        ax.set_title("Feature importance (LGBM)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig.savefig(output_dir / "feature_importance.png", dpi=150)
    except Exception as e:
        print("⚠️ Impossibile calcolare feature importance:", e)

print(f"\n📈 Grafici salvati in {output_dir}")
print(f"📄 PDF report generato in {report_path}")

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
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
report_path = output_dir / "report_analysis.pdf"

# -----------------------
# Caricamento dati e modello
# -----------------------
df = pd.read_parquet(laps_path)
model = joblib.load(model_path)

NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate']
CAT_COLS = ['Driver', 'Compound']
TARGET = 'LapDelta'

X = df[NUM_COLS + CAT_COLS]
y_true = df[TARGET]
y_pred = model.predict(X)

# -----------------------
# Metriche
# -----------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📊 Metriche valutazione:")
print(f"   MAE  : {mae:.3f} sec")
print(f"   RMSE : {rmse:.3f} sec")
print(f"   R²   : {r2:.3f}")

# set style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# inizializza PDF multipagina
with PdfPages(report_path) as pdf:
    # Pred vs Real
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, ax=ax)
    minv, maxv = y_true.min(), y_true.max()
    ax.plot([minv, maxv], [minv, maxv], 'r--', lw=2, label="Ideale")
    ax.set_xlabel("LapDelta Reale [s]")
    ax.set_ylabel("LapDelta Predetto [s]")
    ax.set_title("Predetto vs Reale")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Distribuzione errori
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(errors, bins=60, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Errore (Pred - Real) [s]")
    ax.set_title("Distribuzione degli errori")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Boxplot per driver (solo top N per leggibilità)
    top_drivers = df['Driver'].value_counts().index[:20].tolist()
    df_eval = df.copy()
    df_eval["Error"] = errors
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x="Driver", y="Error", data=df_eval[df_eval['Driver'].isin(top_drivers)], ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Distribuzione errore per pilota (top 20 per numero di giri)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Feature importance (se LightGBM presente nella pipeline)
    try:
        model_lgbm = model.named_steps['model']
        importances = model_lgbm.feature_importances_
        # prova a estrarre nomi dalle trasformazioni
        preproc = model.named_steps['preproc']
        num_features = NUM_COLS
        cat_pipe = preproc.named_transformers_['cat']
        ohe = cat_pipe.named_steps['ohe']
        try:
            cat_features = list(ohe.get_feature_names_out(CAT_COLS))
        except Exception:
            cat_features = []  # fallback
        feature_names = num_features + cat_features
        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances[:len(feature_names)]})
        feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp.head(20), ax=ax)
        ax.set_title("Feature importance (LGBM)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    except Exception as e:
        print("⚠️ Impossibile ottenere feature importance:", e)

    # SHAP summary (opzionale, se shap installato e piccolo sample)
    try:
        import shap
        # usa solo un campione per velocità
        X_sample = X.sample(min(2000, len(X)), random_state=42)
        explainer = shap.Explainer(model_lgbm)
        shap_vals = explainer(X_sample.iloc[:, :len(NUM_COLS)])  # potrebbe variare con pipeline; try/except safe-guard
        fig = shap.plots.beeswarm(shap_vals, show=False)
        # shap.plots returns a matplotlib figure only in some versions; fallback
        pdf.savefig(bbox_inches='tight')
        plt.close('all')
    except Exception:
        pass

print(f"✅ Grafici salvati in {output_dir} e report PDF in {report_path}")

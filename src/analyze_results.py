from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate', 'Stint']
CAT_COLS = ['Driver', 'Compound', 'CircuitId', 'IsOutLap']
TARGET = 'LapDelta'

# colonne mancanti
for col in NUM_COLS:
    if col not in df.columns:
        df[col] = 0.0
for col in CAT_COLS:
    if col not in df.columns:
        df[col] = 'Unknown'

# converti booleani in stringhe per OneHotEncoder
if 'IsOutLap' in df.columns:
    df['IsOutLap'] = df['IsOutLap'].astype(str)

X = df[NUM_COLS + CAT_COLS]
y_true = df[TARGET]
y_pred = model.predict(X)

# -----------------------
# Metriche globali
# -----------------------
errors = y_pred - y_true
df['Error'] = errors

mae_global = np.mean(np.abs(errors))
rmse_global = np.sqrt(np.mean(errors**2))

print(f"\n📊 Metriche valutazione globale:")
print(f"   MAE  : {mae_global:.3f} sec")
print(f"   RMSE : {rmse_global:.3f} sec")

# -----------------------
# Grafici e PDF report
# -----------------------
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

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
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(errors, bins=60, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Errore (Pred - Real) [s]")
    ax.set_title("Distribuzione degli errori")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Boxplot driver
    top_drivers = df['Driver'].value_counts().index[:20]
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x="Driver", y="Error", data=df[df['Driver'].isin(top_drivers)], ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Distribuzione errore per pilota (top 20)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Boxplot circuito
    top_circuits = df['CircuitId'].value_counts().index[:20]
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x="CircuitId", y="Error", data=df[df['CircuitId'].isin(top_circuits)], ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Distribuzione errore per circuito (top 20)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Ranking driver per MAE
    driver_stats = df.groupby('Driver').agg(
        MAE=('Error', lambda x: np.mean(np.abs(x))),
        RMSE=('Error', lambda x: np.sqrt(np.mean(x**2))),
        N=('Error', 'count')
    ).sort_values(by='MAE')
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=driver_stats.reset_index().values,
                     colLabels=driver_stats.reset_index().columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    ax.set_title("Tabella driver: MAE, RMSE, numero giri")
    pdf.savefig(fig)
    plt.close(fig)

    # Ranking circuiti per MAE
    circuit_stats = df.groupby('CircuitId').agg(
        MAE=('Error', lambda x: np.mean(np.abs(x))),
        RMSE=('Error', lambda x: np.sqrt(np.mean(x**2))),
        N=('Error', 'count')
    ).sort_values(by='MAE')
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=circuit_stats.reset_index().values,
                     colLabels=circuit_stats.reset_index().columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    ax.set_title("Tabella circuiti: MAE, RMSE, numero giri")
    pdf.savefig(fig)
    plt.close(fig)

    # Feature importance
    try:
        model_lgbm = model.named_steps['model']
        importances = model_lgbm.feature_importances_
        preproc = model.named_steps['preproc']
        num_features = NUM_COLS
        cat_pipe = preproc.named_transformers_['cat']
        ohe = cat_pipe.named_steps['ohe']
        try:
            cat_features = list(ohe.get_feature_names_out(CAT_COLS))
        except Exception:
            cat_features = []
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

print(f"✅ Analisi completata. Report PDF salvato in {report_path}")

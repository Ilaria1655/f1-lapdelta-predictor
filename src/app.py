import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ------------------- Percorsi
BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"

# ------------------- Caricamento ultimo modello
latest_model_folder = max(models_dir.iterdir(), key=lambda d: d.stat().st_mtime)
feature_info_path = latest_model_folder / "feature_info.joblib"
feature_info = joblib.load(feature_info_path)
model_paths = feature_info["model_paths"]
models = [lgb.Booster(model_file=path) for path in model_paths]

# ------------------- Caricamento dataset storico
laps_path = processed_dir / "laps_clean_final.parquet"
laps_df = pd.read_parquet(laps_path)

# ------------------- Caricamento fold di validazione reali
fold_val_path = latest_model_folder / "X_val_fold0.joblib"
y_val_path = latest_model_folder / "y_val_fold0.joblib"
X_val_fold0 = joblib.load(fold_val_path)
y_val_fold0 = joblib.load(y_val_path)

# ------------------- Streamlit Page
st.set_page_config(page_title="🏎️ F1 LapDelta Predictor 2024", layout="wide")
st.title("🏎️ F1 LapDelta Predictor 2024")
st.markdown("Predizione LapDelta con dashboard professionale e confronto tra piloti.")

# ------------------- Sidebar Input
st.sidebar.header("Input Giro")
drivers = sorted(laps_df['Driver'].unique())
circuits = sorted(laps_df['name'].unique())
compounds = sorted(laps_df['Compound'].unique())

driver_name = st.sidebar.selectbox("Driver principale", drivers)
other_drivers = st.sidebar.multiselect("Confronta con altri piloti", drivers, default=[driver_name])
circuit_name = st.sidebar.selectbox("Circuito", circuits)
lap_number = st.sidebar.slider("Numero del giro", 1, 80, 1)
tyre_age = st.sidebar.slider("Età pneumatico", 0, 50, 0)
compound = st.sidebar.selectbox("Mescola pneumatico", compounds)
is_out_lap = st.sidebar.checkbox("Out Lap?")

# ------------------- Mappatura codice interno per il modello
driver_code = laps_df[laps_df['Driver'] == driver_name]['Driver'].iloc[0]
circuit_code = laps_df[laps_df['name'] == circuit_name]['CircuitId'].iloc[0]
other_driver_codes = [laps_df[laps_df['Driver'] == d]['Driver'].iloc[0] for d in other_drivers]

# ------------------- Funzione Feature Derivate
def compute_features(df, driver_code, circuit_code, lap_number, tyre_age, compound, is_out_lap):
    df_sub = df[(df['Driver'] == driver_code) & (df['CircuitId'] == circuit_code)].sort_values('LapNumber')
    prev_lap_delta = df_sub[df_sub['LapNumber'] < lap_number]['LapDelta'].iloc[-1] if not df_sub.empty and lap_number > 1 else 0
    rolling_avg_lap = df_sub['RollingAvgLap'].mean() if not df_sub.empty else 0
    lap_diff_from_rolling = lap_number - rolling_avg_lap
    tyre_eff = tyre_age * df_sub['DegradationRate'].mean() if not df_sub.empty else 0
    lap_age_factor = lap_number / (tyre_age + 1)
    return pd.DataFrame([{
        'LapNumber': lap_number,
        'RollingAvgLap': rolling_avg_lap,
        'TyreAge': tyre_age,
        'DegradationRate': df_sub['DegradationRate'].mean() if not df_sub.empty else 0,
        'LapDiffFromRollingAvg': lap_diff_from_rolling,
        'TyreEff': tyre_eff,
        'LapAgeFactor': lap_age_factor,
        'PrevLapDelta': prev_lap_delta,
        'Driver': driver_code,
        'CircuitId': circuit_code,
        'IsOutLap': is_out_lap,
        'Compound': compound
    }])

# ------------------- Predizione Input
X_input = compute_features(laps_df, driver_code, circuit_code, lap_number, tyre_age, compound, is_out_lap)

# ✅ Assicura categorie coerenti con training
for col in feature_info['CAT_COLS']:
    X_input[col] = X_input[col].astype(pd.CategoricalDtype(categories=X_val_fold0[col].cat.categories))

# Predizione media su tutti i modelli
preds = [model.predict(X_input)[0] for model in models]
lap_delta_pred = np.mean(preds) / 10  # scaling inverse

# ------------------- Layout Output
col1, col2 = st.columns([1, 2])

# ---------- Colonna 1: Predizione e Feature
with col1:
    st.subheader("Predizione LapDelta")
    st.metric(label=f"Driver {driver_name} - LapDelta stimato (s)", value=f"{lap_delta_pred:.3f}")

    st.subheader("Feature derivate e confronto storico")
    hist_data = laps_df[(laps_df['Driver'] == driver_code) & (laps_df['CircuitId'] == circuit_code)]
    hist_means = {
        'LapDiffFromRollingAvg': hist_data['LapNumber'].mean() - hist_data['RollingAvgLap'].mean() if not hist_data.empty else 0,
        'TyreEff': (hist_data['TyreAge'] * hist_data['DegradationRate']).mean() if not hist_data.empty else 0,
        'LapAgeFactor': (hist_data['LapNumber'] / (hist_data['TyreAge'] + 1)).mean() if not hist_data.empty else 0,
        'PrevLapDelta': hist_data['LapDelta'].shift(1).mean() if not hist_data.empty else 0
    }
    feature_df = pd.DataFrame({
        "Feature": ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta'],
        "Valore Inserito": [X_input['LapDiffFromRollingAvg'][0],
                            X_input['TyreEff'][0],
                            X_input['LapAgeFactor'][0],
                            X_input['PrevLapDelta'][0]],
        "Media Storica": [hist_means[f] for f in ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']],
        "Descrizione": [
            "Differenza tra giro attuale e media dei giri precedenti del pilota sul circuito",
            "Età della gomma moltiplicata per il tasso di degradazione medio del circuito",
            "Rapporto numero giro / età gomme",
            "LapDelta del giro precedente del pilota sul circuito"
        ]
    })
    st.table(feature_df)

# ---------- Colonna 2: Grafico Comparativo
with col2:
    st.subheader("Confronto con valori medi storici")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=feature_df['Feature'],
        y=feature_df['Valore Inserito'],
        name='Valore Inserito',
        marker_color='red'
    ))
    fig_comp.add_trace(go.Bar(
        x=feature_df['Feature'],
        y=feature_df['Media Storica'],
        name='Media Storica',
        marker_color='blue'
    ))
    fig_comp.update_layout(barmode='group', template="plotly_dark",
                           title="Valori feature vs Media Storica")
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------- Boxplot Distribuzione LapDelta
st.subheader("Distribuzione LapDelta per mescola")
box_data = laps_df[(laps_df['CircuitId'] == circuit_code) & (laps_df['Driver'].isin(other_driver_codes))]
if not box_data.empty:
    color_map = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "grey"}
    fig_box = px.box(box_data, x='Compound', y='LapDelta', color='Compound', points="all",
                     color_discrete_map=color_map,
                     title="Distribuzione LapDelta per mescola e pilota", template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- Accuratezza modello (fold 0) reale
st.subheader("Accuratezza modello sul fold di validazione")
y_val = y_val_fold0 / 10  # scaling inverse
pred_val = np.mean([model.predict(X_val_fold0) for model in models], axis=0) / 10
mae = mean_absolute_error(y_val, pred_val)
rmse = np.sqrt(mean_squared_error(y_val, pred_val))
r2 = r2_score(y_val, pred_val)
st.metric("MAE", f"{mae:.3f} s")
st.metric("RMSE", f"{rmse:.3f} s")
st.metric("R²", f"{r2:.3f}")

st.markdown("💡 Tutte le feature complesse vengono calcolate automaticamente e coerenti con il training.")
st.markdown("ℹ️ Dashboard aggiornata in tempo reale con selezione piloti e confronto storico.")

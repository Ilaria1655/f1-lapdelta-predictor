import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime

# ----------------------------- PAGE CONFIG
st.set_page_config(page_title="🏁 F1 LapDelta Predictor — Pro Visual UI", layout="wide")

# ----------------------------- THEME / STYLE
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
        color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        color: #ff1801;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sidebar .sidebar-content {
        background-color: #111;
    }
    div[data-testid="stMetricValue"] {
        color: #00ff88;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- PAGE HEADER
st.title("🏁 F1 LapDelta Predictor — Pro Visual UI")
st.markdown("App focalizzata: inserisci i parametri del giro e ottieni subito la predizione del LapDelta, con insight visivi e metriche tecniche.")

# ----------------------------- PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"

# ----------------------------- HELPERS
@st.cache_data(show_spinner=False)
def list_model_folders(models_dir=models_dir):
    if not models_dir.exists():
        return []
    return sorted([d for d in models_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)

@st.cache_data(show_spinner=False)
def load_latest_model(models_dir=models_dir):
    folders = list_model_folders(models_dir)
    if not folders:
        return None, None
    latest = folders[0]
    feature_info = joblib.load(latest / "feature_info.joblib")
    model_paths = feature_info.get("model_paths", [])
    models = []
    for p in model_paths:
        p = Path(p)
        if p.exists():
            try:
                models.append(lgb.Booster(model_file=str(p)))
            except Exception:
                try:
                    models.append(joblib.load(p))
                except Exception:
                    pass
    return feature_info, models

@st.cache_data(show_spinner=False)
def load_laps(parquet_path=processed_dir / "laps_clean_final.parquet"):
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)

def ensemble_predict(models, X):
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)

def compute_features_for_input(df_ref, driver_code, circuit_code, lap_number, tyre_age, compound, is_out_lap):
    df_sub = df_ref[(df_ref['Driver'] == driver_code) & (df_ref['CircuitId'] == circuit_code)].sort_values('LapNumber')
    prev_lap_delta = df_sub[df_sub['LapNumber'] < lap_number]['LapDelta'].iloc[-1] if (not df_sub.empty and lap_number > 1) else 0.0
    rolling_avg_lap = df_sub['RollingAvgLap'].mean() if not df_sub.empty else 0.0
    degr = df_sub['DegradationRate'].mean() if not df_sub.empty else 0.0
    return {
        'LapNumber': lap_number,
        'RollingAvgLap': rolling_avg_lap,
        'TyreAge': tyre_age,
        'DegradationRate': degr,
        'LapDiffFromRollingAvg': lap_number - rolling_avg_lap,
        'TyreEff': tyre_age * degr,
        'LapAgeFactor': lap_number / (tyre_age + 1),
        'PrevLapDelta': prev_lap_delta,
        'Driver': driver_code,
        'CircuitId': circuit_code,
        'IsOutLap': is_out_lap,
        'Compound': compound
    }

def ensure_categorical(df: pd.DataFrame, cat_cols):
    from pandas.api.types import CategoricalDtype
    for c in cat_cols:
        if c in df.columns and not isinstance(df[c].dtype, CategoricalDtype):
            df[c] = df[c].astype('category')
    return df

# ----------------------------- LOAD ASSETS
feature_info, models = load_latest_model()
laps_df = load_laps()

if feature_info is None or not models:
    st.error("Modelli non trovati in data/models. Esegui prima lo script di training.")
    st.stop()

if laps_df.empty:
    st.error("laps_clean_final.parquet non trovato o vuoto.")
    st.stop()

# ----------------------------- SIDEBAR
st.sidebar.header("⚙️ Configura il giro")
presenter_mode = st.sidebar.toggle("Modalità Presentazione", value=False)
drivers = sorted(laps_df['Driver'].unique())
circuits = sorted(laps_df['name'].unique())
compounds = sorted(laps_df['Compound'].unique())

driver_name = st.sidebar.selectbox("Pilota", drivers)
circuit_name = st.sidebar.selectbox("Circuito", circuits)
lap_number = st.sidebar.slider("Numero giro", 1, 80, 1)
tyre_age = st.sidebar.slider("Età pneumatico (giri)", 0, 50, 0)
compound = st.sidebar.selectbox("Mescola", compounds)
is_out_lap = st.sidebar.checkbox("Out Lap", value=False)

if presenter_mode:
    st.markdown("<style>section[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

# ----------------------------- PREDIZIONE
driver_code = driver_name
circuit_code = laps_df[laps_df['name'] == circuit_name]['CircuitId'].iloc[0]

X_input_dict = compute_features_for_input(laps_df, driver_code, circuit_code, lap_number, tyre_age, compound, is_out_lap)
X_input = pd.DataFrame([X_input_dict])
X_input = ensure_categorical(X_input, feature_info.get('CAT_COLS', []))
pred_scaled = ensemble_predict(models, X_input)[0]
lap_delta_pred = float(pred_scaled) / 10.0

# ----------------------------- METRICHE SUPERIORI
latest = list_model_folders(models_dir)[0]
colA, colB, colC = st.columns(3)
colA.metric("Boosters", len(models))
colB.metric("Folds", feature_info.get("n_folds", "N/A"))
colC.metric("Modello", latest.name)

# ----------------------------- RISULTATO PRINCIPALE
color = "#00FF6A" if lap_delta_pred < 0 else "#FF4C4C"
sign = "più veloce" if lap_delta_pred < 0 else "più lento"
st.markdown(f"""
<div style='background:#111; padding:16px; border-radius:12px; text-align:center'>
<h2 style='color:{color}; margin:0'>{lap_delta_pred:.3f} s</h2>
<div style='color:#bbb'>Rispetto al giro medio del pilota ({sign})</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------- INSIGHT
with st.expander("📈 Insight sulla previsione", expanded=True):
    if lap_delta_pred < 0:
        st.success("🟢 Giro previsto più veloce della media — ottimo rendimento gomme e setup!")
    elif lap_delta_pred < 0.1:
        st.info("⚪ Giro in linea con la media del pilota — prestazione stabile.")
    else:
        st.warning("🔴 Giro più lento — possibile degrado gomme o traffico in pista.")

# ----------------------------- GRAFICO
test_data = laps_df[(laps_df['Driver'] == driver_code) & (laps_df['CircuitId'] == circuit_code)].copy()
if not test_data.empty:
    test_data = test_data.sort_values('LapNumber')
    test_data['PrevLapDelta'] = test_data['LapDelta'].shift(1).fillna(0)
    test_data['LapDiffFromRollingAvg'] = test_data['LapNumber'] - test_data['RollingAvgLap'].mean()
    test_data['TyreEff'] = test_data['TyreAge'] * test_data['DegradationRate']
    test_data['LapAgeFactor'] = test_data['LapNumber'] / (test_data['TyreAge'] + 1)
    X_test_df = test_data[['LapNumber','RollingAvgLap','TyreAge','DegradationRate','LapDiffFromRollingAvg','TyreEff','LapAgeFactor','PrevLapDelta','Driver','CircuitId','IsOutLap','Compound']].copy()

    cat_cols = feature_info.get("CAT_COLS", [])
    for c in cat_cols:
        if c in X_test_df.columns:
            X_test_df[c] = X_test_df[c].astype("category")

    test_data["Predicted"] = ensemble_predict(models, X_test_df) / 10.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['LapNumber'], y=test_data['LapDelta'], mode='lines+markers', name='Storico'))
    fig.add_trace(go.Scatter(x=test_data['LapNumber'], y=test_data['Predicted'], mode='lines+markers', name='Predetto'))
    fig.add_shape(type="line", x0=lap_number, x1=lap_number, y0=min(test_data['LapDelta']), y1=max(test_data['LapDelta']),
                  line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(template='plotly_dark', height=420, xaxis_title='Numero Giro', yaxis_title='LapDelta (s)')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nessuno storico disponibile per questo pilota/circuito.")

# ----------------------------- FOOTER
st.markdown("""
<hr style="border:1px solid #333">
<div style='text-align:center; color:#777'>
🏁 <b>F1 LapDelta Predictor</b> — powered by LightGBM · Streamlit Pro UI<br>
<small>Data-driven performance analysis for race engineering</small>
</div>
""", unsafe_allow_html=True)

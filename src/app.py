import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt

# =================================================================================
# PAGE CONFIG & STYLING
# =================================================================================
def setup_page():
    """Configura la pagina Streamlit e applica lo stile CSS."""
    st.set_page_config(page_title="🏎️ F1 LapDelta Predictor — Engineering Suite", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Titillium Web', sans-serif;
        }
        .stApp {
            background-color: #0a0a0a;
            color: #f1f1f1;
        }
        h1, h2, h3, h4, h5 {
            color: #ff1801;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 700;
        }
        .sidebar .sidebar-content {
            background-color: #111;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
        }
        div[data-testid="stMetricLabel"] {
            text-transform: uppercase;
            color: #a0a0a0;
        }
        hr {
            border: 1px solid #222;
        }
    </style>
    """, unsafe_allow_html=True)
    st.title("🏎️ F1 LapDelta Predictor — Engineering Suite")
    st.markdown("Piattaforma professionale per predizione e analisi del *Lap Delta* con insight ingegneristici e confronto telemetrico in tempo reale.")
    st.markdown("---")

# =================================================================================
# DATA LOADING & CACHING
# =================================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

@st.cache_data(show_spinner="Caricamento modelli...")
def load_model_assets(_models_dir):
    """Carica l'ultimo modello, le feature info e l'explainer SHAP."""
    folders = sorted([d for d in _models_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    if not folders:
        return None, None, None
    latest = folders[0]
    st.session_state.model_name = latest.name

    feature_info = joblib.load(latest / "feature_info.joblib")
    model_paths = feature_info.get("model_paths", [])
    models = [lgb.Booster(model_file=str(p)) for p in model_paths if Path(p).exists()]

    explainer = shap.TreeExplainer(models[0]) if models else None
    return feature_info, models, explainer

@st.cache_data(show_spinner="Caricamento dati pre-calcolati...")
def load_precomputed_laps(_parquet_path):
    """Carica i dati dei giri con le previsioni già calcolate."""
    if not _parquet_path.exists():
        return None
    return pd.read_parquet(_parquet_path)

# =================================================================================
# FEATURE ENGINEERING & PREDICTION (SOLO PER INPUT LIVE)
# =================================================================================
def compute_features_for_input(df_ref, driver_code, circuit_id, lap_number, tyre_age, compound, is_out_lap):
    """Calcola le feature per un singolo input di predizione live."""
    df_sub = df_ref[(df_ref['Driver'] == driver_code) & (df_ref['CircuitId'] == circuit_id)]

    if not df_sub.empty:
        prev_lap_delta = df_sub[df_sub['LapNumber'] < lap_number]['LapDelta'].iloc[-1] if lap_number > 1 else 0.0
        rolling_avg_lap = df_sub['RollingAvgLap'].mean()
        degr = df_sub['DegradationRate'].mean()
    else: # Fallback
        prev_lap_delta = 0.0
        rolling_avg_lap = df_ref['RollingAvgLap'].mean()
        degr = df_ref['DegradationRate'].mean()

    return {
        'LapNumber': lap_number, 'RollingAvgLap': rolling_avg_lap, 'TyreAge': tyre_age,
        'DegradationRate': degr, 'LapDiffFromRollingAvg': lap_number - rolling_avg_lap,
        'TyreEff': tyre_age * degr, 'LapAgeFactor': lap_number / (tyre_age + 1),
        'PrevLapDelta': prev_lap_delta, 'Driver': driver_code, 'CircuitId': circuit_id,
        'IsOutLap': is_out_lap, 'Compound': compound
    }

def ensure_categorical(df: pd.DataFrame, cat_cols, feature_info):
    """Converte le colonne in tipo 'category' usando le categorie salvate."""
    from pandas.api.types import CategoricalDtype
    for c in cat_cols:
        if c in df.columns:
            categories = feature_info.get('categories', {}).get(c)
            if categories:
                cat_dtype = CategoricalDtype(categories=categories, ordered=False)
                df[c] = df[c].astype(cat_dtype)
    return df

def get_live_prediction(models, X):
    """Esegue la predizione live e restituisce media e deviazione standard."""
    all_preds = np.array([m.predict(X) for m in models]) / 10.0
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)

# =================================================================================
# UI COMPONENTS
# =================================================================================
def build_sidebar(laps_df):
    """Costruisce la sidebar e restituisce i parametri di simulazione."""
    st.sidebar.header("⚙️ Parametri di Simulazione")
    drivers = sorted(laps_df['Driver'].unique())
    circuits = sorted(laps_df['name'].unique())
    compounds = sorted(laps_df['Compound'].unique())

    driver_name = st.sidebar.selectbox("Pilota", drivers, index=drivers.index('VER') if 'VER' in drivers else 0)
    circuit_name = st.sidebar.selectbox("Circuito", circuits, index=circuits.index('Bahrain International Circuit') if 'Bahrain International Circuit' in circuits else 0)
    lap_number = st.sidebar.slider("Numero Giro", 1, 80, 25)
    tyre_age = st.sidebar.slider("Età Gomma (giri)", 0, 50, 10)
    compound = st.sidebar.selectbox("Mescola", compounds, index=compounds.index('MEDIUM') if 'MEDIUM' in compounds else 0)
    is_out_lap = st.sidebar.checkbox("È un Out Lap?", value=False)

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Modello Caricato:** `{st.session_state.get('model_name', 'N/A')}`")

    return driver_name, circuit_name, lap_number, tyre_age, compound, is_out_lap

def display_main_dashboard(lap_delta_pred, uncertainty, models, feature_info, r2):
    """Visualizza il pannello principale con la predizione e le metriche chiave."""
    col_pred, col_info = st.columns([2, 1])

    with col_pred:
        color = "#00FF6A" if lap_delta_pred <= 0 else "#FF4C4C"
        sign = "più veloce" if lap_delta_pred <= 0 else "più lento"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%); padding:25px; border-radius:12px; text-align:center; border: 1px solid #333;'>
            <h3 style='color:#a0a0a0; margin:0; text-transform: uppercase;'>Predizione Lap Delta</h3>
            <p style='color:{color}; font-size: 4rem; font-weight: 700; margin:0; line-height: 1.2;'>
                {lap_delta_pred:.3f} s
            </p>
            <p style='color:#bbb; margin-top: 5px;'>
                Incertezza: <span style='color:white;'>±{uncertainty:.3f}s</span>
                <br>
                Rispetto al giro di riferimento del pilota ({sign})
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.metric("Boosters Usati", len(models))
        st.metric("Folds del Modello", feature_info.get("n_folds", "N/A"))
        if r2 is not None:
            st.metric("R² Score (vs Storico)", f"{r2:.3f}")
    st.markdown("---")

def display_historical_tab(hist_data, lap_number, lap_delta_pred, driver_name, circuit_name):
    """Visualizza il tab dell'analisi storica."""
    st.subheader(f"Storico vs. Predizioni per {driver_name} a {circuit_name}")
    if not hist_data.empty:
        color = "#00FF6A" if lap_delta_pred <= 0 else "#FF4C4C"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data['LapNumber'], y=hist_data['LapDelta'], mode='lines+markers', name='Lap Delta Reale', line=dict(color='cyan', width=2)))
        fig.add_trace(go.Scatter(x=hist_data['LapNumber'], y=hist_data['PredictedDelta'], mode='lines', name='Lap Delta Predetto', line=dict(color='magenta', dash='dash')))
        fig.add_trace(go.Scatter(x=[lap_number], y=[lap_delta_pred], mode='markers', name='Predizione Attuale', marker=dict(color=color, size=15, symbol='star', line=dict(color='white', width=1))))
        fig.update_layout(template='plotly_dark', height=450, xaxis_title='Numero Giro', yaxis_title='Lap Delta (s)', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nessuno storico disponibile per questa combinazione pilota/circuito.")

def display_shap_tab(explainer, X_input):
    """Visualizza il tab dell'analisi SHAP."""
    st.subheader("Analisi di Contribuzione delle Feature (SHAP)")
    st.info("SHAP (SHapley Additive exPlanations) aiuta a capire l'impatto di ogni feature sulla singola predizione.")
    if explainer:
        shap_values = explainer.shap_values(X_input)


        st.markdown("##### Waterfall Plot")
        st.markdown("Scompone la predizione, mostrando il contributo cumulativo di ogni feature per arrivare al risultato finale.")
        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP", orientation="h",
            measure=["relative"] * len(X_input.columns),
            y=X_input.columns, text=[f"{val:.3f}" for val in shap_values[0]],
            x=shap_values[0], connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Decomposizione della Predizione (SHAP Waterfall)", template='plotly_dark', height=500)
        st.plotly_chart(fig_waterfall, use_container_width=True)
    else:
        st.error("SHAP explainer non disponibile.")

def display_telemetry_tab(laps_df, drivers, circuit_id):
    """Visualizza il tab della telemetria comparativa usando dati pre-calcolati."""
    st.subheader("Confronto Telemetrico tra Piloti")
    st.info(f"Analisi comparativa del passo gara predetto sullo stesso circuito.")
    colA, colB = st.columns(2)
    driverA_name = colA.selectbox("Pilota A", drivers, key="drvA", index=drivers.index('VER') if 'VER' in drivers else 0)
    driverB_name = colB.selectbox("Pilota B", drivers, key="drvB", index=drivers.index('LEC') if 'LEC' in drivers else 1)

    dataA = laps_df[(laps_df['Driver'] == driverA_name) & (laps_df['CircuitId'] == circuit_id)]
    dataB = laps_df[(laps_df['Driver'] == driverB_name) & (laps_df['CircuitId'] == circuit_id)]

    if not dataA.empty and not dataB.empty:
        merged = pd.merge(dataA[['LapNumber', 'PredictedDelta']], dataB[['LapNumber', 'PredictedDelta']], on='LapNumber', suffixes=('_A', '_B'))
        merged['DeltaPace'] = merged['PredictedDelta_B'] - merged['PredictedDelta_A']

        fig_tlm = go.Figure()
        fig_tlm.add_trace(go.Scatter(x=merged["LapNumber"], y=merged["PredictedDelta_A"], mode="lines", name=f"{driverA_name} Pred.", line=dict(color="#00ff88")))
        fig_tlm.add_trace(go.Scatter(x=merged["LapNumber"], y=merged["PredictedDelta_B"], mode="lines", name=f"{driverB_name} Pred.", line=dict(color="#ff4c4c")))
        fig_tlm.update_layout(template="plotly_dark", height=420, title=f"Confronto Passo Gara Predetto — {driverA_name} vs {driverB_name}", xaxis_title="Numero Giro", yaxis_title="LapDelta Predetto (s)")
        st.plotly_chart(fig_tlm, use_container_width=True)

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(x=merged["LapNumber"], y=merged["DeltaPace"], marker_color=np.where(merged["DeltaPace"] < 0, "#00ff88", "#ff4c4c")))
        fig_delta.update_layout(template="plotly_dark", height=250, title=f"Δ Passo Gara ({driverB_name} vs {driverA_name})", xaxis_title="Numero Giro", yaxis_title="Differenza (s)")
        st.plotly_chart(fig_delta, use_container_width=True)
    else:
        st.warning("Dati insufficienti per uno o entrambi i piloti per il confronto.")

def display_model_details_tab(feature_info):
    """Visualizza il tab con i dettagli del modello."""
    st.subheader("Dettagli del Modello e Performance")

    train_date = feature_info.get("train_date", "N/A")
    model_type = feature_info.get("model_type", "N/A")
    feature_count = len(feature_info.get("ALL_COLS", []))

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Training", str(train_date))
    col2.metric("Tipo Modello", model_type)
    col3.metric("Numero Feature", feature_count)

    st.markdown("##### Performance di Cross-Validation")
    cv_results = feature_info.get("cv_results")
    if cv_results:
        cv_df = pd.DataFrame(cv_results).set_index('fold')
        st.dataframe(cv_df.style.format(precision=4).background_gradient(cmap='Reds', subset=['mae', 'rmse']).format('{:.3f}', subset=['r2']))
    else:
        st.info("Dettagli di cross-validation non disponibili.")

    st.markdown("##### Feature Importance (media sui folds)")
    importances = feature_info.get("feature_importance")
    if importances:
        imp_df = pd.DataFrame(importances.items(), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
        fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'], orientation='h', marker=dict(color=imp_df['Importance'], colorscale='Reds')))
        fig_imp.update_layout(template='plotly_dark', height=500, yaxis=dict(autorange="reversed"), title="Importanza Media delle Feature")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Informazioni di feature importance non disponibili.")

# =================================================================================
# MAIN APP LOGIC
# =================================================================================
def main():
    """Funzione principale che esegue l'applicazione Streamlit."""
    setup_page()

    # --- Caricamento Assets ---
    feature_info, models, explainer = load_model_assets(MODELS_DIR)
    laps_df = load_precomputed_laps(PROCESSED_DIR / "laps_with_predictions.parquet")

    if feature_info is None or not models:
        st.error("❌ **Nessun modello trovato!** Eseguire prima lo script di training `src/train_model.py`.")
        st.stop()
    if laps_df is None:
        st.error("❌ **Dati pre-calcolati non trovati!** Eseguire lo script `src/run_post_processing.py` prima di avviare l'app.")
        st.code("python src/run_post_processing.py")
        st.stop()

    # --- Controlli della Sidebar ---
    driver_name, circuit_name, lap_number, tyre_age, compound, is_out_lap = build_sidebar(laps_df)

    # --- Logica di Predizione Live ---
    driver_code = driver_name
    circuit_id = laps_df[laps_df['name'] == circuit_name]['CircuitId'].iloc[0]

    X_input_dict = compute_features_for_input(laps_df, driver_code, circuit_id, lap_number, tyre_age, compound, is_out_lap)
    X_input = pd.DataFrame([X_input_dict])
    X_input = ensure_categorical(X_input, feature_info.get('CAT_COLS', []), feature_info)
    for col in feature_info['ALL_COLS']:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[feature_info['ALL_COLS']]

    preds, std_devs = get_live_prediction(models, X_input)
    lap_delta_pred = preds[0]
    uncertainty = std_devs[0]

    # --- Filtro dati storici (veloce, no ricalcolo) ---
    hist_data = laps_df[(laps_df['Driver'] == driver_code) & (laps_df['CircuitId'] == circuit_id)]
    r2 = r2_score(hist_data['LapDelta'], hist_data['PredictedDelta']) if not hist_data.empty else 0

    # --- Visualizzazione UI ---
    display_main_dashboard(lap_delta_pred, uncertainty, models, feature_info, r2)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analisi Storica", "🧠 Spiegazione Predizione (SHAP)", "📡 Telemetria Comparativa", "🛠️ Dettagli Modello"])

    with tab1:
        display_historical_tab(hist_data, lap_number, lap_delta_pred, driver_name, circuit_name)
    with tab2:
        display_shap_tab(explainer, X_input)
    with tab3:
        display_telemetry_tab(laps_df, sorted(laps_df['Driver'].unique()), circuit_id)
    with tab4:
        display_model_details_tab(feature_info)

if __name__ == "__main__":
    main()

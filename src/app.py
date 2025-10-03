import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# ------------------- Configurazione pagina
st.set_page_config(
    page_title="🏎️ F1 LapDelta Predictor 2024",
    layout="wide"
)

st.title("🏎️ F1 LapDelta Predictor 2024")
st.markdown("""
Dashboard interattiva che mostra la previsione del **LapDelta** per il pilota selezionato e confronta
la predizione con i dati storici sul circuito scelto.
""")

# ------------------- Percorsi
BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"

# ------------------- Caricamento modello
latest_model_folder = max(models_dir.iterdir(), key=lambda d: d.stat().st_mtime)
feature_info = joblib.load(latest_model_folder / "feature_info.joblib")
model_paths = feature_info["model_paths"]
models = [lgb.Booster(model_file=path) for path in model_paths]

# ------------------- Caricamento dataset con cache
@st.cache_data(show_spinner=False)
def load_laps():
    return pd.read_parquet(processed_dir / "laps_clean_final.parquet")

laps_df = load_laps()

# ------------------- Sidebar Input
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
st.sidebar.markdown("### ⚙️ Configura il giro da predire")

drivers = sorted(laps_df['Driver'].unique())
circuits = sorted(laps_df['name'].unique())
compounds = sorted(laps_df['Compound'].unique())

driver_name = st.sidebar.selectbox("🏁 Pilota principale", drivers)
circuit_name = st.sidebar.selectbox("🌍 Circuito", circuits)
lap_number = st.sidebar.slider("🔢 Numero del giro", 1, 80, 1)
tyre_age = st.sidebar.slider("🛞 Età pneumatico (giri)", 0, 50, 0)
compound = st.sidebar.selectbox("⚡ Mescola", compounds)
is_out_lap = st.sidebar.checkbox("🚦 Out Lap?")

# ------------------- Mapping IDs
driver_code = laps_df[laps_df['Driver'] == driver_name]['Driver'].iloc[0]
circuit_code = laps_df[laps_df['name'] == circuit_name]['CircuitId'].iloc[0]

# ------------------- Funzione feature engineering per predizione singola
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

X_input = compute_features(laps_df, driver_code, circuit_code, lap_number, tyre_age, compound, is_out_lap)
for col in feature_info['CAT_COLS']:
    if col in X_input.columns:
        X_input[col] = pd.Categorical(X_input[col], categories=X_input[col].unique())

preds = [model.predict(X_input)[0] for model in models]
lap_delta_pred = np.mean(preds) / 10

# ------------------- Predizione evidenziata
color_pred = "green" if lap_delta_pred < 0 else "red"
st.markdown(f"## 🏁 Predizione LapDelta: **<span style='color:{color_pred}'>{lap_delta_pred:.3f} s</span>**", unsafe_allow_html=True)
st.info("Il LapDelta indica di quanto il giro stimato è più veloce (verde) o più lento (rosso) rispetto al giro medio del pilota.")

# ------------------- Preparazione dati storici
@st.cache_data(show_spinner=False)
def prepare_test_data(driver_code, circuit_code, cat_cols):
    df = laps_df[(laps_df['Driver'] == driver_code) & (laps_df['CircuitId'] == circuit_code)].copy()
    if df.empty:
        return df
    df = df.sort_values('LapNumber')
    df['PrevLapDelta'] = df['LapDelta'].shift(1).fillna(0)
    rolling_avg = df['RollingAvgLap'].mean()
    df['LapDiffFromRollingAvg'] = df['LapNumber'] - rolling_avg
    df['TyreEff'] = df['TyreAge'] * df['DegradationRate']
    df['LapAgeFactor'] = df['LapNumber'] / (df['TyreAge'] + 1)
    for col in cat_cols:
        if col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                if "N/A" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("N/A")
            df[col] = df[col].fillna("N/A")
            df[col] = pd.Categorical(df[col], categories=df[col].unique())
    return df

test_data = prepare_test_data(driver_code, circuit_code, feature_info['CAT_COLS'])

# ------------------- Metriche e confronto con storico
if not test_data.empty:
    X_test_df = test_data[['LapNumber','RollingAvgLap','TyreAge','DegradationRate',
                           'LapDiffFromRollingAvg','TyreEff','LapAgeFactor',
                           'PrevLapDelta','Driver','CircuitId','IsOutLap','Compound']]
    y_test = test_data['LapDelta'].values
    preds_test = np.mean([m.predict(X_test_df) for m in models], axis=0) / 10
    errors = y_test - preds_test

    # Metriche
    mae = mean_absolute_error(y_test, preds_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    r2 = r2_score(y_test, preds_test)
    medae = median_absolute_error(y_test, preds_test)
    non_zero_mask = y_test != 0
    mape = (np.abs(errors[non_zero_mask] / y_test[non_zero_mask]).mean()) * 100 if non_zero_mask.any() else np.nan

    st.markdown("### 🎯 Metriche di Accuratezza")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{mae:.3f} s")
    col2.metric("RMSE", f"{rmse:.3f} s")
    col3.metric("R²", f"{r2:.3f}")
    col4.metric("MedAE", f"{medae:.3f} s")
    col5.metric("MAPE", f"{mape:.2f} %")

    # Grafico lineplot storico senza tooltip
    test_data_plot = test_data.copy()
    test_data_plot['Predizione'] = np.nan
    test_data_plot.loc[test_data_plot['LapNumber'] == lap_number, 'Predizione'] = lap_delta_pred
    colors = np.where(test_data_plot['LapDelta'] <= 0, 'green', 'red')

    fig_line = go.Figure()

    # Storico
    fig_line.add_trace(go.Scatter(
        x=test_data_plot['LapNumber'],
        y=test_data_plot['LapDelta'],
        mode='lines+markers',
        marker=dict(color=colors, size=8),
        line=dict(color='white', width=2),
        name='Storico',
        hoverinfo='skip'  # tooltip disabilitati
    ))

    # Predizione
    fig_line.add_trace(go.Scatter(
        x=[lap_number],
        y=[lap_delta_pred],
        mode='markers+text',
        marker=dict(color=color_pred, size=14, symbol='star'),
        name='Predizione',
        text=["Predizione"],
        textposition="top center",
        hoverinfo='skip'
    ))

    fig_line.update_layout(
        template='plotly_dark',
        title="LapDelta Storico con Predizione Evidenziata",
        xaxis_title="Numero Giro",
        yaxis_title="LapDelta (s)"
    )

    st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("❌ Nessun giro storico disponibile per questo pilota/circuito.")

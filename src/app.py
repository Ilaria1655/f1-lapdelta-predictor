import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# ------------------- Configurazione pagina
st.set_page_config(
    page_title="🏎️ F1 LapDelta Predictor 2024",
    layout="wide"
)

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

# ------------------- Caricamento dataset
laps_df = pd.read_parquet(processed_dir / "laps_clean_final.parquet")

# ------------------- Sidebar Input pulita
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
st.sidebar.markdown("### ⚙️ Configura il giro da predire")

drivers = sorted(laps_df['Driver'].unique())
circuits = sorted(laps_df['name'].unique())
compounds = sorted(laps_df['Compound'].unique())

driver_name = st.sidebar.selectbox("🏁 Pilota principale", drivers,
                                   help="Scegli il pilota per il quale vuoi stimare il LapDelta")
circuit_name = st.sidebar.selectbox("🌍 Circuito", circuits, help="Seleziona il circuito della gara")
lap_number = st.sidebar.slider("🔢 Numero del giro", 1, 80, 1,
                               help="Indica il numero del giro per il quale vuoi stimare il LapDelta")
tyre_age = st.sidebar.slider("🛞 Età pneumatico (giri)", 0, 50, 0, help="Indica da quanti giri il pneumatico è montato")
compound = st.sidebar.selectbox("⚡ Mescola", compounds, help="Seleziona la mescola della gomma")
is_out_lap = st.sidebar.checkbox("🚦 Out Lap?", help="Spunta se si tratta del giro di uscita dai box")

# ------------------- Mapping IDs
driver_code = laps_df[laps_df['Driver'] == driver_name]['Driver'].iloc[0]
circuit_code = laps_df[laps_df['name'] == circuit_name]['CircuitId'].iloc[0]


# ------------------- Funzione feature engineering
def compute_features(df, driver_code, circuit_code, lap_number, tyre_age, compound, is_out_lap):
    df_sub = df[(df['Driver'] == driver_code) & (df['CircuitId'] == circuit_code)].sort_values('LapNumber')
    prev_lap_delta = df_sub[df_sub['LapNumber'] < lap_number]['LapDelta'].iloc[
        -1] if not df_sub.empty and lap_number > 1 else 0
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

# Categorie coerenti
for col in feature_info['CAT_COLS']:
    if col in X_input.columns:
        X_input[col] = pd.Categorical(X_input[col], categories=X_input[col].unique())

# Predizione
preds = [model.predict(X_input)[0] for model in models]
lap_delta_pred = np.mean(preds) / 10

# ------------------- Tabs layout
tab_pred, tab_hist, tab_acc = st.tabs(["📊 Predizione", "📦 Storico", "🎯 Accuratezza"])

# ---------- Tab Predizione
with tab_pred:
    st.markdown("## 🏁 Predizione LapDelta")
    st.info("Questa sezione mostra la previsione del distacco dal giro medio del pilota selezionato.")
    col1, col2, col3 = st.columns(3)
    col1.metric("LapDelta stimato", f"{lap_delta_pred:.3f} s")
    col2.metric("Numero giro", lap_number)
    col3.metric("Età pneumatico", f"{tyre_age} giri")

# ---------- Tab Storico
with tab_hist:
    st.markdown("## 📦 Confronto con dati storici")
    st.info(
        "Qui puoi vedere come le feature inserite si confrontano con i dati reali del pilota sul circuito selezionato.")

    hist_data = laps_df[(laps_df['Driver'] == driver_code) & (laps_df['CircuitId'] == circuit_code)]
    hist_means = {
        'LapDiffFromRollingAvg': hist_data['LapNumber'].mean() - hist_data[
            'RollingAvgLap'].mean() if not hist_data.empty else 0,
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
        "Media Storica": [hist_means[f] for f in ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']]
    })

    fig_comp = px.bar(feature_df, x='Feature', y=['Valore Inserito', 'Media Storica'],
                      barmode='group', template='plotly_dark',
                      color_discrete_map={"Valore Inserito": "crimson", "Media Storica": "royalblue"},
                      title="Confronto Feature con Storico")
    fig_comp.update_layout(legend_title_text='Legenda', legend=dict(x=0.8, y=1.2))
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------- Tab Accuratezza
with tab_acc:
    st.markdown("## 🎯 Accuratezza Predizioni")
    st.info("Questa sezione mostra quanto il modello è preciso su giri storici del pilota selezionato.")
    test_data = laps_df[(laps_df['Driver'] == driver_code) & (laps_df['CircuitId'] == circuit_code)]

    if not test_data.empty:
        X_test_list = []
        y_test_list = []
        for _, row in test_data.iterrows():
            X_row = compute_features(laps_df, row['Driver'], row['CircuitId'],
                                     row['LapNumber'], row['TyreAge'], row['Compound'], row['IsOutLap'])
            for col in feature_info['CAT_COLS']:
                if col in X_row.columns:
                    X_row[col] = pd.Categorical(X_row[col], categories=X_row[col].unique())
            X_test_list.append(X_row)
            y_test_list.append(row['LapDelta'])

        X_test_df = pd.concat(X_test_list, ignore_index=True)
        y_test = np.array(y_test_list)
        preds_test = np.mean([m.predict(X_test_df) for m in models], axis=0) / 10

        errors = y_test - preds_test
        mae = mean_absolute_error(y_test, preds_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        r2 = r2_score(y_test, preds_test)
        medae = median_absolute_error(y_test, preds_test)
        mape = (np.abs(errors / y_test).mean()) * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{mae:.3f} s")
        col2.metric("RMSE", f"{rmse:.3f} s")
        col3.metric("R²", f"{r2:.3f}")
        col4.metric("MedAE", f"{medae:.3f} s")
        col5.metric("MAPE", f"{mape:.2f} %")

        fig_err = px.histogram(errors, nbins=20, template="plotly_dark",
                               title="Distribuzione Errori Predizione vs Reale",
                               labels={"value": "Errore (s)"})
        fig_err.update_layout(legend_title_text='Errori')
        st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info("❌ Nessun giro storico disponibile per questo pilota/circuito.")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import base64

# ------------------- Configurazione pagina
st.set_page_config(page_title="🏎️ F1 LapDelta Predictor 2024", layout="wide")

# ------------------- Percorsi
BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"

# ------------------- Caricamento ultimo modello
latest_model_folder = max(models_dir.iterdir(), key=lambda d: d.stat().st_mtime)
feature_info = joblib.load(latest_model_folder / "feature_info.joblib")
model_paths = feature_info["model_paths"]
models = [lgb.Booster(model_file=path) for path in model_paths]

# ------------------- Caricamento dataset
laps_df = pd.read_parquet(processed_dir / "laps_clean_final.parquet")
X_val_fold0 = joblib.load(latest_model_folder / "X_val_fold0.joblib")
y_val_fold0 = joblib.load(latest_model_folder / "y_val_fold0.joblib")

# ------------------- Sidebar Input
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
st.sidebar.markdown("### ⚙️ Configura Input Giro")

drivers = sorted(laps_df['Driver'].unique())
circuits = sorted(laps_df['name'].unique())
compounds = sorted(laps_df['Compound'].unique())

driver_name = st.sidebar.selectbox("🏁 Driver principale", drivers)
other_drivers = st.sidebar.multiselect("👥 Confronta con altri piloti", drivers, default=[driver_name])
circuit_name = st.sidebar.selectbox("🌍 Circuito", circuits)
lap_number = st.sidebar.slider("🔢 Numero del giro", 1, 80, 1)
tyre_age = st.sidebar.slider("🛞 Età pneumatico", 0, 50, 0)
compound = st.sidebar.selectbox("⚡ Mescola", compounds)
is_out_lap = st.sidebar.checkbox("🚦 Out Lap?")

# ------------------- Mapping IDs
driver_code = laps_df[laps_df['Driver'] == driver_name]['Driver'].iloc[0]
circuit_code = laps_df[laps_df['name'] == circuit_name]['CircuitId'].iloc[0]
other_driver_codes = [laps_df[laps_df['Driver'] == d]['Driver'].iloc[0] for d in other_drivers]

# ------------------- Feature Engineering
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

# categorie coerenti
for col in feature_info['CAT_COLS']:
    X_input[col] = X_input[col].astype(pd.CategoricalDtype(categories=X_val_fold0[col].cat.categories))

preds = [model.predict(X_input)[0] for model in models]
lap_delta_pred = np.mean(preds) / 10  # scaling inverse

# ------------------- Tabs Layout
tab_pred, tab_hist, tab_acc = st.tabs(["📊 Predizione", "📦 Storico", "🎯 Accuratezza"])

# ---------- Tab Predizione
with tab_pred:
    st.markdown("## 🏁 Predizione LapDelta")
    col1, col2, col3 = st.columns(3)
    col1.metric("LapDelta stimato", f"{lap_delta_pred:.3f} s")
    col2.metric("Numero giro", lap_number)
    col3.metric("Età pneumatico", f"{tyre_age} giri")

# ---------- Tab Storico
with tab_hist:
    st.markdown("## 📦 Confronto con Storico")
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
        "Media Storica": [hist_means[f] for f in ['LapDiffFromRollingAvg', 'TyreEff', 'LapAgeFactor', 'PrevLapDelta']]
    })

    fig_comp = px.bar(feature_df, x='Feature', y=['Valore Inserito', 'Media Storica'],
                      barmode='group', template='plotly_dark', color_discrete_sequence=["crimson","royalblue"],
                      title="Confronto Feature con Storico")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### Distribuzione LapDelta per mescola")
    box_data = laps_df[(laps_df['CircuitId'] == circuit_code) & (laps_df['Driver'].isin(other_driver_codes))]
    if not box_data.empty:
        fig_box = px.box(box_data, x='Compound', y='LapDelta', color='Compound',
                         points="all", template="plotly_dark",
                         title="Distribuzione LapDelta per mescola e pilota")
        st.plotly_chart(fig_box, use_container_width=True)

# ---------- Tab Accuratezza
with tab_acc:
    st.markdown("## 🎯 Accuratezza Previsione")

    # ------------------- GIF centrata
    gif_path = BASE_DIR / "assets" / "gif_1.gif"
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
    encoded_gif = base64.b64encode(gif_bytes).decode("utf-8")
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            <img src="data:image/gif;base64,{encoded_gif}" width="150">
        </div>
        """, unsafe_allow_html=True)

    # ------------------- Funzione metriche
    @st.cache_data
    def compute_metrics(_models, X_val, y_val):
        y_val_scaled = y_val / 10
        pred_val = np.mean([m.predict(X_val) for m in _models], axis=0) / 10
        errors = y_val_scaled - pred_val
        mae = mean_absolute_error(y_val_scaled, pred_val)
        rmse = np.sqrt(mean_squared_error(y_val_scaled, pred_val))
        r2 = r2_score(y_val_scaled, pred_val)
        medae = median_absolute_error(y_val_scaled, pred_val)
        mape = (abs(errors / y_val_scaled).mean()) * 100
        return mae, rmse, r2, medae, mape, pred_val, errors

    mae, rmse, r2, medae, mape, y_pred, errors = compute_metrics(models, X_val_fold0, y_val_fold0)

    # ------------------- Metriche principali
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{mae:.3f} s")
    col2.metric("RMSE", f"{rmse:.3f} s")
    col3.metric("R²", f"{r2:.3f}")
    col4.metric("MedAE", f"{medae:.3f} s")
    col5.metric("MAPE", f"{mape:.2f} %")

    # ------------------- Campionamento dati pesanti
    sample_size = min(len(laps_df), 500)
    df_sample = laps_df.sample(sample_size, random_state=42).copy()

    # Calcolo predizione per il campione
    # Nota: qui assumiamo X_val_fold0 corrisponde a laps_df. Se no, devi mappare X_input al campione
    pred_sample = np.mean([m.predict(X_val_fold0) for m in models], axis=0) / 10
    df_sample['Predizione'] = pred_sample[:sample_size]  # prendi i primi N predizioni corrispondenti
    df_sample['Errore'] = (df_sample['LapDelta'] / 10) - df_sample['Predizione']

    # ------------------- Grafici opzionali
    st.markdown("### Grafici di dettaglio")
    show_scatter = st.checkbox("📈 Predizioni vs Reali", value=True)
    show_error_dist = st.checkbox("📊 Distribuzione errori", value=True)
    show_box_driver = st.checkbox("🏎️ Boxplot per Pilota", value=False)
    show_trend_lap = st.checkbox("🕒 Trend errore medio per giro", value=False)
    show_heatmap = st.checkbox("🌡️ Heatmap Pilota x Mescola x Giro", value=False)
    show_cum = st.checkbox("📊 Errore cumulativo per Pilota", value=False)

    # Scatter Predizioni vs Reali
    if show_scatter:
        st.markdown("#### Predizioni vs Valori Reali")
        df_scatter = pd.DataFrame({"Valore Reale": df_sample['LapDelta'] / 10, "Predizione": df_sample['Predizione']})
        fig_scatter = px.scatter(df_scatter, x="Valore Reale", y="Predizione",
                                 trendline="ols", template="plotly_dark",
                                 labels={"Valore Reale": "Valore Reale (s)", "Predizione": "Predizione (s)"},
                                 color_discrete_sequence=["#00cc96"])
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Distribuzione errori
    if show_error_dist:
        st.markdown("#### Distribuzione degli errori")
        fig_hist = px.histogram(df_sample, x="Errore", nbins=30, template="plotly_dark",
                                color_discrete_sequence=["#ff6361"])
        st.plotly_chart(fig_hist, use_container_width=True)

    # Boxplot errore per pilota
    if show_box_driver:
        st.markdown("#### Boxplot errore per Pilota")
        fig_box_driver = px.box(df_sample, x='Driver', y='Errore', template="plotly_dark",
                                color='Driver', color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_box_driver, use_container_width=True)

    # Trend errore medio per giro
    if show_trend_lap:
        st.markdown("#### Trend errore medio per giro")
        lap_error = df_sample.groupby('LapNumber')['Errore'].mean().reset_index()
        fig_line = px.line(lap_error, x='LapNumber', y='Errore', template="plotly_dark",
                           line_shape='spline', markers=True, title="Errore medio per giro")
        st.plotly_chart(fig_line, use_container_width=True)

    # Heatmap Pilota x Mescola x Giro
    if show_heatmap:
        st.markdown("#### Heatmap Pilota x Mescola x Giro")
        df_heat = df_sample.groupby(['Driver','Compound','LapNumber'])['Errore'].mean().reset_index()
        fig_heat = px.density_heatmap(df_heat, x='LapNumber', y='Driver', z='Errore',
                                      facet_col='Compound', template='plotly_dark',
                                      color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_heat, use_container_width=True)

    # Errore cumulativo per pilota
    if show_cum:
        st.markdown("#### Errore cumulativo per pilota")
        df_cum = df_sample.groupby(['Driver','LapNumber'])['Errore'].mean().groupby(level=0).cumsum().reset_index()
        fig_cum = px.line(df_cum, x='LapNumber', y='Errore', color='Driver', template='plotly_dark',
                          color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_cum, use_container_width=True)


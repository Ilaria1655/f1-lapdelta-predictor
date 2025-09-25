import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import io

# -----------------------
# Setup cartelle e file
# -----------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
model_path = processed_dir / "lapdelta_model.joblib"
laps_path = processed_dir / "laps_clean.parquet"

# Carica modello e dataset
model = joblib.load(model_path)
df = pd.read_parquet(laps_path)

# Ottieni liste uniche per UI
drivers = sorted(df['Driver'].unique())
compounds = sorted(df['Compound'].unique())
circuits = sorted(df['CircuitId'].unique())

# -----------------------
# Layout Streamlit
# -----------------------
st.set_page_config(page_title="F1 LapDelta Predictor", layout="wide")
st.title("🏎️ F1 LapDelta Predictor")
st.markdown("""
Questa app predice di quanto un giro sarà più lento rispetto al **miglior giro del pilota** (LapDelta).  
Inserisci i parametri e ottieni predizioni interattive e metriche intuitive.
""")

# Sidebar per input
st.sidebar.header("Impostazioni simulazione")

driver = st.sidebar.selectbox("Pilota", drivers, index=0)
circuit = st.sidebar.selectbox("Circuito", circuits, index=0)
compound = st.sidebar.selectbox("Tipo di gomma", compounds, index=0)
lap_number = st.sidebar.slider("Numero giro", 1, int(df['LapNumber'].max()), 10)

# Preset esempio
if st.sidebar.button("Esempio: Verstappen Soft giro 12"):
    driver = "VER"
    compound = "Soft"
    lap_number = 12
    circuit = circuits[0]

# -----------------------
# Calcolo dinamico variabili necessarie
# -----------------------
# RollingAvgLap: media dei 3 giri precedenti per il driver sul circuito
driver_laps = df[(df['Driver'] == driver) & (df['CircuitId'] == circuit)]
recent_laps = driver_laps[driver_laps['LapNumber'] < lap_number].sort_values('LapNumber').tail(3)
rolling_avg = recent_laps['LapTimeSeconds'].mean() if not recent_laps.empty else driver_laps['LapTimeSeconds'].mean()
# TyreAge: numero di giri sull’attuale gomma, approssimato
tyre_age = 1 if lap_number == 1 else min(10, lap_number - 1)
# DegradationRate: differenza media tra giri precedenti
degradation = recent_laps['LapTimeSeconds'].diff().mean() if len(recent_laps) > 1 else 0
# Stint: ogni 20 giri un nuovo stint
stint = (lap_number - 1) // 20 + 1
# IsOutLap: default 0 (non è giro out)
is_outlap = 0

# -----------------------
# Predizione
# -----------------------
input_data = pd.DataFrame([{
    "Driver": driver,
    "Compound": compound,
    "CircuitId": circuit,
    "LapNumber": lap_number,
    "RollingAvgLap": rolling_avg,
    "TyreAge": tyre_age,
    "DegradationRate": degradation,
    "Stint": stint,
    "IsOutLap": is_outlap
}])

prediction = model.predict(input_data)[0]

# Miglior giro del pilota sul circuito
best_lap = driver_laps['LapTimeSeconds'].min()
predicted_lap = best_lap + prediction

# -----------------------
# Output principale
# -----------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📌 LapDelta Predetto")
    st.metric(label=f"Giro previsto vs miglior giro di {driver}", value=f"{prediction:.2f} s")

    # Download CSV
    csv_buffer = io.StringIO()
    input_data.assign(PredictedLapTime=predicted_lap).to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Scarica predizione (CSV)",
        data=csv_buffer.getvalue(),
        file_name="prediction.csv",
        mime="text/csv"
    )

with col1:
    # Grafico interattivo Pred vs Miglior giro
    fig = px.bar(
        x=["Miglior Giro", "Predizione"],
        y=[best_lap, predicted_lap],
        color=["Miglior Giro", "Predizione"],
        color_discrete_map={"Miglior Giro": "green", "Predizione": "blue"},
        labels={"x": "Tipo", "y": "Tempo giro [s]"},
        text=[f"{best_lap:.2f}", f"{predicted_lap:.2f}"]
    )
    fig.update_layout(showlegend=False, yaxis_title="Tempo giro [s]")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Espander: Come funziona
# -----------------------
with st.expander("ℹ️ Come funziona?"):
    st.markdown("""
- Inserisci **Pilota, Circuito, Gomma e Giro**.  
- Il modello ML predice di quanto quel giro sarà più lento rispetto al miglior giro del pilota sul circuito.  
- Tutte le altre variabili (Stint, IsOutLap, RollingAvg, Degradation, TyreAge) vengono calcolate automaticamente.  
- Grafici e download rendono la simulazione intuitiva anche per chi non conosce la F1.
""")

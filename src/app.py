import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------
# Setup
# -----------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
model_path = processed_dir / "lapdelta_model.joblib"
laps_path = processed_dir / "laps_clean.parquet"

# Carica modello e dataset
model = joblib.load(model_path)
df = pd.read_parquet(laps_path)

# -----------------------
# UI Streamlit
# -----------------------
st.set_page_config(page_title="F1 LapDelta Predictor", layout="centered")

st.title("🏎️ Formula 1 LapDelta Predictor")
st.markdown("""
Questa applicazione predice di **quanto un giro sarà più lento** rispetto al miglior giro personale del pilota (*LapDelta*).  
Inserisci i parametri e ottieni la predizione 📊
""")

# Dropdown per pilota e gomma
drivers = sorted(df['Driver'].unique())
compounds = sorted(df['Compound'].unique())

col1, col2 = st.columns(2)
driver = col1.selectbox("Seleziona pilota", drivers)
compound = col2.selectbox("Seleziona gomma", compounds)

# Slider per valori numerici
lap_number = st.slider("Numero giro", 1, int(df['LapNumber'].max()), 10)
tyre_age = st.slider("Età gomme (giri)", 0, 40, 5)
rolling_avg = st.slider("Media mobile ultimi 3 giri [s]", 60, 120, 90)
degradation = st.slider("Degradazione rispetto al giro precedente [s]", -2.0, 5.0, 0.2)

# Bottone predizione
if st.button("Predici LapDelta"):
    input_data = pd.DataFrame([{
        "Driver": driver,
        "Compound": compound,
        "LapNumber": lap_number,
        "TyreAge": tyre_age,
        "RollingAvgLap": rolling_avg,
        "DegradationRate": degradation
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("📌 Risultato")
    st.metric(label="LapDelta Predetto", value=f"{prediction:.2f} sec")

    # Grafico confronto
    best_lap = df[df['Driver'] == driver]['LapTimeSeconds'].min()
    predicted_lap = best_lap + prediction

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(["Miglior Giro", "Predizione"], [best_lap, predicted_lap], color=["green", "blue"])
    ax.set_ylabel("Tempo giro [s]")
    ax.set_title(f"Confronto tempi {driver}")
    st.pyplot(fig)

    st.markdown(f"➡️ Questo giro è previsto **{prediction:.2f} secondi più lento** rispetto al miglior giro di {driver}.")

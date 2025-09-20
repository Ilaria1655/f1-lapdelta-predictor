import streamlit as st
import pandas as pd
import plotly.express as px
from src.simulate import simulate_strategy
from src.model_utils import load_model

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")


@st.cache_resource
def load_pipeline():
    return load_model('model_pipeline.joblib')


model = load_pipeline()

st.title("🏎️ F1 Strategy Simulator")
st.markdown("Simula e confronta strategie gomme in Formula 1 con ML.")

col1, col2 = st.columns(2)

with col1:
    total_laps = st.number_input("Numero totale giri", min_value=10, max_value=80, value=50)
    pit_laps = st.text_input("Pit stops (es. 15,35)", value="15,35")
    pit_laps = [int(x.strip()) for x in pit_laps.split(',') if x.strip().isdigit()]

with col2:
    compounds = st.text_input("Strategia gomme (es. Soft,Medium,Hard)", value="Soft,Medium,Hard")
    compounds = [x.strip() for x in compounds.split(',')]

base_state = {
    'Driver': 'HAM',
    'Team': 'Mercedes',
    'RollingAvgLap': 90.0  # placeholder medio
}

if st.button("Simula Strategia"):
    df_sim = simulate_strategy(model, base_state, pit_laps, compounds, total_laps)
    total_time = df_sim['PredictedLapTime'].sum()

    st.subheader("Risultato Simulazione")
    st.write(f"Tempo totale previsto: **{total_time:.1f} s**")

    fig = px.line(df_sim, x='Lap', y='PredictedLapTime', color='Compound', title="Tempi per giro")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_sim)

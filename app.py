import streamlit as st
import pandas as pd
import plotly.express as px
from src.simulate import simulate_strategy
from src.model_utils import load_model

# ----------------------------
# Config pagina
# ----------------------------
st.set_page_config(
    page_title="🏎️ F1 Strategy Simulator",
    layout="wide",
    page_icon="🏁"
)
st.title("🏎️ F1 Strategy Simulator")
st.markdown(
    """
    Benvenuto nel simulatore di strategie di Formula 1!  
    Inserisci i tuoi giri, pit stop e tipi di gomme per vedere quale strategia è più veloce.
    Anche se non conosci la F1, vedrai grafici chiari e tempi stimati in secondi.
    """
)

# ----------------------------
# Caricamento modello con cache
# ----------------------------
@st.cache_resource
def load_pipeline():
    return load_model('lapdelta_model.joblib')  # usa il tuo modello salvato

model = load_pipeline()

# ----------------------------
# Input utente
# ----------------------------
st.header("🛠️ Imposta la tua simulazione")
col1, col2 = st.columns(2)

with col1:
    total_laps = st.number_input(
        "Numero totale di giri della gara",
        min_value=10,
        max_value=80,
        value=50,
        help="Inserisci il numero totale di giri della gara."
    )
    pit_input = st.text_input(
        "Pit stop (es. 15,35)",
        value="15,35",
        help="Inserisci i giri in cui vuoi fermarti ai box, separati da virgola."
    )
    try:
        pit_laps = [int(x.strip()) for x in pit_input.split(',') if x.strip().isdigit()]
    except Exception:
        pit_laps = []
        st.warning("Formato pit stop non valido. Lasciato vuoto.")

with col2:
    compounds_input = st.text_input(
        "Strategia gomme (es. Soft,Medium,Hard)",
        value="Soft,Medium,Hard",
        help="Inserisci il tipo di gomme da usare per ciascun stint della gara."
    )
    compounds = [x.strip() for x in compounds_input.split(',') if x.strip()]

# Stato base per simulazione
base_state = {
    'Driver': 'HAM',
    'Team': 'Mercedes',
    'RollingAvgLap': 90.0  # placeholder medio
}

# ----------------------------
# Bottone per simulazione
# ----------------------------
if st.button("🏁 Simula Strategia"):
    if not pit_laps or not compounds:
        st.error("Inserisci almeno un pit stop e una strategia gomme valida!")
    else:
        try:
            df_sim = simulate_strategy(model, base_state, pit_laps, compounds, total_laps)
            total_time = df_sim['PredictedLapTime'].sum()

            st.subheader("📊 Risultato Simulazione")
            st.metric(label="Tempo totale stimato", value=f"{total_time:.1f} s")

            # Grafico tempi per giro
            fig = px.line(
                df_sim,
                x='Lap',
                y='PredictedLapTime',
                color='Compound',
                markers=True,
                title="⏱️ Tempi previsti per giro",
                labels={"PredictedLapTime": "Tempo stimato [s]", "Lap": "Giro", "Compound": "Tipo di gomma"},
                hover_data={"PredictedLapTime": True, "Lap": True, "Compound": True}
            )
            fig.update_layout(
                legend_title_text='Gomme',
                plot_bgcolor='white',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Grafico accumulato
            df_sim["CumulativeTime"] = df_sim["PredictedLapTime"].cumsum()
            fig_cum = px.line(
                df_sim,
                x='Lap',
                y='CumulativeTime',
                color='Compound',
                markers=True,
                title="⏳ Tempo cumulativo durante la gara",
                labels={"CumulativeTime": "Tempo totale [s]", "Lap": "Giro"}
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # Tabella dettagliata
            st.subheader("📋 Tabella dettagliata giri")
            st.dataframe(df_sim)

        except Exception as e:
            st.error(f"⚠️ Errore nella simulazione: {e}")

# 🏎️ F1 LapDelta Predictor — Engineering Suite

> **Predizione e analisi avanzata delle performance in Formula 1** con modelli *LightGBM*, analisi *SHAP* e visualizzazioni interattive in tempo reale.

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-00B300?logo=lightgbm&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🧩 Cos’è

**F1 LapDelta Predictor — Engineering Suite** è una piattaforma interattiva per **analizzare, predire e visualizzare il *Lap Delta*** (cioè la differenza di tempo tra giri) nelle gare di Formula 1.  
Combina **machine learning**, **feature engineering** e **telemetria comparativa** per fornire insight ingegneristici sul passo gara dei piloti.

L’applicazione include:
- 🚀 **Predizione live del Lap Delta** basata su modelli LightGBM multi-fold  
- 📊 **Dashboard interattiva Streamlit** con analisi storiche e telemetria comparativa  
- 🧠 **Interpretabilità con SHAP** per capire il contributo di ogni feature  
- ⚙️ **Suite completa di preprocessing e training script** per gestire l’intero ciclo ML  

---

## 🧱 Architettura del Progetto

f1-lapdelta-predictor/
│
├── data/
│ ├── processed/
│ │ ├── laps_clean_final.parquet
│ │ ├── laps_with_predictions.parquet
│ └── models/
│ └── <timestamped_model_folder>/
│ ├── model.txt
│ ├── feature_info.joblib
│ └── ...
│
├── src/
│ ├── app.py # App Streamlit principale
│ ├── features.py # Generazione feature da dati grezzi
│ ├── train_model.py # Addestramento modelli LightGBM
│ ├── run_post_processing.py # Post-elaborazione predizioni
│ └── clean_features.py # Pulizia e normalizzazione dati
│
├── requirements.txt
└── README.md


## ⚙️ Setup & Installazione

### 1️⃣ Clona il repository
git clone https://github.com/Ilaria1655/f1-lapdelta-predictor
cd f1-lapdelta-predictor

### 2️⃣ Crea l’ambiente virtuale e installa le dipendenze
python -m venv .venv
source .venv/bin/activate   # su macOS/Linux
.venv\Scripts\activate      # su Windows
pip install -r requirements.txt

### 3️⃣ Prepara i dati
Assicurati di avere i dati pre-elaborati nella cartella `data/processed/`.  
Se non li hai, esegui in ordine:
python src/clean_features.py
python src/train_model.py
python src/run_post_processing.py
NB: I dataset originali sono stati scaricati da Kaggle, e sono necessari per il funzionamento dell'app.

### 4️⃣ Avvia l’applicazione Streamlit
streamlit run app.py

---

## 🧠 Principali Funzionalità

Funzione: 🏎️ Predizione Lap Delta  
Descrizione: Calcola in tempo reale la variazione di performance per pilota/circuito  

Funzione: 📈 Analisi Storica  
Descrizione: Visualizza il confronto tra dati reali e predetti per ogni sessione  

Funzione: 🧮 SHAP Insights  
Descrizione: Mostra il contributo di ogni feature nella predizione  

Funzione: 📡 Telemetria Comparativa  
Descrizione: Confronta il passo gara tra due piloti sullo stesso circuito  

Funzione: 🛠️ Model Analytics  
Descrizione: Mostra importanza feature, cross-validation e dettagli di training

---

## 🧩 Stack Tecnologico

- Python 3.10+  
- LightGBM — Modello di machine learning principale  
- Streamlit — Interfaccia web interattiva  
- Plotly — Visualizzazioni dinamiche e grafiche  
- SHAP — Explainability delle predizioni  
- Pandas / NumPy / Scikit-learn — Analisi e preprocessing dati

---

## 🧪 Esempio di utilizzo

Una volta avviata l’app:  
Seleziona il pilota, il circuito e i parametri gomma nella sidebar.  
Ottieni la predizione del Lap Delta con confidenza e insight SHAP.  

Esplora i tab:  
- Analisi Storica → confronto tra giri reali e previsti  
- SHAP → contributo delle feature alla singola predizione  
- Telemetria Comparativa → confronto tra due piloti  
- Dettagli Modello → performance, importanza e metadati del training

---

## 📊 Esempi di Visualizzazione

Analisi Storica | SHAP Waterfall | Telemetria Comparativa  
![Example1](https://github.com/user-attachments/assets/placeholder1) | ![Example2](https://github.com/user-attachments/assets/placeholder2) | ![Example3](https://github.com/user-attachments/assets/placeholder3)  


---


## 👨‍💻 Autore

F1 LapDelta Predictor — Engineering Suite  
Realizzato con ❤️ da Ilaria Fantasia

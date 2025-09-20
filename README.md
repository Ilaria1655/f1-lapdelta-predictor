# F1 Strategy Simulator


Un'app Streamlit che utilizza Machine Learning (Random Forest / LightGBM) per simulare e confrontare strategie di gara in Formula 1.


## 🚀 Struttura
- `src/data_prep.py` — caricamento e preprocessing dati (FastF1 + Kaggle)
- `src/features.py` — feature engineering
- `src/train_model.py` — training e tuning modelli ML
- `src/model_utils.py` — funzioni di utilità per salvare/caricare pipeline
- `src/simulate.py` — logica simulazione strategie
- `app.py` — Streamlit app per simulare strategie e visualizzare risultati
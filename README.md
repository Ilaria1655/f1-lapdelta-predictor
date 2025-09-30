# 🏎️ F1 LapDelta Predictor 2024

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-success)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-orange)](https://optuna.org/)
![Accuracy](https://img.shields.io/badge/R²≈1.0-brightgreen)
![Dataset Size](https://img.shields.io/badge/Dataset->50k%20laps-lightgrey)

---

## 📌 Overview

Questo progetto introduce un **modello predittivo per il LapDelta in Formula 1**, costruito con **LightGBM** e integrato in una **dashboard interattiva Streamlit**.
Il sistema è in grado di prevedere con estrema precisione la differenza di tempo per giro (`LapDelta`) a partire da feature tecniche e contestuali, rendendolo uno strumento utile per analisi strategiche e simulazioni.

La pipeline completa combina:

* **Feature engineering avanzata** (indicatori di degrado gomma, rolling average, fattori età giro, ecc.)
* **Validazione rigorosa con GroupKFold** (pilota + circuito)
* **Ottimizzazione iperparametri con Optuna**
* **Dashboard user-friendly** per interazioni in tempo reale

---

## ⚙️ Architettura del Progetto

```
├── data/
│   ├── raw/                # Dataset raw (.csv)
│   ├── processed/          # Dataset pulito (.parquet)
│   └── models/             # Modelli addestrati e validazione salvata
├── src/
│   ├── data_prep.py        # Raccolta dati da *.csv
│   ├── features.py         # Feature engeneering
│   ├── clean_features.py   # Pulizia dataset
│   ├── model_utils.py      # Supporto salvataggio modelli
│   ├── train_model.py      # Training, feature engineering, salvataggio modelli
│   └── app.py              # Dashboard Streamlit per predizione e visualizzazione
├── README.md
├── requirements.txt
```

---

## 🧩 Feature Principali

* **Input dinamico**: selezione pilota, circuito, giro, mescola, età pneumatico
* **Predizione LapDelta in tempo reale** con media su più fold del modello
* **Metriche reali di validazione** mostrate nella dashboard (MAE, RMSE, R²)
* **Confronto storico** con performance medie passate dello stesso pilota
* **Visualizzazioni interattive** con Plotly (bar chart, boxplot, distribuzioni)

---

## 🔬 Modello Predittivo

* **Algoritmo**: LightGBM Regressor
* **Validazione**: GroupKFold (3 fold, raggruppati per pilota + circuito)
* **Iperparametri**: ottimizzati con Optuna (25 trial, early stopping)
* **Target**: `LapDelta` (scalato ×10 per stabilità numerica durante il training)
* **Feature derivate principali**:

  * `LapDiffFromRollingAvg`
  * `TyreEff` (età gomma × degradazione)
  * `LapAgeFactor`
  * `PrevLapDelta`

---

## 📊 Metriche (su fold di validazione)

| Metric | Mean (3 fold) |
| ------ | ------------- |
| MAE    | ~0.02 s       |
| RMSE   | ~0.03 s       |
| R²     | ≈ 1.00        |

---

## 🎛️ Dashboard Streamlit

La dashboard è stata sviluppata per **analizzare, confrontare e predire** in modo immediato.

### 🚀 Demo (GIF)

👉 Inserisci qui la tua registrazione GIF della dashboard:

![Demo](demo.gif)

*Tip*: puoi registrare con [LICEcap](https://www.cockos.com/licecap/) o OBS e salvare direttamente in `.gif`.

---

## 📥 Installazione & Utilizzo

1. **Clona la repo**

   ```bash
   git clone https://github.com/tuouser/f1-lapdelta-predictor.git
   cd f1-lapdelta-predictor
   ```

2. **Crea l’ambiente**

   ```bash
   conda create -n f1predictor python=3.9 -y
   conda activate f1predictor
   pip install -r requirements.txt
   ```

3. **Avvia il training**

   ```bash
   python src/train_model.py
   ```

4. **Lancia la dashboard**

   ```bash
   streamlit run src/app.py
   ```

---

## 📖 Riferimenti

* [LightGBM Documentation](https://lightgbm.readthedocs.io/)
* [Optuna HPO](https://optuna.org/)
* [Streamlit](https://streamlit.io/)
* Dataset Formula 1 rielaborato da telemetria/lap data ufficiale (pre-elaborato in `laps_clean_final.parquet`)

---


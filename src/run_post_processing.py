import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
import logging

# =================================================================================
# CONFIGURAZIONE
# =================================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# =================================================================================
# FUNZIONI HELPER
# =================================================================================
def load_latest_model_assets(_models_dir):
    """Carica il modello più recente e le informazioni sulle feature."""
    logger.info("Caricamento del modello più recente...")
    folders = sorted([d for d in _models_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    if not folders:
        logger.error("Nessuna cartella di modello trovata.")
        return None, None
    latest = folders[0]
    logger.info(f"Modello trovato: {latest.name}")

    feature_info = joblib.load(latest / "feature_info.joblib")
    model_paths = feature_info.get("model_paths", [])
    models = [lgb.Booster(model_file=str(p)) for p in model_paths if Path(p).exists()]

    return feature_info, models

def ensure_categorical(df: pd.DataFrame, cat_cols, feature_info):
    """Converte le colonne in tipo 'category' usando le categorie salvate."""
    from pandas.api.types import CategoricalDtype
    for c in cat_cols:
        if c in df.columns:
            categories = feature_info.get('categories', {}).get(c)
            if categories:
                cat_dtype = CategoricalDtype(categories=categories, ordered=False)
                df[c] = df[c].astype(cat_dtype)
    return df

def ensemble_predict(models, X):
    """Esegue la predizione con l'ensemble di modelli."""
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)

# =================================================================================
# SCRIPT PRINCIPALE
# =================================================================================
def main():
    """
    Esegue lo script di post-processing:
    1. Carica i dati dei giri puliti.
    2. Carica l'ultimo modello addestrato.
    3. Calcola le previsioni per l'intero dataset.
    4. Salva il dataset arricchito con le previsioni.
    """
    logger.info("Avvio dello script di post-processing...")

    # --- Caricamento assets ---
    feature_info, models = load_latest_model_assets(MODELS_DIR)
    if feature_info is None or not models:
        logger.error("Impossibile caricare il modello. Uscita.")
        return

    logger.info("Caricamento dati dei giri...")
    laps_df = pd.read_parquet(PROCESSED_DIR / "laps_clean_final.parquet")
    if laps_df.empty:
        logger.error("Il file laps_clean_final.parquet è vuoto. Uscita.")
        return

    # --- Feature Engineering ---
    logger.info("Esecuzione del feature engineering sui dati...")
    required_features = {
        'LapDiffFromRollingAvg': lambda d: d['LapNumber'] - d['RollingAvgLap'],
        'TyreEff': lambda d: d['TyreAge'] * d['DegradationRate'],
        'LapAgeFactor': lambda d: d['LapNumber'] / (d['TyreAge'] + 1),
        'PrevLapDelta': lambda d: d.groupby(['Driver', 'CircuitId'])['LapDelta'].shift(1)
    }
    for feature, func in required_features.items():
        if feature not in laps_df.columns:
            laps_df[feature] = func(laps_df)
    laps_df['PrevLapDelta'] = laps_df['PrevLapDelta'].fillna(0)

    # --- Preparazione dati per la predizione ---
    feature_cols = feature_info['ALL_COLS']
    cat_cols = feature_info.get('CAT_COLS', [])

    # Assicura che tutte le colonne siano presenti
    for col in feature_cols:
        if col not in laps_df.columns:
            laps_df[col] = 0
            logger.warning(f"Colonna '{col}' mancante, aggiunta con valore 0.")

    X = laps_df[feature_cols].copy()
    X = ensure_categorical(X, cat_cols, feature_info)

    # --- Esecuzione Predizioni ---
    logger.info(f"Esecuzione delle predizioni su {len(X)} giri...")
    predictions = ensemble_predict(models, X) / 10.0
    laps_df['PredictedDelta'] = predictions
    logger.info("Predizioni calcolate con successo.")

    # --- Salvataggio del file arricchito ---
    output_path = PROCESSED_DIR / "laps_with_predictions.parquet"
    laps_df.to_parquet(output_path, index=False)
    logger.info(f"Dati arricchiti salvati in: {output_path}")
    logger.info("Post-processing completato.")

if __name__ == "__main__":
    main()


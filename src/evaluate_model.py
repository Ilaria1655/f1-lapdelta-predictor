# src/evaluate_model.py

from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from features import add_features  # se hai feature aggiuntive, altrimenti rimuovi

def evaluate_model(df: pd.DataFrame, pit_df: pd.DataFrame, model_path=None):
    """Valuta il modello lap-based e mostra le prime predizioni"""

    # 🔹 Imposta percorso modello se non fornito
    if model_path is None:
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        model_path = processed_dir / "lap_model.joblib"

    # 🔹 Carica modello
    model = joblib.load(model_path)
    print(f"✅ Modello caricato da {model_path}")

    # 🔹 Applica eventuali feature aggiuntive
    df = add_features(df, pit_df)

    # 🔹 Aggiungi pit count cumulativo
    pit_count = pit_df.groupby(['raceId', 'driverId'])['lap'].count().reset_index()
    pit_count.rename(columns={'lap': 'TotalPits'}, inplace=True)
    df = df.merge(pit_count, on=['raceId', 'driverId'], how='left')
    df['TotalPits'] = df['TotalPits'].fillna(0)

    # 🔹 Definizione feature e target
    NUM_COLS = ['LapNumber', 'TotalPits']
    if 'Speed' in df.columns:
        NUM_COLS.append('Speed')
    CAT_COLS = ['Driver', 'Circuit', 'Compound']

    # 🔹 Assicurati che tutte le colonne categoriche esistano
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = 'Unknown'

    X = df[NUM_COLS + CAT_COLS]
    y_true = df['LapTimeSeconds']

    # 🔹 Predizioni
    y_pred = model.predict(X)

    # 🔹 Metriche
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Metriche valutazione modello:")
    print(f"   MAE  : {mae:.3f} sec")
    print(f"   RMSE : {rmse:.3f} sec")
    print(f"   R²   : {r2:.3f}")

    # 🔹 Mostra prime 10 predizioni con dettagli
    sample = pd.DataFrame({
        "Driver": df["Driver"],
        "LapNumber": df["LapNumber"],
        "LapTime_Real": y_true,
        "LapTime_Pred": y_pred
    }).head(10)

    print("\n🔎 Prime 10 predizioni:")
    print(sample)


if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    laps_path = processed_dir / "laps_features.parquet"
    pit_path = processed_dir / "pit_stops.parquet"

    df = pd.read_parquet(laps_path)
    pit_df = pd.read_parquet(pit_path)

    evaluate_model(df, pit_df)

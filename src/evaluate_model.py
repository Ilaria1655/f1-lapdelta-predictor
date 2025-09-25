from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(df: pd.DataFrame, model_path=None):
    """
    Valuta modello LapDelta su df.
    df deve avere le colonne numeriche/categoriche usate nel modello.
    """
    # -------------------------
    # Caricamento modello
    # -------------------------
    if model_path is None:
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        model_path = processed_dir / "lapdelta_model.joblib"

    model = joblib.load(model_path)
    print(f"✅ Modello caricato da {model_path}")

    # -------------------------
    # Colonne usate dal modello
    # -------------------------
    NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate', 'Stint']
    CAT_COLS = ['Driver', 'Compound', 'CircuitId', 'IsOutLap']
    TARGET = 'LapDelta'

    # -------------------------
    # Gestione colonne mancanti
    # -------------------------
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0.0  # default numerico
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = 'Unknown'  # default categoriale

    # converti boolean in stringa per OneHotEncoder
    if 'IsOutLap' in df.columns:
        df['IsOutLap'] = df['IsOutLap'].astype(str)

    # -------------------------
    # Preparazione dati
    # -------------------------
    X = df[NUM_COLS + CAT_COLS]
    y_true = df[TARGET]

    # -------------------------
    # Predizione e metriche
    # -------------------------
    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Metriche valutazione modello LapDelta:")
    print(f"   MAE  : {mae:.3f} sec")
    print(f"   RMSE : {rmse:.3f} sec")
    print(f"   R²   : {r2:.3f}")

    # -------------------------
    # Prime 10 predizioni
    # -------------------------
    sample = pd.DataFrame({
        "Driver": df.get("Driver"),
        "CircuitId": df.get("CircuitId"),
        "LapNumber": df.get("LapNumber"),
        "LapDelta_Real": y_true,
        "LapDelta_Pred": y_pred
    }).head(10)

    print("\n🔎 Prime 10 predizioni:")
    print(sample)

if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    laps_path = processed_dir / "laps_clean.parquet"

    if not laps_path.exists():
        print(f"⚠️ File non trovato: {laps_path}")
    else:
        df = pd.read_parquet(laps_path)
        if df.empty:
            print(f"⚠️ File vuoto: {laps_path}")
        else:
            evaluate_model(df)

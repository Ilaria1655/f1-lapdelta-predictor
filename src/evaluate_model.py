from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(df: pd.DataFrame, model_path=None):
    """
    Valuta modello LapDelta su df che deve avere le colonne numeriche/categoriche usate per il model.
    """
    if model_path is None:
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        model_path = processed_dir / "lapdelta_model.joblib"

    model = joblib.load(model_path)
    print(f"✅ Modello caricato da {model_path}")

    NUM_COLS = ['LapNumber', 'RollingAvgLap', 'TyreAge', 'DegradationRate']
    CAT_COLS = ['Driver', 'Compound']
    TARGET = 'LapDelta'

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = 'Unknown'

    X = df[NUM_COLS + CAT_COLS]
    y_true = df[TARGET]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Metriche valutazione modello LapDelta:")
    print(f"   MAE  : {mae:.3f} sec")
    print(f"   RMSE : {rmse:.3f} sec")
    print(f"   R²   : {r2:.3f}")

    sample = pd.DataFrame({
        "Driver": df.get("Driver"),
        "LapNumber": df.get("LapNumber"),
        "LapDelta_Real": y_true,
        "LapDelta_Pred": y_pred
    }).head(10)

    print("\n🔎 Prime 10 predizioni:")
    print(sample)

if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    laps_path = processed_dir / "laps_clean.parquet"

    df = pd.read_parquet(laps_path)

    evaluate_model(df)

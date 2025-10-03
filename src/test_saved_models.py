import joblib
import logging
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------- Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------------- Directory setup
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
processed_dir = data_dir / "processed"
models_dir = data_dir / "models"

def get_latest_model_folder():
    """Trova la cartella modello più recente in data/models"""
    subdirs = [f for f in models_dir.iterdir() if f.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"Nessuna cartella trovata in {models_dir}")
    latest = max(subdirs, key=lambda x: x.stat().st_mtime)
    return latest

def test_on_final_set(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Cartella {folder} non trovata!")

    # Carico info feature
    feature_info_path = folder / "feature_info.joblib"
    if not feature_info_path.exists():
        raise FileNotFoundError(f"feature_info.joblib non trovato in {folder}")

    feature_info = joblib.load(feature_info_path)
    model_paths = feature_info["model_paths"]

    # Carico test set
    X_test = joblib.load(processed_dir / "X_test.joblib")
    y_test = joblib.load(processed_dir / "y_test.joblib")

    logger.info(f"Caricamento test set finale: {X_test.shape[0]} esempi")
    metrics = []

    for fold, model_path in enumerate(model_paths):
        # Carico modello LightGBM
        model = lgb.Booster(model_file=model_path)

        # Predizione sul test
        y_pred = model.predict(X_test)

        # Calcolo metriche
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics.append((mae, rmse, r2))
        logger.info(f"[Model {fold+1}] MAE: {mae/10:.3f} sec | RMSE: {rmse/10:.3f} sec | R²: {r2:.3f}")

    # Media tra i modelli
    mae_mean = np.mean([m[0] for m in metrics])
    rmse_mean = np.mean([m[1] for m in metrics])
    r2_mean = np.mean([m[2] for m in metrics])

    logger.info("\n📊 Metriche medie sul test finale:")
    logger.info(f"MAE: {mae_mean/10:.3f} sec")
    logger.info(f"RMSE: {rmse_mean/10:.3f} sec")
    logger.info(f"R²: {r2_mean:.3f}")

if __name__ == "__main__":
    latest_folder = get_latest_model_folder()
    logger.info(f"Ultima cartella modello trovata: {latest_folder}")
    test_on_final_set(latest_folder)

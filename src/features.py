# src/features.py

from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle e file
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / 'data' / 'processed'
raw_dir = root_dir / 'data' / 'raw'

laps_path = processed_dir / 'laps_processed.parquet'
pit_stops_path = processed_dir / 'pit_stops.parquet'  # ora parquet

# CSV storici
races_path = raw_dir / 'races.csv'
results_path = raw_dir / 'results.csv'
status_path = raw_dir / 'status.csv'
circuits_path = raw_dir / 'circuits.csv'
drivers_path = raw_dir / 'drivers.csv'

# -------------------------
# 2️⃣ Carica dati
# -------------------------
laps_df = pd.read_parquet(laps_path)
pit_df = pd.read_parquet(pit_stops_path)

races_df = pd.read_csv(races_path)
results_df = pd.read_csv(results_path)
status_df = pd.read_csv(status_path)
circuits_df = pd.read_csv(circuits_path)
drivers_df = pd.read_csv(drivers_path)

# -------------------------
# 3️⃣ Dizionario driverId -> sigla FastF1
# -------------------------
piloti_f1 = drivers_df[['driverId', 'code']].dropna(subset=['code'])
DRIVER_MAP = dict(zip(piloti_f1['driverId'], piloti_f1['code']))

# -------------------------
# 4️⃣ Funzione per aggiungere feature
# -------------------------
def add_features(df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature chiave per il modello:
    - Stint corretto
    - TyreAge
    - DegradationRate
    - IsOutLap
    - RollingAvgLap
    """
    # 1️⃣ Calcola Stint corretto
    pit_laps_map = pit_df.groupby('driverId')['lap'].apply(list).to_dict()

    def get_stint(row):
        driver = row['Driver']
        lap = row['LapNumber']
        driver_ids = [k for k, v in DRIVER_MAP.items() if v == driver]
        if not driver_ids:
            return 1
        driver_id = driver_ids[0]
        pit_laps = sorted(pit_laps_map.get(driver_id, []))
        stint = 1
        for p in pit_laps:
            if lap > p:
                stint += 1
        return stint

    df['Stint'] = df.apply(get_stint, axis=1)

    # 2️⃣ TyreAge: numero di giri nello stesso stint
    df['TyreAge'] = df.groupby(['Driver', 'Stint']).cumcount()

    # 3️⃣ IsOutLap: True se giro subito dopo un pit stop
    def is_out_lap(row):
        driver = row['Driver']
        lap = row['LapNumber']
        driver_ids = [k for k, v in DRIVER_MAP.items() if v == driver]
        if not driver_ids:
            return False
        driver_id = driver_ids[0]
        pit_laps = pit_laps_map.get(driver_id, [])
        return (lap - 1) in pit_laps

    df['IsOutLap'] = df.apply(is_out_lap, axis=1)

    # 4️⃣ DegradationRate: differenza col giro precedente nello stesso stint
    df['DegradationRate'] = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].diff().fillna(0)

    # 5️⃣ Rolling average ultimi 3 giri
    df['RollingAvgLap'] = df.groupby('Driver')['LapTimeSeconds'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # 6️⃣ Compound placeholder se non c'è
    if 'Compound' not in df.columns:
        df['Compound'] = 'Soft'

    return df

# -------------------------
# 5️⃣ Applica features
# -------------------------
laps_features = add_features(laps_df, pit_df)

# -------------------------
# 6️⃣ Salva nuovo Parquet pronto per il modello
# -------------------------
output_path = processed_dir / 'laps_features.parquet'
laps_features.to_parquet(output_path, index=False)
print(f"✅ Dataset con feature salvato in {output_path}")

from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
raw_dir = root_dir / 'data' / 'raw'
processed_dir = root_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 2️⃣ Caricamento dati
# -------------------------
laps_path = processed_dir / 'laps_processed.parquet'
pit_stops_path = processed_dir / 'pit_stops.parquet'
drivers_path = raw_dir / 'drivers.csv'

laps_df = pd.read_parquet(laps_path) if laps_path.exists() else pd.DataFrame()
pit_df = pd.read_parquet(pit_stops_path) if pit_stops_path.exists() else pd.DataFrame()
drivers_df = pd.read_csv(drivers_path)

# Mappa driverId -> codice
piloti_f1 = drivers_df[['driverId', 'code']].dropna(subset=['code'])
DRIVER_MAP = dict(zip(piloti_f1['driverId'], piloti_f1['code']))

# -------------------------
# 3️⃣ Funzione di feature engineering
# -------------------------
def add_features(df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea feature:
    - Stint
    - TyreAge
    - IsOutLap
    - DegradationRate
    - RollingAvgLap
    - LapDelta (rispetto al best lap del driver nella stessa gara)
    """
    if df.empty:
        return df

    # mappa pit laps per driverId
    pit_laps_map = pit_df.groupby('driverId')['lap'].apply(list).to_dict()

    # Stint
    def get_stint(row):
        driver_id = row.get('driverId', None)
        if pd.isna(driver_id):
            # fallback mapping da codice
            codes = [k for k, v in DRIVER_MAP.items() if v == row.get('Driver')]
            driver_id = codes[0] if codes else None
        pit_laps = sorted(pit_laps_map.get(driver_id, [])) if driver_id is not None else []
        stint = 1
        for p in pit_laps:
            if row['LapNumber'] > p:
                stint += 1
        return stint

    df = df.copy()
    df['Stint'] = df.apply(get_stint, axis=1)

    # TyreAge
    df['TyreAge'] = df.groupby(['Driver', 'Stint']).cumcount()

    # IsOutLap
    df['IsOutLap'] = df.apply(
        lambda row: (row['LapNumber'] - 1) in pit_laps_map.get(row.get('driverId'), []), axis=1
    )

    # DegradationRate
    df['DegradationRate'] = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].diff().fillna(0)

    # RollingAvgLap
    df['RollingAvgLap'] = df.groupby('Driver')['LapTimeSeconds'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Compound fallback
    if 'Compound' not in df.columns:
        df['Compound'] = 'Unknown'

    # LapDelta
    if 'RaceId' in df.columns:
        df['MinLapByDriverRace'] = df.groupby(['Driver', 'RaceId'])['LapTimeSeconds'].transform('min')
    else:
        df['MinLapByDriverRace'] = df.groupby('Driver')['LapTimeSeconds'].transform('min')

    df['LapDelta'] = df['LapTimeSeconds'] - df['MinLapByDriverRace']

    return df

# -------------------------
# 4️⃣ Applica e salva
# -------------------------
laps_features = add_features(laps_df, pit_df)
output_path = processed_dir / 'laps_features.parquet'
laps_features.to_parquet(output_path, index=False)
print(f"✅ Dataset con feature e LapDelta salvato in {output_path}")

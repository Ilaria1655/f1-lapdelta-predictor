from pathlib import Path
import pandas as pd

root_dir = Path(__file__).parent.parent
processed_dir = root_dir / 'data' / 'processed'

laps_path = processed_dir / 'laps_processed.parquet'
pit_stops_path = processed_dir / 'pit_stops.parquet'

# Carica
laps_df = pd.read_parquet(laps_path)
pit_df = pd.read_parquet(pit_stops_path)

# Mappa driverId -> codice (se servisse)
drivers_path = root_dir / 'data' / 'raw' / 'drivers.csv'
drivers_df = pd.read_csv(drivers_path)
piloti_f1 = drivers_df[['driverId', 'code']].dropna(subset=['code'])
DRIVER_MAP = dict(zip(piloti_f1['driverId'], piloti_f1['code']))

def add_features(df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea Stint, TyreAge, IsOutLap, DegradationRate, RollingAvgLap (senza leakage)
    e LapDelta (target): LapTimeSeconds - best lap del driver nella stessa gara (RaceId).
    """
    # mappa pit laps per driverId
    pit_laps_map = pit_df.groupby('driverId')['lap'].apply(list).to_dict()

    # costruisco Stint usando driverId quando possibile
    def get_stint(row):
        driver_id = row.get('driverId', None)
        if pd.isna(driver_id):
            # try mapping by code
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

    # TyreAge: numero di giro nello stesso stint (0-based)
    df['TyreAge'] = df.groupby(['Driver', 'Stint']).cumcount()

    # IsOutLap: True se il giro è subito dopo un pit
    def is_out_lap(row):
        driver_id = row.get('driverId', None)
        if pd.isna(driver_id):
            codes = [k for k, v in DRIVER_MAP.items() if v == row.get('Driver')]
            driver_id = codes[0] if codes else None
        pit_laps = pit_laps_map.get(driver_id, []) if driver_id is not None else []
        return (row['LapNumber'] - 1) in pit_laps

    df['IsOutLap'] = df.apply(is_out_lap, axis=1)

    # DegradationRate: differenza rispetto al giro precedente nello stesso stint
    df['DegradationRate'] = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].diff().fillna(0)

    # RollingAvgLap: media ultimi 3 giri **precedenti** al giro corrente (shift)
    df['RollingAvgLap'] = df.groupby('Driver')['LapTimeSeconds'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Compound fallback
    if 'Compound' not in df.columns:
        df['Compound'] = 'Unknown'

    # LapDelta: rispetto al best lap dello stesso pilota nella stessa gara (RaceId)
    if 'RaceId' in df.columns:
        df['MinLapByDriverRace'] = df.groupby(['Driver', 'RaceId'])['LapTimeSeconds'].transform('min')
    else:
        # se manca RaceId, fallback a best driver in tutto il dataset (meno ideale)
        df['MinLapByDriverRace'] = df.groupby('Driver')['LapTimeSeconds'].transform('min')

    df['LapDelta'] = df['LapTimeSeconds'] - df['MinLapByDriverRace']

    return df

# applica e salva
laps_features = add_features(laps_df, pit_df)
output_path = processed_dir / 'laps_features.parquet'
laps_features.to_parquet(output_path, index=False)
print(f"✅ Dataset con feature e LapDelta salvato in {output_path}")

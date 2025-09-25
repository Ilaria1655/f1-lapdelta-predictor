from pathlib import Path
import fastf1 as ff1
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
raw_dir = root_dir / 'data' / 'raw'
processed_dir = root_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

cache_dir = root_dir / 'fastf1-cache'
cache_dir.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)

# -------------------------
# 2️⃣ Funzioni di utilità
# -------------------------
def read_csv_safe(path: Path, na_values='\\N') -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️ File non trovato: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, na_values=na_values)

def save_parquet(df: pd.DataFrame, path: Path):
    if df.empty:
        print(f"⚠️ DataFrame vuoto, non salvo {path}")
        return
    df.to_parquet(path, index=False)
    print(f"✅ Salvato parquet in {path}")

def load_fastf1_laps(year: int, gp: str, session: str = 'R') -> pd.DataFrame:
    sess = ff1.get_session(year, gp, session)
    sess.load()
    laps = sess.laps.reset_index()
    print("Colonne FastF1 disponibili:", laps.columns.tolist())
    if 'LapTime' in laps.columns:
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    print("Driver unici FastF1:", laps['Driver'].unique())
    print("LapNumber unici FastF1:", laps['LapNumber'].unique()[:10])
    return laps

def merge_kaggle_with_fastf1(kaggle_path: Path, laps: pd.DataFrame, races_parquet: Path, gp_name: str) -> pd.DataFrame:
    kaggle = read_csv_safe(kaggle_path)
    if kaggle.empty:
        return pd.DataFrame()

    kaggle['Driver'] = kaggle['driverId'].map(DRIVER_MAP)
    laps_filtered = laps[laps['Driver'].isin(VALID_DRIVER_CODES)]
    kaggle_filtered = kaggle[kaggle['Driver'].isin(VALID_DRIVER_CODES)]

    merged = pd.merge(
        laps_filtered,
        kaggle_filtered,
        left_on=['Driver', 'LapNumber'],
        right_on=['Driver', 'lap'],
        how='inner',
        suffixes=('_ff1', '_kaggle')
    )

    if merged.empty:
        print("⚠️ Merge vuoto: controlla driver e numeri giri.")
        return merged

    races_df = read_csv_safe(raw_dir / 'races.csv')
    save_parquet(races_df, processed_dir / 'races.parquet')  # sempre salvato

    race_match = races_df[races_df['name'] == gp_name]
    if race_match.empty:
        print(f"⚠️ Gara '{gp_name}' non trovata")
        return merged

    merged['RaceId'] = race_match['raceId'].values[0]
    merged['CircuitId'] = race_match['circuitId'].values[0]
    print(f"✅ Merge completato: {len(merged)} righe")
    return merged

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' in df.columns:
        def time_to_seconds(t):
            try:
                if pd.isna(t):
                    return None
                mins, secs = t.split(":")
                return int(mins)*60 + float(secs)
            except:
                return None
        df['LapTimeSeconds'] = df['time'].apply(time_to_seconds)
    if 'LapTimeSeconds' not in df.columns:
        df['LapTimeSeconds'] = 90.0
    if 'Compound' not in df.columns:
        df['Compound'] = 'Unknown'
    return df

# -------------------------
# 3️⃣ Caricamento CSV e creazione mappe driver
# -------------------------
drivers_df = read_csv_safe(raw_dir / 'drivers.csv')
drivers_df['number'] = pd.to_numeric(drivers_df['number'], errors='coerce')
piloti_f1 = drivers_df.dropna(subset=['number', 'code'])
DRIVER_MAP = dict(zip(piloti_f1['number'].astype(int), piloti_f1['code']))
VALID_DRIVER_CODES = set(DRIVER_MAP.values())

# Salvataggio CSV come parquet
for csv_name in ['drivers.csv', 'circuits.csv', 'results.csv', 'status.csv', 'pit_stops.csv']:
    df = read_csv_safe(raw_dir / csv_name)
    save_parquet(df, processed_dir / csv_name.replace('.csv', '.parquet'))

# -------------------------
# 4️⃣ Esecuzione script
# -------------------------
if __name__ == "__main__":
    print("Eseguo data_prep...")

    try:
        laps_fastf1 = load_fastf1_laps(2023, 'Monza', 'R')
    except Exception as e:
        print("⚠️ Errore FastF1:", e)
        laps_fastf1 = pd.DataFrame()

    kaggle_csv_path = raw_dir / 'lap_times.csv'
    if not laps_fastf1.empty and kaggle_csv_path.exists():
        merged = merge_kaggle_with_fastf1(kaggle_csv_path, laps_fastf1, processed_dir / "races.parquet", gp_name='Italian Grand Prix')
        if not merged.empty:
            merged = add_features(merged)
            save_parquet(merged, processed_dir / 'laps_processed.parquet')
        else:
            print("⚠️ Merge vuoto, laps_processed.parquet non salvato.")
    else:
        print("⚠️ Skip merge: FastF1 vuoto o CSV Kaggle mancante.")

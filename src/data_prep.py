from pathlib import Path
import fastf1 as ff1
import pandas as pd

# -------------------------
# 1️⃣ Carica drivers.csv e crea mappa piloti
# -------------------------
drivers_path = Path(__file__).parent.parent / 'data' / 'raw' / 'drivers.csv'
drivers_df = pd.read_csv(drivers_path)

piloti_f1 = drivers_df[['driverId', 'code', 'forename', 'surname']].dropna(subset=['code'])

DRIVER_MAP = dict(zip(piloti_f1['driverId'], piloti_f1['code']))
DRIVER_NAME_MAP = dict(zip(piloti_f1['driverId'], piloti_f1['forename'] + " " + piloti_f1['surname']))
VALID_DRIVER_CODES = set(DRIVER_MAP.values())

# -------------------------
# 2️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
cache_dir = root_dir / 'fastf1-cache'
cache_dir.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)

raw_dir = root_dir / 'data' / 'raw'
processed_dir = root_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 3️⃣ Funzioni
# -------------------------
def load_fastf1_laps(year: int, gp: str, session: str = 'R') -> pd.DataFrame:
    """
    Scarica sessione FastF1 (usa cache se presente) e ritorna laps con LapTimeSeconds.
    """
    sess = ff1.get_session(year, gp, session)
    sess.load()
    laps = sess.laps.reset_index()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    return laps


def merge_kaggle_with_fastf1(kaggle_path: str, laps: pd.DataFrame, races_parquet: str, gp_name: str) -> pd.DataFrame:
    """
    Merge dei dati FastF1 con il CSV Kaggle e assegna RaceId (basato su races.parquet).
    Il CSV Kaggle deve contenere 'driverId' (numerico) e 'lap' (numero giro).
    Dopo il merge la tabella contiene driverId (numerico) e Driver (sigla).
    """
    kaggle = pd.read_csv(kaggle_path)
    kaggle['Driver'] = kaggle['driverId'].map(DRIVER_MAP)

    # Filtri su driver validi
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

    races_df = pd.read_parquet(races_parquet)
    race_match = races_df[races_df['name'] == gp_name]
    if race_match.empty:
        raise ValueError(f"⚠️ Gara '{gp_name}' non trovata in races.parquet")

    race_id = race_match['raceId'].values[0]
    merged['RaceId'] = race_id

    print(f"✅ Merge completato: {merged.shape[0]} righe con RaceId={race_id}")
    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funzione di comodità per convertire colonne 'time' (se esistono) e assicurare LapTimeSeconds.
    Non fa feature complesse: le feature avanzate sono in src/features.py
    """
    if 'time' in df.columns:
        def time_to_seconds(t):
            try:
                if pd.isna(t):
                    return None
                mins, secs = t.split(":")
                return int(mins) * 60 + float(secs)
            except:
                return None
        df['LapTimeSeconds'] = df['time'].apply(time_to_seconds)

    # fallback / placeholder
    df['LapTimeSeconds'] = df.get('LapTimeSeconds', pd.Series()).fillna(df['LapTimeSeconds'].median() if 'LapTimeSeconds' in df.columns else 90.0)

    if 'Compound' not in df.columns:
        df['Compound'] = 'Unknown'

    return df


def save_processed(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False)


def save_pit_stops():
    pit_csv_path = raw_dir / "pit_stops.csv"
    if pit_csv_path.exists():
        pit_df = pd.read_csv(pit_csv_path)
        output_path = processed_dir / "pit_stops.parquet"
        pit_df.to_parquet(output_path, index=False)
        print(f"✅ Pit stop salvati in {output_path}")
    else:
        print("⚠️ pit_stops.csv non trovato in raw/")


def save_csv_as_parquet(csv_name: str, parquet_name: str, map_dict=None):
    csv_path = raw_dir / csv_name
    output_path = processed_dir / parquet_name
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if map_dict:
            for col, mapping in map_dict.items():
                df[col] = df[col].map(mapping)
        df.to_parquet(output_path, index=False)
        print(f"✅ {csv_name} salvato come {parquet_name}")
    else:
        print(f"⚠️ {csv_name} non trovato in raw/")


def save_all_csvs():
    save_csv_as_parquet("races.csv", "races.parquet")
    save_csv_as_parquet("circuits.csv", "circuits.parquet")
    save_csv_as_parquet("status.csv", "status.parquet")
    save_csv_as_parquet("drivers.csv", "drivers.parquet", map_dict={"driverId": DRIVER_MAP})
    save_csv_as_parquet("results.csv", "results.parquet", map_dict={"driverId": DRIVER_MAP})


# -------------------------
# 4️⃣ Esecuzione se lanciato come script
# -------------------------
if __name__ == "__main__":
    print("Eseguo data_prep: attenzione, questo script prova a scaricare la sessione FastF1 impostata al suo interno.")
    # esempio predefinito: Monza 2023 finale
    try:
        laps_fastf1 = load_fastf1_laps(2023, 'Monza', 'R')
    except Exception as ex:
        print("⚠️ Errore FastF1 (forse mancano i dati in cache):", ex)
        laps_fastf1 = pd.DataFrame()  # si continua se user vuole usare solo i CSV

    kaggle_csv_path = raw_dir / 'lap_times.csv'
    races_parquet_path = processed_dir / "races.parquet"

    if not kaggle_csv_path.exists():
        print(f"⚠️ File {kaggle_csv_path} non trovato. Mettilo in data/raw/")
    else:
        if laps_fastf1.empty:
            print("⚠️ FastF1 laps vuoto: il merge non verrà eseguito. Puoi usare direttamente i CSV Kaggle.")
        else:
            merged = merge_kaggle_with_fastf1(kaggle_csv_path, laps_fastf1, races_parquet_path, gp_name='Italian Grand Prix')
            merged = add_features(merged)
            output_path = processed_dir / 'laps_processed.parquet'
            save_processed(merged, output_path)
            print(f"✅ Dataset preprocessato salvato in {output_path}")

    save_pit_stops()
    save_all_csvs()

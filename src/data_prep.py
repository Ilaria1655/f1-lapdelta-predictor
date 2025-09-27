from pathlib import Path
import fastf1 as ff1
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
raw_dir = root_dir / "data" / "raw"
processed_dir = root_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

cache_dir = root_dir / "fastf1-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)

# -------------------------
# 2️⃣ Funzioni di utilità
# -------------------------
def read_csv_safe(path: Path, na_values="\\N") -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️ File non trovato: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, na_values=na_values)

def save_parquet(df: pd.DataFrame, path: Path):
    if df.empty:
        print(f"⚠️ DataFrame vuoto, non salvo {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Salvato parquet in {path}")

def load_fastf1_laps(year: int, gp: str, session: str = "R") -> pd.DataFrame:
    sess = ff1.get_session(year, gp, session)
    sess.load()
    laps = sess.laps.reset_index()
    if "LapTime" in laps.columns:
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    return laps

def merge_kaggle_with_fastf1(
    kaggle_path: Path, laps: pd.DataFrame, races_parquet: Path, gp_name: str
) -> pd.DataFrame:
    kaggle = read_csv_safe(kaggle_path)
    if kaggle.empty:
        return pd.DataFrame()

    kaggle["Driver"] = kaggle["driverId"].map(DRIVER_MAP)
    laps_filtered = laps[laps["Driver"].isin(VALID_DRIVER_CODES)]
    kaggle_filtered = kaggle[kaggle["Driver"].isin(VALID_DRIVER_CODES)]

    merged = pd.merge(
        laps_filtered,
        kaggle_filtered,
        left_on=["Driver", "LapNumber"],
        right_on=["Driver", "lap"],
        how="inner",
        suffixes=("_ff1", "_kaggle"),
    )

    if merged.empty:
        return merged

    races_df = read_csv_safe(raw_dir / "races.csv")
    save_parquet(races_df, processed_dir / "races.parquet")

    race_match = races_df[races_df["name"] == gp_name]
    if not race_match.empty:
        merged["RaceId"] = race_match["raceId"].values[0]
        merged["CircuitId"] = race_match["circuitId"].values[0]

    return merged

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        def time_to_seconds(t):
            try:
                if pd.isna(t):
                    return None
                mins, secs = t.split(":")
                return int(mins) * 60 + float(secs)
            except:
                return None
        df["LapTimeSeconds"] = df["time"].apply(time_to_seconds)
    if "LapTimeSeconds" not in df.columns:
        df["LapTimeSeconds"] = 90.0
    if "Compound" not in df.columns:
        df["Compound"] = "Unknown"
    return df

# -------------------------
# 3️⃣ Driver map
# -------------------------
drivers_df = read_csv_safe(raw_dir / "drivers.csv")
drivers_df["number"] = pd.to_numeric(drivers_df["number"], errors="coerce")
piloti_f1 = drivers_df.dropna(subset=["number", "code"])
DRIVER_MAP = dict(zip(piloti_f1["number"].astype(int), piloti_f1["code"]))
VALID_DRIVER_CODES = set(DRIVER_MAP.values())

# Salva altri CSV in parquet
for csv_name in ["drivers.csv", "circuits.csv", "results.csv", "status.csv", "pit_stops.csv"]:
    df = read_csv_safe(raw_dir / csv_name)
    save_parquet(df, processed_dir / csv_name.replace(".csv", ".parquet"))

# -------------------------
# 4️⃣ Funzione principale per una stagione
# -------------------------
def process_season(year: int):
    print(f"\n📅 Elaboro stagione {year}...")

    try:
        schedule = ff1.get_event_schedule(year)
    except Exception as e:
        print(f"❌ Errore caricamento calendario {year}: {e}")
        return

    kaggle_csv_path = raw_dir / "lap_times.csv"
    if not kaggle_csv_path.exists():
        print("❌ lap_times.csv mancante")
        return

    season_dir = processed_dir / str(year)
    season_dir.mkdir(parents=True, exist_ok=True)

    all_laps = []

    for _, event in schedule.iterrows():
        gp_name = event["EventName"]
        round_no = event["RoundNumber"]
        print(f"\n➡️ Processing GP {round_no}: {gp_name}")

        # FastF1
        try:
            laps_fastf1 = load_fastf1_laps(year, round_no, "R")
            if laps_fastf1.empty or len(laps_fastf1) < 20:
                print(f"⚠️ Dati FastF1 insufficienti per {gp_name}, skip.")
                continue
        except Exception as e:
            print(f"⚠️ Errore FastF1 per {gp_name}: {e}")
            continue

        # Merge Kaggle
        merged = merge_kaggle_with_fastf1(
            kaggle_csv_path, laps_fastf1, processed_dir / "races.parquet", gp_name
        )
        if merged.empty or len(merged) < 20:
            print(f"⚠️ Merge non valido per {gp_name}, skip.")
            continue

        merged = add_features(merged)
        merged = merged.dropna(subset=["LapTimeSeconds"])
        if merged.empty:
            print(f"⚠️ Tutti i giri invalidi per {gp_name}, skip.")
            continue

        safe_name = gp_name.replace(" ", "_").replace("/", "_")
        save_parquet(merged, season_dir / f"laps_{round_no}_{safe_name}.parquet")
        all_laps.append(merged)

    # Cumulativo per anno
    if all_laps:
        df_all = pd.concat(all_laps, ignore_index=True)
        save_parquet(df_all, processed_dir / f"laps_{year}_all.parquet")
        print(f"✅ Stagione {year} completata con {len(df_all)} giri validi")
    else:
        print(f"❌ Nessun dato valido per {year}")

# -------------------------
# 5️⃣ Unione multi-anno in laps_processed
# -------------------------
def update_laps_processed():
    parquet_files = sorted(processed_dir.glob("laps_*_all.parquet"))
    if not parquet_files:
        print("❌ Nessun cumulativo annuale trovato")
        return

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Errore caricando {f}: {e}")

    if not dfs:
        print("❌ Nessun dato valido per laps_processed")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    save_parquet(df_all, processed_dir / "laps_processed.parquet")
    print(f"✅ laps_processed aggiornato con {len(df_all)} giri totali")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # 👇 qui scegli gli anni da processare
    for year in [2024]:
        process_season(year)

    update_laps_processed()

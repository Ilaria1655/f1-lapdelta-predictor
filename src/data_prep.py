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
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"✅ Salvato parquet in {path}")
    test = pd.read_parquet(path, engine="pyarrow")
    print(f"🔎 Ricaricato {path}: {test.shape}")

def load_fastf1_laps(year: int, gp: str, session: str = "R") -> pd.DataFrame:
    sess = ff1.get_session(year, gp, session)
    sess.load()
    laps = sess.laps.reset_index()
    if "LapTime" in laps.columns:
        laps["LapTimeSecondsFF1"] = laps["LapTime"].dt.total_seconds()
    return laps

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
        df["LapTimeSecondsKaggle"] = df["time"].apply(time_to_seconds)

    if "LapTimeSecondsKaggle" not in df.columns:
        df["LapTimeSecondsKaggle"] = None
    if "Compound" not in df.columns:
        df["Compound"] = "Unknown"
    return df

# -------------------------
# 3️⃣ Carica CSV Kaggle
# -------------------------
drivers_df = read_csv_safe(raw_dir / "drivers.csv")
circuits_df = read_csv_safe(raw_dir / "circuits.csv")
results_df = read_csv_safe(raw_dir / "results.csv")
status_df = read_csv_safe(raw_dir / "status.csv")
pit_df = read_csv_safe(raw_dir / "pit_stops.csv")
races_df = read_csv_safe(raw_dir / "races.csv")
laps_csv_df = read_csv_safe(raw_dir / "lap_times.csv")

for name, df in {
    "drivers.parquet": drivers_df,
    "circuits.parquet": circuits_df,
    "results.parquet": results_df,
    "status.parquet": status_df,
    "pit_stops.parquet": pit_df,
    "races.parquet": races_df,
    "lap_times.parquet": laps_csv_df,
}.items():
    save_parquet(df, processed_dir / name)

# Driver map
drivers_df["number"] = pd.to_numeric(drivers_df["number"], errors="coerce")
piloti_f1 = drivers_df.dropna(subset=["number", "code"])
DRIVER_MAP = dict(zip(piloti_f1["number"].astype(int), piloti_f1["code"]))
VALID_DRIVER_CODES = set(DRIVER_MAP.values())

# -------------------------
# 4️⃣ Merge FastF1 + Kaggle
# -------------------------
def merge_kaggle_with_fastf1(laps_ff1: pd.DataFrame, gp_name: str) -> pd.DataFrame:
    if laps_csv_df.empty:
        return pd.DataFrame()

    # Map drivers
    laps_csv_df["Driver"] = laps_csv_df["driverId"].map(drivers_df.set_index("driverId")["code"])
    laps_ff1 = laps_ff1[laps_ff1["Driver"].isin(VALID_DRIVER_CODES)]

    merged = pd.merge(
        laps_ff1,
        laps_csv_df,
        left_on=["Driver", "LapNumber"],
        right_on=["Driver", "lap"],
        how="inner",
        suffixes=("_ff1", "_kaggle"),
    )

    if merged.empty:
        return merged

    # Races info
    race_match = races_df[races_df["name"] == gp_name]
    if not race_match.empty:
        race_id = race_match["raceId"].values[0]
        circuit_id = race_match["circuitId"].values[0]
        merged["RaceId"] = race_id
        merged["CircuitId"] = circuit_id

        # Join extra CSVs
        merged = merged.merge(results_df, on=["raceId", "driverId"], how="left", suffixes=("", "_res"))
        merged = merged.merge(pit_df, on=["raceId", "driverId", "lap"], how="left", suffixes=("", "_pit"))
        merged = merged.merge(circuits_df, left_on="CircuitId", right_on="circuitId", how="left")
        merged = merged.merge(status_df, on="statusId", how="left")
        merged = merged.merge(drivers_df, on="driverId", how="left", suffixes=("", "_drv"))

    merged = add_features(merged)
    return merged

# -------------------------
# 5️⃣ Funzione principale
# -------------------------
def process_season(year: int):
    print(f"\n📅 Elaboro stagione {year}...")

    try:
        schedule = ff1.get_event_schedule(year)
    except Exception as e:
        print(f"❌ Errore caricamento calendario {year}: {e}")
        return

    season_dir = processed_dir / str(year)
    season_dir.mkdir(parents=True, exist_ok=True)

    all_laps = []
    report = []

    for _, event in schedule.iterrows():
        gp_name = event["EventName"]
        round_no = event["RoundNumber"]
        print(f"\n➡️ Processing GP {round_no}: {gp_name}")

        try:
            laps_fastf1 = load_fastf1_laps(year, round_no, "R")
            if laps_fastf1.empty:
                print(f"⚠️ Nessun dato FastF1 per {gp_name}, skip.")
                continue
        except Exception as e:
            print(f"⚠️ Errore FastF1 per {gp_name}: {e}")
            continue

        merged = merge_kaggle_with_fastf1(laps_fastf1, gp_name)
        if merged.empty:
            print(f"⚠️ Merge vuoto per {gp_name}, skip.")
            continue

        merged = merged.dropna(subset=["LapTimeSecondsFF1", "LapTimeSecondsKaggle"], how="all")
        if merged.empty:
            print(f"⚠️ Tutti i giri invalidi per {gp_name}, skip.")
            continue

        safe_name = gp_name.replace(" ", "_").replace("/", "_")
        out_path = season_dir / f"laps_{round_no}_{safe_name}.parquet"
        save_parquet(merged, out_path)
        all_laps.append(merged)
        report.append((gp_name, len(merged)))

    if all_laps:
        df_all = pd.concat(all_laps, ignore_index=True)
        save_parquet(df_all, processed_dir / f"laps_{year}_all.parquet")
        print(f"✅ Stagione {year} completata con {len(df_all)} giri validi")

        print("\n📊 Report finale:")
        for gp, n in report:
            print(f"   - {gp}: {n} giri salvati")
    else:
        print(f"❌ Nessun dato valido per {year}")

# -------------------------
# 6️⃣ Unione multi-anno
# -------------------------
def update_laps_processed():
    parquet_files = sorted(processed_dir.glob("laps_*_all.parquet"))
    if not parquet_files:
        print("❌ Nessun cumulativo annuale trovato")
        return

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, engine="pyarrow")
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
    for year in [2024]:
        process_season(year)

    update_laps_processed()

from pathlib import Path
import pandas as pd

# -------------------------
# 1️⃣ Setup cartelle
# -------------------------
root_dir = Path(__file__).parent.parent
processed_dir = root_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 2️⃣ Caricamento dati
# -------------------------
laps_path = processed_dir / "laps_processed.parquet"
pit_stops_path = processed_dir / "pit_stops.parquet"
drivers_path = processed_dir / "drivers.parquet"  # Ora usiamo il parquet già processato

laps_df = pd.read_parquet(laps_path) if laps_path.exists() else pd.DataFrame()
pit_df = pd.read_parquet(pit_stops_path) if pit_stops_path.exists() else pd.DataFrame()
drivers_df = pd.read_parquet(drivers_path) if drivers_path.exists() else pd.DataFrame()

if laps_df.empty:
    print("❌ Nessun laps_processed.parquet trovato o vuoto")
    exit(1)

# Mappa driverId -> code (fallback)
if not drivers_df.empty:
    DRIVER_MAP = dict(zip(drivers_df["driverId"], drivers_df["code"]))
else:
    DRIVER_MAP = {}

# -------------------------
# 3️⃣ Feature engineering dettagliata con log
# -------------------------
def add_features(df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering con log dettagliato:
    - Stint
    - TyreAge
    - IsOutLap
    - DegradationRate
    - RollingAvgLap
    - LapDelta
    """

    if df.empty:
        return df.copy()

    df = df.copy()

    # fallback driverId
    df["driverId_filled"] = df["driverId"].copy()
    missing_driver_id = df["driverId_filled"].isna()
    df.loc[missing_driver_id, "driverId_filled"] = df.loc[missing_driver_id, "Driver"].map(
        {v: k for k, v in DRIVER_MAP.items()}
    )

    # inizializzazione
    df["Stint"] = 1
    df["TyreAge"] = 0
    df["IsOutLap"] = False

    if not pit_df.empty:
        pit_map = pit_df.groupby("driverId")["lap"].apply(list).to_dict()
        pit_set_map = pit_df.groupby("driverId")["lap"].apply(set).to_dict()

    total_groups = df.groupby(["RaceId", "driverId_filled"]).ngroups
    print(f"🚀 Inizio elaborazione dettagliata di {total_groups} driver-gara...")

    stint_list = []
    tyreage_list = []
    outlap_list = []

    for i, ((race_id, driver_id), group) in enumerate(df.groupby(["RaceId", "driverId_filled"]), start=1):
        laps = group["LapNumber"].tolist()
        pits = sorted(pit_map.get(driver_id, [])) if not pit_df.empty else []
        stint = 1
        tyre_age = 0

        stint_col = []
        tyre_col = []
        outlap_col = []

        for lap in laps:
            # outlap
            is_outlap = (lap - 1) in pit_set_map.get(driver_id, set()) if not pit_df.empty else False
            outlap_col.append(is_outlap)

            # stint e tyre age
            if pits and lap > pits[0]:
                pits.pop(0)
                stint += 1
                tyre_age = 0
            stint_col.append(stint)
            tyre_col.append(tyre_age)
            tyre_age += 1

        stint_list.extend(stint_col)
        tyreage_list.extend(tyre_col)
        outlap_list.extend(outlap_col)

        # log dettagliato ogni 100 gruppi o ultimo gruppo
        if i % 100 == 0 or i == total_groups:
            print(f"📊 Elaborati {i}/{total_groups} gruppi | driver {driver_id} | race {race_id} | {len(laps)} giri | max stint {max(stint_col)}")

    df["Stint"] = stint_list
    df["TyreAge"] = tyreage_list
    df["IsOutLap"] = outlap_list

    # DegradationRate
    df["DegradationRate"] = df.groupby(["Driver", "Stint"])["LapTimeSecondsFF1"].diff().fillna(0)

    # RollingAvgLap
    df["RollingAvgLap"] = df.groupby("Driver")["LapTimeSecondsFF1"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Compound fallback
    if "Compound" not in df.columns:
        df["Compound"] = "Unknown"

    # LapDelta
    if "RaceId" in df.columns:
        df["MinLapByDriverRace"] = df.groupby(["Driver", "RaceId"])["LapTimeSecondsFF1"].transform("min")
    else:
        df["MinLapByDriverRace"] = df.groupby("Driver")["LapTimeSecondsFF1"].transform("min")

    df["LapDelta"] = df["LapTimeSecondsFF1"] - df["MinLapByDriverRace"]

    # pulizia colonna temporanea
    df.drop(columns=["driverId_filled"], inplace=True)

    print("✅ Feature engineering dettagliata completata")
    return df

# -------------------------
# 4️⃣ Applica e salva
# -------------------------
laps_features = add_features(laps_df, pit_df)

if laps_features.empty:
    print("❌ Nessun dato valido per feature engineering")
    exit(1)

output_path = processed_dir / "laps_features.parquet"
laps_features.to_parquet(output_path, index=False)
print(f"✅ Dataset con feature salvato in {output_path} ({len(laps_features)} righe)")

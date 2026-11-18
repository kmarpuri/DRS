# Downloads data from fastf1 from 2018-2025.

from pathlib import Path
import pandas as pd
import fastf1
from fastf1.req import RateLimitExceededError

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RAW = ROOT / "data" / "raw"
CACHE.mkdir(exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE))
fastf1.set_log_level("WARNING")

COLS = ["DriverNumber", "Abbreviation", "DriverId", "FullName", "TeamName", "TeamId", "Position", 
        "ClassifiedPosition", "GridPosition", "Status", "Points", "Q1", "Q2", "Q3"]

failed = []

for year in range(2018, 2026):
    out = RAW / f"{year}.csv"
    if out.exists():
        print(f"Skipping {year} (already exists)")
        continue
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    frames, year_failed = [], False

    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        for code in ("Q", "R"):
            try:
                s = fastf1.get_session(year, rnd, code)
                s.load(laps=False, telemetry=False, weather=False, messages=False)
                df = s.results.reindex(columns=COLS).copy()
                if df["Position"].isna().all():
                    raise ValueError("results came back empty")
                for q in ("Q1", "Q2", "Q3"):          # lap times -> seconds
                    df[q] = pd.to_timedelta(df[q]).dt.total_seconds()
                df.insert(0, "Session", code)
                df.insert(0, "EventName", ev["EventName"])
                df.insert(0, "EventDate", ev["EventDate"])
                df.insert(0, "Round", rnd)
                df.insert(0, "Year", year)
                frames.append(df)
                print(f"{year} R{rnd:02d} {code}: {len(df)} drivers")
            except RateLimitExceededError:
                raise SystemExit("Hit the API rate limit. Wait ~1 hour and rerun; "
                                 "cached sessions won't be downloaded again.")
            except Exception as e:
                year_failed = True
                failed.append((year, rnd, code, repr(e)))
                print(f"{year} R{rnd:02d} {code}: FAILED ({e})")

    if year_failed:
        print(f"{year}: not saved because some sessions failed; rerun to retry")
    else:
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"{year}: saved {out}")

# Combine once every season is present
files = sorted(RAW.glob("*.csv"))
if len(files) == 8:
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_df.to_csv(ROOT / "data" / "results_2018_2025.csv", index=False)
    races = all_df[all_df.Session == "R"]
    print("\nRaces per season:\n", races.groupby("Year")["Round"].nunique())
    print("Race rows:", len(races), "| Quali rows:", (all_df.Session == "Q").sum())

if failed:
    print("\nFailures:", *failed, sep="\n")
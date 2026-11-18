# Cleans the raw FastF1 download into one row per driver per race.
#   input:  data/results_2018_2025.csv  (from download_data.py)
#   output: data/clean.csv

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "results_2018_2025.csv"
OUT = ROOT / "data" / "clean.csv"

# One fixed ID per team, so a rebrand doesn't reset the team's history.
# Keys are FastF1/Jolpica TeamIds; teams not listed keep their own ID.
TEAM_MAP = {
    "toro_rosso": "rb", "alphatauri": "rb", "rb": "rb",                  # Toro Rosso -> AlphaTauri -> RB / Racing Bulls
    "force_india": "aston_martin", "racing_point": "aston_martin",
    "aston_martin": "aston_martin",                                       # Force India -> Racing Point -> Aston Martin
    "renault": "alpine", "alpine": "alpine",                              # Renault -> Alpine
    "sauber": "sauber", "alfa": "sauber",                                 # Sauber -> Alfa Romeo -> Kick Sauber
}

df = pd.read_csv(RAW, dtype={"ClassifiedPosition": str})

# ---- qualifying position, joined onto the race rows ----------------------
quali = (df[df.Session == "Q"][["Year", "Round", "DriverId", "Position"]]
         .rename(columns={"Position": "QualiPosition"}))

race = df[df.Session == "R"].drop(columns=["Session", "Q1", "Q2", "Q3"])
race = race.merge(quali, on=["Year", "Round", "DriverId"], how="left")

# ---- drop drivers who were never on the grid -----------------------------
# W = withdrawn. Most W drivers had a grid slot and failed at or near the start
# (or retired); before the race nobody knew that, so they stay in as DNFs.
# Only W drivers with no grid slot (never entered the race) are dropped.
withdrawn = (race.ClassifiedPosition == "W") & race.GridPosition.isna()
race = race[~withdrawn].copy()

# ---- team ID ---------------------------------------------------------------
race["Team"] = race.TeamId.map(TEAM_MAP).fillna(race.TeamId)

# ---- DNF flag ---------------------------------------------------------------
# ClassifiedPosition is a number for classified finishers (lapped cars included)
# and a letter otherwise: R = retired, D = disqualified, W = did not start,
# N = not classified.
# Status wording changed in 2023 ("+1 Lap" -> "Lapped"), so it isn't used here.
race["DNF"] = pd.to_numeric(race.ClassifiedPosition, errors="coerce").isna().astype(int)

# ---- grid ---------------------------------------------------------------------
# GridPosition 0 = pit-lane start -> put the driver at the back of the grid.
race["FieldSize"] = race.groupby(["Year", "Round"]).DriverId.transform("size")
race["Grid"] = race.GridPosition.where(race.GridPosition > 0, race.FieldSize)

# ---- era flag + chronological order ----------------------------------------
race["Era"] = (race.Year >= 2022).astype(int)          # 2022 regulation change
race["EventDate"] = pd.to_datetime(race.EventDate)
race = race.sort_values(["Year", "Round", "Position"]).reset_index(drop=True)
race["RaceId"] = race.Year * 100 + race.Round          # one group per race, e.g. 202305

cols = ["RaceId", "Year", "Round", "EventDate", "EventName", "Era",
        "DriverId", "Abbreviation", "FullName", "TeamName", "TeamId", "Team",
        "GridPosition", "Grid", "QualiPosition", "Position", "ClassifiedPosition",
        "DNF", "Status", "Points", "FieldSize"]
race[cols].to_csv(OUT, index=False)

# ---- summary (numbers for the report's data table) -------------------------
train = race[race.Year <= 2023]
test = race[race.Year >= 2024]
print(f"Saved {OUT}\n")
print(f"{'split':<7}{'seasons':<12}{'races':>6}{'rows':>7}{'drivers':>9}{'teams':>7}{'DNF rate':>10}")
for name, part, seasons in [("train", train, "2018-2023"), ("test", test, "2024-2025"),
                            ("total", race, "2018-2025")]:
    print(f"{name:<7}{seasons:<12}{part.RaceId.nunique():>6}{len(part):>7}"
          f"{part.DriverId.nunique():>9}{part.Team.nunique():>7}{part.DNF.mean():>10.1%}")
print(f"\nDropped {withdrawn.sum()} rows for drivers never on the grid")
print(f"Pit-lane starts moved to back of grid: {(race.GridPosition == 0).sum()}")
print(f"Team names -> team IDs: {race.TeamName.nunique()} -> {race.Team.nunique()}")
print(f"Race rows missing a qualifying position: {race.QualiPosition.isna().sum()}")

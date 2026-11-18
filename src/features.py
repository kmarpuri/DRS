# Builds the model features: one row per driver per race.
#   input:  data/clean.csv     (from clean.py)
#   output: data/features.csv
#
# Leakage rule: every history feature for race n uses only races before n.
# That is done with .shift(1) BEFORE .rolling()/.expanding(), per driver or team,
# on rows sorted by race.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 5         # races in the "recent form" averages
DNF_WINDOW = 10    # races in the reliability average

# Events that ran at the same circuit under a different name
SAME_TRACK = {
    "Styrian Grand Prix": "Austrian Grand Prix",
    "70th Anniversary Grand Prix": "British Grand Prix",
    "Sakhir Grand Prix": "Bahrain Grand Prix",
    "Mexico City Grand Prix": "Mexican Grand Prix",
    "São Paulo Grand Prix": "Brazilian Grand Prix",
}

df = pd.read_csv(ROOT / "data" / "clean.csv")
df = df.sort_values(["RaceId", "Position"]).reset_index(drop=True)
df["Track"] = df.EventName.replace(SAME_TRACK)


def past_mean(series, window=None):
    """Mean of earlier values only: shift(1) drops the current race."""
    shifted = series.shift(1)
    if window is None:
        return shifted.expanding().mean()
    return shifted.rolling(window, min_periods=1).mean()


# ---- driver features (follow the driver across team changes) -------------
by_driver = df.groupby("DriverId")
df["driver_avg_finish_5"] = by_driver.Position.transform(lambda s: past_mean(s, WINDOW))
df["driver_dnf_rate_10"] = by_driver.DNF.transform(lambda s: past_mean(s, DNF_WINDOW))

# Driver's average finish at this circuit in earlier races
df["driver_track_avg"] = (df.groupby(["DriverId", "Track"]).Position
                            .transform(lambda s: past_mean(s)))

# ---- team features (the car) ------------------------------------------------
# First summarise each team's race (mean finish of its cars, total points),
# then average the team's previous WINDOW races.
team_race = (df.groupby(["Team", "RaceId"])
               .agg(team_finish=("Position", "mean"), team_points=("Points", "sum"))
               .reset_index()
               .sort_values(["Team", "RaceId"]))
by_team = team_race.groupby("Team")
team_race["team_avg_finish_5"] = by_team.team_finish.transform(lambda s: past_mean(s, WINDOW))
team_race["team_avg_points_5"] = by_team.team_points.transform(lambda s: past_mean(s, WINDOW))
df = df.merge(team_race[["Team", "RaceId", "team_avg_finish_5", "team_avg_points_5"]],
              on=["Team", "RaceId"], how="left")

# ---- label: higher = better -------------------------------------------------
# 21 - position for 20-car fields (winner = 20), and 0 for a DNF.
df["label"] = (21 - df.Position).where(df.DNF == 0, 0)

FEATURES = ["Grid", "driver_avg_finish_5", "team_avg_finish_5", "team_avg_points_5",
            "driver_track_avg", "driver_dnf_rate_10", "Era"]

df = df.sort_values(["RaceId", "Position"]).reset_index(drop=True)
df.to_csv(ROOT / "data" / "features.csv", index=False)

print(f"Saved data/features.csv: {len(df)} rows, {len(FEATURES)} features\n")
print("Missing values (no earlier history -> left as NaN; XGBoost handles these):")
print(df[FEATURES].isna().sum().to_string())

# Trains XGBRanker on 2018-2023 and tests it on 2024-2025 against the grid baseline.
#   input:  data/features.csv  (from features.py)
#   output: results/ranker_per_race.csv, results/comparison.csv,
#           results/feature_importance.csv

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from metrics import evaluate, summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = ["Grid", "driver_avg_finish_5", "team_avg_finish_5", "team_avg_points_5",
            "driver_track_avg", "driver_dnf_rate_10", "Era"]
GAMMA = 0.8          # season weight decay: a 2022 race counts 0.8x a 2023 race, etc.
LAST_TRAIN_YEAR = 2023

df = pd.read_csv(ROOT / "data" / "features.csv")
df = df.sort_values(["RaceId", "Position"]).reset_index(drop=True)  # rows grouped by race

train = df[df.Year <= LAST_TRAIN_YEAR]
test = df[df.Year > LAST_TRAIN_YEAR].copy()

# One weight per race (query group), in the same order the races appear
train_races = train.drop_duplicates("RaceId")[["RaceId", "Year"]]
race_weights = GAMMA ** (LAST_TRAIN_YEAR - train_races.Year.to_numpy())

model = xgb.XGBRanker(
    objective="rank:pairwise",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
)
model.fit(train[FEATURES], train.label, qid=train.RaceId, sample_weight=race_weights)

# Higher score = predicted to finish higher
test["score"] = model.predict(test[FEATURES])

ranker = evaluate(test, "score", higher_is_better=True)
grid = evaluate(test, "Grid", higher_is_better=False)
ranker.to_csv(RESULTS / "ranker_per_race.csv", index=False)

comparison = pd.DataFrame({
    "Grid-order baseline": summarize(grid),
    "XGBRanker (pairwise)": summarize(ranker),
}).T
comparison.to_csv(RESULTS / "comparison.csv")

importance = (pd.Series(model.get_booster().get_score(importance_type="gain"))
                .reindex(FEATURES).fillna(0))
importance = (importance / importance.sum()).sort_values(ascending=False)
importance.rename("gain_share").to_csv(RESULTS / "feature_importance.csv")

print(f"Trained on {train.RaceId.nunique()} races (2018-{LAST_TRAIN_YEAR}), "
      f"tested on {test.RaceId.nunique()} races (2024-2025)\n")
print(comparison.to_string(float_format=lambda x: f"{x:.3f}"))
print("\nPer season (XGBRanker):")
for year, part in ranker.groupby("Year"):
    s = summarize(part)
    print(f"  {year}: spearman {s.spearman:.3f}  winner {s.winner_acc:.3f}  podium {s.podium_acc:.3f}")
print("\nFeature importance (share of total gain):")
print(importance.to_string(float_format=lambda x: f"{x:.3f}"))

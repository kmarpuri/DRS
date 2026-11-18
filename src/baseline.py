# Grid-order baseline: predict that every driver finishes where they started.
#   input:  data/clean.csv  (from clean.py)
#   output: results/baseline_per_race.csv, results/baseline_summary.csv

from pathlib import Path
import pandas as pd
from metrics import evaluate, summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "clean.csv")

# Lower grid slot = better predicted finish
per_race = evaluate(df, score_col="Grid", higher_is_better=False)
per_race["Split"] = per_race.Year.map(lambda y: "test" if y >= 2024 else "train")

# Headline result: the test seasons (2024-2025). Train seasons shown for context.
summary = pd.DataFrame({
    "test 2024-2025": summarize(per_race[per_race.Split == "test"]),
    "2024": summarize(per_race[per_race.Year == 2024]),
    "2025": summarize(per_race[per_race.Year == 2025]),
    "train 2018-2023": summarize(per_race[per_race.Split == "train"]),
}).T

per_race.to_csv(RESULTS / "baseline_per_race.csv", index=False)
summary.to_csv(RESULTS / "baseline_summary.csv")

print("Grid-order baseline\n")
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

# Ranking metrics, shared by the baseline and the XGBRanker.
#
# Every model produces a score per driver per race. Drivers are ranked within
# each race by that score, and the ranking is compared with the real result.

import pandas as pd
from scipy.stats import spearmanr


def predicted_rank(df, score_col, higher_is_better):
    """Rank drivers within each race by score (1 = predicted winner).
    Ties are broken by qualifying position, then grid."""
    order = df.sort_values(
        ["RaceId", score_col, "QualiPosition", "Grid"],
        ascending=[True, not higher_is_better, True, True],
        na_position="last",
    )
    return order.groupby("RaceId").cumcount().add(1).reindex(df.index)


def evaluate(df, score_col, higher_is_better=False):
    """Per-race metrics.
    spearman: rank correlation between predicted and actual finishing order
    winner:   1 if the predicted winner actually won
    podium:   share of the actual top 3 that the model put in its top 3
    """
    df = df.copy()
    df["PredRank"] = predicted_rank(df, score_col, higher_is_better)

    rows = []
    for race_id, race in df.groupby("RaceId"):
        pred_top3 = set(race.loc[race.PredRank <= 3, "DriverId"])
        real_top3 = set(race.loc[race.Position <= 3, "DriverId"])
        rows.append({
            "RaceId": race_id,
            "Year": race.Year.iloc[0],
            "EventName": race.EventName.iloc[0],
            "spearman": spearmanr(race.PredRank, race.Position).statistic,
            "winner": int((race.loc[race.PredRank == 1, "Position"] == 1).any()),
            "podium": len(pred_top3 & real_top3) / 3,
        })
    return pd.DataFrame(rows)


def summarize(per_race):
    """Average the per-race metrics."""
    return pd.Series({
        "races": len(per_race),
        "spearman": per_race.spearman.mean(),
        "winner_acc": per_race.winner.mean(),
        "podium_acc": per_race.podium.mean(),
    })

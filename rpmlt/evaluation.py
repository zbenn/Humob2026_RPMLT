"""The ten diagnostic windows and eight earlier windows used in the paper."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .data import score_prediction
from .models import components, bridge_prediction


@dataclass(frozen=True)
class Window:
    name: str
    start: str
    end: str
    tier: str

    def mask(self, dates):
        return (dates >= pd.Timestamp(self.start)) & (dates <= pd.Timestamp(self.end))


WINDOWS = [
    Window("early_jan", "2024-01-08", "2024-01-31", "shock_stress"),
    Window("jan_tail", "2024-01-15", "2024-01-31", "recovery_edge"),
    Window("jan_last10", "2024-01-22", "2024-01-31", "recovery_edge"),
    Window("april_full", "2024-04-01", "2024-04-30", "recovery_edge"),
    Window("april_first14", "2024-04-01", "2024-04-14", "recovery_edge"),
    Window("gap_may_june", "2024-05-01", "2024-06-29", "gap_geometry"),
    Window("gap_july_aug", "2024-07-01", "2024-08-29", "gap_geometry"),
    Window("normal_timetable_event", "2024-07-06", "2024-08-03", "event_transfer"),
    Window("rain_shock", "2024-09-01", "2024-10-30", "shock_stress"),
    Window("winter_control", "2023-12-06", "2023-12-31", "season_control"),
]
LEGACY = [
    Window("A_jan_trough", "2024-01-08", "2024-01-31", "primary"),
    Window("B_jan_anchored", "2024-04-01", "2024-04-30", "primary"),
    Window("S_apr_may", "2024-04-01", "2024-05-30", "rolling"),
    Window("S_may_jun", "2024-05-01", "2024-06-29", "rolling"),
    Window("S_jun_jul", "2024-06-01", "2024-07-30", "rolling"),
    Window("S_jul_aug", "2024-07-01", "2024-08-29", "rolling"),
    Window("S_aug_sep", "2024-08-01", "2024-09-29", "rolling"),
    Window("C_rain_shock", "2024-09-01", "2024-10-30", "rolling"),
]


def evaluate(data, output, protocol="all"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    windows = WINDOWS if protocol == "diagnostic" else LEGACY if protocol == "legacy" else WINDOWS + LEGACY
    rows, daily, errors = [], [], []
    for window in windows:
        held = window.mask(data.dates)
        if not held.any() or held.all():
            raise ValueError(f"Window {window.name} has no usable train/test split")
        train, target = data.subset(~held), data.dates[held]
        local, temporal = components(train, target)
        candidates = {"local": local, "temporal": temporal, "combined": .5 * (local + temporal),
                      "boundary": bridge_prediction(train, target, local, temporal)}
        for name, pred in candidates.items():
            scores = score_prediction(data.X[held], pred, data.is_diag)
            rows.append({"model": name, "window": window.name, "tier": window.tier,
                         **{k: scores[k] for k in ["combined", "diag", "off"]}})
            daily.extend({"model": name, "window": window.name, "date": str(day.date()), "combined": score}
                         for day, score in zip(target, scores["daily"]))
            if window.name == "jan_tail":
                for pair in [(40, 46, 40, 46), (54, 56, 54, 56), (41, 47, 41, 47), (58, 44, 58, 44)]:
                    if pair not in data.pairs:
                        continue
                    j = data.pairs.index(pair)
                    errors.append({"model": name, "pair": "_".join(map(str, pair)),
                                   "rmse": float(np.sqrt(np.mean((pred[:, j] - data.X[held, j]) ** 2)))})
        print(f"Finished {window.name}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(daily).to_csv(output / "daily_scores.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "selected_errors.csv", index=False)
    summary = scores.groupby(["model", "tier"]).combined.mean().unstack()
    summary.to_csv(output / "summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return scores

"""Public water-service and calendar inputs used by the final model."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"


class WaterStore:
    def __init__(self, cells, daily):
        self.cells = cells.set_index(["cell_y", "cell_x"])
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily.date)
        self.first_date, self.last_date = daily.date.min(), daily.date.max()
        self.daily = daily.set_index(["date", "en"]).sort_index()

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls):
        return cls(pd.read_csv(ASSETS / "cells.csv"), pd.read_csv(ASSETS / "water_daily.csv"))

    def levels(self, cells, date):
        # Preserve original endpoint holding, even before the earthquake.
        # This is a lookup convention, not a claim of observed pre-event outages.
        date = min(max(pd.Timestamp(date), self.first_date), self.last_date)
        index = pd.MultiIndex.from_tuples(cells, names=["cell_y", "cell_x"])
        municipalities = self.cells.reindex(index).muni_en.fillna("").astype(str).to_numpy()
        day = self.daily.xs(date, level="date")
        water = pd.Series(municipalities).map(day.water_out_frac).fillna(0).to_numpy(float)
        return np.nan_to_num(water, nan=0., posinf=0., neginf=0.)


@lru_cache(maxsize=1)
def holiday_dates():
    return pd.DatetimeIndex(pd.read_csv(ASSETS / "holidays.csv").date)


@lru_cache(maxsize=1)
def events():
    return json.loads((ASSETS / "festivals.json").read_text(encoding="utf-8"))["events"]


def raw_features(dates):
    dates = pd.DatetimeIndex(dates)
    t = np.asarray((dates - pd.Timestamp("2024-01-01")).days, float)
    post = (t >= 0).astype(float)
    columns = [post, np.maximum(t, 0) / 365,
               ((t >= 0) & (t < 7)).astype(float), ((t >= 0) & (t < 3)).astype(float)]
    columns += [post * np.exp(-np.maximum(t, 0) / tau) for tau in [7., 30., 90.]]
    columns += [np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25)]
    columns += [(dates.dayofweek == d).astype(float) for d in range(6)]
    features = np.column_stack(columns)
    features[dates.isin(holiday_dates()), -6:] = 0.  # Sunday is omitted category.
    pulses = np.column_stack([(dates >= pd.Timestamp(e["start"])) &
                              (dates <= pd.Timestamp(e["end"])) for e in events()])
    return np.column_stack([features, pulses.astype(float)])

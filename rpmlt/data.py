"""Read the user-supplied restricted competition TSV without shared caches."""
from dataclasses import dataclass
from pathlib import Path
import ast
import math
import numpy as np
import pandas as pd

N_CELLS = 1476
NORMALIZERS = (26.57, 0.0176)


def in_box(y, x):
    return 35 <= y <= 70 and 30 <= x <= 70


def cell_key(value):
    if not isinstance(value, str):
        raise ValueError("Grid identifiers must be strings in y_x format")
    if value == "-1_-1":
        return -1, -1  # Official unknown/outside-area sentinel.
    parts = value.split("_")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid grid identifier: {value!r}")
    y, x = map(int, parts)
    if not (1 <= y <= 70 and 1 <= x <= 100):
        raise ValueError(f"Out-of-domain grid: {value!r}")
    return y, x


def records(path):
    """Yield valid dated records; skip NA days, never convert them to zero."""
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            day, payload = line.rstrip("\r\n").split("\t", 1)
            if len(day) != 8 or not day.isdigit():
                raise ValueError("Dates must use YYYYMMDD")
            date = pd.to_datetime(day, format="%Y%m%d")
            if date in seen:
                raise ValueError(f"Duplicate date: {day}")
            seen.add(date)
            if payload.strip() == "NA":
                continue
            od = ast.literal_eval(payload)
            if not isinstance(od, dict):
                raise ValueError("Expected a nested OD dictionary")
            for origin, dests in od.items():
                cell_key(origin)
                if not isinstance(dests, dict) or not dests:
                    raise ValueError("Expected a destination dictionary")
                for dest, value in dests.items():
                    cell_key(dest)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError("OD values must be finite, nonnegative numbers")
            yield date, od


@dataclass
class TrainView:
    X: np.ndarray
    dates: pd.DatetimeIndex
    pairs: list[tuple]
    is_diag: np.ndarray

    def subset(self, mask):
        return TrainView(self.X[mask], self.dates[mask], self.pairs, self.is_diag)


def load_data(path):
    """Keep the sparse union of observed pairs; unrepresented pairs are zero.

    Float32 storage matches the original evaluation pipeline. Fitting converts
    to float64 where the original did. Models zero out fold-unseen columns.
    """
    rows, dates = [], []
    for date, od in records(path):
        row = {}
        for origin, dests in od.items():
            oy, ox = cell_key(origin)
            if not in_box(oy, ox):
                continue
            for dest, value in dests.items():
                dy, dx = cell_key(dest)
                if in_box(dy, dx):
                    row[oy, ox, dy, dx] = value
        rows.append(row)
        dates.append(date)
    if not dates:
        raise ValueError("No usable observations")
    order = np.argsort(dates)
    dates = pd.DatetimeIndex([dates[i] for i in order])
    rows = [rows[i] for i in order]
    pairs = sorted({p for row in rows for p in row})
    if not pairs:
        raise ValueError("No observations inside the scoring box")
    lookup = {p: j for j, p in enumerate(pairs)}
    values = np.zeros((len(dates), len(pairs)), dtype=np.float32)
    for i, row in enumerate(rows):
        for pair, value in row.items():
            values[i, lookup[pair]] = value
    return TrainView(values, dates, pairs, np.array([p[:2] == p[2:] for p in pairs]))


def submission_dates():
    dates = pd.date_range("2024-02-01", "2024-03-31")
    return dates[~dates.isin(pd.to_datetime(["2024-02-02", "2024-03-05"]))]


def score_prediction(truth, prediction, is_diag):
    """Official daily-vector NRMSE, including ALL implicit-zero pairs."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    is_diag = np.asarray(is_diag, bool)
    if truth.ndim != 2 or truth.shape != prediction.shape or len(is_diag) != truth.shape[1]:
        raise ValueError("Incompatible truth/prediction/mask shapes")
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Cannot score empty or nonfinite arrays")
    error = (truth - prediction) ** 2
    diagonal = np.sqrt(error[:, is_diag].sum(1) / N_CELLS) / NORMALIZERS[0]
    offdiagonal = np.sqrt(error[:, ~is_diag].sum(1) / (N_CELLS * (N_CELLS - 1))) / NORMALIZERS[1]
    daily = .5 * (diagonal + offdiagonal)
    return {"combined": float(daily.mean()), "diag": float(diagonal.mean()),
            "off": float(offdiagonal.mean()), "daily": daily}

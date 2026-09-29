"""Full-area submission serialization and independent safety checks."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import records, cell_key, in_box, submission_dates


def april_outside_box(path):
    sums, days = defaultdict(float), 0
    for date, od in records(path):
        if not "20240401" <= date.strftime("%Y%m%d") <= "20240430":
            continue
        days += 1
        for origin, dests in od.items():
            for dest, value in dests.items():
                if not (in_box(*cell_key(origin)) and in_box(*cell_key(dest))):
                    sums[origin, dest] += float(value)
    if not days:
        raise ValueError("April observations are required for the outside-box fallback")
    return {k: value / days for k, value in sums.items()}


def validate(path):
    days = []
    for day, od in records(path):
        if not any(dests for dests in od.values()):
            raise ValueError("Empty prediction")
        days.append(day)
    if days != list(submission_dates()):
        raise ValueError("Expected exactly 58 ordered valid target dates")
    # records() deliberately skips NA input rows; reject them for submissions.
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 58:
        raise ValueError("Expected exactly 58 submission rows")
    return {"days": len(days), "local_validation": "passed"}


def write_submission(path, datafile, data, prediction):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    dates = submission_dates()
    if prediction.shape != (len(dates), len(data.pairs)):
        raise ValueError("Unexpected prediction shape")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid prediction values")
    outside = april_outside_box(datafile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        for date, values in zip(dates, prediction):
            od = {}
            for (origin, dest), value in outside.items():
                if value > 0:
                    od.setdefault(origin, {})[dest] = float(value)
            for (oy, ox, dy, dx), value in zip(data.pairs, values):
                if value > 0:
                    od.setdefault(f"{oy}_{ox}", {})[f"{dy}_{dx}"] = float(value)
            stream.write(date.strftime("%Y%m%d") + "\t" + repr(od) + "\n")
    result = validate(path)
    for i, (_, od) in enumerate(records(path)):
        recovered = [od.get(f"{oy}_{ox}", {}).get(f"{dy}_{dx}", 0.) for oy, ox, dy, dx in data.pairs]
        np.testing.assert_array_equal(recovered, prediction[i])
    return {**result, "box_roundtrip_exact": True}

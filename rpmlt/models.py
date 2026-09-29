"""Paper models, preserving the original numerical fitting operations."""
import numpy as np
import pandas as pd
from scipy.linalg import solve
from .features import WaterStore, raw_features


def local_predict(train, target, store=None):
    """Local level + month-centered weekday residual + water-outage scaling."""
    target = pd.DatetimeIndex(target)
    x = np.asarray(train.X, float)
    diag = train.is_diag
    pre = x[train.dates < target[0]][-30:]
    post = x[train.dates > target[-1]][:30]
    if not len(pre):
        pre = x
    if len(post) < 14:
        post = pre
    base = np.zeros(x.shape[1])
    base[diag] = post[:, diag].mean(0)
    base[~diag] = .5 * pre[:, ~diag].mean(0) + .5 * post[:, ~diag].mean(0)
    active = (x > 0).sum(0)
    base[~diag] *= active[~diag] / (active[~diag] + 7. + 1e-12)
    out = np.repeat(base[None, :], len(target), axis=0)
    reliability = active / (active + 7.)
    periods = train.dates.to_period("M")
    residual = x.copy()
    for month in periods.unique():
        mask = periods == month
        residual[mask] -= x[mask].mean(0)
    for dow in range(7):
        rows = train.dates.dayofweek == dow
        if rows.any():
            out[target.dayofweek == dow] += residual[rows].mean(0) * reliability
    store = store or WaterStore.load()
    origins, destinations = [p[:2] for p in train.pairs], [p[2:] for p in train.pairs]
    after, before = train.dates[train.dates > target[-1]], train.dates[train.dates < target[0]]
    ref = after[0] if len(after) else before[-1]
    ro, rd = store.levels(origins, ref), store.levels(destinations, ref)
    for i, date in enumerate(target):
        mo = np.exp(-.15 * (store.levels(origins, date) - ro))
        md = np.exp(-.15 * (store.levels(destinations, date) - rd))
        out[i] *= np.where(diag, mo, np.sqrt(mo * md))
    out[:, active == 0] = 0
    return np.maximum(out, 0.)


def robust_fit(a, b, y, ridge):
    z = np.column_stack([np.ones(len(a)), a])
    target = np.column_stack([np.ones(len(b)), b])
    penalty = np.diag(np.r_[0., np.full(a.shape[1], ridge)])
    coef = solve(z.T @ z + penalty, z.T @ y, assume_a="pos")
    initial = np.linalg.norm(y - z @ coef, axis=1)
    scale = max(float(np.median(initial)), 1e-8)
    for _ in range(25):
        residual = np.linalg.norm(y - z @ coef, axis=1)
        weights = scale / np.sqrt(residual ** 2 + (.05 * scale) ** 2)
        new = solve(z.T @ (weights[:, None] * z) + penalty,
                    z.T @ (weights[:, None] * y), assume_a="pos")
        if np.linalg.norm(new - coef) < 1e-7 * max(np.linalg.norm(coef), 1.):
            coef = new
            break
        coef = new
    return target @ coef


def temporal_predict(train, target):
    a, b = raw_features(train.dates), raw_features(target)
    mean, std = a.mean(0), a.std(0)
    keep = std > 1e-8
    a = (a[:, keep] - mean[keep]) / std[keep]
    b = (b[:, keep] - mean[keep]) / std[keep]
    b = np.clip(b, a.min(0), a.max(0))
    result = np.zeros((len(target), len(train.pairs)))
    for mask, ridge in [(train.is_diag, 10.), (~train.is_diag, 100.)]:
        if mask.any():
            result[:, mask] = robust_fit(a, b, train.X[:, mask], ridge)
    result[:, train.X.mean(0) == 0] = 0
    return np.maximum(result, 0.)


def components(train, target):
    target = pd.DatetimeIndex(target)
    if not len(target) or not len(train.dates):
        raise ValueError("Training and target dates must not be empty")
    if not target.is_monotonic_increasing or not train.dates.is_monotonic_increasing:
        raise ValueError("Dates must be sorted")
    if target.has_duplicates or train.dates.has_duplicates or target.isin(train.dates).any():
        raise ValueError("Dates must be unique and training must exclude targets")
    return local_predict(train, target), temporal_predict(train, target)


def predict(train, target):
    local, temporal = components(train, target)
    return .5 * (local + temporal)


def bridge_prediction(train, target, local, temporal, window=7, bridge_mix=.5):
    """Rejected boundary-interpolation comparator; NOT the final model."""
    target = pd.DatetimeIndex(target)
    left_ix = np.flatnonzero(train.dates < target[0])
    right_ix = np.flatnonzero(train.dates > target[-1])
    left = train.X[left_ix[-window:]].mean(0) if len(left_ix) else train.X.mean(0)
    right = train.X[right_ix[:window]].mean(0) if len(right_ix) else left
    left_date = train.dates[left_ix[-1]] if len(left_ix) else target[0] - pd.Timedelta(days=1)
    right_date = train.dates[right_ix[0]] if len(right_ix) else target[-1] + pd.Timedelta(days=1)
    alpha = np.clip(np.asarray((target - left_date).days, float) / max((right_date - left_date).days, 1), 0, 1)
    bridge = (1 - alpha[:, None]) * left + alpha[:, None] * right
    bridge = np.maximum(bridge + (local - local.mean(0)), 0.)
    result = .5 * (local + temporal)
    result[:, train.is_diag] = (1 - bridge_mix) * result[:, train.is_diag] + bridge_mix * bridge[:, train.is_diag]
    result[:, (train.X > 0).sum(0) == 0] = 0
    return np.maximum(result, 0.)

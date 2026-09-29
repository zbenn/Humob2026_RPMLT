"""Synthetic tests run without restricted data, credentials, or network."""
import numpy as np
import pandas as pd
import pytest
from rpmlt.data import TrainView, load_data, score_prediction, submission_dates, records, cell_key
from rpmlt.features import raw_features, WaterStore
from rpmlt.models import components, predict, temporal_predict, local_predict
from rpmlt.submission import validate, write_submission


@pytest.fixture
def synthetic():
    dates = pd.date_range("2023-11-01", "2024-05-31")
    target = pd.date_range("2024-02-01", "2024-03-31")
    dates = dates[~dates.isin(target)]
    rng = np.random.default_rng(42)
    values = np.column_stack([10 + rng.random(len(dates)), rng.random(len(dates)), np.zeros(len(dates))]).astype(np.float32)
    return TrainView(values, dates, [(41, 47, 41, 47), (41, 47, 58, 44), (58, 44, 58, 44)],
                     np.array([True, False, True])), target


def test_required_dates():
    dates = submission_dates()
    assert len(dates) == 58 and dates.is_unique
    assert pd.Timestamp("2024-02-02") not in dates
    assert pd.Timestamp("2024-03-05") not in dates


def test_official_unknown_sentinel():
    assert cell_key("-1_-1") == (-1, -1)
    with pytest.raises(ValueError):
        cell_key("-1_47")


def test_metric_full_domain():
    result = score_prediction([[0, 0]], [[26.57 * np.sqrt(1476), .0176 * np.sqrt(1476 * 1475)]], [True, False])
    assert result["combined"] == pytest.approx(1.)
    assert score_prediction([[2, 3]], [[2, 3]], [True, False])["combined"] == 0


def test_metric_averages_daily_norms():
    result = score_prediction([[0], [0]], [[0], [2 * 26.57 * np.sqrt(1476)]], [True])
    assert result["combined"] == pytest.approx(.5)


def test_components_and_unseen(synthetic):
    train, target = synthetic
    local, temporal = components(train, target)
    actual = predict(train, target)
    np.testing.assert_array_equal(actual, .5 * (local + temporal))
    assert actual.shape == (60, 3) and np.isfinite(actual).all() and (actual >= 0).all()
    assert (actual[:, 2] == 0).all()


def test_heldout_january_support_guard(synthetic):
    train, _ = synthetic
    target = pd.date_range("2024-01-01", "2024-01-31")
    train = train.subset(~train.dates.isin(target))
    result = temporal_predict(train, target)
    assert np.isfinite(result).all() and result.max() < 100


def test_holiday_as_sunday_and_event():
    a = raw_features(pd.to_datetime(["2024-01-08", "2024-01-09", "2024-07-05", "2024-08-03"]))
    assert (a[0, 9:15] == 0).all()
    assert a[1, 10] == 1
    np.testing.assert_array_equal(a[2, -2:], [1, 0])
    np.testing.assert_array_equal(a[3, -2:], [0, 1])


def test_water_boundary_and_unknown():
    store = WaterStore.load()
    np.testing.assert_array_equal(store.levels([(41, 47), (35, 30)], "2023-12-01"),
                                  store.levels([(41, 47), (35, 30)], store.first_date))
    assert store.levels([(35, 30)], "2024-02-01")[0] == 0


def test_overlap_rejected(synthetic):
    train, _ = synthetic
    with pytest.raises(ValueError, match="exclude"):
        components(train, train.dates[:3])


def test_short_post_fallback():
    dates = pd.DatetimeIndex(["2024-04-01", "2024-04-02", "2024-04-20"])
    train = TrainView(np.array([[2.], [2.], [100.]], np.float32), dates, [(35, 30, 35, 30)], np.array([True]))
    # Tuesday correction is nonzero, but compare two post levels: fallback
    # affects the baseline, whereas both use the same training weekday residual.
    out = local_predict(train, pd.DatetimeIndex(["2024-04-10"]))
    assert out[0, 0] == pytest.approx(2.)  # Wednesday has no training examples.


def test_loader_na_sort_and_sparse(tmp_path):
    path = tmp_path / "input.tsv"
    path.write_text("20240403\t{'41_47': {'41_47': 2.5}}\n20240402\tNA\n20240401\t{'41_47': {'58_44': 1.0}, '1_1': {'1_1': 99.0}}\n", encoding="utf-8")
    data = load_data(path)
    assert data.X.shape == (2, 2) and data.dates[0] == pd.Timestamp("2024-04-01")
    assert data.X.dtype == np.float32
    assert all(35 <= p[0] <= 70 for p in data.pairs)


@pytest.mark.parametrize("payload", ["{'41_47': {'41_47': -1}}", "{'41_47': {'41_47': 1e309}}", "{'71_1': {'41_47': 1}}", "[]"])
def test_bad_input_rejected(tmp_path, payload):
    path = tmp_path / "input.tsv"
    path.write_text("20240401\t" + payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(records(path))


def test_submission_roundtrip_and_overwrite(tmp_path):
    source = tmp_path / "input.tsv"
    source.write_text("20240401\t{'41_47': {'41_47': 2.0}, '1_1': {'1_1': 3.0}}\n", encoding="utf-8")
    data = load_data(source)
    pred = np.arange(1., 59.)[:, None]
    output = tmp_path / "out.tsv"
    assert write_submission(output, source, data, pred)["box_roundtrip_exact"]
    assert validate(output)["days"] == 58
    assert len({str(od) for _, od in records(output)}) == 58
    with pytest.raises(FileExistsError):
        write_submission(output, source, data, pred)


def test_invalid_submission_dates(tmp_path):
    path = tmp_path / "out.tsv"
    path.write_text("20240201\t{'41_47': {'41_47': 1.}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="58"):
        validate(path)

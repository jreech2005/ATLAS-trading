"""Behavioral tests for causal OHLCV features (Phase 1C).

Every frame is small, synthetic and deterministic. Timestamps are UTC bar closes, as in the data
contract; sessions are America/New_York dates of the minute each bar covers.
"""

import math
from collections.abc import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from atlas.data import OHLCVValidationError
from atlas.features import compute_features

NEW_YORK = ZoneInfo("America/New_York")
ONE_MINUTE = pd.Timedelta(minutes=1)
FEATURES = [
    "return_1m",
    "return_5m",
    "return_15m",
    "realized_volatility_15m",
    "relative_volume_15m",
]
OUTPUT_COLUMNS = ["timestamp", *FEATURES, "feature_ready"]
# Consecutive one-minute intervals each feature needs behind the current bar.
REQUIRED_HISTORY = {
    "return_1m": 1,
    "return_5m": 5,
    "return_15m": 15,
    "realized_volatility_15m": 15,
    "relative_volume_15m": 15,
}


def utc_minutes(start: str, count: int) -> pd.DatetimeIndex:
    """``count`` consecutive one-minute UTC bar closes beginning at ``start``."""
    return pd.date_range(start, periods=count, freq="min", tz="UTC")


def join(*parts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return parts[0].append(list(parts[1:]))


def make_frame(
    timestamps: pd.DatetimeIndex,
    close: Sequence[float] | np.ndarray | None = None,
    volume: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Valid canonical frame; ``close`` defaults to 100, 101, ... and ``volume`` to 1000."""
    n = len(timestamps)
    close = np.asarray(100.0 + np.arange(n) if close is None else close, dtype=np.float64)
    volume = np.asarray(np.full(n, 1000.0) if volume is None else volume, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": volume,
        }
    )


def random_frame(timestamps: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    volume = rng.integers(0, 3000, n).astype(np.float64)
    volume[rng.random(n) < 0.05] = 0.0
    return make_frame(timestamps, close, volume)


# A multi-session scenario covering every boundary type the features must respect.
SCENARIO = join(
    utc_minutes("2024-01-02 20:31", 30),  # Tue 15:31-16:00 EST
    utc_minutes("2024-01-03 14:31", 20),  # Wed open (overnight boundary)
    utc_minutes("2024-01-03 14:52", 20),  # one missing minute (14:51 close)
    utc_minutes("2024-01-03 15:42", 10),  # 30-minute gap; never warms up
    utc_minutes("2024-01-04 04:51", 20),  # crosses New York midnight at the 05:01Z close
    utc_minutes("2024-03-10 06:41", 40),  # crosses the spring-forward DST change at 07:00Z
)


def session_date(close: pd.Timestamp) -> object:
    return (close - ONE_MINUTE).tz_convert(NEW_YORK).date()


def reference_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row reference written directly from the feature definitions."""
    ts = list(frame["timestamp"])
    close = [float(c) for c in frame["close"]]
    volume = [float(v) for v in frame["volume"]]

    def consecutive(i: int, k: int) -> bool:
        return i >= k and all(
            ts[j] - ts[j - 1] == ONE_MINUTE and session_date(ts[j]) == session_date(ts[j - 1])
            for j in range(i - k + 1, i + 1)
        )

    rows = []
    for i in range(len(frame)):
        row = dict.fromkeys(FEATURES, math.nan)
        for name, k in (("return_1m", 1), ("return_5m", 5), ("return_15m", 15)):
            if consecutive(i, k):
                row[name] = close[i] / close[i - k] - 1.0
        if consecutive(i, 15):
            squares = [math.log(close[j] / close[j - 1]) ** 2 for j in range(i - 14, i + 1)]
            row["realized_volatility_15m"] = math.sqrt(sum(squares))
            mean_volume = sum(volume[i - 15 : i]) / 15
            if mean_volume > 0:
                row["relative_volume_15m"] = volume[i] / mean_volume
        row["feature_ready"] = all(math.isfinite(row[name]) for name in FEATURES)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*FEATURES, "feature_ready"])


def assert_rows_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True
    )


# --- Output shape -------------------------------------------------------------------------------


def test_output_has_timestamp_features_and_ready_flag() -> None:
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 20)))
    assert list(result.columns) == OUTPUT_COLUMNS
    for name in FEATURES:
        assert result[name].dtype == np.float64
    assert result["feature_ready"].dtype == np.bool_


# --- 1. Mathematical correctness ----------------------------------------------------------------


def test_returns_are_close_to_close_percentage_changes() -> None:
    close = 100.0 + np.arange(16)  # 100, 101, ..., 115
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 16), close))
    last = result.iloc[15]
    assert last["return_1m"] == pytest.approx(115 / 114 - 1, rel=1e-12)
    assert last["return_5m"] == pytest.approx(115 / 110 - 1, rel=1e-12)
    assert last["return_15m"] == pytest.approx(115 / 100 - 1, rel=1e-12)


def test_realized_volatility_is_root_sum_of_squared_log_returns() -> None:
    close = 100.0 * np.exp(0.01 * np.arange(16))  # every one-minute log return is 0.01
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 16), close))
    assert result["realized_volatility_15m"].iloc[15] == pytest.approx(math.sqrt(15) * 0.01, rel=1e-9)


def test_realized_volatility_window_is_exactly_15_intervals_and_not_annualized() -> None:
    # The only nonzero log return is ln(2), between rows 0 and 1.
    close = np.r_[50.0, np.full(16, 100.0)]
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 17), close))
    volatility = result["realized_volatility_15m"]
    assert volatility.iloc[15] == pytest.approx(math.log(2), rel=1e-12)  # log, not simple return 1.0
    assert volatility.iloc[16] == 0.0  # the jump is now the 16th interval back: outside the window


def test_relative_volume_divides_by_mean_of_preceding_15_bars_excluding_current() -> None:
    volume = np.r_[np.full(15, 100.0), 1600.0]
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 16), volume=volume))
    assert result["relative_volume_15m"].iloc[15] == 16.0  # 8.0 if the current bar were included


def test_relative_volume_window_is_exactly_15_preceding_bars() -> None:
    volume = np.r_[10_000.0, np.full(15, 100.0), 200.0]
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 17), volume=volume))
    assert result["relative_volume_15m"].iloc[16] == 2.0


def test_features_match_reference_on_multi_session_data() -> None:
    frame = random_frame(SCENARIO)
    result = compute_features(frame)
    expected = reference_features(frame)
    for name in FEATURES:
        np.testing.assert_array_equal(result[name].isna(), expected[name].isna(), err_msg=name)
        np.testing.assert_allclose(result[name], expected[name], rtol=1e-12, err_msg=name)
    np.testing.assert_array_equal(result["feature_ready"], expected["feature_ready"])
    assert result["feature_ready"].sum() == 15 + 5 + 5 + 0 + 0 + 25  # per SCENARIO segment


def test_integer_and_nullable_dtypes_give_the_same_features() -> None:
    frame = make_frame(utc_minutes("2024-01-02 14:31", 20), volume=np.arange(20) * 10 + 5)
    as_int = frame.assign(volume=frame["volume"].astype(np.int64))
    nullable = frame.astype({c: "Float64" for c in ("open", "high", "low", "close")}).astype(
        {"volume": "Int64"}
    )
    expected = compute_features(frame)
    pd.testing.assert_frame_equal(compute_features(as_int), expected, check_exact=True)
    pd.testing.assert_frame_equal(compute_features(nullable), expected, check_exact=True)


# --- 2. Future perturbation invariance ----------------------------------------------------------


def perturb_after(frame: pd.DataFrame, t: int, seed: int) -> pd.DataFrame:
    """Copy of ``frame`` with every price and volume after row position ``t`` changed."""
    rng = np.random.default_rng(seed)
    later = np.arange(len(frame)) > t
    factor = np.where(later, rng.uniform(0.5, 2.0, len(frame)), 1.0)
    volume = np.where(later, rng.integers(0, 50_000, len(frame)), frame["volume"])
    return frame.assign(
        **{c: frame[c] * factor for c in ("open", "high", "low", "close")},
        volume=volume.astype(np.float64),
    )


def test_changing_future_bars_never_changes_earlier_features() -> None:
    frame = random_frame(SCENARIO)
    full = compute_features(frame)
    for t in range(len(frame)):
        perturbed = compute_features(perturb_after(frame, t, seed=t))
        assert_rows_equal(perturbed.iloc[: t + 1], full.iloc[: t + 1])


def test_removing_or_shifting_future_bars_never_changes_earlier_features() -> None:
    frame = random_frame(SCENARIO)
    full = compute_features(frame)
    for t in range(len(frame)):
        truncated = compute_features(frame.iloc[: t + 1])
        assert_rows_equal(truncated, full.iloc[: t + 1])
        later = np.arange(len(frame)) > t
        delay = pd.to_timedelta(np.where(later, 3, 0), unit="min").to_numpy()
        shifted = frame.assign(timestamp=frame["timestamp"] + delay)  # opens a gap after t
        assert_rows_equal(compute_features(shifted).iloc[: t + 1], full.iloc[: t + 1])


def test_perturbation_check_is_sensitive_to_the_current_bar() -> None:
    frame = random_frame(SCENARIO)
    t = int(np.flatnonzero(compute_features(frame)["feature_ready"])[0])
    changed = compute_features(perturb_after(frame, t - 1, seed=0))
    assert changed["return_1m"].iloc[t] != compute_features(frame)["return_1m"].iloc[t]


# --- 3/4. Session boundaries and missing minutes ------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            utc_minutes("2024-01-02 20:41", 20), utc_minutes("2024-01-03 14:31", 20), id="overnight"
        ),
        pytest.param(
            utc_minutes("2024-01-05 20:41", 20), utc_minutes("2024-01-08 14:31", 20), id="weekend"
        ),
        pytest.param(
            utc_minutes("2024-01-02 14:31", 20), utc_minutes("2024-01-02 14:52", 20), id="one_missing"
        ),
        pytest.param(
            utc_minutes("2024-01-02 14:31", 20), utc_minutes("2024-01-02 15:21", 20), id="long_gap"
        ),
    ],
)
def test_windows_restart_after_a_session_boundary_or_missing_minutes(
    before: pd.DatetimeIndex, after: pd.DatetimeIndex
) -> None:
    frame = random_frame(join(before, after))
    result = compute_features(frame)
    tail = result.iloc[len(before) :]
    # Features after the break equal those computed from the post-break bars alone.
    alone = compute_features(frame.iloc[len(before) :])
    assert_rows_equal(tail, alone)
    assert tail["return_1m"].isna().tolist() == [True] + [False] * 19
    assert tail["feature_ready"].tolist() == [False] * 15 + [True] * 5


def test_single_missing_minute_inside_a_window_breaks_it() -> None:
    # Closes 14:31..14:50 then 14:52..14:56: the 14:51 bar is missing.
    timestamps = join(utc_minutes("2024-01-02 14:31", 20), utc_minutes("2024-01-02 14:52", 5))
    result = compute_features(random_frame(timestamps))
    after_gap = result.iloc[20:]
    assert after_gap["return_5m"].isna().all()  # each would reach back across the gap
    assert not after_gap["feature_ready"].any()


def test_new_york_midnight_is_a_session_boundary() -> None:
    # 04:51Z..05:10Z in January is 23:51..00:10 EST. The 05:00Z bar covers 23:59-24:00 (Jan 3);
    # the 05:01Z bar covers 00:00-00:01 (Jan 4) and starts a new session.
    timestamps = utc_minutes("2024-01-04 04:51", 20)
    result = compute_features(random_frame(timestamps))
    first_of_new_session = timestamps.get_loc(pd.Timestamp("2024-01-04 05:01", tz="UTC"))
    assert first_of_new_session == 10
    assert not math.isnan(result["return_1m"].iloc[9])
    assert math.isnan(result["return_1m"].iloc[10])
    assert not result["feature_ready"].any()


# --- 5. Warm-up ---------------------------------------------------------------------------------


def test_warm_up_rows_are_nan_and_not_ready() -> None:
    result = compute_features(random_frame(utc_minutes("2024-01-02 14:31", 20)))
    for name, history in REQUIRED_HISTORY.items():
        assert result[name].isna().tolist() == [True] * history + [False] * (20 - history), name
    assert result["feature_ready"].tolist() == [False] * 15 + [True] * 5


@pytest.mark.parametrize("rows", range(1, 16))
def test_frames_shorter_than_the_window_have_no_ready_rows(rows: int) -> None:
    # Regression: lags longer than the frame once raised instead of yielding NaN.
    result = compute_features(random_frame(utc_minutes("2024-01-02 14:31", rows)))
    assert len(result) == rows
    assert result[["return_15m", "realized_volatility_15m", "relative_volume_15m"]].isna().all(
        axis=None
    )
    assert not result["feature_ready"].any()


# --- 6. Zero volume and undefined ratios --------------------------------------------------------


@pytest.mark.parametrize(
    "current_volume",
    [pytest.param(0.0, id="zero_over_zero"), pytest.param(500.0, id="positive_over_zero")],
)
def test_zero_mean_volume_makes_relative_volume_undefined(current_volume: float) -> None:
    volume = np.r_[np.zeros(15), current_volume]
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 16), volume=volume))
    assert math.isnan(result["relative_volume_15m"].iloc[15])
    assert not result["feature_ready"].iloc[15]
    # Returns do not depend on volume and stay defined.
    assert result[["return_1m", "return_5m", "return_15m"]].iloc[15].notna().all()


def test_zero_current_volume_with_positive_history_is_zero_and_ready() -> None:
    volume = np.r_[np.zeros(14), 150.0, 0.0]
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 16), volume=volume))
    assert result["relative_volume_15m"].iloc[15] == 0.0
    assert result["feature_ready"].iloc[15]


def test_overflowing_calculations_are_nan_and_not_ready() -> None:
    close = np.r_[np.full(15, 1e-300), 1e300]  # the return ratio overflows to inf
    result = compute_features(make_frame(utc_minutes("2024-01-02 14:31", 16), close))
    last = result.iloc[15]
    assert math.isnan(last["return_1m"])
    assert math.isnan(last["return_15m"])
    assert math.isnan(last["realized_volatility_15m"])
    assert not last["feature_ready"]
    assert np.isfinite(result[FEATURES].to_numpy()[result[FEATURES].notna().to_numpy()]).all()


# --- 7. No mutation -----------------------------------------------------------------------------


def test_input_frame_is_not_modified() -> None:
    frame = random_frame(SCENARIO).astype({"volume": "Int64"}).set_axis(np.arange(len(SCENARIO))[::-1])
    snapshot = frame.copy(deep=True)
    result = compute_features(frame)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    assert result is not frame
    result.loc[:, "timestamp"] = result["timestamp"] + ONE_MINUTE
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


# --- 8. Timezone and timestamp semantics --------------------------------------------------------


def test_output_timestamps_are_the_input_utc_bar_closes() -> None:
    frame = random_frame(SCENARIO)
    result = compute_features(frame)
    pd.testing.assert_series_equal(result["timestamp"], frame["timestamp"], check_exact=True)
    assert str(result["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize(
    "timestamps",
    [
        pytest.param(utc_minutes("2024-01-02 23:51", 20), id="utc_midnight"),
        pytest.param(utc_minutes("2024-03-10 06:51", 20), id="spring_forward_0700Z"),
        pytest.param(utc_minutes("2024-11-03 05:51", 20), id="fall_back_0600Z"),
    ],
)
def test_utc_midnight_and_dst_changes_are_not_boundaries(timestamps: pd.DatetimeIndex) -> None:
    # Consecutive UTC minutes within one New York date are continuous even when the UTC date or
    # the New York wall-clock offset changes inside the window.
    result = compute_features(random_frame(timestamps))
    assert result["feature_ready"].tolist() == [False] * 15 + [True] * 5


@pytest.mark.parametrize(
    "timestamps",
    [
        pytest.param(utc_minutes("2024-01-02 14:31", 20).tz_localize(None), id="naive"),
        pytest.param(utc_minutes("2024-01-02 14:31", 20).tz_convert(NEW_YORK), id="new_york"),
    ],
)
def test_non_utc_timestamps_are_rejected(timestamps: pd.DatetimeIndex) -> None:
    with pytest.raises(OHLCVValidationError, match="timestamp"):
        compute_features(make_frame(timestamps))


def test_invalid_frames_are_rejected_by_the_data_contract() -> None:
    frame = make_frame(utc_minutes("2024-01-02 14:31", 20))
    with pytest.raises(OHLCVValidationError, match="strictly increasing"):
        compute_features(frame.iloc[::-1])
    with pytest.raises(TypeError):
        compute_features(frame.to_numpy())  # type: ignore[arg-type]


# --- 9. Row alignment ---------------------------------------------------------------------------


def test_index_and_row_order_are_preserved() -> None:
    frame = random_frame(SCENARIO)
    labels = np.r_[[7, 7, 3], np.arange(len(frame) - 3)[::-1]]  # duplicate, non-monotonic labels
    relabelled = frame.set_axis(labels)
    result = compute_features(relabelled)
    np.testing.assert_array_equal(result.index, labels)
    assert_rows_equal(result, compute_features(frame))


# --- 10. Determinism ----------------------------------------------------------------------------


def test_output_is_deterministic() -> None:
    frame = random_frame(SCENARIO)
    first = compute_features(frame)
    second = compute_features(frame.copy(deep=True))
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first[FEATURES].to_numpy().tobytes() == second[FEATURES].to_numpy().tobytes()

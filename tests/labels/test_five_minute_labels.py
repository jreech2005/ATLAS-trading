"""Behavioral tests for five-minute prediction labels (Phase 1D).

Every frame is small, synthetic and deterministic. Timestamps are UTC bar closes, as in the data
contract; sessions are America/New_York dates of the minute each bar covers, as in Phase 1C.
"""

import ast
import inspect
import math
from collections.abc import Callable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import atlas.features
from atlas.data import OHLCVValidationError
from atlas.features import compute_features
from atlas.labels import compute_labels

NEW_YORK = ZoneInfo("America/New_York")
ONE_MINUTE = pd.Timedelta(minutes=1)
HORIZON = 5
OUTPUT_COLUMNS = [
    "timestamp",
    "future_return_5m",
    "direction_5m",
    "label_end_timestamp",
    "label_ready",
]
FLOAT_MAX = np.finfo(np.float64).max
SMALLEST_SUBNORMAL = 5e-324


def utc_minutes(start: str, count: int) -> pd.DatetimeIndex:
    """``count`` consecutive one-minute UTC bar closes beginning at ``start``."""
    return pd.date_range(start, periods=count, freq="min", tz="UTC")


def join(*parts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return parts[0].append(list(parts[1:]))


def make_frame(
    timestamps: pd.DatetimeIndex | Sequence[str],
    close: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Valid canonical frame with open = high = low = close; ``close`` defaults to 100, 101, ..."""
    timestamps = pd.DatetimeIndex(timestamps)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    n = len(timestamps)
    close = np.asarray(100.0 + np.arange(n) if close is None else close, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def random_frame(timestamps: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(timestamps))))
    close[rng.random(len(timestamps)) < 0.1] = 100.0  # some exactly-zero returns
    return make_frame(timestamps, close)


# A multi-session scenario covering every boundary type the labels must respect.
SCENARIO = join(
    utc_minutes("2024-01-02 20:41", 20),  # Tue 15:41-16:00 EST
    utc_minutes("2024-01-03 14:31", 12),  # Wed open (overnight boundary)
    utc_minutes("2024-01-03 14:44", 12),  # one missing minute (14:43 close)
    utc_minutes("2024-01-05 20:51", 10),  # Fri close
    utc_minutes("2024-01-08 14:31", 8),  # Mon open (weekend boundary)
    utc_minutes("2024-01-09 04:51", 20),  # New York midnight (EST) at the 05:01Z close
    utc_minutes("2024-03-10 06:41", 40),  # spring-forward DST change at 07:00Z, one NY date
    utc_minutes("2024-07-02 03:51", 20),  # New York midnight (EDT) at the 04:01Z close
    utc_minutes("2024-11-03 05:41", 40),  # fall-back DST change at 06:00Z, one NY date
)
# Ready labels per SCENARIO segment: each uninterrupted run of n bars has n - 5.
SCENARIO_READY = 15 + 7 + 7 + 5 + 3 + (5 + 5) + 35 + (5 + 5) + 35


def session_date(close: pd.Timestamp) -> object:
    return (close - ONE_MINUTE).tz_convert(NEW_YORK).date()


def reference_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row reference written directly from the label definition."""
    ts = list(frame["timestamp"])
    close = [float(c) for c in frame["close"]]
    returns, directions, ends, ready = [], [], [], []
    for i in range(len(ts)):
        j = i + HORIZON
        value = math.nan
        if j < len(ts) and all(
            ts[k] - ts[k - 1] == ONE_MINUTE and session_date(ts[k]) == session_date(ts[k - 1])
            for k in range(i + 1, j + 1)
        ):
            value = close[j] / close[i] - 1.0
        ok = math.isfinite(value)
        returns.append(value if ok else math.nan)
        directions.append(int(value > 0) if ok else pd.NA)
        ends.append(ts[j] if ok else pd.NaT)
        ready.append(ok)
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].array,
            "future_return_5m": np.array(returns, dtype=np.float64),
            "direction_5m": pd.array(directions, dtype="Int64"),
            "label_end_timestamp": pd.array(ends, dtype=frame["timestamp"].dtype),
            "label_ready": np.array(ready, dtype=bool),
        }
    )


def assert_rows_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True
    )


def assert_undefined(row: pd.Series) -> None:
    assert math.isnan(row["future_return_5m"])
    assert row["direction_5m"] is pd.NA
    assert row["label_end_timestamp"] is pd.NaT
    assert not row["label_ready"]


# --- Output contract (16, 17) -------------------------------------------------------------------


def test_output_columns_and_dtypes() -> None:
    frame = make_frame(utc_minutes("2024-01-02 14:31", 10))
    result = compute_labels(frame)
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["future_return_5m"].dtype == np.float64
    assert result["direction_5m"].dtype == pd.Int64Dtype()
    assert result["label_ready"].dtype == np.bool_
    for column in ("timestamp", "label_end_timestamp"):
        dtype = result[column].dtype
        assert isinstance(dtype, pd.DatetimeTZDtype), column
        assert str(dtype.tz) == "UTC", column
        assert dtype == frame["timestamp"].dtype, column


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timestamp_resolution_is_preserved(unit: str) -> None:
    frame = make_frame(utc_minutes("2024-01-02 14:31", 10).as_unit(unit))
    result = compute_labels(frame)
    assert result["timestamp"].dtype == pd.DatetimeTZDtype(unit, "UTC")
    assert result["label_end_timestamp"].dtype == pd.DatetimeTZDtype(unit, "UTC")
    assert result["label_end_timestamp"].iloc[0] == pd.Timestamp("2024-01-02 14:36", tz="UTC")


def test_dtypes_are_stable_when_no_label_is_defined() -> None:
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 3)))
    assert result["future_return_5m"].dtype == np.float64
    assert result["direction_5m"].dtype == pd.Int64Dtype()
    assert result["label_ready"].dtype == np.bool_
    assert isinstance(result["label_end_timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(result["label_end_timestamp"].dt.tz) == "UTC"


# --- 1-3. Direction and return ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("end_close", "expected_return", "expected_direction"),
    [
        pytest.param(110.0, 110.0 / 100.0 - 1.0, 1, id="positive"),
        pytest.param(95.0, 95.0 / 100.0 - 1.0, 0, id="negative"),
        pytest.param(100.0, 0.0, 0, id="zero"),
    ],
)
def test_return_and_direction(
    end_close: float, expected_return: float, expected_direction: int
) -> None:
    close = [100.0, 150.0, 50.0, 120.0, 80.0, end_close]  # intermediate bars must not matter
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 6), close))
    first = result.iloc[0]
    assert first["future_return_5m"] == expected_return
    assert first["direction_5m"] == expected_direction
    assert first["label_ready"]


def test_return_is_measured_against_the_bar_exactly_five_minutes_later() -> None:
    close = 100.0 + np.arange(12)  # 100, 101, ..., 111
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 12), close))
    for i in range(7):
        assert result["future_return_5m"].iloc[i] == close[i + 5] / close[i] - 1.0


@pytest.mark.parametrize(
    ("step", "expected_direction"),
    [pytest.param(np.inf, 1, id="one_ulp_up"), pytest.param(-np.inf, 0, id="one_ulp_down")],
)
def test_smallest_representable_moves_have_the_right_direction(
    step: float, expected_direction: int
) -> None:
    close = [100.0] * 5 + [np.nextafter(100.0, step)]
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 6), close))
    assert result["direction_5m"].iloc[0] == expected_direction
    assert (result["future_return_5m"].iloc[0] > 0) == bool(expected_direction)
    assert result["future_return_5m"].iloc[0] != 0.0


# --- 4/5. End of data and short frames ----------------------------------------------------------


def test_last_five_observations_are_undefined() -> None:
    result = compute_labels(random_frame(utc_minutes("2024-01-02 14:31", 20)))
    assert result["label_ready"].tolist() == [True] * 15 + [False] * 5
    for position in range(15, 20):
        assert_undefined(result.iloc[position])


@pytest.mark.parametrize("rows", range(1, 6))
def test_fewer_than_six_observations_have_no_labels(rows: int) -> None:
    result = compute_labels(random_frame(utc_minutes("2024-01-02 14:31", rows)))
    assert len(result) == rows
    assert not result["label_ready"].any()
    assert result["future_return_5m"].isna().all()
    assert result["direction_5m"].isna().all()
    assert result["label_end_timestamp"].isna().all()


def test_exactly_six_observations_give_one_label() -> None:
    result = compute_labels(random_frame(utc_minutes("2024-01-02 14:31", 6)))
    assert result["label_ready"].tolist() == [True] + [False] * 5


# --- 6. Missing minutes: five rows are not five minutes -----------------------------------------


def test_missing_minute_inside_the_future_window_makes_labels_undefined() -> None:
    # Closes 14:31..14:40, then 14:42..14:51: the 14:41 bar is missing.
    timestamps = join(utc_minutes("2024-01-02 14:31", 10), utc_minutes("2024-01-02 14:42", 10))
    result = compute_labels(random_frame(timestamps))
    expected = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
    assert result["label_ready"].tolist() == expected
    for position in range(5, 10):
        assert_undefined(result.iloc[position])


def test_a_later_nonconsecutive_bar_never_substitutes_for_the_target_bar() -> None:
    # Row 5 is the 14:37 bar, six minutes after row 0; the 14:36 target bar does not exist.
    timestamps = join(utc_minutes("2024-01-02 14:31", 5), utc_minutes("2024-01-02 14:37", 6))
    result = compute_labels(random_frame(timestamps))
    assert not result["label_ready"].iloc[:5].any()
    assert result["label_ready"].iloc[5]  # the first full window after the gap


@pytest.mark.parametrize(
    "timestamps",
    [
        # Row 5 is exactly five minutes after row 0, but the steps are not one-minute bars.
        pytest.param(
            [
                "2024-01-02 14:31:00",
                "2024-01-02 14:31:30",
                "2024-01-02 14:32:30",
                "2024-01-02 14:33:30",
                "2024-01-02 14:34:30",
                "2024-01-02 14:36:00",
            ],
            id="irregular_steps_summing_to_five_minutes",
        ),
        # An extra sub-minute bar: the bar five minutes later exists, but six rows ahead.
        pytest.param(
            [
                "2024-01-02 14:31:00",
                "2024-01-02 14:32:00",
                "2024-01-02 14:32:30",
                "2024-01-02 14:33:00",
                "2024-01-02 14:34:00",
                "2024-01-02 14:35:00",
                "2024-01-02 14:36:00",
            ],
            id="extra_sub_minute_bar",
        ),
    ],
)
def test_labels_require_five_consecutive_one_minute_steps(timestamps: list[str]) -> None:
    result = compute_labels(make_frame(timestamps))
    assert not result["label_ready"].any()


# --- 7-10. Session boundaries, UTC and DST ------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            utc_minutes("2024-01-02 20:51", 10), utc_minutes("2024-01-03 14:31", 10), id="overnight"
        ),
        pytest.param(
            utc_minutes("2024-01-05 20:51", 10), utc_minutes("2024-01-08 14:31", 10), id="weekend"
        ),
    ],
)
def test_labels_never_cross_overnight_or_weekend_gaps(
    before: pd.DatetimeIndex, after: pd.DatetimeIndex
) -> None:
    frame = random_frame(join(before, after))
    result = compute_labels(frame)
    assert result["label_ready"].tolist() == ([True] * 5 + [False] * 5) * 2
    # The last label before the break ends at the session's final bar.
    assert result["label_end_timestamp"].iloc[4] == before[-1]
    # Labels before the break equal those computed from the pre-break bars alone.
    assert_rows_equal(result.iloc[: len(before)], compute_labels(frame.iloc[: len(before)]))


@pytest.mark.parametrize(
    ("start", "first_new_date_close"),
    [
        pytest.param("2024-01-04 04:51", "2024-01-04 05:01", id="EST_midnight_0500Z"),
        pytest.param("2024-07-02 03:51", "2024-07-02 04:01", id="EDT_midnight_0400Z"),
    ],
)
def test_new_york_midnight_is_a_boundary(start: str, first_new_date_close: str) -> None:
    # The bar closing at 00:00 New York time covers 23:59-24:00 and belongs to the earlier date.
    timestamps = utc_minutes(start, 20)
    result = compute_labels(random_frame(timestamps))
    boundary = timestamps.get_loc(pd.Timestamp(first_new_date_close, tz="UTC"))
    assert boundary == 10
    assert result["label_ready"].tolist() == ([True] * 5 + [False] * 5) * 2
    # The label at row 4 ends at the 00:00 New York close, the last bar of the earlier date.
    assert result["label_end_timestamp"].iloc[4] == timestamps[9]


@pytest.mark.parametrize(
    "timestamps",
    [
        pytest.param(utc_minutes("2024-01-02 23:51", 20), id="utc_midnight"),
        pytest.param(utc_minutes("2024-03-10 06:51", 20), id="spring_forward_0700Z"),
        pytest.param(utc_minutes("2024-11-03 05:51", 20), id="fall_back_0600Z"),
    ],
)
def test_utc_midnight_and_dst_changes_are_not_boundaries(timestamps: pd.DatetimeIndex) -> None:
    # Labels are five minutes of elapsed UTC time, even when the wall clock jumps or repeats.
    result = compute_labels(random_frame(timestamps))
    assert result["label_ready"].tolist() == [True] * 15 + [False] * 5
    ready = result[result["label_ready"]]
    assert ((ready["label_end_timestamp"] - ready["timestamp"]) == HORIZON * ONE_MINUTE).all()


def test_no_label_is_formed_across_sessions() -> None:
    frame = random_frame(SCENARIO)
    result = compute_labels(frame)
    ready = result[result["label_ready"]]
    assert len(ready) == SCENARIO_READY
    assert ((ready["label_end_timestamp"] - ready["timestamp"]) == HORIZON * ONE_MINUTE).all()
    for start, end in zip(ready["timestamp"], ready["label_end_timestamp"], strict=True):
        window = frame["timestamp"][(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        assert len(window) == HORIZON + 1
        assert len({session_date(close) for close in window}) == 1


# --- 15. Label end timestamps -------------------------------------------------------------------


def test_label_end_timestamp_is_the_close_of_the_target_bar() -> None:
    frame = random_frame(SCENARIO)
    result = compute_labels(frame)
    positions = np.flatnonzero(result["label_ready"])
    np.testing.assert_array_equal(
        result["label_end_timestamp"].iloc[positions].to_numpy(),
        frame["timestamp"].iloc[positions + HORIZON].to_numpy(),
    )
    assert result["label_end_timestamp"][~result["label_ready"]].isna().all()


# --- 18/19. Numerical extremes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "close",
    [
        pytest.param([1e-300, 1.0, 1.0, 1.0, 1.0, 1e300], id="ratio_overflows"),
        pytest.param([0.5, 1.0, 1.0, 1.0, 1.0, FLOAT_MAX], id="ratio_just_overflows"),
    ],
)
def test_nonfinite_returns_are_undefined(close: list[float]) -> None:
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 6), close))
    assert_undefined(result.iloc[0])
    assert not result["label_ready"].any()


def test_largest_finite_ratio_is_a_valid_label() -> None:
    close = [1.0, 1.0, 1.0, 1.0, 1.0, FLOAT_MAX]
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 6), close))
    assert result["future_return_5m"].iloc[0] == FLOAT_MAX - 1.0
    assert result["direction_5m"].iloc[0] == 1
    assert result["label_ready"].iloc[0]


def test_ratio_underflow_is_a_finite_full_loss() -> None:
    # 1e-300 / 1e300 underflows to 0.0; the correctly rounded return is -1.0, still "not up".
    close = [1e300, 1.0, 1.0, 1.0, 1.0, 1e-300]
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 6), close))
    assert result["future_return_5m"].iloc[0] == -1.0
    assert result["direction_5m"].iloc[0] == 0
    assert result["label_ready"].iloc[0]


@pytest.mark.parametrize(
    ("start", "end", "expected_return", "expected_direction"),
    [
        pytest.param(SMALLEST_SUBNORMAL, 2 * SMALLEST_SUBNORMAL, 1.0, 1, id="subnormal_up"),
        pytest.param(2 * SMALLEST_SUBNORMAL, SMALLEST_SUBNORMAL, -0.5, 0, id="subnormal_down"),
        pytest.param(1e-300, 1.5e-300, 1.5e-300 / 1e-300 - 1.0, 1, id="tiny_normal_up"),
        pytest.param(1e-300, 1e-300, 0.0, 0, id="tiny_flat"),
    ],
)
def test_tiny_positive_prices(
    start: float, end: float, expected_return: float, expected_direction: int
) -> None:
    close = [start] * 5 + [end]
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", 6), close))
    assert result["future_return_5m"].iloc[0] == expected_return
    assert result["direction_5m"].iloc[0] == expected_direction
    assert result["label_ready"].iloc[0]


def test_defined_returns_are_always_finite() -> None:
    close = np.r_[1e-300, 1e300, 1e-300, FLOAT_MAX, 1.0, 1e-300, 1e300, SMALLEST_SUBNORMAL, 1.0]
    close = np.tile(close, 3)
    result = compute_labels(make_frame(utc_minutes("2024-01-02 14:31", len(close)), close))
    defined = result["future_return_5m"].notna()
    assert np.isfinite(result["future_return_5m"][defined]).all()
    np.testing.assert_array_equal(defined, result["label_ready"])
    np.testing.assert_array_equal(result["direction_5m"].notna(), result["label_ready"])
    np.testing.assert_array_equal(result["label_end_timestamp"].notna(), result["label_ready"])


def test_integer_and_nullable_dtypes_give_the_same_labels() -> None:
    frame = make_frame(utc_minutes("2024-01-02 14:31", 20), close=np.arange(20) % 7 + 10)
    as_int = frame.astype({c: np.int64 for c in ("open", "high", "low", "close", "volume")})
    nullable = frame.astype({c: "Float64" for c in ("open", "high", "low", "close")}).astype(
        {"volume": "Int64"}
    )
    expected = compute_labels(frame)
    pd.testing.assert_frame_equal(compute_labels(as_int), expected, check_exact=True)
    pd.testing.assert_frame_equal(compute_labels(nullable), expected, check_exact=True)


# --- 20. Reference implementation and properties ------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_labels_match_reference_on_multi_session_data(seed: int) -> None:
    frame = random_frame(SCENARIO, seed=seed)
    result = compute_labels(frame)
    pd.testing.assert_frame_equal(result, reference_labels(frame), check_exact=True)
    assert result["label_ready"].sum() == SCENARIO_READY


def test_direction_is_one_exactly_when_the_target_close_is_higher() -> None:
    frame = random_frame(SCENARIO)
    result = compute_labels(frame)
    positions = np.flatnonzero(result["label_ready"])
    close = frame["close"].to_numpy()
    up = close[positions + HORIZON] > close[positions]
    np.testing.assert_array_equal(result["direction_5m"].iloc[positions].to_numpy(), up.astype(int))
    assert 0 < up.sum() < len(up)
    assert (close[positions + HORIZON] == close[positions]).any()  # zero returns are exercised


def test_each_label_depends_only_on_its_own_six_bar_window() -> None:
    frame = random_frame(SCENARIO)
    full = compute_labels(frame)
    for i in range(len(frame)):
        window = compute_labels(frame.iloc[i : i + HORIZON + 1])
        assert_rows_equal(window.iloc[[0]], full.iloc[[i]])


def test_target_bar_close_changes_the_label() -> None:
    frame = random_frame(utc_minutes("2024-01-02 14:31", 6))
    before = compute_labels(frame)
    changed = frame.copy()
    changed.loc[5, ["open", "high", "low", "close"]] *= 1.5
    after = compute_labels(changed)
    assert after["future_return_5m"].iloc[0] != before["future_return_5m"].iloc[0]


def test_labels_agree_with_phase_1c_return_5m_continuity() -> None:
    # The feature return_5m at bar t + 5 looks back over exactly the bars the label at t looks
    # forward over, so the two must agree value-for-value, including where each is undefined.
    frame = random_frame(SCENARIO)
    labels = compute_labels(frame)["future_return_5m"].to_numpy()
    features = compute_features(frame)["return_5m"].to_numpy()
    np.testing.assert_array_equal(labels[:-HORIZON], features[HORIZON:])
    assert np.isnan(labels[-HORIZON:]).all()
    assert np.isnan(features[:HORIZON]).all()


# --- 11/12/22. Index and row order --------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(lambda n: [f"bar-{k}" for k in range(n)], id="strings"),
        pytest.param(lambda n: np.arange(n)[::-1] * 10, id="descending_integers"),
        pytest.param(lambda n: pd.date_range("2000-01-01", periods=n, freq="D"), id="dates"),
        pytest.param(lambda n: [3, 3, 7, 7, *range(n - 4)], id="duplicates"),
        pytest.param(lambda n: [0] * n, id="all_identical"),
    ],
)
def test_index_and_row_order_are_preserved(labels: Callable[[int], object]) -> None:
    frame = random_frame(SCENARIO)
    index = pd.Index(labels(len(frame)))
    result = compute_labels(frame.set_axis(index))
    pd.testing.assert_index_equal(result.index, index)
    assert_rows_equal(result, compute_labels(frame))
    pd.testing.assert_series_equal(
        result["timestamp"].reset_index(drop=True), frame["timestamp"], check_exact=True
    )


def test_output_timestamps_are_the_input_utc_bar_closes() -> None:
    frame = random_frame(SCENARIO)
    result = compute_labels(frame)
    pd.testing.assert_series_equal(result["timestamp"], frame["timestamp"], check_exact=True)


# --- 13. No mutation ----------------------------------------------------------------------------


def test_input_frame_is_not_modified() -> None:
    frame = random_frame(SCENARIO).astype({"volume": "Int64"})
    frame = frame.set_axis(np.r_[[5, 5], np.arange(len(frame) - 2)[::-1]])
    snapshot = frame.copy(deep=True)
    result = compute_labels(frame)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    assert result is not frame
    result.loc[:, "timestamp"] = result["timestamp"] + ONE_MINUTE
    result.loc[:, "label_end_timestamp"] = result["label_end_timestamp"] + ONE_MINUTE
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)


# --- 14. Invalid input --------------------------------------------------------------------------


def _with(frame: pd.DataFrame, position: int, column: str, value: object) -> pd.DataFrame:
    changed = frame.copy()
    changed[column] = changed[column].astype(object if isinstance(value, str) else np.float64)
    changed.loc[position, column] = value
    return changed


VALID = make_frame(utc_minutes("2024-01-02 14:31", 10))


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        pytest.param(VALID.iloc[::-1], "strictly increasing", id="descending"),
        pytest.param(
            VALID.assign(timestamp=VALID["timestamp"].dt.tz_localize(None)), "naive", id="naive"
        ),
        pytest.param(
            VALID.assign(timestamp=VALID["timestamp"].dt.tz_convert(NEW_YORK)), "UTC", id="new_york"
        ),
        pytest.param(
            VALID.assign(timestamp=VALID["timestamp"].where(VALID.index != 3, VALID["timestamp"][2])),
            "duplicate timestamps",
            id="duplicate_timestamp",
        ),
        pytest.param(_with(VALID, 4, "close", np.nan), "missing values", id="nan_close"),
        pytest.param(_with(VALID, 4, "close", np.inf), "non-finite", id="inf_close"),
        pytest.param(VALID.assign(open=0.0, low=0.0), "strictly positive", id="zero_price"),
        pytest.param(_with(VALID, 4, "volume", -1.0), "nonnegative", id="negative_volume"),
        pytest.param(_with(VALID, 4, "close", "101"), "integer or float dtype", id="object_close"),
        pytest.param(VALID.assign(high=VALID["low"] - 1.0), "high >= low", id="ohlc_relation"),
        pytest.param(VALID.drop(columns="volume"), "columns must be exactly", id="missing_column"),
        pytest.param(VALID.assign(label=1), "columns must be exactly", id="extra_column"),
        pytest.param(VALID.iloc[:0], "empty", id="empty"),
    ],
)
def test_invalid_frames_are_rejected_by_the_data_contract(bad: pd.DataFrame, message: str) -> None:
    with pytest.raises(OHLCVValidationError, match=message):
        compute_labels(bad)


def test_non_dataframe_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        compute_labels(VALID.to_numpy())  # type: ignore[arg-type]


# --- Isolation from features --------------------------------------------------------------------


def test_feature_module_does_not_import_labels() -> None:
    tree = ast.parse(inspect.getsource(atlas.features))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any("labels" in name for name in imported), imported


def test_label_columns_are_disjoint_from_feature_columns() -> None:
    frame = random_frame(utc_minutes("2024-01-02 14:31", 20))
    features = compute_features(frame)
    labels = compute_labels(frame)
    assert set(features.columns) & set(labels.columns) == {"timestamp"}
    pd.testing.assert_frame_equal(compute_features(frame), features, check_exact=True)


# --- Determinism --------------------------------------------------------------------------------


def test_output_is_deterministic() -> None:
    frame = random_frame(SCENARIO)
    pd.testing.assert_frame_equal(
        compute_labels(frame), compute_labels(frame.copy(deep=True)), check_exact=True
    )

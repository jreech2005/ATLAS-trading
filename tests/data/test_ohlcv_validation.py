"""Behavioral tests for ``atlas.data.validate_ohlcv``.

The specification is docs/data_contract.md.
"""

import datetime as dt
from collections.abc import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from atlas.data import OHLCV_COLUMNS, OHLCVValidationError, validate_ohlcv

Transform = Callable[[pd.DataFrame], pd.DataFrame]
PRICE_COLUMNS = ("open", "high", "low", "close")


def set_value(frame: pd.DataFrame, column: str, row: int, value: object) -> pd.DataFrame:
    """Return a copy of ``frame`` with one cell replaced; pandas infers the column dtype."""
    values = frame[column].tolist()
    values[row] = value
    return frame.assign(**{column: values})


def reorder_timestamps(frame: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Return a copy whose timestamp column is taken from ``positions`` (dtype preserved)."""
    return frame.assign(timestamp=frame["timestamp"].iloc[positions].set_axis(frame.index))


# ---------------------------------------------------------------------------
# Invalid inputs: (transform of valid_bars, regex expected in the error message)
# ---------------------------------------------------------------------------

SCHEMA_CASES: dict[str, tuple[Transform, str]] = {
    "missing_column": (lambda f: f.drop(columns=["volume"]), r"missing.*'volume'"),
    "unexpected_column": (lambda f: f.assign(vwap=f["close"]), r"unexpected.*'vwap'"),
    "wrong_case_column": (
        lambda f: f.rename(columns={"close": "Close"}),
        r"missing.*'close'",
    ),
    "duplicate_column_label": (
        lambda f: pd.concat([f, f[["close"]]], axis=1),
        r"duplicate column.*'close'",
    ),
    "empty_with_columns": (lambda f: f.iloc[0:0], r"empty"),
    "empty_without_columns": (lambda f: pd.DataFrame(), r"missing"),
}

TIMESTAMP_CASES: dict[str, tuple[Transform, str]] = {
    "missing_timestamp": (
        lambda f: f.assign(timestamp=f["timestamp"].where(f.index != 1)),
        r"'timestamp'.*missing",
    ),
    "naive_timestamp": (
        lambda f: f.assign(timestamp=f["timestamp"].dt.tz_localize(None)),
        r"timezone-naive",
    ),
    "string_timestamp": (
        lambda f: f.assign(timestamp=f["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")),
        r"'timestamp'.*dtype",
    ),
    "object_timestamp": (
        lambda f: f.assign(timestamp=f["timestamp"].astype(object)),
        r"'timestamp'.*dtype",
    ),
    "duplicate_timestamp": (lambda f: reorder_timestamps(f, [0, 1, 1]), r"duplicate"),
    "out_of_order_timestamp": (
        lambda f: reorder_timestamps(f, [0, 2, 1]),
        r"strictly increasing",
    ),
}


def _numeric_cases(column: str, *, allow_zero: bool) -> dict[str, tuple[Transform, str]]:
    col = f"'{column}'"
    cases: dict[str, tuple[Transform, str]] = {
        f"{column}_nan": (lambda f: set_value(f, column, 1, np.nan), rf"{col}.*missing"),
        f"{column}_pos_inf": (lambda f: set_value(f, column, 1, np.inf), rf"{col}.*non-finite"),
        f"{column}_neg_inf": (lambda f: set_value(f, column, 1, -np.inf), rf"{col}.*non-finite"),
        f"{column}_negative": (
            lambda f: set_value(f, column, 1, -1.0),
            rf"{col}.*(strictly positive|nonnegative)",
        ),
        f"{column}_string": (
            lambda f: f.assign(**{column: f[column].astype(str)}),
            rf"{col}.*integer or float",
        ),
        f"{column}_object_numbers": (
            lambda f: f.assign(**{column: f[column].astype(object)}),
            rf"{col}.*integer or float",
        ),
        f"{column}_boolean": (
            lambda f: f.assign(**{column: [True, True, True]}),
            rf"{col}.*integer or float",
        ),
        f"{column}_nullable_na": (
            lambda f: f.assign(**{column: pd.array([1, pd.NA, 1], dtype="Int64")}),
            rf"{col}.*missing",
        ),
    }
    if not allow_zero:
        cases[f"{column}_zero"] = (
            lambda f: set_value(f, column, 1, 0.0),
            rf"{col}.*strictly positive",
        )
    return cases


NUMERIC_CASES: dict[str, tuple[Transform, str]] = {
    name: case
    for column in PRICE_COLUMNS
    for name, case in _numeric_cases(column, allow_zero=False).items()
} | _numeric_cases("volume", allow_zero=True)

# Row 1 of valid_bars: open 100.25, high 100.75, low 99.75, close 100.50.
OHLC_RELATION_CASES: dict[str, tuple[Transform, str]] = {
    "high_below_open": (lambda f: set_value(f, "open", 1, 100.80), r"high >= open"),
    "high_below_close": (lambda f: set_value(f, "close", 1, 100.80), r"high >= close"),
    "high_below_low": (lambda f: set_value(f, "high", 1, 99.70), r"high >= low"),
    "low_above_open": (lambda f: set_value(f, "open", 1, 99.70), r"low <= open"),
    "low_above_close": (lambda f: set_value(f, "close", 1, 99.70), r"low <= close"),
    "low_above_high": (lambda f: set_value(f, "low", 1, 100.80), r"high >= low"),
}

INVALID_CASES = SCHEMA_CASES | TIMESTAMP_CASES | NUMERIC_CASES | OHLC_RELATION_CASES


def _params(cases: dict[str, tuple[Transform, str]]) -> list:
    return [pytest.param(transform, match, id=name) for name, (transform, match) in cases.items()]


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


def test_canonical_columns_are_the_documented_six() -> None:
    assert OHLCV_COLUMNS == ("timestamp", "open", "high", "low", "close", "volume")


def test_valid_three_bar_frame_passes(valid_bars: pd.DataFrame) -> None:
    assert validate_ohlcv(valid_bars) is None


def test_single_bar_frame_passes(valid_bars: pd.DataFrame) -> None:
    validate_ohlcv(valid_bars.iloc[:1])


def test_column_order_is_not_significant(valid_bars: pd.DataFrame) -> None:
    validate_ohlcv(valid_bars[list(reversed(OHLCV_COLUMNS))])


def test_exact_high_low_boundary_equality_is_accepted(valid_bars: pd.DataFrame) -> None:
    frame = valid_bars.assign(
        open=[100.0, 100.0, 101.0],  # row 0 flat bar; row 1 open == low; row 2 open == high
        high=[100.0, 101.0, 101.0],
        low=[100.0, 100.0, 100.0],
        close=[100.0, 101.0, 100.0],  # row 1 close == high; row 2 close == low
    )
    validate_ohlcv(frame)


def test_zero_volume_is_permitted(valid_bars: pd.DataFrame) -> None:
    validate_ohlcv(set_value(valid_bars, "volume", 1, 0))


@pytest.mark.parametrize(
    "timestamps",
    [
        pytest.param(("2024-01-02 14:31", "2024-01-02 14:45", "2024-01-02 15:10"), id="intraday_gap"),
        pytest.param(("2024-01-02 21:00", "2024-01-03 14:31", "2024-01-03 14:32"), id="overnight_gap"),
        pytest.param(("2024-01-05 21:00", "2024-01-08 14:31", "2024-01-08 14:32"), id="weekend_gap"),
        pytest.param(("2024-07-03 17:00", "2024-07-05 13:31", "2024-07-05 13:32"), id="holiday_gap"),
        pytest.param(("2024-01-02 23:59", "2024-01-03 00:00", "2024-01-03 00:01"), id="utc_midnight"),
    ],
)
def test_gaps_between_bars_are_not_errors(make_bars, timestamps: tuple[str, ...]) -> None:
    validate_ohlcv(make_bars(timestamps))


@pytest.mark.parametrize(
    "tz",
    [
        pytest.param("UTC", id="str_UTC"),
        pytest.param(dt.timezone.utc, id="datetime_timezone_utc"),
        pytest.param(ZoneInfo("UTC"), id="zoneinfo_UTC"),
        pytest.param(ZoneInfo("Etc/UTC"), id="zoneinfo_Etc_UTC"),
    ],
)
def test_utc_timezone_representations_are_accepted(valid_bars: pd.DataFrame, tz) -> None:
    validate_ohlcv(valid_bars.assign(timestamp=valid_bars["timestamp"].dt.tz_convert(tz)))


def test_index_labels_are_not_part_of_the_contract(valid_bars: pd.DataFrame) -> None:
    validate_ohlcv(valid_bars.set_axis([7, 7, 3]))


def test_nullable_numeric_dtypes_without_missing_values_are_accepted(
    valid_bars: pd.DataFrame,
) -> None:
    frame = valid_bars.astype({c: "Float64" for c in PRICE_COLUMNS} | {"volume": "Int64"})
    validate_ohlcv(frame)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("transform", "match"), _params(INVALID_CASES))
def test_invalid_frame_is_rejected(
    valid_bars: pd.DataFrame, transform: Transform, match: str
) -> None:
    with pytest.raises(OHLCVValidationError, match=match):
        validate_ohlcv(transform(valid_bars))


@pytest.mark.parametrize("column", [*PRICE_COLUMNS, "volume"])
def test_complex_numeric_dtype_is_rejected(valid_bars: pd.DataFrame, column: str) -> None:
    frame = valid_bars.assign(**{column: valid_bars[column].astype(complex)})
    with pytest.raises(OHLCVValidationError, match=rf"'{column}'.*integer or float"):
        validate_ohlcv(frame)


@pytest.mark.parametrize(
    "tz",
    [
        pytest.param("America/New_York", id="new_york"),
        pytest.param("Europe/London", id="london_zero_offset_in_january"),
        pytest.param(dt.timezone(dt.timedelta(hours=-5)), id="fixed_offset"),
    ],
)
def test_non_utc_aware_timestamps_are_rejected_not_normalized(
    valid_bars: pd.DataFrame, tz
) -> None:
    frame = valid_bars.assign(timestamp=valid_bars["timestamp"].dt.tz_convert(tz))
    with pytest.raises(OHLCVValidationError, match=r"UTC"):
        validate_ohlcv(frame)


@pytest.mark.parametrize("not_a_frame", [None, {"timestamp": []}, pd.Series([1.0])])
def test_non_dataframe_input_raises_type_error(not_a_frame) -> None:
    with pytest.raises(TypeError, match=r"DataFrame"):
        validate_ohlcv(not_a_frame)


def test_error_reports_offending_row_positions(make_bars) -> None:
    frame = make_bars(["2024-01-02 14:31", "2024-01-02 14:32", "2024-01-02 14:33", "2024-01-02 14:34"])
    frame = set_value(set_value(frame, "volume", 1, -5), "volume", 3, -1)
    with pytest.raises(OHLCVValidationError, match=r"row positions \[1, 3\]"):
        validate_ohlcv(frame)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_valid_input_is_not_mutated(valid_bars: pd.DataFrame) -> None:
    snapshot = valid_bars.copy(deep=True)
    validate_ohlcv(valid_bars)
    pd.testing.assert_frame_equal(valid_bars, snapshot, check_exact=True)


@pytest.mark.parametrize(("transform", "match"), _params(INVALID_CASES))
def test_invalid_input_is_not_mutated(
    valid_bars: pd.DataFrame, transform: Transform, match: str
) -> None:
    frame = transform(valid_bars)
    snapshot = frame.copy(deep=True)
    with pytest.raises(OHLCVValidationError):
        validate_ohlcv(frame)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)

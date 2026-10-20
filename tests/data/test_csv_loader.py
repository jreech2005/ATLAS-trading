"""Behavioral tests for ``atlas.data.csv_loader.load_ohlcv_csv`` (Phase 1B specification).

Interface under test::

    load_ohlcv_csv(path, *, provider_timezone, timestamp_convention) -> pd.DataFrame

Contract these tests pin down (finalized by the Phase 1B architect):

* ``path`` is a ``str`` or ``os.PathLike``. ``provider_timezone`` and ``timestamp_convention``
  are keyword-only, required and have no defaults: a naive timestamp never gets a guessed zone.
* ``provider_timezone`` is an IANA name such as ``"America/New_York"`` or ``"UTC"``.
  ``timestamp_convention`` is exactly ``"bar_open"`` or ``"bar_close"``. With ``"bar_open"`` the
  provider stamp is the bar's start, so the result is the stamp plus one minute of *elapsed* time.
* Three failure classes:

  - bad API arguments (unknown or non-IANA timezone, unknown convention) raise ``ValueError``
    that is *not* an ``OHLCVValidationError``;
  - a missing file raises ``FileNotFoundError``;
  - invalid CSV content raises ``OHLCVValidationError``: empty file, wrong columns,
    unparseable or non-finite numbers, malformed or non-minute-aligned timestamps, ambiguous or
    nonexistent DST local times, duplicate or out-of-order timestamps, and OHLC range violations.
    Errors about specific rows name their 0-based data-row positions (the header is not row 0),
    in the validator's style: ``row positions [1, 3]``.
* The CSV header is exactly the six canonical column names. File column order is not
  significant. Missing, extra and wrong-case columns are rejected.
* Every timestamp is a naive local string in one format, ``YYYY-MM-DD HH:MM:SS``, read in
  ``provider_timezone``, and minute-aligned (seconds are ``00``). Other formats (offset or ``Z``
  suffixes, ``T`` separator, missing seconds, fractional seconds, dates only, ``MM/DD/YYYY``,
  empty or mixed) and nonzero seconds are rejected: formats are never inferred or truncated.
* Ambiguous (fall-back) and nonexistent (spring-forward) local times are rejected with a message
  naming the problem and the offending local timestamp. They are never resolved by a guess.
* Result: exactly the canonical columns in canonical order, a fresh ``RangeIndex`` starting at
  0, UTC bar-close timestamps, source row order, nothing dropped, sorted, filled or coerced.
  The result satisfies ``validate_ohlcv``, whose rules are enforced on the data. That is tested
  through observable behavior (every ``validate_ohlcv`` rule family is rejected through the
  loader, and every returned frame is re-validated), not by spying on internal calls.
* The source file is only read, never modified.

The loader lives in ``atlas.data.csv_loader`` and is re-exported from ``atlas.data``.
"""

import csv
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

import atlas.data as atlas_data
from atlas.data import (
    OHLCV_COLUMNS,
    OHLCVValidationError,
    csv_loader,
    load_ohlcv_csv,
    validate_ohlcv,
)

MakeBars = Callable[[Sequence[str]], pd.DataFrame]

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
NEW_YORK = "America/New_York"
UTC_BAR_CLOSE = {"provider_timezone": "UTC", "timestamp_convention": "bar_close"}
NEW_YORK_BAR_CLOSE = {"provider_timezone": NEW_YORK, "timestamp_convention": "bar_close"}
NEW_YORK_BAR_OPEN = {"provider_timezone": NEW_YORK, "timestamp_convention": "bar_open"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def minutes(start: str, count: int = 3) -> list[str]:
    """``count`` consecutive one-minute timestamp strings beginning at ``start``."""
    return list(pd.date_range(start, periods=count, freq="min").strftime(TIMESTAMP_FORMAT))


def csv_frame(make_bars: MakeBars, local_timestamps: Sequence[str]) -> pd.DataFrame:
    """Valid OHLCV values for each row, with ``local_timestamps`` as raw text.

    Row ``i`` carries the same numbers as ``make_bars(...)`` row ``i``, so the expected result
    of loading it is ``make_bars(<expected UTC closes>)``.
    """
    numbers = make_bars(minutes("2024-01-02 14:31", len(local_timestamps)))
    return numbers.assign(timestamp=list(local_timestamps))


def write_frame(directory: Path, frame: pd.DataFrame, name: str = "bars.csv") -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def write_bytes(directory: Path, data: bytes, name: str = "bars.csv") -> Path:
    path = directory / name
    path.write_bytes(data)
    return path


def write_text(directory: Path, text: str, name: str = "bars.csv") -> Path:
    return write_bytes(directory, text.encode("utf-8"), name)


def csv_lines(make_bars: MakeBars, rows: int = 3) -> list[str]:
    """Header plus ``rows`` valid data lines at UTC 14:31, 14:32, ... (six fields each).

    With the default three rows, line 2 (row 1) is
    ``2024-01-02 14:32:00,100.25,100.75,99.75,100.5,10100``.
    """
    return csv_frame(make_bars, minutes("2024-01-02 14:31", rows)).to_csv(index=False).splitlines()


def write_lines(directory: Path, lines: Sequence[str]) -> Path:
    return write_text(directory, "\n".join(lines) + "\n")


def edit_field(line: str, index: int, edit: Callable[[str], str]) -> str:
    """Apply ``edit`` to the ``index``-th comma-separated field of ``line``."""
    fields = line.split(",")
    fields[index] = edit(fields[index])
    return ",".join(fields)


def write_bars(tmp_path: Path, make_bars: MakeBars, local_timestamps: Sequence[str]) -> Path:
    return write_frame(tmp_path, csv_frame(make_bars, local_timestamps))


def replace_rows(rows: Sequence[str], replacements: dict[int, str]) -> list[str]:
    """Copy of ``rows`` with the given 0-based positions replaced."""
    result = list(rows)
    for position, value in replacements.items():
        result[position] = value
    return result


def with_cell(frame: pd.DataFrame, column: str, row: int, value: object) -> pd.DataFrame:
    """Copy of ``frame`` with one cell replaced (column forced to object so text can go in)."""
    values = frame[column].astype(object).tolist()
    values[row] = value
    return frame.assign(**{column: values})


def assert_bars_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Same canonical columns and the same values, ignoring numeric dtype and datetime unit."""
    assert isinstance(actual, pd.DataFrame)
    assert list(actual.columns) == list(OHLCV_COLUMNS)
    validate_ohlcv(actual)
    for column in OHLCV_COLUMNS:
        assert actual[column].tolist() == expected[column].tolist(), column


def snapshot(path: Path) -> tuple[bytes, int, list[str]]:
    """Everything observable about a source file and its directory that a read must not change."""
    return path.read_bytes(), path.stat().st_mtime_ns, sorted(os.listdir(path.parent))


# ---------------------------------------------------------------------------
# Valid input: timestamp convention and timezone conversion
# ---------------------------------------------------------------------------


def test_loader_is_exported_from_the_data_package() -> None:
    assert atlas_data.load_ohlcv_csv is csv_loader.load_ohlcv_csv
    assert "load_ohlcv_csv" in atlas_data.__all__


def test_utc_close_timestamps_are_unchanged(tmp_path: Path, make_bars: MakeBars) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 14:31"))
    result = load_ohlcv_csv(path, **UTC_BAR_CLOSE)
    assert_bars_equal(result, make_bars(minutes("2024-01-02 14:31")))


def test_utc_open_timestamps_become_bar_closes(tmp_path: Path, make_bars: MakeBars) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 14:30"))
    result = load_ohlcv_csv(path, provider_timezone="UTC", timestamp_convention="bar_open")
    assert_bars_equal(result, make_bars(minutes("2024-01-02 14:31")))


@pytest.mark.parametrize(
    ("timezone", "local_start", "utc_start"),
    [
        pytest.param("UTC", "2024-01-02 14:31:00", "2024-01-02 14:31:00", id="utc"),
        pytest.param(NEW_YORK, "2024-01-02 09:31:00", "2024-01-02 14:31:00", id="new_york_winter"),
        pytest.param(NEW_YORK, "2024-07-02 09:31:00", "2024-07-02 13:31:00", id="new_york_summer"),
        pytest.param("Europe/London", "2024-07-02 14:31:00", "2024-07-02 13:31:00", id="london_bst"),
        pytest.param("Asia/Tokyo", "2024-01-03 09:01:00", "2024-01-03 00:01:00", id="tokyo_no_dst"),
    ],
)
def test_provider_timezone_is_converted_to_utc(
    tmp_path: Path, make_bars: MakeBars, timezone: str, local_start: str, utc_start: str
) -> None:
    path = write_bars(tmp_path, make_bars, minutes(local_start))
    result = load_ohlcv_csv(path, provider_timezone=timezone, timestamp_convention="bar_close")
    assert_bars_equal(result, make_bars(minutes(utc_start)))


@pytest.mark.parametrize(
    ("local_start", "utc_start"),
    [
        pytest.param("2024-01-02 09:30:00", "2024-01-02 14:31:00", id="winter"),
        pytest.param("2024-07-02 09:30:00", "2024-07-02 13:31:00", id="summer"),
    ],
)
def test_bar_open_convention_combines_with_a_local_timezone(
    tmp_path: Path, make_bars: MakeBars, local_start: str, utc_start: str
) -> None:
    """The 09:30 regular-session open bar closes at 09:31 local, 14:31Z (EST) or 13:31Z (EDT)."""
    path = write_bars(tmp_path, make_bars, minutes(local_start))
    result = load_ohlcv_csv(path, **NEW_YORK_BAR_OPEN)
    assert_bars_equal(result, make_bars(minutes(utc_start)))


@pytest.mark.parametrize(
    ("local", "utc"),
    [
        pytest.param(  # Fri 16:00 EST, then Mon 09:31 EDT after the 2024-03-10 spring-forward
            ["2024-03-08 15:59:00", "2024-03-08 16:00:00", "2024-03-11 09:31:00", "2024-03-11 09:32:00"],
            ["2024-03-08 20:59:00", "2024-03-08 21:00:00", "2024-03-11 13:31:00", "2024-03-11 13:32:00"],
            id="spring_forward_weekend",
        ),
        pytest.param(  # Fri 16:00 EDT, then Mon 09:31 EST after the 2024-11-03 fall-back
            ["2024-11-01 15:59:00", "2024-11-01 16:00:00", "2024-11-04 09:31:00", "2024-11-04 09:32:00"],
            ["2024-11-01 19:59:00", "2024-11-01 20:00:00", "2024-11-04 14:31:00", "2024-11-04 14:32:00"],
            id="fall_back_weekend",
        ),
    ],
)
def test_dst_offset_is_applied_per_timestamp_within_one_file(
    tmp_path: Path, make_bars: MakeBars, local: list[str], utc: list[str]
) -> None:
    path = write_bars(tmp_path, make_bars, local)
    assert_bars_equal(load_ohlcv_csv(path, **NEW_YORK_BAR_CLOSE), make_bars(utc))


@pytest.mark.parametrize(
    ("local_open", "utc_close"),
    [
        # 01:59 EST + 1 min elapsed = 03:00 EDT (wall clock jumps over the nonexistent 02:xx hour).
        pytest.param("2024-03-10 01:59:00", "2024-03-10 07:00:00", id="before_spring_forward"),
        # 00:59 EDT + 1 min elapsed = 01:00 EDT, a wall-clock time that is itself ambiguous.
        pytest.param("2024-11-03 00:59:00", "2024-11-03 05:00:00", id="before_fall_back"),
    ],
)
def test_open_to_close_shift_is_elapsed_time_not_wall_clock(
    tmp_path: Path, make_bars: MakeBars, local_open: str, utc_close: str
) -> None:
    path = write_bars(tmp_path, make_bars, [local_open])
    result = load_ohlcv_csv(path, **NEW_YORK_BAR_OPEN)
    assert_bars_equal(result, make_bars([utc_close]))


def test_first_unambiguous_times_after_dst_transitions_are_valid(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    path = write_bars(tmp_path, make_bars, ["2024-03-10 03:00:00", "2024-11-03 02:00:00"])
    result = load_ohlcv_csv(path, **NEW_YORK_BAR_CLOSE)
    assert_bars_equal(result, make_bars(["2024-03-10 07:00:00", "2024-11-03 07:00:00"]))


# ---------------------------------------------------------------------------
# Valid input: content, order and columns
# ---------------------------------------------------------------------------


def test_values_and_row_order_match_the_file(tmp_path: Path) -> None:
    """Literal, independent of the helpers. Includes zero volume and a nonconsecutive row."""
    path = write_text(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02 14:31:00,475.25,475.75,475.0,475.5,1200\n"
        "2024-01-02 14:32:00,475.5,476.0,475.25,475.75,0\n"
        "2024-01-02 14:40:00,475.75,476.5,475.5,476.25,3300\n",
    )
    result = load_ohlcv_csv(path, **UTC_BAR_CLOSE)
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 14:31", tz="UTC"),
        pd.Timestamp("2024-01-02 14:32", tz="UTC"),
        pd.Timestamp("2024-01-02 14:40", tz="UTC"),
    ]
    assert result["open"].tolist() == [475.25, 475.5, 475.75]
    assert result["high"].tolist() == [475.75, 476.0, 476.5]
    assert result["low"].tolist() == [475.0, 475.25, 475.5]
    assert result["close"].tolist() == [475.5, 475.75, 476.25]
    assert result["volume"].tolist() == [1200, 0, 3300]


def test_output_has_exactly_the_canonical_columns_in_canonical_order(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    shuffled = ["volume", "close", "timestamp", "low", "open", "high"]
    frame = csv_frame(make_bars, minutes("2024-01-02 14:31"))[shuffled]
    result = load_ohlcv_csv(write_frame(tmp_path, frame), **UTC_BAR_CLOSE)
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert_bars_equal(result, make_bars(minutes("2024-01-02 14:31")))


def test_nonconsecutive_timestamps_are_kept_without_synthesizing_bars(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    local = [  # intraday close, next-day open, then the next Monday: only these rows exist
        "2024-01-02 15:59:00",
        "2024-01-02 16:00:00",
        "2024-01-03 09:31:00",
        "2024-01-05 16:00:00",
        "2024-01-08 09:31:00",
    ]
    utc = [
        "2024-01-02 20:59:00",
        "2024-01-02 21:00:00",
        "2024-01-03 14:31:00",
        "2024-01-05 21:00:00",
        "2024-01-08 14:31:00",
    ]
    result = load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **NEW_YORK_BAR_CLOSE)
    assert len(result) == len(local)
    assert_bars_equal(result, make_bars(utc))


def test_result_has_a_fresh_range_index(tmp_path: Path, make_bars: MakeBars) -> None:
    local = ["2024-01-02 15:59:00", "2024-01-02 16:00:00", "2024-01-03 09:31:00"]
    result = load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **NEW_YORK_BAR_CLOSE)
    assert isinstance(result.index, pd.RangeIndex)
    assert (result.index.start, result.index.stop, result.index.step) == (0, 3, 1)


def test_str_path_is_accepted(tmp_path: Path, make_bars: MakeBars) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 14:31"))
    assert_bars_equal(load_ohlcv_csv(str(path), **UTC_BAR_CLOSE), make_bars(minutes("2024-01-02 14:31")))


def test_loading_does_not_modify_the_source_file(tmp_path: Path, make_bars: MakeBars) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 09:31"))
    before = snapshot(path)
    load_ohlcv_csv(path, **NEW_YORK_BAR_CLOSE)
    assert snapshot(path) == before


# ---------------------------------------------------------------------------
# validate_ohlcv rules enforced through the loader (public behavior only)
#
# Replaces two monkeypatch tests that spied on `validate_ohlcv`. Their coverage now comes from
# (a) assert_bars_equal, which re-validates every returned frame, and (b) the rejection tests
# below, which assert the validator's own rule wording and `row positions`, so the verdict
# can only come from validate_ohlcv's rules. The "called exactly once" requirement is dropped:
# it is an implementation detail, not a correctness property.
# ---------------------------------------------------------------------------


def test_every_returned_frame_satisfies_the_data_contract(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    local = ["2024-03-08 15:59:00", "2024-03-11 09:31:00", "2024-03-11 09:40:00"]
    result = load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **NEW_YORK_BAR_OPEN)
    assert validate_ohlcv(result) is None


def test_violations_name_zero_based_data_row_positions(tmp_path: Path, make_bars: MakeBars) -> None:
    frame = with_cell(csv_frame(make_bars, minutes("2024-01-02 14:31", 4)), "volume", 2, -5)
    with pytest.raises(OHLCVValidationError, match=r"'volume'.*row positions \[2\]"):
        load_ohlcv_csv(write_frame(tmp_path, frame), **UTC_BAR_CLOSE)


# ---------------------------------------------------------------------------
# Invalid input: file and CSV structure
# ---------------------------------------------------------------------------


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ohlcv_csv(tmp_path / "does_not_exist.csv", **UTC_BAR_CLOSE)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="zero_bytes"),
        pytest.param("timestamp,open,high,low,close,volume\n", id="header_only"),
    ],
)
def test_empty_csv_is_rejected(tmp_path: Path, text: str) -> None:
    path = write_text(tmp_path, text)
    with pytest.raises(OHLCVValidationError, match=r"(?i)empty|no (data|rows|columns)"):
        load_ohlcv_csv(path, **UTC_BAR_CLOSE)


@pytest.mark.parametrize(
    ("transform", "match"),
    [
        pytest.param(lambda f: f.drop(columns=["volume"]), r"volume", id="missing_volume"),
        pytest.param(lambda f: f.drop(columns=["timestamp"]), r"timestamp", id="missing_timestamp"),
        pytest.param(lambda f: f.assign(vwap=f["close"]), r"vwap", id="extra_column"),
        pytest.param(
            lambda f: f.rename(columns={"close": "Close"}), r"(?i)close", id="wrong_case_header"
        ),
    ],
)
def test_csv_columns_must_be_exactly_the_canonical_six(
    tmp_path: Path,
    make_bars: MakeBars,
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    match: str,
) -> None:
    frame = transform(csv_frame(make_bars, minutes("2024-01-02 14:31")))
    with pytest.raises(OHLCVValidationError, match=match):
        load_ohlcv_csv(write_frame(tmp_path, frame), **UTC_BAR_CLOSE)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        pytest.param("close", "abc", id="close_text"),
        pytest.param("volume", "", id="volume_empty"),
        pytest.param("open", "nan", id="open_nan"),
        pytest.param("high", "inf", id="high_inf"),
    ],
)
def test_invalid_numeric_fields_are_rejected(
    tmp_path: Path, make_bars: MakeBars, column: str, bad_value: str
) -> None:
    frame = with_cell(csv_frame(make_bars, minutes("2024-01-02 14:31")), column, 1, bad_value)
    with pytest.raises(OHLCVValidationError, match=column):
        load_ohlcv_csv(write_frame(tmp_path, frame), **UTC_BAR_CLOSE)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        # Row 1 of make_bars is: open 100.25, high 100.75, low 99.75, close 100.50.
        pytest.param("close", 100.80, r"high >= close", id="high_below_close"),
        pytest.param("open", 100.80, r"high >= open", id="high_below_open"),
        pytest.param("high", 99.70, r"high >= low", id="high_below_low"),
        pytest.param("open", 99.70, r"low <= open", id="low_above_open"),
        pytest.param("close", 99.70, r"low <= close", id="low_above_close"),
        pytest.param("open", 0.0, r"'open'.*strictly positive", id="zero_price"),
        pytest.param("close", -1.0, r"'close'.*strictly positive", id="negative_price"),
        pytest.param("volume", -5, r"'volume'.*nonnegative", id="negative_volume"),
    ],
)
def test_values_violating_the_ohlcv_contract_are_rejected_by_validation(
    tmp_path: Path, make_bars: MakeBars, column: str, value: float, match: str
) -> None:
    frame = with_cell(csv_frame(make_bars, minutes("2024-01-02 14:31")), column, 1, value)
    with pytest.raises(OHLCVValidationError, match=match):
        load_ohlcv_csv(write_frame(tmp_path, frame), **UTC_BAR_CLOSE)


def test_duplicate_timestamps_are_rejected_not_deduplicated(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    local = ["2024-01-02 14:31:00", "2024-01-02 14:32:00", "2024-01-02 14:32:00"]
    with pytest.raises(OHLCVValidationError, match=r"duplicate"):
        load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **UTC_BAR_CLOSE)


def test_out_of_order_timestamps_are_rejected_not_sorted(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    local = ["2024-01-02 14:31:00", "2024-01-02 14:33:00", "2024-01-02 14:32:00"]
    with pytest.raises(OHLCVValidationError, match=r"strictly increasing"):
        load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **UTC_BAR_CLOSE)


def test_rejected_file_is_not_modified(tmp_path: Path, make_bars: MakeBars) -> None:
    local = ["2024-01-02 14:32:00", "2024-01-02 14:31:00", "2024-01-02 14:31:00"]
    path = write_bars(tmp_path, make_bars, local)
    before = snapshot(path)
    with pytest.raises(OHLCVValidationError):
        load_ohlcv_csv(path, **UTC_BAR_CLOSE)
    assert snapshot(path) == before


# ---------------------------------------------------------------------------
# Invalid input: arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_timezone",
    [
        pytest.param("Mars/Olympus", id="unknown_name"),
        pytest.param("", id="empty"),
        pytest.param("+05:00", id="fixed_offset_is_not_iana"),
    ],
)
def test_invalid_provider_timezone_is_rejected(
    tmp_path: Path, make_bars: MakeBars, bad_timezone: str
) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 14:31"))
    with pytest.raises(ValueError, match=r"(?i)time ?zone") as excinfo:
        load_ohlcv_csv(path, provider_timezone=bad_timezone, timestamp_convention="bar_close")
    assert not isinstance(excinfo.value, OHLCVValidationError)  # an API error, not a data error


@pytest.mark.parametrize(
    "bad_convention",
    [
        pytest.param("middle", id="unknown"),
        pytest.param("", id="empty"),
        pytest.param("BAR_OPEN", id="wrong_case"),
        pytest.param("open", id="retired_name_open"),
        pytest.param("close", id="retired_name_close"),
    ],
)
def test_invalid_timestamp_convention_is_rejected(
    tmp_path: Path, make_bars: MakeBars, bad_convention: str
) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 14:31"))
    with pytest.raises(ValueError, match=r"timestamp_convention") as excinfo:
        load_ohlcv_csv(path, provider_timezone="UTC", timestamp_convention=bad_convention)
    assert not isinstance(excinfo.value, OHLCVValidationError)  # an API error, not a data error


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((), {}, id="neither_given"),
        pytest.param((), {"provider_timezone": "UTC"}, id="convention_missing"),
        pytest.param((), {"timestamp_convention": "bar_close"}, id="timezone_missing"),
        pytest.param(("UTC", "bar_close"), {}, id="positional"),
    ],
)
def test_timezone_and_convention_are_required_keyword_arguments(
    tmp_path: Path, make_bars: MakeBars, args: tuple, kwargs: dict
) -> None:
    path = write_bars(tmp_path, make_bars, minutes("2024-01-02 14:31"))
    with pytest.raises(TypeError):
        load_ohlcv_csv(path, *args, **kwargs)


# ---------------------------------------------------------------------------
# Invalid input: timestamp text and DST
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        pytest.param("not-a-timestamp", id="garbage"),
        pytest.param("2024-13-45 14:32:00", id="impossible_date"),
        pytest.param("2024-01-02", id="date_only"),
        pytest.param("2024-01-02 14:32", id="no_seconds"),
        pytest.param("2024-01-02 14:32:00.500", id="fractional_seconds"),
        pytest.param("01/02/2024 14:32:00", id="us_style"),
        pytest.param("2024-01-02T14:32:00Z", id="iso_with_z_suffix"),
        pytest.param("2024-01-02 14:32:00+00:00", id="explicit_offset"),
        pytest.param("", id="empty_field"),
    ],
)
def test_malformed_timestamp_in_one_row_is_rejected(
    tmp_path: Path, make_bars: MakeBars, bad_timestamp: str
) -> None:
    local = minutes("2024-01-02 14:31")
    local[1] = bad_timestamp
    with pytest.raises(OHLCVValidationError, match=r"timestamp"):
        load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **UTC_BAR_CLOSE)


def test_consistent_but_unsupported_timestamp_format_is_rejected(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    local = [t.replace(" ", "T") for t in minutes("2024-01-02 14:31")]
    with pytest.raises(OHLCVValidationError, match=r"timestamp"):
        load_ohlcv_csv(write_bars(tmp_path, make_bars, local), **UTC_BAR_CLOSE)


@pytest.mark.parametrize("convention", ["bar_open", "bar_close"])
def test_ambiguous_fall_back_local_time_is_rejected(
    tmp_path: Path, make_bars: MakeBars, convention: str
) -> None:
    """01:30 on 2024-11-03 happens twice in New York; the loader must not pick one."""
    local = ["2024-11-01 16:00:00", "2024-11-03 01:30:00", "2024-11-04 09:31:00"]
    path = write_bars(tmp_path, make_bars, local)
    with pytest.raises(OHLCVValidationError, match=r"(?i)ambiguous") as excinfo:
        load_ohlcv_csv(path, provider_timezone=NEW_YORK, timestamp_convention=convention)
    assert "2024-11-03 01:30:00" in str(excinfo.value)


@pytest.mark.parametrize("convention", ["bar_open", "bar_close"])
def test_nonexistent_spring_forward_local_time_is_rejected(
    tmp_path: Path, make_bars: MakeBars, convention: str
) -> None:
    """02:30 on 2024-03-10 never happens in New York; the loader must not shift it."""
    local = ["2024-03-08 16:00:00", "2024-03-10 02:30:00", "2024-03-11 09:31:00"]
    path = write_bars(tmp_path, make_bars, local)
    with pytest.raises(OHLCVValidationError, match=r"(?i)non-?existent") as excinfo:
        load_ohlcv_csv(path, provider_timezone=NEW_YORK, timestamp_convention=convention)
    assert "2024-03-10 02:30:00" in str(excinfo.value)


@pytest.mark.parametrize(
    ("seconds", "convention"),
    [
        pytest.param("01", "bar_close", id="one_second_past"),
        pytest.param("59", "bar_close", id="fifty_nine_seconds"),
        pytest.param("30", "bar_open", id="thirty_seconds_with_bar_open"),
    ],
)
def test_timestamps_with_nonzero_seconds_are_rejected_not_truncated(
    tmp_path: Path, make_bars: MakeBars, seconds: str, convention: str
) -> None:
    local = replace_rows(minutes("2024-01-02 14:31"), {1: f"2024-01-02 14:32:{seconds}"})
    path = write_bars(tmp_path, make_bars, local)
    with pytest.raises(OHLCVValidationError, match=r"(?i)second|minute"):
        load_ohlcv_csv(path, provider_timezone="UTC", timestamp_convention=convention)


FIVE_MINUTES = minutes("2024-01-02 14:31", 5)
ROW_POSITION_CASES = [
    pytest.param(
        "UTC",
        replace_rows(FIVE_MINUTES, {3: "not-a-timestamp"}),
        r"row positions \[3\]",
        id="malformed_one_row",
    ),
    pytest.param(
        "UTC",
        replace_rows(FIVE_MINUTES, {1: "not-a-timestamp", 3: "01/02/2024 14:34:00"}),
        r"row positions \[1, 3\]",
        id="malformed_two_rows",
    ),
    pytest.param(
        "UTC",
        replace_rows(FIVE_MINUTES, {3: "2024-01-02 14:34:30"}),
        r"row positions \[3\]",
        id="nonzero_seconds",
    ),
    pytest.param(
        NEW_YORK,
        ["2024-11-01 15:58:00", "2024-11-01 15:59:00", "2024-11-01 16:00:00"]
        + ["2024-11-03 01:30:00", "2024-11-04 09:31:00"],
        r"row positions \[3\]",
        id="ambiguous_local_time",
    ),
    pytest.param(
        NEW_YORK,
        ["2024-03-08 15:58:00", "2024-03-08 15:59:00", "2024-03-08 16:00:00"]
        + ["2024-03-10 02:30:00", "2024-03-11 09:31:00"],
        r"row positions \[3\]",
        id="nonexistent_local_time",
    ),
]


@pytest.mark.parametrize(("timezone", "local", "match"), ROW_POSITION_CASES)
def test_timestamp_errors_name_the_zero_based_row_positions(
    tmp_path: Path, make_bars: MakeBars, timezone: str, local: list[str], match: str
) -> None:
    """Row 3 of 5 is data row 3, not line 4 (1-based) nor line 5 (counting the header)."""
    path = write_bars(tmp_path, make_bars, local)
    with pytest.raises(OHLCVValidationError, match=match):
        load_ohlcv_csv(path, provider_timezone=timezone, timestamp_convention="bar_close")


# ---------------------------------------------------------------------------
# Regressions from the independent Phase 1B audit
#
# These tests reject malformed *content*, so every one asserts OHLCVValidationError. They
# neutralize pytest's global `filterwarnings = error` (see pyproject.toml): pandas reports some
# malformed structure (e.g. surplus fields) as a ParserWarning, and a warning that pytest merely
# turns into an exception is not evidence that the loader rejects the file. With warnings
# ignored, a loader that silently drops or shifts fields returns a frame and fails the test.
# ---------------------------------------------------------------------------

malformed_content = pytest.mark.filterwarnings("ignore")

FIELD_INDEX = {"timestamp": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}


@malformed_content
@pytest.mark.parametrize(
    ("column", "suffix"),
    [
        # Each suffix follows a *valid* value, so truncating at the NUL silently yields a
        # plausible number or timestamp that was never in the file.
        pytest.param("volume", "\x00999", id="nul_in_volume"),
        pytest.param("close", "\x009", id="nul_in_price"),
        pytest.param("timestamp", "\x00junk", id="nul_in_timestamp"),
    ],
)
def test_nul_byte_in_a_field_is_rejected_not_truncated(
    tmp_path: Path, make_bars: MakeBars, column: str, suffix: str
) -> None:
    lines = csv_lines(make_bars)
    lines[2] = edit_field(lines[2], FIELD_INDEX[column], lambda field: field + suffix)
    assert "\x00" in lines[2]
    with pytest.raises(OHLCVValidationError):
        load_ohlcv_csv(write_lines(tmp_path, lines), **UTC_BAR_CLOSE)


@malformed_content
@pytest.mark.parametrize("convention", ["bar_close", "bar_open"])
def test_sixty_seconds_is_rejected_not_rolled_into_the_next_minute(
    tmp_path: Path, make_bars: MakeBars, convention: str
) -> None:
    """``14:34:60`` must not become ``14:35:00``; the following row leaves room for that roll."""
    local = [
        "2024-01-02 14:31:00",
        "2024-01-02 14:32:00",
        "2024-01-02 14:33:00",
        "2024-01-02 14:34:60",
        "2024-01-02 14:40:00",
    ]
    path = write_bars(tmp_path, make_bars, local)
    with pytest.raises(OHLCVValidationError, match=r"row positions \[3\]"):
        load_ohlcv_csv(path, provider_timezone="UTC", timestamp_convention=convention)


@malformed_content
@pytest.mark.parametrize(
    "bad_timestamp",
    [
        pytest.param("2024-1-02 14:32:00", id="month"),
        pytest.param("2024-01-2 14:32:00", id="day"),
        pytest.param("2024-01-02 4:32:00", id="hour"),
        pytest.param("2024-01-02 14:3:00", id="minute"),
        pytest.param("2024-01-02 14:32:0", id="second"),
    ],
)
def test_one_digit_timestamp_components_are_rejected(
    tmp_path: Path, make_bars: MakeBars, bad_timestamp: str
) -> None:
    lines = csv_lines(make_bars)
    lines[2] = edit_field(lines[2], 0, lambda _: bad_timestamp)
    with pytest.raises(OHLCVValidationError, match=r"row positions \[1\]"):
        load_ohlcv_csv(write_lines(tmp_path, lines), **UTC_BAR_CLOSE)


def rows_only(edit: Callable[[str], str]) -> Callable[[list[str]], list[str]]:
    """Case builder: apply ``edit`` to every data line and leave the header alone."""
    return lambda lines: [lines[0], *(edit(line) for line in lines[1:])]


# Each builder maps [header, row0, row1, row2] to the lines actually written.
STRUCTURE_CASES: dict[str, Callable[[list[str]], list[str]]] = {
    "extra_field_in_every_row": rows_only(lambda line: line + ",7"),
    # Seven fields per row under a six-name header: pandas would silently make the leading
    # field the index and shift every column, unless the loader refuses to infer an index.
    "extra_leading_field_in_every_row": rows_only(lambda line: "X," + line),
    "extra_field_in_first_row": lambda L: [L[0], L[1] + ",7", L[2], L[3]],
    "extra_field_in_middle_row": lambda L: [L[0], L[1], L[2] + ",7", L[3]],
    "extra_field_in_last_row": lambda L: [L[0], L[1], L[2], L[3] + ",7"],
    "trailing_empty_field_in_every_row_only": rows_only(lambda line: line + ","),
    "trailing_empty_field_in_header_and_rows": lambda L: [line + "," for line in L],
    "unbalanced_quote_in_middle_row": lambda L: [
        L[0], L[1], edit_field(L[2], 0, lambda f: '"' + f), L[3],
    ],
    "unbalanced_quote_in_last_row": lambda L: [
        L[0], L[1], L[2], edit_field(L[3], 0, lambda f: '"' + f),
    ],
    "short_row": lambda L: [L[0], L[1], L[2].rsplit(",", 1)[0], L[3]],
    "every_row_short": rows_only(lambda line: line.rsplit(",", 1)[0]),
    "duplicate_header_label_with_extra_column": lambda L: (
        [L[0] + ",volume"] + [line + ",1" for line in L[1:]]
    ),
    "duplicate_header_label_replacing_a_column": lambda L: [
        L[0].replace("volume", "close"), *L[1:],
    ],
}


@malformed_content
@pytest.mark.parametrize("build", STRUCTURE_CASES.values(), ids=STRUCTURE_CASES.keys())
def test_malformed_csv_structure_is_rejected(
    tmp_path: Path, make_bars: MakeBars, build: Callable[[list[str]], list[str]]
) -> None:
    path = write_lines(tmp_path, build(csv_lines(make_bars)))
    with pytest.raises(OHLCVValidationError):
        load_ohlcv_csv(path, **UTC_BAR_CLOSE)


@malformed_content
def test_entirely_missing_timestamp_values_are_rejected_naming_every_row(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    lines = [csv_lines(make_bars)[0]] + [
        edit_field(line, 0, lambda _: "") for line in csv_lines(make_bars)[1:]
    ]
    assert lines[1].startswith(",")
    with pytest.raises(OHLCVValidationError, match=r"row positions \[0, 1, 2\]"):
        load_ohlcv_csv(write_lines(tmp_path, lines), **UTC_BAR_CLOSE)


@malformed_content
@pytest.mark.parametrize(
    ("original", "corrupted"),
    [
        pytest.param(b"14:32:00", b"14:32:\xff0", id="invalid_utf8_in_timestamp"),
        pytest.param(b"100.75", b"100.\xe975", id="latin1_byte_in_price"),
        pytest.param(b"volume", b"vol\xffume", id="invalid_utf8_in_header"),
    ],
)
def test_invalid_utf8_is_rejected(
    tmp_path: Path, make_bars: MakeBars, original: bytes, corrupted: bytes
) -> None:
    good = ("\n".join(csv_lines(make_bars)) + "\n").encode("utf-8")
    assert original in good
    path = write_bytes(tmp_path, good.replace(original, corrupted))
    with pytest.raises(OHLCVValidationError):
        load_ohlcv_csv(path, **UTC_BAR_CLOSE)


@malformed_content
@pytest.mark.parametrize("column", ["timestamp", "volume"])
def test_field_longer_than_the_csv_module_limit_is_rejected_as_content(
    tmp_path: Path, make_bars: MakeBars, column: str
) -> None:
    """A huge field is malformed content, not a stdlib ``csv.Error`` escaping to the caller."""
    lines = csv_lines(make_bars)
    huge = "9" * (csv.field_size_limit() + 1)  # reads the limit; does not change it
    lines[2] = edit_field(lines[2], FIELD_INDEX[column], lambda _: huge)
    with pytest.raises(OHLCVValidationError):
        load_ohlcv_csv(write_lines(tmp_path, lines), **UTC_BAR_CLOSE)


@malformed_content
@pytest.mark.parametrize(
    ("separator", "expected_positions"),
    [
        # pandas skips empty and space/tab-only lines, so they are not data rows.
        pytest.param("", "[0, 1, 2]", id="blank_line"),
        pytest.param("   ", "[0, 1, 2]", id="spaces_only_line"),
        pytest.param("\t", "[0, 1, 2]", id="tab_only_line"),
        pytest.param(" \t ", "[0, 1, 2]", id="space_and_tab_line"),
        # A quoted blank field is a (short) data record to pandas, so it keeps its position.
        pytest.param('"  "', "[0, 2, 3]", id="quoted_spaces_line_is_a_record"),
    ],
)
def test_surplus_field_positions_count_parsed_records_not_skipped_lines(
    tmp_path: Path, make_bars: MakeBars, separator: str, expected_positions: str
) -> None:
    """Every row has a trailing empty field (caught only by the field-count scan)."""
    header, *rows = csv_lines(make_bars)
    lines = [header, rows[0] + ",", separator, rows[1] + ",", rows[2] + ","]
    with pytest.raises(OHLCVValidationError, match=rf"row positions {re.escape(expected_positions)}$"):
        load_ohlcv_csv(write_lines(tmp_path, lines), **UTC_BAR_CLOSE)


@malformed_content
def test_whitespace_only_line_inside_a_quoted_field_is_data_not_a_skipped_record(
    tmp_path: Path, make_bars: MakeBars
) -> None:
    """Record 0 spans three physical lines, the middle one blank-looking but inside quotes.

    Every record has a trailing empty field (so the loader's own field-count scan reports them,
    as in the sibling test), hence all three records are surplus: positions [0, 1, 2]. Counting
    the quoted blank line as a skipped line would shift records 1 and 2 down to [0, 0, 1].
    """
    header, *rows = csv_lines(make_bars)
    multiline = edit_field(rows[0], 0, lambda stamp: f'"{stamp}\n   \n"')
    lines = [header, multiline + ",", rows[1] + ",", rows[2] + ","]
    assert lines[1].count("\n") == 2
    with pytest.raises(OHLCVValidationError, match=r"row positions \[0, 1, 2\]$"):
        load_ohlcv_csv(write_lines(tmp_path, lines), **UTC_BAR_CLOSE)

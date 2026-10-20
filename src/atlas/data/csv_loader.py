"""Local CSV loader for historical one-minute OHLCV bars (Phase 1B).

The CSV carries naive local wall-clock timestamps in one fixed format. The caller states the IANA
zone they are read in and whether each stamp marks the bar open or close; the loader converts them
to the UTC bar-close timestamps required by docs/data_contract.md and hands the result to
``validate_ohlcv``. Nothing is sorted, deduplicated, filled, coerced or guessed: bad input is
rejected with the offending 0-based data-row positions.
"""

import csv
import io
import warnings
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from atlas.data.contracts import OHLCV_COLUMNS, TIMESTAMP_COLUMN, OHLCVValidationError
from atlas.data.validation import _fail_if, validate_ohlcv

_BAR_DURATION = pd.Timedelta(minutes=1)
_CONVENTIONS = ("bar_open", "bar_close")
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
# strptime also accepts one-digit fields such as "2024-1-2", so the shape is checked separately.
_TIMESTAMP_SHAPE = r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"


def load_ohlcv_csv(
    path: str | Path,
    *,
    provider_timezone: str,
    timestamp_convention: Literal["bar_open", "bar_close"],
) -> pd.DataFrame:
    """Load ``path`` into a canonical, validated OHLCV frame with UTC bar-close timestamps.

    ``provider_timezone`` is the IANA zone of the naive ``YYYY-MM-DD HH:MM:SS`` timestamps.
    ``timestamp_convention`` is ``"bar_close"`` (stamps are bar closes) or ``"bar_open"`` (stamps
    are bar opens; one minute of elapsed time is added after conversion to UTC).

    Raises ``ValueError`` for a bad argument, ``FileNotFoundError`` for a missing file and
    ``OHLCVValidationError`` for any problem with the file's content.
    """
    zone = _parse_timezone(provider_timezone)
    if timestamp_convention not in _CONVENTIONS:
        raise ValueError(
            f"timestamp_convention must be one of {list(_CONVENTIONS)}, got {timestamp_convention!r}"
        )
    raw = _read_csv(path)
    if set(raw.columns) != set(OHLCV_COLUMNS):
        validate_ohlcv(raw)  # always raises, naming the missing and unexpected columns
    timestamps = _to_utc(raw[TIMESTAMP_COLUMN], zone)
    if timestamp_convention == "bar_open":
        timestamps = timestamps + _BAR_DURATION  # elapsed time, applied after the UTC conversion
    frame = raw.assign(**{TIMESTAMP_COLUMN: timestamps})[list(OHLCV_COLUMNS)]
    validate_ohlcv(frame)
    return frame


def _parse_timezone(name: str) -> ZoneInfo:
    if not isinstance(name, str):
        raise ValueError(f"provider_timezone must be an IANA timezone name, got {name!r}")  # noqa: TRY004
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as error:
        raise ValueError(f"provider_timezone is not a known IANA timezone: {name!r}") from error


def _read_csv(path: str | Path) -> pd.DataFrame:
    # Opening the file ourselves keeps pandas from treating ``path`` as a URL.
    with open(path, "rb") as handle:
        content = handle.read()
    # pandas silently truncates a field at a NUL byte, so such files never reach it.
    if b"\x00" in content:
        raise OHLCVValidationError(f"CSV file contains NUL bytes: {path}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OHLCVValidationError(f"CSV file is malformed: {error}") from error
    with warnings.catch_warnings():
        # A ParserWarning means pandas is about to drop or shift fields; reject instead.
        warnings.simplefilter("error", pd.errors.ParserWarning)
        try:
            # index_col=False stops pandas turning a surplus leading field into the index.
            frame = pd.read_csv(io.StringIO(text), index_col=False, dtype={TIMESTAMP_COLUMN: str})
        except pd.errors.EmptyDataError as error:
            raise OHLCVValidationError(f"CSV file is empty (no columns to parse): {path}") from error
        except (pd.errors.ParserError, pd.errors.ParserWarning) as error:
            raise OHLCVValidationError(f"CSV file is malformed: {error}") from error
    _reject_surplus_fields(text)
    return frame


def _reject_surplus_fields(text: str) -> None:
    """Reject data rows with more fields than the header.

    pandas ignores a trailing empty field (``...,10,``) when ``index_col=False``, so a row with a
    surplus field can pass unnoticed. Row positions are 0-based data rows, as in the validator.
    """
    lines = io.StringIO(text, newline="").readlines()
    reader = csv.reader(lines)
    records: list[list[str]] = []
    start = 0
    try:
        for fields in reader:
            # reader.line_num counts the physical lines consumed, so this record's raw text is the
            # slice between the previous record's end and this one (a quoted record spans several).
            end = reader.line_num
            raw = "".join(lines[start:end])
            start = end
            # pandas skips empty and space/tab-only physical lines, but not other whitespace such
            # as form feed or NBSP. Quoted content is never skipped, even when it is only
            # whitespace: the quote characters survive the strip, so the quote test is redundant.
            if raw.strip(" \t\r\n") or '"' in raw:
                records.append(fields)
    except csv.Error as error:
        raise OHLCVValidationError(f"CSV file is malformed: {error}") from error
    header, rows = records[0], records[1:]  # pandas has already rejected an empty file
    _fail_if(
        np.array([len(row) > len(header) for row in rows], dtype=bool),
        f"CSV rows must not have more than the {len(header)} fields of the header",
    )


def _to_utc(local_text: pd.Series, zone: ZoneInfo) -> pd.Series:
    """Parse naive local timestamp text in ``zone`` and convert it to UTC.

    The rules are checked as ordered groups: format or missing value, whole-minute alignment,
    ambiguous local time, nonexistent local time. The first failing group is reported with its
    offending row positions (the first 10, plus the total count when there are more).
    """
    name = TIMESTAMP_COLUMN
    well_formed = local_text.str.fullmatch(_TIMESTAMP_SHAPE).fillna(False).to_numpy(dtype=bool)
    # errors="coerce" only locates unparseable rows (impossible dates); they are rejected below.
    parsed = pd.to_datetime(local_text, format=_TIMESTAMP_FORMAT, errors="coerce")
    _fail_if(
        ~well_formed | parsed.isna().to_numpy(),
        f"'{name}' must be a naive local time formatted YYYY-MM-DD HH:MM:SS "
        "(no offset, 'T' separator or fractional seconds)",
    )
    # Read from the text: pandas rolls "14:34:60" into 14:35:00, so parsed seconds are never 60.
    _fail_if(
        (local_text.str[-2:] != "00").to_numpy(),
        f"'{name}' must be aligned to a whole minute (seconds must be 00)",
    )
    shifted = parsed.dt.tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward")
    _fail_if(
        shifted.isna().to_numpy(),
        f"'{name}' contains ambiguous local times in {zone.key} (repeated by a DST fall-back): "
        f"{_quoted(local_text, shifted.isna())}",
    )
    localized = parsed.dt.tz_localize(zone, ambiguous="raise", nonexistent="NaT")
    _fail_if(
        localized.isna().to_numpy(),
        f"'{name}' contains nonexistent local times in {zone.key} (skipped by a DST spring-forward): "
        f"{_quoted(local_text, localized.isna())}",
    )
    return localized.dt.tz_convert("UTC")


def _quoted(local_text: pd.Series, bad: pd.Series, limit: int = 10) -> str:
    return str(local_text[bad].head(limit).tolist())

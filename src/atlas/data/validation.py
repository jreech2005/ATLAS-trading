"""Structural validation of OHLCV frames against the ATLAS data contract.

See docs/data_contract.md. Validation only reads the frame: it never sorts,
fills, drops, coerces or otherwise repairs data.
"""

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from atlas.data.contracts import (
    OHLCV_COLUMNS,
    PRICE_COLUMNS,
    TIMESTAMP_COLUMN,
    VOLUME_COLUMN,
    OHLCVValidationError,
)

_UTC_NAMES = frozenset({"UTC", "Etc/UTC"})
_MAX_REPORTED_ROWS = 10


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Validate ``frame`` against the OHLCV contract.

    Returns ``None`` if the frame is valid. Raises ``OHLCVValidationError`` at the
    first violated rule group (or ``TypeError`` if ``frame`` is not a DataFrame).
    The input is never modified.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    _check_columns(frame)
    if len(frame) == 0:
        raise OHLCVValidationError("frame is empty; at least one bar is required")
    _check_timestamps(frame[TIMESTAMP_COLUMN])
    prices = {column: _finite_numeric(frame, column) for column in PRICE_COLUMNS}
    volume = _finite_numeric(frame, VOLUME_COLUMN)
    for column, values in prices.items():
        _fail_if(values <= 0, f"'{column}' must be strictly positive")
    _fail_if(volume < 0, f"'{VOLUME_COLUMN}' must be nonnegative")
    _check_price_relations(prices)


def _check_columns(frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    duplicated = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
    if duplicated:
        raise OHLCVValidationError(f"duplicate column labels: {duplicated}")
    missing = [c for c in OHLCV_COLUMNS if c not in columns]
    unexpected = [str(c) for c in columns if c not in OHLCV_COLUMNS]
    if missing or unexpected:
        raise OHLCVValidationError(
            f"columns must be exactly {list(OHLCV_COLUMNS)}; "
            f"missing: {missing}, unexpected: {unexpected}"
        )


def _check_timestamps(timestamps: pd.Series) -> None:
    dtype = timestamps.dtype
    if ptypes.is_datetime64_dtype(dtype):
        raise OHLCVValidationError(
            f"'{TIMESTAMP_COLUMN}' is timezone-naive ({dtype}); timestamps must be "
            "timezone-aware UTC and are never assumed to be UTC"
        )
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise OHLCVValidationError(
            f"'{TIMESTAMP_COLUMN}' must have a timezone-aware datetime dtype, got {dtype}"
        )
    if str(dtype.tz) not in _UTC_NAMES:
        raise OHLCVValidationError(
            f"'{TIMESTAMP_COLUMN}' must be in UTC, got timezone {dtype.tz}; "
            "convert explicitly with .dt.tz_convert('UTC')"
        )
    name = TIMESTAMP_COLUMN
    _fail_if(timestamps.isna().to_numpy(), f"'{name}' contains missing values (NaT)")
    _fail_if(timestamps.duplicated(keep=False).to_numpy(), f"'{name}' contains duplicate timestamps")
    # Row i is flagged when it is not strictly later than row i - 1. Gaps of any size are fine.
    not_increasing = np.zeros(len(timestamps), dtype=bool)
    not_increasing[1:] = (timestamps.diff().iloc[1:] <= pd.Timedelta(0)).to_numpy()
    _fail_if(not_increasing, f"'{name}' must be strictly increasing in row order")


def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return ``column`` as a float64 array after checking its dtype and finiteness."""
    series = frame[column]
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype) or not (
        ptypes.is_integer_dtype(dtype) or ptypes.is_float_dtype(dtype)
    ):
        raise OHLCVValidationError(f"'{column}' must have an integer or float dtype, got {dtype}")
    _fail_if(series.isna().to_numpy(), f"'{column}' contains missing values (NaN/NA)")
    values = series.to_numpy(dtype=np.float64)
    _fail_if(~np.isfinite(values), f"'{column}' contains non-finite values (inf)")
    return values


def _check_price_relations(prices: dict[str, np.ndarray]) -> None:
    o, h, l, c = (prices[k] for k in PRICE_COLUMNS)
    rules = {
        "high >= low": h >= l,
        "high >= open": h >= o,
        "high >= close": h >= c,
        "low <= open": l <= o,
        "low <= close": l <= c,
    }
    failures = [f"{rule} fails at {_positions(~ok)}" for rule, ok in rules.items() if not ok.all()]
    if failures:
        raise OHLCVValidationError("OHLC price relationship violated: " + "; ".join(failures))


def _fail_if(bad: np.ndarray, message: str) -> None:
    if bad.any():
        raise OHLCVValidationError(f"{message} at {_positions(bad)}")


def _positions(bad: np.ndarray) -> str:
    rows = np.flatnonzero(bad)
    shown = ", ".join(str(r) for r in rows[:_MAX_REPORTED_ROWS])
    more = f", ... ({len(rows)} total)" if len(rows) > _MAX_REPORTED_ROWS else ""
    return f"row positions [{shown}{more}]"

"""Causal one-minute OHLCV features (Phase 1C).

See docs/architecture.md (Features). Each feature at a bar uses only that bar and earlier bars of
the same uninterrupted run: consecutive one-minute closes on one America/New_York date. A feature
whose window is incomplete, broken or numerically undefined is NaN, never imputed.
"""

import numpy as np
import pandas as pd

from atlas.data.contracts import TIMESTAMP_COLUMN
from atlas.data.validation import validate_ohlcv

_ONE_MINUTE = pd.Timedelta(minutes=1)
_SESSION_TIMEZONE = "America/New_York"
_WINDOW = 15


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute causal features for each bar of a contract-valid OHLCV frame.

    Returns a new frame with the input's index and row order: ``timestamp`` (the input's UTC bar
    closes), ``return_1m``, ``return_5m``, ``return_15m``, ``realized_volatility_15m``,
    ``relative_volume_15m`` (float64, NaN when undefined) and ``feature_ready`` (bool, True only
    when all five are finite). Raises as ``validate_ohlcv`` does for an invalid frame. The input
    is never modified.
    """
    validate_ohlcv(bars)
    close = bars["close"].to_numpy(dtype=np.float64)
    volume = bars["volume"].to_numpy(dtype=np.float64)
    history = _continuous_history(pd.DatetimeIndex(bars[TIMESTAMP_COLUMN]))
    # Undefined ratios (x / 0) and overflow become NaN below, so their warnings are expected.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_return = np.log(close / _lag(close, 1))
        computed = {
            "return_1m": (close / _lag(close, 1) - 1.0, 1),
            "return_5m": (close / _lag(close, 5) - 1.0, 5),
            "return_15m": (close / _lag(close, _WINDOW) - 1.0, _WINDOW),
            "realized_volatility_15m": (
                np.sqrt(sum(_lag(log_return, k) ** 2 for k in range(_WINDOW))),
                _WINDOW,
            ),
            "relative_volume_15m": (
                volume / (sum(_lag(volume, k) for k in range(1, _WINDOW + 1)) / _WINDOW),
                _WINDOW,
            ),
        }
    features = {
        name: np.where((history >= needed) & np.isfinite(values), values, np.nan)
        for name, (values, needed) in computed.items()
    }
    ready = np.logical_and.reduce([np.isfinite(values) for values in features.values()])
    return pd.DataFrame(
        {TIMESTAMP_COLUMN: bars[TIMESTAMP_COLUMN].array.copy(), **features, "feature_ready": ready},
        index=bars.index.copy(),
    )


def _continuous_history(closes: pd.DatetimeIndex) -> np.ndarray:
    """Number of consecutive one-minute intervals ending at each bar within one session.

    A bar continues its predecessor only if it closes exactly one minute later and covers a minute
    on the same America/New_York date. The bar closing at 00:00 local covers 23:59-24:00, so it
    belongs to the earlier date. This is not exchange-calendar validation.
    """
    session = (closes - _ONE_MINUTE).tz_convert(_SESSION_TIMEZONE).date
    continues = np.zeros(len(closes), dtype=bool)
    continues[1:] = (closes[1:] - closes[:-1] == _ONE_MINUTE) & (session[1:] == session[:-1])
    starts = np.flatnonzero(~continues)  # row 0 always starts a run
    run = np.cumsum(~continues) - 1
    return np.arange(len(closes)) - starts[run]


def _lag(values: np.ndarray, k: int) -> np.ndarray:
    """``values`` shifted ``k`` rows later (row i holds row i - k); the first ``k`` rows are NaN."""
    lagged = np.full(len(values), np.nan)
    lagged[k:] = values[: max(len(values) - k, 0)]  # empty when k exceeds the frame
    return lagged

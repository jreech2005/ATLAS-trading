"""Five-minute prediction labels (Phase 1D).

See docs/architecture.md (Labels). The label at a bar is built from that bar and the bar exactly
five minutes later in the same uninterrupted run, with the continuity rule used by the Phase 1C
features. Labels read future bars by design, so they are training targets only and must never
reach features or decisions.
"""

import numpy as np
import pandas as pd

from atlas.data.contracts import TIMESTAMP_COLUMN
from atlas.data.validation import validate_ohlcv
from atlas.features import _continuous_history

_HORIZON = 5


def compute_labels(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute five-minute forward labels for each bar of a contract-valid OHLCV frame.

    Returns a new frame with the input's index and row order: ``timestamp`` (the input's UTC bar
    closes), ``future_return_5m`` (float64, ``c[t+5] / c[t] - 1``), ``direction_5m`` (Int64, 1 if
    that return is positive, else 0), ``label_end_timestamp`` (the close of the bar the label is
    measured at) and ``label_ready`` (bool). A label is defined only when the bar and the five
    bars after it are one run and the return is finite; otherwise its values are NaN, ``<NA>``
    and NaT and ``label_ready`` is False. Raises as ``validate_ohlcv`` does for an invalid frame.
    The input is never modified.
    """
    validate_ohlcv(bars)
    close = bars["close"].to_numpy(dtype=np.float64)
    timestamps = bars[TIMESTAMP_COLUMN].array
    positions = np.arange(len(bars))
    target = positions + _HORIZON
    # Bar t + 5 has five intervals of history exactly when bars t, ..., t + 5 form one run.
    history = _continuous_history(pd.DatetimeIndex(timestamps))
    spans_run = np.zeros(len(bars), dtype=bool)
    spans_run[: max(len(bars) - _HORIZON, 0)] = history[_HORIZON:] >= _HORIZON
    # A ratio of valid prices can overflow to inf; such labels are undefined, so the warning is
    # expected. An underflow to 0.0 is the correctly rounded ratio and stays a valid -1.0 return.
    with np.errstate(over="ignore"):
        future_return = _lead(close, _HORIZON) / close - 1.0
    ready = spans_run & np.isfinite(future_return)
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps.copy(),
            "future_return_5m": np.where(ready, future_return, np.nan),
            "direction_5m": pd.arrays.IntegerArray(
                (future_return > 0).astype(np.int64), mask=~ready
            ),
            "label_end_timestamp": timestamps.take(np.where(ready, target, -1), allow_fill=True),
            "label_ready": ready,
        },
        index=bars.index.copy(),
    )


def _lead(values: np.ndarray, k: int) -> np.ndarray:
    """``values`` shifted ``k`` rows earlier (row i holds row i + k); the last ``k`` rows are NaN."""
    led = np.full(len(values), np.nan)
    led[: max(len(values) - k, 0)] = values[k:]  # empty when k exceeds the frame
    return led

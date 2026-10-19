"""Shared fixtures: tiny, deterministic, synthetic OHLCV frames."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

# 2024-01-02 is a Tuesday; 14:31Z is the close of the first regular-session bar (EST).
BASE_TIMESTAMPS = ("2024-01-02 14:31", "2024-01-02 14:32", "2024-01-02 14:33")


def build_bars(timestamps: Sequence[str]) -> pd.DataFrame:
    """Build a valid canonical OHLCV frame with UTC bar-close ``timestamps``."""
    n = len(timestamps)
    base = 100.0 + 0.25 * np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps).tz_localize("UTC"),
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.25,
            "volume": 10_000 + 100 * np.arange(n, dtype=np.int64),
        }
    )


@pytest.fixture
def make_bars() -> Callable[[Sequence[str]], pd.DataFrame]:
    """Factory for valid frames at arbitrary UTC timestamps."""
    return build_bars


@pytest.fixture
def valid_bars() -> pd.DataFrame:
    """Three consecutive valid one-minute bars.

    Row 1 is: open 100.25, high 100.75, low 99.75, close 100.50.
    """
    return build_bars(BASE_TIMESTAMPS)

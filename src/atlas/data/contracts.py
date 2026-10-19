"""Canonical OHLCV schema for ATLAS V1 market data.

The full contract, including what validation does *not* prove, is in
docs/data_contract.md.
"""

TIMESTAMP_COLUMN = "timestamp"
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMN = "volume"
OHLCV_COLUMNS: tuple[str, ...] = (TIMESTAMP_COLUMN, *PRICE_COLUMNS, VOLUME_COLUMN)


class OHLCVValidationError(ValueError):
    """Raised when a frame violates the OHLCV data contract."""

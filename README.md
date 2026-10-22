# ATLAS

ML-first quantitative trading research and simulation platform. **Research and
simulation only; there is no live trading.**

**Status: Phase 1D.** Implemented: the repository foundation, structural OHLCV
validation (Phase 1A), a loader for local one-minute OHLCV CSV files
(Phase 1B), causal features (Phase 1C) and five-minute prediction labels
(Phase 1D). There are no data downloads, provenance records, train/test
splits, models, backtests or performance results yet.

## V1 research question (for later phases)

> Can information available at a prediction timestamp predict SPY price direction
> five minutes ahead, and can those forecasts support a trading strategy with
> positive out-of-sample returns after realistic costs?

## Setup (macOS / Linux)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest            # full suite
python -m pytest -v tests/data/test_ohlcv_validation.py
python -m pytest -v tests/data/test_csv_loader.py
python -m pytest -v tests/features/test_causal_features.py
python -m pytest -v tests/labels/test_five_minute_labels.py
```

When testing from a Git worktree that shares another checkout's virtual
environment, put the worktree's source first so its code is the code under test:
`PYTHONPATH="$PWD/src" python -m pytest`.

## Usage

Load a local CSV of one-minute bars. Both keyword arguments are required: the
IANA timezone of the file's naive `YYYY-MM-DD HH:MM:SS` timestamps, and whether
each timestamp marks the bar's open or close.

```csv
timestamp,open,high,low,close,volume
2024-01-02 09:30:00,470.10,470.50,469.90,470.20,1200
2024-01-02 09:31:00,470.20,470.60,470.00,470.40,900
```

```python
from atlas.data import load_ohlcv_csv, OHLCVValidationError

bars = load_ohlcv_csv(
    "spy_1min.csv",                       # local file only; nothing is downloaded
    provider_timezone="America/New_York",
    timestamp_convention="bar_open",
)
# bars["timestamp"] is UTC bar close: 2024-01-02 14:31:00+00:00, 14:32:00+00:00
```

The result is a validated frame in the canonical schema. Bad content raises
`OHLCVValidationError` naming the rule and the offending row positions; nothing is
repaired, sorted or dropped. The loader **assumes** the file holds one-minute
bars and cannot verify it. See `docs/data_contract.md` for the full rules and
known limitations.

To check a frame you built yourself:

```python
from atlas.data import validate_ohlcv

validate_ohlcv(frame)   # returns None if valid; raises OHLCVValidationError otherwise
```

Compute causal features from a validated frame:

```python
from atlas.features import compute_features

features = compute_features(bars)  # same index and row order as bars
usable = features[features["feature_ready"]]
```

Features are NaN and `feature_ready` is `False` wherever a window is incomplete:
the first 15 minutes of each run of consecutive bars, after any missing minute,
and after each America/New_York date change. See `docs/architecture.md` for the
exact definitions and limitations.

Compute five-minute training labels from the same frame:

```python
from atlas.labels import compute_labels

labels = compute_labels(bars)  # same index and row order as bars
# future_return_5m = close[t+5] / close[t] - 1; direction_5m = 1 if > 0 else 0
trainable = features["feature_ready"] & labels["label_ready"]
```

A label is defined only when the bar and the five bars after it are consecutive
one-minute bars on one America/New_York date. Otherwise the return is NaN, the
direction is `<NA>`, `label_end_timestamp` is NaT and `label_ready` is `False`.
Labels use future prices: they are training targets only and must never be used
as features or at decision time. See `docs/architecture.md` for the full
contract.

Validation is structural only. Data provenance and licensing are not recorded
or checked yet; record them separately for every dataset (see
`docs/data_contract.md`).

## Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | Engineering rules (TDD, causality, data provenance, no live trading) |
| `CLAUDE.md` | Claude Code operating instructions |
| `docs/architecture.md` | V1 pipeline, what is implemented, feature and label definitions |
| `docs/data_contract.md` | OHLCV schema, timestamp and validity rules, CSV loading rules and limitations |
| `docs/testing.md` | Test commands, TDD workflow, test categories |
| `docs/decisions/` | Architecture decision records |

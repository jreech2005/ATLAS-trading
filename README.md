# ATLAS

ML-first quantitative trading research and simulation platform. **Research and
simulation only; there is no live trading.**

**Status: Phase 1A.** Only the repository foundation and structural OHLCV
validation are implemented. There are no models, backtests, data ingestion or
performance results yet.

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
```

## Usage

```python
from atlas.data import validate_ohlcv, OHLCVValidationError

validate_ohlcv(frame)   # returns None if valid; raises OHLCVValidationError otherwise
```

## Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | Engineering rules (TDD, causality, data provenance, no live trading) |
| `CLAUDE.md` | Claude Code operating instructions |
| `docs/architecture.md` | V1 pipeline and what is implemented |
| `docs/data_contract.md` | OHLCV schema, timestamp and validity rules |
| `docs/testing.md` | Test commands, TDD workflow, test categories |
| `docs/decisions/` | Architecture decision records |

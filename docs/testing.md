# Testing

## Commands

```bash
python -m pytest                                   # full suite
python -m pytest -v tests/data/test_ohlcv_validation.py
python -m pytest -k timestamp                      # subset by name
```

The pytest configuration in `pyproject.toml` turns **warnings into errors**, so
pandas deprecations surface immediately. Unknown markers and config keys also
fail.

## TDD workflow

1. **RED:** write the test, run it, and confirm it fails because the behavior is
   missing. An `ImportError` or a missing dependency does not count as RED.
   A stub raising `NotImplementedError` is acceptable.
2. **GREEN:** write the minimum implementation, then rerun.
3. **REFACTOR:** clean up with the tests green, then rerun.
4. **REVIEW:** check edge cases. Add a regression test for each real defect.
   Periodically break a rule on purpose (mutation check) to confirm a test
   catches it, then restore the code.

## Test categories

| Category | What it proves | Status |
|----------|----------------|--------|
| **Software correctness** (unit and behavioral) | Code does what the contract says on small, synthetic, deterministic inputs | Active |
| **Leakage / causality** | Features and labels never use data after `t` | Future (features phase) |
| **Financial performance evaluation** | Out-of-sample predictive and trading performance after costs | Future. These are *experiments*, not pass/fail tests |

A passing test suite means the software is correct against its specification.
It says **nothing** about predictive power or profitability. Performance claims
need recorded, reproducible walk-forward experiments.

## Rules

- Unit tests use no network access and no real market data.
- Test observable behavior through public APIs such as `atlas.data`, not
  internals.
- Never weaken a test to make code pass (see `AGENTS.md`).

# Testing

## Commands

```bash
python -m pytest                                   # full suite
python -m pytest -v tests/data/test_ohlcv_validation.py
python -m pytest -v tests/features/test_causal_features.py
python -m pytest -v tests/labels/test_five_minute_labels.py
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
| **Leakage / causality** | Features never use data after `t`; labels use exactly their own future window and stay out of features | Active (features Phase 1C, labels Phase 1D) |
| **Financial performance evaluation** | Out-of-sample predictive and trading performance after costs | Future. These are *experiments*, not pass/fail tests |

A passing test suite means the software is correct against its specification.
It says **nothing** about predictive power or profitability. Performance claims
need recorded, reproducible walk-forward experiments.

## Leakage tests for features

`tests/features/test_causal_features.py` checks causality directly rather than
by inspection:

- **Future perturbation:** for every row position `t` of a multi-session frame,
  prices and volumes after `t` are changed at random. Features at rows `<= t`
  must stay exactly equal (`check_exact=True`).
- **Truncation and timestamp shifts:** computing on the first `t + 1` rows, or
  after delaying every later timestamp, gives exactly the same rows `<= t`.
- **Boundary isolation:** after an overnight, weekend or missing-minute break,
  the features equal those computed from the post-break bars alone.
- **Reference check:** every feature is compared with a row-by-row reference
  written directly from the definitions in `docs/architecture.md`.

Mutation checks run during Phase 1C confirmed that a negative shift, a centered
window, back-filling, a missing gap or session check, and an off-by-one window
each make these tests fail.

## Label tests

`tests/labels/test_five_minute_labels.py` checks the label contract in
`docs/architecture.md`:

- **Reference check:** every output column is compared exactly with a
  row-by-row reference written from the definition. It runs on a multi-session
  frame with overnight, weekend, missing-minute, New York midnight (EST and
  EDT) and DST boundaries.
- **Window locality:** for every row, labelling just that row and the five rows
  after it gives exactly the full-frame label. Labels depend on no earlier bar
  and no bar beyond `t+5`.
- **Phase 1C agreement:** `future_return_5m` at `t` equals the `return_5m`
  feature at `t+5`, including where each is undefined, so both phases share one
  continuity rule.
- **Five rows are not five minutes:** missing minutes, a later non-consecutive
  bar, and irregular sub-minute steps never produce a label.
- **Numerics:** overflow to `inf` gives an undefined label; underflow,
  subnormal prices, the largest finite ratio and one-ulp moves keep their exact
  value and direction.
- **Isolation:** `atlas.features` does not import `atlas.labels`, and the two
  outputs share only `timestamp`.

Mutation checks run during Phase 1D each made these tests fail: row-offset-only
continuity, a 4-interval window, counting zero as up, dropping the finiteness
check, gap-only continuity, two wrong session-date rules, an off-by-one target
or lead, dropping the index, skipping validation and leaving undefined returns
unmasked.

## Rules

- Unit tests use no network access and no real market data.
- Test observable behavior through public APIs such as `atlas.data`, not
  internals.
- Never weaken a test to make code pass (see `AGENTS.md`).

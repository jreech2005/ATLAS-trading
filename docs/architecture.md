# ATLAS V1 Architecture

V1 is a single-process Python library: plain functions over pandas DataFrames,
tested with pytest. It has no services, database or orchestration layer.

## Pipeline

```
raw OHLCV ──► validation ──► features ──► models ──► evaluation ──► simulation
 (data)        (contract)    (causal)    (baselines)  (walk-forward) (long/flat + costs)
```

| Stage | Responsibility | Status |
|-------|----------------|--------|
| Data loading | Read 1-minute OHLCV from a local CSV file into the canonical UTC bar-close schema (`atlas.data.load_ohlcv_csv`) | **Implemented (Phase 1B)** |
| Provenance | Record source, license, retrieval date and adjustment policy for each dataset; provider integration | Not implemented (deferred) |
| Validation | Structural OHLCV contract checks (`atlas.data.validate_ohlcv`) | **Implemented (Phase 1A)** |
| Features | Causal features that use only bars with `timestamp <= t` (`atlas.features.compute_features`) | **Implemented (Phase 1C)** |
| Labels | Direction 5 minutes ahead, built strictly from future bars | Not implemented |
| Models | Logistic regression and LightGBM baselines | Not implemented |
| Evaluation | Chronological walk-forward validation | Not implemented |
| Simulation | Long/flat strategy with realistic fees and slippage | Not implemented |
| Experiments | Reproducible configs, seeds and recorded results | Not implemented |

## Code layout

```
src/atlas/data/contracts.py    canonical column names, OHLCVValidationError
src/atlas/data/validation.py   validate_ohlcv
src/atlas/data/csv_loader.py   load_ohlcv_csv (local CSV -> UTC bar-close frame, then validate_ohlcv)
src/atlas/features.py          compute_features (causal one-minute features)
tests/data/                    behavioral tests for validation and CSV loading
tests/features/                behavioral and leakage tests for features
```

Each future stage gets a module when it is implemented, not before. No
placeholder packages.

## Features (Phase 1C)

```python
compute_features(bars: pd.DataFrame) -> pd.DataFrame
```

`bars` must satisfy the OHLCV contract; it is passed to `validate_ohlcv` first,
so an invalid frame raises `OHLCVValidationError` (or `TypeError`). The input is
never modified. The result is a new frame with the **same index and row order**
as the input and these columns:

| Column | Definition at bar `t` (close prices `c`, volumes `v`) | Needs |
|--------|--------------------------------------------------------|-------|
| `timestamp` | The input's UTC bar-close timestamp, unchanged | - |
| `return_1m` | `c[t] / c[t-1] - 1` | 1 interval |
| `return_5m` | `c[t] / c[t-5] - 1` | 5 intervals |
| `return_15m` | `c[t] / c[t-15] - 1` | 15 intervals |
| `realized_volatility_15m` | `sqrt(sum over k = 0..14 of ln(c[t-k] / c[t-k-1])^2)`; not annualized | 15 intervals |
| `relative_volume_15m` | `v[t] / mean(v[t-15], ..., v[t-1])`; the current bar is excluded from the mean | 15 intervals |
| `feature_ready` | `True` only if all five features are finite | - |

Here `t-k` means the bar `k` rows earlier. Returns are fractions (`0.01` is
1%), not percentage points.

**Continuity.** A feature uses only bar `t` and the bars before it in the same
*run*. Bar `t` continues the run of bar `t-1` only if:

1. it closes exactly one minute after bar `t-1`, and
2. both bars cover a minute on the same **America/New_York calendar date**. The
   date is taken from the minute the bar covers, `[close - 1 min, close)`, so
   the bar closing at 00:00 New York time belongs to the previous date.

A feature that needs `k` intervals is defined only when the `k` bars before `t`
are in `t`'s run. Missing minutes, overnight, weekend and holiday gaps, and New
York midnight all start a new run. Windows never reach across them, and missing
bars are never filled or interpolated. UTC midnight and DST changes are not
boundaries in themselves, because timestamps are compared as elapsed UTC time.

**Undefined values are NaN.** Warm-up rows (the first 1, 5 or 15 bars of a
run), broken windows, a zero mean volume in the relative-volume denominator, and
any calculation that overflows to `inf` all produce NaN, and the row's
`feature_ready` is `False`. Nothing is imputed, back-filled or carried forward.
A zero *current* volume with a positive mean gives `relative_volume_15m == 0.0`,
which is valid. Downstream stages must use only rows where `feature_ready` is
`True`.

**Causality by construction.** Every window is built from non-negative lags of
the bar's own run. There are no centered windows, negative shifts or fills, and
continuity is decided only from bar `t` and earlier timestamps. Window sums are
explicit sums over the lagged bars, not running sums, so each value depends only
on the bars inside its window.

**Not verified.** The New York date rule only separates sessions. It is **not**
exchange-calendar validation. It does not know trading hours, holidays, early
closes or halts. Pre-market, regular and after-hours bars on the same date form
one run if they are consecutive minutes. Like the loader, the features assume
the input holds one-minute bars and cannot verify bar duration.

## Principles

- **Validate at the boundary.** Data is validated once on entry. Downstream code
  may assume the contract holds. The CSV loader handles only file structure and
  timestamp conversion; every price, volume and ordering rule is delegated to
  `validate_ohlcv`.
- **State conventions explicitly.** The provider's timezone and its bar-open or
  bar-close convention are required arguments, never inferred. The loader
  assumes one-minute bars and cannot verify bar duration (see
  `docs/data_contract.md`).
- **Fail loudly.** Bad data is rejected, never repaired silently.
- **Causality by construction.** Every stage takes a decision time `t` and must
  not read data after it.
- **Portable paths.** No hard-coded absolute paths. macOS development and the
  future Linux/CUDA research machines do not share a filesystem layout.

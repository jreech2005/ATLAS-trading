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
| Labels | Five-minute forward return and direction, built from future bars for training only (`atlas.labels.compute_labels`) | **Implemented (Phase 1D)** |
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
src/atlas/labels.py            compute_labels (five-minute forward labels; training targets only)
tests/data/                    behavioral tests for validation and CSV loading
tests/features/                behavioral and leakage tests for features
tests/labels/                  behavioral, boundary and isolation tests for labels
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

## Labels (Phase 1D)

```python
compute_labels(bars: pd.DataFrame) -> pd.DataFrame
```

`bars` must satisfy the OHLCV contract; it is passed to `validate_ohlcv` first,
so an invalid frame raises `OHLCVValidationError` (or `TypeError`). The input is
never modified. The result is a new frame with the **same index and row order**
as the input, including duplicate or non-monotonic index labels, and these
columns:

| Column | dtype | Definition at bar `t` (close prices `c`) |
|--------|-------|-------------------------------------------|
| `timestamp` | input's UTC datetime dtype | The input's UTC bar-close timestamp, unchanged |
| `future_return_5m` | `float64` | `c[t+5] / c[t] - 1`; NaN when undefined |
| `direction_5m` | `Int64` (nullable) | `1` if `future_return_5m > 0`, `0` if it is zero or negative; `<NA>` when undefined |
| `label_end_timestamp` | input's UTC datetime dtype | Close of the bar `t+5` the label is measured at (always `timestamp + 5 min`); NaT when undefined |
| `label_ready` | `bool` | `True` only when the label is defined |

Here `t+5` means the bar five rows later, and the label is defined only when
those five rows are also exactly five elapsed minutes. Returns are fractions,
not percentage points. A zero return is labelled `0` ("not up").

**Continuity.** A label is defined only when bars `t, t+1, ..., t+5` are one
*run* under exactly the Phase 1C rule (see [Features](#features-phase-1c)): each
bar closes exactly one minute after the previous one, and all six bars cover
minutes on the same America/New_York date. The implementation reuses the
features' run computation, so the two phases cannot drift apart. It is
equivalent to "bar `t+5` has at least five intervals of history", and it means
the label at `t` equals the `return_5m` feature at `t+5`, value for value.
Consequently:

- The last five bars of every run have no label. That covers the end of the
  data, the five bars before any missing minute, and the last five bars before
  an overnight, weekend or holiday gap or a New York midnight.
- A later, non-consecutive bar never stands in for a missing target bar, and five
  rows ahead are not assumed to be five minutes ahead. Irregular sub-minute steps
  break a run even when they add up to five minutes.
- UTC midnight and DST changes are not boundaries in themselves: the horizon is
  five minutes of elapsed UTC time, even when the New York wall clock jumps or
  repeats inside the window.

**Undefined values.** Wherever a label is undefined, `future_return_5m` is NaN,
`direction_5m` is `<NA>`, `label_end_timestamp` is NaT and `label_ready` is
`False`. Rows are never dropped, filled or repaired. Downstream stages must use
only rows where `label_ready` is `True`.

**Numerical limits.** Prices are converted to `float64`. The ratio of two valid
prices can overflow to `inf`, for example `1e300 / 1e-300`. Such a label is
undefined, not clipped. A ratio that underflows to `0.0` is the correctly rounded
result and gives a valid `future_return_5m == -1.0` with direction `0`. Tiny and
subnormal positive prices are handled exactly as IEEE 754 division handles them.
For valid prices the sign of `future_return_5m` matches `c[t+5]` compared with
`c[t]`, down to one-ulp moves. Precision is otherwise that of `float64`, and no
rounding tolerance is applied to the zero threshold.

**Labels are not features.** Labels read future bars by design. They are
training and evaluation targets only. `atlas.features` does not import
`atlas.labels`, and the two outputs share only the `timestamp` column. Tests
check both. Joining features
and labels on row position or `timestamp` gives, for each decision time `t`,
inputs knowable at `t` and a target realized at `label_end_timestamp`. A label
column must never be used as a model input or by a decision at `t`.

**Chronological purging (for Phase 1E).** Each ready row's target depends on
bars closing in `(timestamp, label_end_timestamp]`, so a training sample's
outcome is not known until `label_end_timestamp`. A walk-forward splitter can
purge any training row whose `label_end_timestamp` is at or after the start of
the following test window, so no training target overlaps test-period prices.
Feature lookbacks (up to 15 minutes) may also call for an embargo. Rows that are
not ready have NaT and must be excluded before splitting. Splitting itself is
not implemented.

**Not verified.** As with features, the New York date rule is not exchange-calendar
validation. Labels can be formed in pre-market or after-hours bars and across
the regular open or close when bars are consecutive. Bar duration is assumed to
be one minute and is not verified.

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

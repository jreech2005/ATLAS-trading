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
| Features | Causal features that use only bars with `timestamp <= t` | Not implemented |
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
tests/data/                    behavioral tests for validation and CSV loading
```

Each future stage gets a module when it is implemented, not before. No
placeholder packages.

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

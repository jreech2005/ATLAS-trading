# OHLCV Data Contract (V1)

Applies to every OHLCV frame entering ATLAS. Enforced by
`atlas.data.validate_ohlcv(frame: pd.DataFrame) -> None`, which returns `None`
for a valid frame and raises `atlas.data.OHLCVValidationError` (a `ValueError`)
otherwise. Passing something other than a `pd.DataFrame` raises `TypeError`.
The validator never mutates, reorders, repairs, fills, drops or coerces input.

## Schema

Exactly these six columns (order not significant, no extras, no duplicate labels):

| Column      | Required dtype                                      | Rules                                   |
|-------------|-----------------------------------------------------|-----------------------------------------|
| `timestamp` | pandas tz-aware datetime (`datetime64[*, UTC]`)     | UTC, no NaT, strictly increasing        |
| `open`      | integer or float (numpy or nullable); **not bool**  | finite, > 0                             |
| `high`      | integer or float; not bool                          | finite, > 0, `high >= open, close, low` |
| `low`       | integer or float; not bool                          | finite, > 0, `low <= open, close, high` |
| `close`     | integer or float; not bool                          | finite, > 0                             |
| `volume`    | integer or float; not bool                          | finite, >= 0 (zero volume is valid)     |

Additional rules:

- The frame must contain at least one row.
- NaN, `pd.NA`, NaT, `+inf` and `-inf` are invalid everywhere.
- `object`, string, boolean and complex columns are rejected even if their
  contents look numeric. Convert explicitly and upstream, where the conversion
  can be reviewed.
- The DataFrame index is not part of the contract and is ignored.
- Error messages name the rule and the offending 0-based **row positions**.

## Timestamps

- Each `timestamp` is the **close** of a completed one-minute bar. A bar
  stamped `14:31:00Z` covers `[14:30:00Z, 14:31:00Z)`, and its values are
  first knowable at `14:31:00Z`.
- Timestamps must be timezone-aware **UTC**. A timezone is accepted only if it
  is named `UTC` or `Etc/UTC`, for example `"UTC"`, `datetime.timezone.utc` or
  `ZoneInfo("UTC")`.
- **Naive timestamps are rejected.** They are never assumed to be UTC.
- **Tz-aware non-UTC timestamps are rejected**, not silently converted. That
  includes zones that happen to sit at offset zero, such as `Europe/London` in
  winter. Convert explicitly first: `frame["timestamp"].dt.tz_convert("UTC")`.
- Duplicate timestamps are invalid. Timestamps must be strictly increasing in
  row order. Rows are never sorted for you.
- **Gaps are legal.** Overnight, weekend, holiday and intraday gaps (halts,
  missing prints) are not errors. Consecutive bars do not have to be one minute
  apart, and the validator never synthesizes missing bars.
- Not enforced in Phase 1A: seconds/sub-second alignment to whole minutes, and
  membership in exchange session hours.

## Causality rules (binding on future phases)

- A bar may be used for a decision at time `t` only if `timestamp <= t`.
- Features computed at `t` may use only bars with `timestamp <= t`.
- Labels such as "direction 5 minutes ahead" are built from future bars and must
  never feed features. Train/test splits are chronological.

## Valid is not the same as verified

Passing `validate_ohlcv` means the frame is **structurally valid**. It does **not**
establish that:

- the data came from a trustworthy or licensed provider,
- splits, dividends or other corporate actions were handled correctly,
- a trading session is complete, or no bars are missing,
- the prices were actually available to a strategy at the stated timestamp
  (vendor latency, late prints, revisions),
- the license permits model training or redistribution.

Provenance must be recorded separately for every dataset: source, retrieval
date, license terms, adjustment policy and timestamp convention. Treat that
record as a prerequisite for using the data in research.

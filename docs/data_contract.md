# OHLCV Data Contract (V1)

Applies to every OHLCV frame entering ATLAS. Enforced by
`atlas.data.validate_ohlcv(frame: pd.DataFrame) -> None`, which returns `None`
for a valid frame and raises `atlas.data.OHLCVValidationError` (a `ValueError`)
otherwise. Passing something other than a `pd.DataFrame` raises `TypeError`.
The validator never mutates, reorders, repairs, fills, drops or coerces input.

Local CSV files are loaded with `atlas.data.load_ohlcv_csv`, which converts
provider timestamps to this contract and then calls `validate_ohlcv` (see
[Loading local CSV files](#loading-local-csv-files-phase-1b)).

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
- Error messages name the rule and the offending 0-based **row positions**:
  the first 10 positions, followed by the total count when there are more
  than 10, for example `row positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... (15 total)]`.
  Positions are counted in row order, never from index labels.
- Validation stops at the first violated rule group and reports only that group.

## Timestamps

- Each `timestamp` is the **close** of a completed one-minute bar. A bar
  stamped `14:31:00Z` covers `[14:30:00Z, 14:31:00Z)`, and its values are
  first knowable at `14:31:00Z`.
- Timestamps must be timezone-aware **UTC**. A timezone is accepted only if it
  is named `UTC` or `Etc/UTC`, for example `"UTC"`, `datetime.timezone.utc` or
  `ZoneInfo("UTC")`, and every timestamp must also have a zero UTC offset. A
  tzinfo that is named `UTC` but carries a nonzero offset is rejected.
- **Naive timestamps are rejected.** They are never assumed to be UTC.
- **Tz-aware non-UTC timestamps are rejected**, not silently converted. That
  includes zones that happen to sit at offset zero, such as `Europe/London` in
  winter. Convert explicitly first: `frame["timestamp"].dt.tz_convert("UTC")`.
- Duplicate timestamps are invalid. Timestamps must be strictly increasing in
  row order. Rows are never sorted for you.
- **Gaps are legal.** Overnight, weekend, holiday and intraday gaps (halts,
  missing prints) are not errors. Consecutive bars do not have to be one minute
  apart, and the validator never synthesizes missing bars.
- Not enforced by `validate_ohlcv`: alignment to whole minutes, and membership
  in exchange session hours. `load_ohlcv_csv` does require whole-minute
  timestamps in its input (see below); session hours are not checked anywhere.

## Loading local CSV files (Phase 1B)

```python
load_ohlcv_csv(
    path: str | Path,
    *,
    provider_timezone: str,
    timestamp_convention: Literal["bar_open", "bar_close"],
) -> pd.DataFrame
```

Reads one local CSV file of **one-minute** OHLCV bars and returns a frame that
satisfies this contract. Nothing is downloaded; `path` is opened as a local file
and is never treated as a URL.

### Arguments

Both keyword arguments are **required**. There are no defaults, because a
wrong guess shifts every timestamp.

- `provider_timezone`: the IANA timezone the file's timestamps are written in,
  for example `"America/New_York"` or `"UTC"`. It must resolve with Python's
  `zoneinfo`; otherwise `ValueError` is raised.
- `timestamp_convention`:
  - `"bar_close"`: each timestamp is the bar's close. It is converted to UTC
    and used unchanged.
  - `"bar_open"`: each timestamp is the bar's open. It is converted to UTC and
    then **one minute of elapsed time** is added to give the bar close.
  - Any other value raises `ValueError`.

### Input file

- UTF-8 text with a header row naming exactly the six canonical columns
  (`timestamp, open, high, low, close, volume`). Column order in the file is
  not significant. Missing, extra or duplicate columns are rejected.
- `timestamp` values must be naive local wall-clock times in exactly the form
  `YYYY-MM-DD HH:MM:SS`: two-digit fields, a single space separator, no `T`,
  no UTC offset or zone suffix, and no fractional seconds. One-digit fields such
  as `2024-1-2` and impossible dates are rejected.
- Seconds must be `00`. A stamp such as `14:34:60` is rejected, not rolled into
  the next minute, and `14:34:30` is rejected, not truncated.
- `open`, `high`, `low`, `close` and `volume` are parsed by pandas' default
  CSV number parsing. A value that does not parse as a number leaves the column
  non-numeric, and the file is rejected. Empty fields and pandas' missing-value
  tokens (such as `NA` or `NaN`) become missing values, which are rejected.

### Strict CSV handling

- A file containing any NUL byte is rejected before parsing (pandas would
  otherwise silently truncate the field).
- Invalid UTF-8 is rejected.
- A file with no content, or only blank lines, is rejected as empty. A header
  with no data rows is rejected because a frame needs at least one bar.
- Malformed CSV structure that pandas or Python's `csv` module refuses (for
  example an unterminated quote, or a field over the `csv` module's field-size
  limit) is rejected. A pandas `ParserWarning`, which means fields are about
  to be dropped or shifted, is turned into a rejection.
- A data row with **more** fields than the header is rejected, including a
  surplus trailing empty field (`...,1200,`). A row with **fewer** fields
  produces missing values, which are rejected.
- **Blank lines:** empty lines and lines containing only spaces and/or tabs
  are skipped, as pandas skips them. They are not data rows and do not count
  toward row positions. A line containing other whitespace (form feed,
  vertical tab, NBSP and so on) is a data row and is then rejected by the
  rules above. Whitespace inside a quoted field is always data.

### Timezone conversion and DST

Each local timestamp is interpreted in `provider_timezone`, converted to UTC
and, for `"bar_open"`, shifted forward one minute. The shift is applied after
conversion, so it is real elapsed time even across a DST change. The UTC offset
is determined separately for each row, so a file may span DST transitions.

- **Ambiguous** local times (repeated by a DST fall-back, for example
  `2024-11-03 01:30:00` in `America/New_York`) are rejected. The loader never
  guesses which of the two instants was meant.
- **Nonexistent** local times (skipped by a DST spring-forward, for example
  `2024-03-10 02:30:00` in `America/New_York`) are rejected, not shifted.
- These messages quote up to the first 10 offending local timestamps as well as
  the row positions.
- The DST rules come from the machine's timezone database, so results for
  far-future or recently changed rules can differ between machines.

### Output

- Exactly the six canonical columns, in canonical order.
- `timestamp` is a timezone-aware UTC datetime marking the bar **close**.
- Prices and volume keep the integer or float dtype pandas inferred.
- Rows are in **file order**. They are never sorted, deduplicated or filled;
  duplicate or out-of-order timestamps (after conversion) are rejected.
  Gaps are preserved and no bars are synthesized.
- The index is a fresh `RangeIndex` starting at 0, so index labels equal the
  row positions used in error messages.
- The returned frame has passed `validate_ohlcv`.

### Errors

| Exception | Raised for |
|-----------|------------|
| `ValueError` | Unknown `provider_timezone`, or a `timestamp_convention` other than `"bar_open"`/`"bar_close"` |
| `FileNotFoundError` | `path` does not exist (other `OSError`s, such as a directory or a permission error, propagate unchanged) |
| `OHLCVValidationError` | Any problem with the file's **content**: NUL bytes, invalid UTF-8, empty file, malformed CSV, surplus fields, wrong columns, bad timestamp text, ambiguous or nonexistent local times, and every `validate_ohlcv` rule |

`OHLCVValidationError` is a subclass of `ValueError`. Checks run in a fixed
order (arguments, file bytes and CSV structure, columns, timestamp text, then
`validate_ohlcv`) and the first failing rule is reported. Row-level messages
use the same format as the validator: 0-based data-row positions, the first 10
listed, plus the total count when there are more than 10. Data rows are
counted after the header and after skipped blank lines.

### What the loader assumes but does not verify

- **The file contains one-minute bars.** Bar duration is **not**
  independently authenticated. The loader checks only that timestamps are on
  whole minutes, unique and strictly increasing after conversion. Because gaps
  are legal, a file of five-minute bars, for example, would load without error.
  With `"bar_open"` it would also be mislabelled, since the shift is always one
  minute.
- **`provider_timezone` and `timestamp_convention` are correct.** They come
  from the caller's knowledge of the provider and cannot be inferred from the
  file.
- **Provenance and licensing are out of scope.** The loader records no source,
  license, retrieval date or adjustment policy, and integration with any data
  provider is deferred (see [Valid is not the same as
  verified](#valid-is-not-the-same-as-verified)).

### Known limitations

These were reported as non-blocking by the independent Phase 1B release audit,
or are accepted by V1 scope. They are documented here, **not** fixed.

- **Whole-file reading.** The file is read fully into memory and parsed twice
  (by pandas and by Python's `csv` module for the surplus-field check). This is
  intended for local research files, not very large inputs.
- **Field-size limit.** A field longer than the `csv` module's limit (131,072
  characters by default) is rejected. The process-wide limit is not changed.
- **Lenient numeric parsing.** pandas' number parsing accepts some
  representations that are not strictly canonical. For example, spaces around a
  number and quoted numbers (`"470.10"`) are accepted as numbers, and a UTF-8
  byte-order mark before the header is accepted.
- **Float precision.** Prices are parsed by pandas' default float parser, which
  can differ from Python's `float()` by one unit in the last place for values
  with 15 or more significant digits.
- **Integer paths.** An `int` passed as `path` is treated by `open()` as an
  already-open file descriptor, which is then read and closed. Pass a `str` or
  `Path`.
- **Malformed input with bare carriage returns.** On some malformed files that
  contain bare CR (`\r`) line endings mixed with quotes, pandas' C parser can
  produce spurious empty rows, and the reported surplus-field positions may not
  match pandas' rows. Such files are still rejected. No accepted frame was
  found to be wrong.
- **Warnings-as-errors.** On a small class of malformed files with bare CR line
  endings, running under `warnings.simplefilter("error")` makes a pandas
  `DtypeWarning` escape as an exception instead of `OHLCVValidationError`.
  Under default warning filters these files raise `OHLCVValidationError`.
  Valid files are not affected.
- **Timezone impostors.** The validator rejects tzinfo objects named `UTC` that
  carry a nonzero offset. Two contrived custom tzinfo implementations found
  during the audit are not rejected. These cannot be produced by the CSV loader,
  which always converts with `"UTC"`.
- **Test coverage gap.** No test pins that lines consisting of form feed,
  vertical tab, NBSP or NEL count as data rows. The behavior was verified by the
  audit but is not locked by a regression test.

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

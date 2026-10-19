# AGENTS.md — Engineering Rules for ATLAS

These rules apply to every contributor, human or AI. The tool-specific notes in
`CLAUDE.md` add to these rules and never override them.

## Scope
- Current scope is **ATLAS V1** (see `docs/decisions/0001-v1-scope.md`). Work
  outside it needs an approved decision record first.
- **No live trading.** No brokerage connections, live or paper orders, or
  credentials in the repository.
- No unauthorized or bulk data extraction. Use only sources whose terms permit
  the use. No scraping and no unofficial endpoints.

## Test-driven development
1. **RED:** write the test first, run it, and confirm it fails for the
   *expected missing behavior*. Import or environment errors do not count.
2. **GREEN:** write the minimum code that passes, then rerun.
3. **REFACTOR:** improve structure with tests green. Rerun.
4. **REVIEW:** look for edge cases, look-ahead and silent repairs. Add a
   regression test for every real defect found.

## Modifying tests
- Never weaken, delete or skip a test to make code pass.
- Change a test expectation only when the *specification* changes, and say so
  explicitly in the change description.
- Tests use small, deterministic, synthetic data. No network, no real market
  data in unit tests.

## Causality and leakage
- Every value used at decision time `t` must be knowable at `t`. Bar timestamps
  mark bar **close** (see `docs/data_contract.md`).
- Labels are built from future data and must never reach features.
- Splits are chronological (walk-forward). Never shuffle across time.
- Fitted transforms (scalers and so on) are fitted on training windows only.

## Financial data
- Timestamps are timezone-aware UTC. Never assume a timezone for naive data.
- Never silently drop, fill, sort, or coerce records. Reject them with a clear
  error.
- "Validated" means structurally valid, not verified provenance. Record source,
  license, adjustment policy and retrieval date for each dataset.
- Raw market data is never committed to Git.

## Changes and verification
- Keep changes small, bounded and reviewable. No placeholder modules for
  deferred features.
- Before claiming that something works, run `python -m pytest` and report the
  actual output.
- **Never fabricate results.** That covers test runs, GPU runs, backtests and
  performance numbers. If something was not run, say so.
- Do not commit, push or run destructive Git commands without explicit owner
  approval.

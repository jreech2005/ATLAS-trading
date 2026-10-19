# CLAUDE.md

Operating instructions for Claude Code in this repository.

1. **Read `AGENTS.md` first.** Its rules are binding. Relevant references:
   `docs/data_contract.md`, `docs/testing.md`, `docs/architecture.md`,
   `docs/decisions/`.
2. **Inspect before editing.** Read the files you will touch and their tests.
3. **State the task** before implementing it: context, constraints and
   acceptance criteria.
4. **Follow RED → GREEN → REFACTOR.** Write the test, run it and observe the
   expected failure. Then implement, rerun and refactor.
5. **Report what you observed.** Give the actual commands and results. If a step
   was not run, say so.
6. **Never silently weaken tests.** Flag any change to a test expectation and
   give the reason.
7. **Ask first** before destructive or high-impact actions: deleting files,
   rewriting history, committing or pushing, network or data downloads, new
   infrastructure.
8. **Don't overengineer.** Make the smallest change that meets the acceptance
   criteria, within V1 scope.

Setup and test commands are in `README.md` and `docs/testing.md`.

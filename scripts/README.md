# `scripts/`

Helper scripts for cobre-bridge, grouped by role.

- **`ci/`** — quality gates and advisory reports (see below). The gates run in
  CI (`.github/workflows/ci.yml`) and from the `pre-commit` hook; every script
  locates the repo root from its own path, so they run from anywhere.
- **`pre-commit`** — the git pre-commit hook. Install with
  `ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`.
- **`gen-cli-docs.sh`** — regenerates `docs/cli.md` from the Typer app
  (content-guarded by `tests/test_docs.py`; never hand-edit the output).
- **`analyze_results.py`**, **`presentation_charts.py`** — local analysis
  utilities, not gates.

## `ci/` — quality gates

Gates marked _advisory_ never fail the build; the rest exit 1 on violation.
The conventions each gate enforces live in `.claude/rules/` (comments,
doc-integrity, testing, bridge contracts).

| Script                   | Purpose                                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `check_no_plan_leaks.py` | No plan-structure vocabulary in shipped docs (hard) + src/ prose burndown count (advisory). `--all` lists it.                |
| `check_comment_refs.py`  | No `file.py:NNN` / `MEMORY.md` / private-tooling refs in src/ prose (hard); machine-local + gitignored path refs (advisory). |
| `check_doc_paths.py`     | Repo-relative paths cited in README/docs resolve against the tree (hard); CLAUDE.md (advisory).                              |
| `check_comment_bloat.py` | _Advisory_: long comment blocks + repeated clauses, candidates for a comment-skeptic pass.                                   |
| `quality_report.py`      | _Advisory_: churn/size/long-fn/suppression/plan-token hotspot ranking.                                                       |
| `_scan.py`               | Shared comment/docstring extraction (tokenize + ast), imported by the gates.                                                 |

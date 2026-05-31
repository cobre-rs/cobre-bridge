# Architecture & Code-Quality Assessment — cobre-bridge

**Method:** adversarial attacker → defender rounds (the saved `/assess` workflow for
round 1, run 3× independently on the whole source; then two focused inline rounds
for special-casing and dashboard/error-handling/types, with the round-1 findings
threaded in as "do not repeat"). Every finding was read in code by an attacker and
independently verified by a defender.

**Outcome:** ~49 confirmed pain points, **1 defended** (ROBUST-04), **0 disputed**.
Three independent whole-repo architecture passes converged tightly on the same top
structural problems (high confidence). Defender severity adjustments are reflected
below.

> Note on method: the saved `assess` workflow did not thread the per-round `args`
> (scope/files/rubric/previousFindings), so round 1 ran as a generic whole-repo
> architecture pass three times. The focused rounds used the assess skill's
> documented **inline fallback** (`adversarial-attacker`/`adversarial-defender`
> dispatched directly) — functionally identical — to reach the special-casing,
> dashboard/UI, and error-handling/type dimensions the workflow couldn't be pointed at.

---

## The big picture

The codebase is functionally rich and well-tested (919 tests), but it has the
classic shape of a converter that grew by accretion: **there is no domain layer.**
Converters receive only file _paths_ (`NewaveFiles`), so each one re-parses the raw
NEWAVE files and re-derives the same domain invariants inline. That single missing
seam is the root cause of most findings below — it forces the duplication (horizon
formula, active-plant filter, readers), it strands domain logic wherever it was
first needed (including the dashboard), and it makes the comparator re-implement the
converter instead of sharing it. On top of that sits a layer of **accreted
special-cases and abandoned calibration** (contradictory penalty constants, dead
named-plant code, no-op tie-break factors) and a pervasive **error-swallowing**
pattern that lets degraded conversions and comparisons exit as success.

Five things deserve attention first; everything else is solid backlog.

---

## Priority 1 — The validation tool can report a false "match" 🔴

This is the most dangerous cluster because it defeats the project's core purpose
(detecting NEWAVE↔Cobre divergence).

| ID(s)         | Problem                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Sev          |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| ARCH-01 (B/C) | `comparators/bounds_from_inputs.py` **re-implements the converter's bounds logic verbatim** — `_build_step_function`, `_build_stage_dates`, `_BIG_M`, the seasonalize-vs-freeze post-study rule are byte-for-byte hand copies of `hydro.py`/`thermal.py`/`network.py` (its own docstrings say "Mirrors … exactly"). The comparator validates the converter **against a copy of itself**, so a bug in the shared logic is present on both sides and `compare bounds` reports a perfect match while the case is wrong. | **CRITICAL** |
| ROBUST-01     | `comparators/cobre_readers.py` has **~23 `except Exception: warn; return empty`** blocks around real aggregation. A read/schema failure on Cobre output → empty frame → `results.py` skips that section → `cli.py` `sys.exit(0)`. Unreadable/broken output is reported as "no divergence".                                                                                                                                                                                                                           | major        |
| ARCH-04 (B)   | `compare bounds`/`compare results` import the private `pipeline._build_id_map` and **rebuild** the NEWAVE→Cobre id map from inputs instead of reading it from the case under test — so the comparator can map entities differently than the case it is validating.                                                                                                                                                                                                                                                   | high         |

**Direction:** (1) Establish a _single_ source of the reference bounds — either have
the comparator read the pipeline's own emitted bounds, or extract the bounds math into
one module imported by both converter and comparator (no copies). (2) Make readers
distinguish "no data present" from "data present but failed to read"; surface the
latter and exit non-zero. (3) Read the id-map from the converted case, or persist it
during conversion, rather than rebuilding.

---

## Priority 2 — No domain model: the master invariants are copy-pasted 🟠

The structural root of most duplication.

| ID(s)                      | Problem                                                                                                                                                                                                                                                                                                                                       | Sev   |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| ARCH-01/02/03 (all 3 runs) | The horizon formula `study_months = (13 - start_month) + (num_anos-1)*12` and `total_stages = study_months + num_anos_pos*12` is hand-copied at **~25 sites across 9 modules**. It sizes every per-stage Parquet/JSON; one missed edit silently mis-aligns stages. Already partially inconsistent (`num_anos or 1` vs `or 0` between copies). | high  |
| ARCH-02/04 (A/B)           | **No parsed case object.** Converters get only paths, so one run re-parses `Dger` 27×, `Hidr` 11×, `Confhd` 16×. `NewaveFiles` caches paths but not content.                                                                                                                                                                                  | high  |
| SPECIAL-09                 | The active-plant filter (`usina_existente=="EX"` + `~startswith("FICT.")`) is recomputed at **~11 sites**, each re-reading Confhd; `resolve_cascade`/`_apply_permanent_overrides` re-walked many times. "Which plants are in the LP" has no canonical definition.                                                                             | minor |
| SPECIAL-06/08              | `_total_study_stages` helper exists but is bypassed by an inline copy in the same file (and the two disagree on the `num_anos` fallback); `_is_na` defined twice with **different** exception handling; the `9999` post-study sentinel hand-coded ~10×.                                                                                       | minor |

**Direction:** Introduce a `CaseContext` built once (parsed `Dger`/`Confhd`/`Hidr`/… +
derived `horizon`, `stage_dates`, `active_hydros`, cascade) and thread it through the
converters. Provide canonical `horizon(ctx)`, `active_hydros(ctx)`, and a named
`POST_STUDY_YEAR = 9999`. This removes the duplication _and_ the drift surface that
Priority 1 depends on.

---

## Priority 3 — Layering inversion: domain logic stranded in the UI 🟠

| ID(s)            | Problem                                                                                                                                                                                                                                                                                                                                                                                                                           | Sev   |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| ARCH-02/03 (A/C) | `comparators/constraints_compare.py` imports core constraint-expression logic **from the dashboard**: `from cobre_bridge.dashboard.tabs.constraints_utils import _parse_expression, _resolve_param_to_column` and `from cobre_bridge.dashboard.data import scan_entity`. The dependency points comparator→dashboard (backwards); it even builds an `artificial_case_dir = cobre_output_dir.parent` to bridge the layout mismatch. | high  |
| PRES-05          | NPV discounting, the cost-component taxonomy, productivity, and LP-load reconstruction live in `dashboard/chart_helpers.py` + `dashboard/data.py`; the comparators re-derive cost categories independently → two definitions that can diverge.                                                                                                                                                                                    | minor |
| ARCH-06 (B/C)    | **Two independent Cobre-output reader stacks** (`dashboard/data.py` vs `comparators/cobre_readers.py`), each hardcoding the `output/simulation/...` layout; the dashboard doesn't import the comparator readers.                                                                                                                                                                                                                  | minor |
| ARCH-05 (C)      | The "Cobre output dir is one level below the case dir" contract is hardcoded via `cobre_output_dir.parent` at ~14 sites; breaks silently for custom `--output` paths.                                                                                                                                                                                                                                                             | minor |

**Direction:** Extract a shared, presentation-free domain module owning (a) the
generic-constraint expression parser, (b) the cost taxonomy + NPV/LP-load math, and
(c) **one** Cobre-output reader that owns the on-disk layout. Dashboard and comparators
both consume it. Make the case-dir↔output-dir relationship an explicit typed object,
not a repeated `.parent`.

---

## Priority 4 — Accreted special-cases, contradictory constants & dead code 🟡

The maintainer's stated concern, made concrete. Mostly low-effort, high-clarity wins.

| ID         | Problem                                                                                                                                                                                                                                                                               | Sev   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| SPECIAL-02 | `network.py:_EVAPORATION_MULT = 10.0` but its own comments variously claim **1.1×, 2×, and 10×**; it sets the defaulted DESVIO/VAZMIN/GHMIN/TURBMN/TURBMX penalties + evaporation cost. A maintainer trusting the prose would shift every defaulted penalty 5–9×.                     | major |
| SPECIAL-12 | `stochastic.py:_parse_cadical` hand-rolls a fixed-width `C_ADIC.DAT` parser (magic `_MONTH_COLS=[7,15,…,95]`, silent `continue` on parse error) when **`inewave` already ships a `Cadic` reader** that does exactly this. Wrong/dropped additional load feeds the LP demand silently. | major |
| SPECIAL-01 | Hardcoded `PIMENTAL(314)→BELO MONTE(288)` diversion with magic `13000` m³/s — **dead code** (only caller is commented out; live code hardcodes `diversion=None`). A landmine: re-enabling routes water only for one plant code.                                                       | minor |
| SPECIAL-03 | `_TURBINED_BELOW_FACTOR/_OUTFLOW_BELOW_FACTOR/_OUTFLOW_ABOVE_FACTOR` are all `1.00` and `_MICRO_UPLIFT = 1.0`, so the "1% tie-break spacing" their 14-line comments describe is a **no-op**. Abandoned calibration left in place.                                                     | minor |
| SPECIAL-07 | `_get_individualizado_cutoff` returns magic `9999` while its docstring promises `num_stages`; feeds the (separate, major) silent RE-bounds drop in ROBUST-02.                                                                                                                         | minor |
| SPECIAL-10 | `anticipated.py` (241 LOC) computes block-weighted MW that is **unconditionally discarded** (`values_mw=[0.0]*lead_stages`, a documented Cobre-validator limitation); only `lead_stages` is consumed. ~130 LOC run each conversion to feed a warning string.                          | minor |
| SPECIAL-11 | `_compute_max_turbined_simple` docstring frames it as dormant "restore-with-a-switch" code, but it is a **live fallback** for plants without head data — inviting an incorrect deletion.                                                                                              | minor |

**Direction:** Reconcile each penalty constant to one validated value (the
`project_forward_penalty_validation` memory has ground truth) and rewrite the comment;
delete or re-enable the dead diversion with a test; remove the no-op factors or make
them real; swap `_parse_cadical` for `inewave.Cadic`; fix the misleading docstrings.

---

## Priority 5 — Errors & degraded output pass as success 🟠

| ID          | Problem                                                                                                                                                                                                                                                                        | Sev   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| ROBUST-03   | Broad `except Exception` around `inewave` parses substitutes empty results (`vazpast→[]`, `c_adic→no load`, EXPT skipped, RE skipped). A **corrupt-but-present** input yields a structurally complete, semantically incomplete Cobre case; the pipeline still reports success. | major |
| ROBUST-02   | On `REE.DAT` failure the `9999` sentinel makes `is_post_indiv` always False → post-individualized RE bounds **silently never emitted**.                                                                                                                                        | major |
| ARCH-05 (A) | `ConversionReport.warnings` is declared and faithfully printed by the CLI but **never populated** — the structured warning channel is dead. (Log warnings _are_ emitted at WARNING level, but no warning affects success/exit status.)                                         | minor |
| ROBUST-07   | The dashboard tab loader `except Exception: continue` **silently drops a whole tab** on any render error.                                                                                                                                                                      | minor |
| ROBUST-08   | One broad `except` (`data.py:77`) lacks the `# noqa: BLE001` the other 44 carry and over-catches around a date parse.                                                                                                                                                          | minor |

**Direction:** Add a conversion/compare _result status_ that records degraded/skipped
sections; populate `ConversionReport.warnings`; make the CLI exit non-zero (or at least
print an end-of-run summary) when inputs were substituted-empty or outputs were
unreadable. Render an in-tab error placeholder instead of dropping tabs.

---

## Secondary backlog

**God-functions / god-modules** 🟡

- SPECIAL-05 [major]: `thermal.convert_thermal_bounds` is a ~360-line, 6-step
  mutate-in-place monolith over shared locals — **the known FCMAX/GTMIN bug lives
  here** and can't be step-tested. Extract each step into a tested helper over an
  explicit state object.
- SPECIAL-04 [major]: the CFUGA/CMONT per-stage step-function sweep is duplicated
  near-verbatim across `_per_stage_integrated_productivities` and
  `_per_stage_drop_overrides` (dedup half-done). Route the integrated variant through
  the shared helper.
- ARCH-06/07/09 (A/B/C): `pipeline.convert_newave_case` is a 300+-line orchestrator
  that threads generic-constraint **ID offsets by hand** (`start_id=vminop_count`,
  `+ electric_count`) and unpacks **anonymous 4-tuples** from the constraint
  converters. Give constraint converters a named result type and an ID allocator.
- PRES-04/06: presentation god-modules (`performance_charts.py` 2181, `energy_balance.py`
  1740, `stochastic.py` 1644) repeat `go.Figure()/update_layout/fig_to_html` boilerplate
  ~dozens of times and embed interactive Plotly logic as `{{}}`-escaped f-string JS in
  5 tabs (duplicating `ui/js.py` + the Python builders). Add a figure factory; move
  shared client JS into `ui/js.py`.

**No public API surface** 🟡

- ARCH-03/04/05 (all runs): underscore "private" functions are imported across module
  and package boundaries everywhere (`cli`←`pipeline._build_id_map`,
  `constraints`←`hydro._compute_productivity`/`_apply_permanent_overrides`,
  `fict_cascade`←`hydro._compute_productivity`, `results`←`alignment._read_reference_names`,
  `newave_readers`←`stochastic._parse_cadical`, `charts`←`bounds._is_effectively_infinite`,
  dashboard internals); both package `__init__.py` are empty with no `__all__`. The
  privacy convention is meaningless and `hydro.py` has silently become a shared util
  library exposed only through its private symbols. Decide a public API (or move shared
  helpers to a `_util`/domain module) and stop reaching across boundaries.

**Untyped output contract** 🟡

- ROBUST-06: converter entry points return bare `-> dict`; the whole Cobre output JSON
  schema is stringly-keyed with raw-subscript consumers and no `TypedDict`/dataclass
  (conflicts with the repo's own `python.md` "return types must be concrete"). Blast
  radius is bounded by external `cobre-python --validate`. Introduce TypedDicts for the
  major output shapes.
- ROBUST-05: `data._stage_avg_mw` returns an argument-discriminated union
  (`dict | pl.DataFrame`) imported across modules → forces caller-side asserts. Split
  into two concretely-typed functions.

**Security — dashboard escaping** 🟡

- PRES-01 [major]: there is **no `html.escape` anywhere in src**. NEWAVE plant/line
  names flow raw into `<td>`/`data-` attributes and via `json.dumps` into `<script>`
  blobs (`json.dumps` does not neutralize `</script>`). A crafted case name → stored
  HTML/JS injection in the generated dashboard. Escape on HTML interpolation and use a
  safe JSON-in-`<script>` embedding.

**Presentation drift** ⚪

- PRES-02: `ui/theme.py` declares itself the single source of truth for colour but is
  contradicted by ~250 inline hex literals and an internal collision (hydro is
  `#4A90B8` in one tab, `#3B82F6` in another). PRES-03: `_hex_to_rgba` triplicated.

**Misleading abstractions** ⚪

- ARCH-07/08 (A/C): `comparators/alignment.py` carries a docstring/dead abstraction that
  describes reverse logic.

---

## Defended (not a flaw)

- **ROBUST-04** — `energy_balance.py:221-223` uses `assert isinstance(..., dict)` to
  narrow a union. The asserts are type-narrowing hints, not safety guards:
  `_stage_avg_mw` deterministically returns a `dict` when `group_cols == []`, which all
  three call sites pass, so `-O` stripping does not break correctness. (The underlying
  union-return smell is captured by ROBUST-05.)

---

## Suggested sequencing

1. **Stop false-positive validations** (P1): make readers fail loudly + exit non-zero;
   read (don't rebuild) the id-map; de-duplicate the bounds reference. _Highest risk._
2. **Introduce `CaseContext` + canonical horizon/active-plant helpers** (P2): unlocks
   most de-duplication and removes the drift behind P1. _Highest leverage._
3. **Quick-win cleanups** (P4): reconcile the contradictory constants, delete/decide the
   dead code, swap `_parse_cadical`→`inewave.Cadic`. _Low effort, high clarity._
4. **Extract the shared domain module out of the dashboard** (P3).
5. **Conversion/compare result status + warning surfacing** (P5).
6. Backlog: god-function extraction (start with `convert_thermal_bounds`, where the known
   bug lives), public-API decision, output TypedDicts, dashboard escaping, presentation
   de-duplication.

# Scope: the two large remaining assessment items

Two items from the assessment backlog are large enough to plan before starting.
Both are **incremental-friendly** — each splits into phases that can land
independently, so we never need a single big-bang change.

---

## ARCH-06 — the two Cobre-output reader stacks

### Current state (facts)

| Stack                          | LOC  | Reads relative to                                   | Purpose                                                             |
| ------------------------------ | ---- | --------------------------------------------------- | ------------------------------------------------------------------- |
| `dashboard/data.py`            | 899  | **case dir** (appends `output/simulation/<entity>`) | lazy frames + metadata for interactive charts (`DashboardData`)     |
| `comparators/cobre_readers.py` | 1711 | **output dir** (appends `simulation/<entity>`)      | aggregated reads (weighted means, percentiles) to compare vs NEWAVE |

**"Merge into one stack" is the wrong framing.** The high-level functions serve
genuinely different consumption patterns (lazy LazyFrames for charts vs eager
aggregates for comparison). What actually duplicates is the **low-level I/O
foundation**:

- `_productivity_from_energy_parquet` and `_productivity_from_production_models`
  exist in **both** files and have already **diverged** (different fallback
  logic, one logs, different locals — same intent, drift-prone). This is the
  literal "two definitions that can diverge" the assessment warns about.
- The entity-scan primitive: `scan_entity(case_dir)` vs
  `_scan_simulation_entity(output_dir)` — same hive-partition scan, different
  base-path convention.
- Metadata / stage-label / bus-map readers: `load_hydro_metadata`,
  `load_thermal_metadata`, `load_stage_labels`, `load_hydro_bus_map`,
  `load_ncs_bus_map` (dashboard) vs `read_cobre_hydro_metadata`,
  `read_cobre_thermal_metadata`, `_load_entity_bus_map` (comparator).
- The fail-loud `CobreReadError` probe lives only in `cobre_readers`; the
  dashboard scanner returns empty silently (inconsistent robustness).

### Proposed approach (phased)

- **Phase 1 — kill the diverged productivity duplicates** _(small, low risk)_.
  Promote one definition of `_productivity_from_energy_parquet` /
  `_productivity_from_production_models` into a shared module; both stacks call
  it. Pick the more-correct of the two diverged copies (verify they agree on the
  example first). ~1 commit, behavior-preserving once reconciled.

- **Phase 2 — a typed `CobreCaseLayout`** _(medium; subsumes ARCH-05/C)_.
  One object holding `case_dir` ↔ `output_dir` (default `output_dir =
case_dir/"output"`), replacing the ~18 `.parent` sites and the dashboard's
  implicit case-dir convention. Makes a custom `--output` path expressible
  instead of silently broken. Touches reader signatures → moderate test churn,
  but mechanical.

- **Phase 3 — a shared `cobre_io` foundation** _(medium-large)_.
  Move the scan primitive + metadata/labels/bus-map readers + the
  `CobreReadError` probe into one low-level module consumed by both stacks. The
  high-level `read_cobre_*` aggregations (comparator) and `DashboardData`
  (dashboard) **stay where they are** — they just stop re-implementing the I/O
  layer.

### Effort / risk

- Phase 1: ~0.5 day, low risk (matches the dedup pattern we've used all along).
- Phase 2: ~1 day, medium risk (signature churn across ~15 reader functions + dashboard).
- Phase 3: ~1–2 days, medium risk (consolidating two metadata/scan layers; needs
  the example dashboard + a compare run to verify both consumers still work).

### Recommendation

Do **Phase 1 now** (it removes a real drift bug — two diverged copies of the same
productivity logic). Phases 2–3 are worth it only if the reader layout/robustness
keeps causing friction; otherwise they're cleanup, not bug-prevention.

---

## PRES-04/06 — presentation god-modules

### Current state (facts)

- Dashboard tabs total **~12.4k LOC**; the giants are `performance_charts.py`
  (2181), `energy_balance.py` (1740), `stochastic.py` (1644), `plants.py`
  (1202), `costs.py` (1128).
- Figure boilerplate repeated across the tabs: `go.Figure(` ×61,
  `update_layout(` ×76, `add_trace(` ×89, `to_html(`/`fig_to_html` ×47.
- **Partial infrastructure already exists**:
  - `ui/plotly_helpers.py` — `fig_to_html`, `plotly_div`, `LEGEND_DEFAULTS`,
    `MARGIN_DEFAULTS`, `stage_x_labels`. A figure factory is half-started.
  - `ui/js.py` (420 LOC) — already has shared client JS: `plotlyBand`,
    `plotlyLine`, `plotlyRef`, `plotlyLayout`, `syncHover`, `showTab`,
    `filterTable`, `sortTable`, `initPlantExplorer`, `initComparisonMode`, …
- Embedded `<script>` f-string JS still lives inline in **6 tabs** (network,
  constraints, stochastic, costs, plants, energy_balance), some of it
  duplicating the `ui/js.py` plotly helpers.

### Proposed approach (phased)

- **Phase 1 — finish the figure factory** _(small, incremental, low risk)_.
  Add a `make_figure(traces, *, layout_overrides=None, height=...) -> go.Figure`
  (or a thin builder) to `ui/plotly_helpers.py` that encapsulates the
  `go.Figure()` + `update_layout(LEGEND/MARGIN/template defaults)` + `fig_to_html`
  sequence. Apply it **a few tabs at a time** (start with `performance_charts.py`,
  the worst offender). Each tab is an independent commit; the rendered HTML is
  visually comparable before/after.

- **Phase 2 — consolidate embedded JS into `ui/js.py`** _(medium; needs visual
  verification)_. Move the inline `<script>` blocks out of the 6 tabs into named
  `ui/js.py` functions (reusing the existing `plotly*`/`sync*` helpers). Risk is
  higher because it changes client-side behavior in the rendered HTML — verify
  with the `verify`/Playwright path on the example dashboard, not just pytest
  (pytest only checks the HTML contains expected substrings).

### Effort / risk

- Phase 1: ~1–2 days spread over commits, low risk per tab (pure rendering;
  pytest asserts on HTML substrings, and the figure factory is opt-in per call
  site). Big readability win (kills the 61× `go.Figure`/76× `update_layout`
  repetition).
- Phase 2: ~1–2 days, medium risk (client JS behavior). Needs browser verification.

### Recommendation

Do **Phase 1 incrementally** (one tab per commit, starting with
`performance_charts.py`) — it's the highest readability ROI and low risk. Defer
**Phase 2** unless the inline-JS duplication causes a real bug, since it carries
client-side-behavior risk and needs browser verification each time.

---

## Suggested order across both

1. **ARCH-06 Phase 1** — kill the diverged productivity duplicates (real drift bug).
2. **PRES Phase 1** — figure factory, one tab at a time (readability ROI).
3. Then decide on the heavier phases (CobreCaseLayout, cobre_io foundation,
   inline-JS consolidation) based on whether they keep causing friction.

All phases keep the suite green and are behavior-preserving except where a
divergence is reconciled (Phase 1 of ARCH-06), which we verify against the
example first.

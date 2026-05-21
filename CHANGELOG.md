# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-05-21

### Added

- New **Network** tab in `compare results` HTML comparing directional
  submarket flow per line: NWLISTOP `int*.out` files vs Cobre
  `output/simulation/exchanges`. Per-line small-multiples with Cobre
  P10/P90 band, NEWAVE mean, and finite capacity bounds (the ±99999
  NEWAVE big-M sentinel for fictitious connections is filtered so it
  doesn't compress real flows into a flat strip).
- New **Performance** tab comparing wall-clock timings: NEWAVE
  `newave.tim` stage totals + per-iteration forward/backward breakdown
  (parsed directly — inewave doesn't expose it) vs Cobre
  `training/convergence.parquet` + `training/metadata.json`. Headline
  cards for total/policy/training duration and a speedup ratio.
- **Hydro Operation** tab now opens with per-bus facets
  (SUDESTE / SUL / NORDESTE / NORTE) for storage, generation, spillage,
  turbined, inflow, and water value, then stacks the EARM/ENA aggregate
  charts and finally a **System Totals (SIN)** section that brings the
  six original system-aggregate charts back at the bottom.
- **Hydro Details** picker gains four new comparison panels per plant:
  evaporation, water withdrawal, total inflow (≡ NEWAVE `QAFLUH`,
  computed Cobre-side from incremental + upstream cascade outflow), and
  total outflow (turbined + spilled). Generation panel carries a dashed
  Cobre LP `gen_max` overlay. Tab renamed from "Plant Details".
- **Hydro Details** and **Thermal Details** tabs open with a
  per-plant **max relative difference** summary table — one column per
  variable, NEWAVE-referenced, colored green/amber/red at ≤1%/≤10%/>10%
  thresholds, sorted worst-first. Bold last row is a per-column
  **Median** across plants (robust to outliers like LAJES's water-value
  divergence).
- **Overview** tab cards switched to thermal-generation NPV
  (NEWAVE/Cobre/Δ R$/Δ%) — the prior meta cards
  (Total Comparisons / Entity Types / Variables) offered no operational
  insight. The cost-breakdown chart now uses a vertical right-side
  legend (the previous horizontal layout collided with the chart
  title), and the companion table has been restyled with proper CSS.
- New readers: `read_nwlistop_intercambio`,
  `read_cobre_line_means/percentiles`,
  `read_cobre_hydro_total_flows`, `read_cobre_hydro_withdrawal`,
  `read_cobre_lp_max_generation`, `read_cobre_spillage_energy`,
  `read_cobre_training_duration`, `read_cobre_iteration_timing`,
  `read_newave_tim_iterations`, `read_newave_tim_stages`.

### Changed

- `compare results` reconstructs _realized_ water-withdrawal and
  evaporation on both sides via the LP slack convention
  `realized = scheduled + violation_pos − violation_neg`. Sign matches
  Cobre's `lp_builder/matrix.rs` water-balance row. Applied symmetrically
  to NEWAVE's `VRETIRUH`/`VEVAPUH` + `VIOL_POS/VIOL_NEG` so the
  comparison is realized-vs-realized.
- NEWAVE `VEVAPUH` and `VRETIRUH` (reported in hm³/month, not flow) are
  divided by 2.63 — NEWAVE's rounded `730 h × 3600 s / 10⁶` factor —
  before comparison against Cobre's m³/s values. The converter side
  still uses the exact 2.628 since it operates on input data with no
  analogous rounding.

- `convert_water_withdrawal` honours `dger.outros_usos_da_agua`: when
  the flag is `0`, the converter short-circuits and emits no
  withdrawal rows, matching NEWAVE's own behaviour (the solver
  ignores `dsvagua.dat` regardless of its content when the flag is
  off). When the flag is `1` (the default) it proceeds normally.

### Fixed

- `convert_water_withdrawal` was treating `dsvagua.dat::codigo_usina`
  as a **posto** and routing it through `confhd.dat`'s posto→plant
  map. The field is actually the **plant code** directly. The
  miscoding swapped data between any plant pair whose code/posto values
  collided (e.g. PICADA's plant code = SIMPLICIO's posto = 126 in the
  bundled case) and silently dropped entries for plants whose code
  didn't coincidentally exist as a posto. On the bundled example,
  SIMPLICIO's withdrawal target moved from 1.09 m³/s to 89.71 m³/s,
  dropping the `withdrawal_m3s` mean abs diff from 4.17 to 0.0009 m³/s
  (r = 0.65 → 1.00). Users with `dsvagua.dat` in their case should
  regenerate after upgrading.
- Network tab small-multiples y-domain formula assumed ≤2 rows and
  positioned panels 5–6 outside the chart area for cases with 5–6
  lines. Now distributes any number of rows evenly across `[0, 1]`.

### Removed

- The dashboard-style Cobre-only capacity-utilisation heatmap from the
  Network tab — a Cobre-only view doesn't belong in the comparison.
- The "System Spillage (Energy Units)" section from the Hydro
  Operation tab — superseded by the new per-bus facets.
- The NEWAVE `GHMAX_FPHC` overlay trace from the Hydro Details
  generation panel — wasn't helping interpretation.

### Notes (carried forward from earlier "Unreleased")

- `convert_non_controllable_sources` emits
  `"allow_curtailment": false` on every NCS entity derived from
  `sistema.dat::geracao_usinas_nao_simuladas`. NEWAVE pre-nets these
  aggregates (PCH, PCT, EOL, UFV, MMGD) from MERC before the dispatch
  LP runs, which makes them effectively must-run; setting
  `allow_curtailment=false` instructs Cobre's LP to pin dispatch to
  the realized per-scenario availability instead of leaving curtailment
  as a cheap LP slack. On the bundled deterministic 1983 case this
  restores parity with NEWAVE — eliminates ≈ 18 % of total NCS supply
  being artificially curtailed, a ≈ +15 % hydro-dispatch swing, and a
  ≈ −23 % spillage divergence. Requires Cobre with the
  `non_controllable_sources.allow_curtailment` field. See
  `docs/findings/ncs-must-run-treatment.md`.
- Regenerated `example/convertido/system/non_controllable_sources.json`
  to reflect the new emission. The 32 NCS aggregates in the bundled
  NEWAVE-derived case now carry `"allow_curtailment": false`; all
  other fields are unchanged.

## [0.6.1] - 2026-05-18

First public release for the cobre v0.6.x line — bundles every change
since v0.5.1 (the v0.6.0 milestone was never published to PyPI).

### Added

- **Cobre v0.6 compatibility**: schema migration to
  `hydro_production_models.json` + `hydro_energy_productivity.parquet`,
  per-(hydro, stage) productivity overrides for CFUGA/CMONT temporal
  changes, scalar parameters, and `scalar_parameters.json` emission.
- **Energy-based hydro outputs**: dashboards and comparison reports now
  surface EARM (stored energy, MWh) and ENA (natural energy from
  incremental inflow, MW) at system, per-bus, and per-plant level.
- **FICT-plant cascade resolution**
  (`src/cobre_bridge/converters/fict_cascade.py`): walks NEWAVE
  fictitious-plant chains so that real plants whose energy cascade
  traverses `FICT.<NAME>` topology bridges (e.g. `TRES MARIAS → FICT.TRES
MA → SOBRADINHO`) are correctly wired in `hydros.json::downstream_id`.
  Restores cobre `ρ_acum` to within 0.004 of NEWAVE's
  `produtibilidade_acumulada_calculo_earm` on the bundled case (was off
  by up to 2.77 MW/(m³/s) on 7 plants). Includes ambiguity warnings
  when the 7-char name-truncation key is shared by multiple real plants.
- **PAR(p)-A `order_selection` mapping**: `dger.consideracao_media_anual`
  → `"pacf"` (classic PAR(p)) or `"pacf_annual"` (NEWAVE option 3) so
  cobre's stochastic estimator matches NEWAVE's configured method.
- **Per-stage VminOP RHS**: bound now uses per-stage ρ_acum (built from
  per-stage own productivities) rather than a static base productivity,
  so the curva.dat percentage targets translate to correct absolute
  bounds for plants with CFUGA temporal overrides.
- **Lighter dashboards**: cached chart data, reduced JSON payload, and
  smaller box-plot point counts in the performance tab. ~40 %
  reduction in HTML file size on a 60-stage case.
- **Comparison-report energy charts**: new aggregate-vs-NEWAVE SIN
  comparison of EARM and ENA in the Hydro Operation and Energy Balance
  tabs, plus per-plant cobre-only entries in the Plant Details tab.
- Findings documentation in `docs/findings/`:
  `evaporation-unidirectional-q_ev.md` (handoff that drove the cobre
  v0.6.1 hotfix) and `fict-cascade-resolution.md` (rationale and
  quantitative verification of the cascade fix).

### Changed

- **`cobre-python>=0.6.1`** in the `validation` extra — required so
  `cobre-bridge convert --validate` picks up cobre's signed `Q_ev`
  evaporation fix.
- **NEWAVE-aligned penalty conversion**
  (`src/cobre_bridge/converters/network.py`): full rewrite of the
  PENALID-to-cobre mapping using the four-family taxonomy from the
  NEWAVE manual v29 §3.24. Documented module-level constants for
  ρ_avg / ρ_max_acum derivation, `_EVAPORATION_MULT = 1.1` (down from a
  literal 10× per the manual to keep the LP coefficient range below
  HiGHS's 1e10 conditioning warning), tie-breaking factors on
  flow-domain slacks that would otherwise share ρ_avg, and a 100×
  uplift on NEWAVE micro-penalties to lift the LP coefficient floor
  off HiGHS's noise threshold.
- **Cost Breakdown chart** in the comparison report now truncates
  Cobre to NEWAVE's reported stage range so the totals compare
  like-for-like when Cobre simulates a longer horizon.
- **All comparison-report charts** now use `hovermode: 'x unified'`
  matching the dashboard convention so NEWAVE-vs-Cobre lines snap to
  the same x-tooltip.
- **Truncation to common-stage horizon** applied to every chart in the
  comparison report (energy balance, hydro operation, plant details)
  so visualizations never show unmatched Cobre-only stages.

### Fixed

- **Evaporation penalty conversion**: restored the `× ρ_max_acum`
  factor on `evaporation_violation_cost` (was dropped when the
  docstring on cobre's side appeared to suggest hm³ units), and
  derived `inflow_nonnegativity_cost` as 1 % above the strictest
  flow-domain slack so the slack is the LP's last resort.
- **`config.json::modeling.inflow_non_negativity.penalty_cost`** is no
  longer emitted — cobre treats this field as deprecated and reads the
  live value from `penalties.json::hydro.inflow_nonnegativity_cost`.
  The legacy field stayed in the converted output and confused users
  about which value the LP would actually use.

## [0.5.1] - earlier

## [0.5.0] - earlier

## [0.4.x] - earlier

See git history (`git log v0.4.0..v0.5.1`) for the 0.4 / 0.5 entries —
those were never recorded in this CHANGELOG.

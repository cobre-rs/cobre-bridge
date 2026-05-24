# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-05-24

### Added

- **GNL anticipated thermal dispatch**. New `converters/anticipated.py`
  reads `adterm.dat` (gated by `dger.despacho_antecipado_gnl`) and
  aggregates per-(thermal, lag, patamar) MW values via a
  block-duration-weighted mean, preserving total committed MWh under
  Cobre's constant-MW-per-stage convention. Each thermal now declares
  `anticipated_config` (with `lead_stages`) in `thermals.json`,
  replacing the dead `gnl_config: null` field. `adterm` is added as an
  optional file to `NewaveFiles` resolved via `arquivos.dat`.
- **Head-corrected turbined-flow cap** for hydro plants. NEWAVE's
  effective turbination cap applies an affinity-law correction with the
  volume-integrated head and a `pinst/prodt` clamp, not the nameplate
  `Σ(n·q_nom)`. New `_compute_max_turbined_hypothesis` implements

      qtur_max = min(Σ_c n_c · q_nom_c · (h_op / h_nom_c)^k_turb,
                     pinst / (ρ_esp · h_int)) · (1 - teif) · (1 - ip)

  with `h_op = mean_cota(V_min, V_65) - cota_jus - perdas` for reservoir
  plants and the machine-count-weighted nominal head for D/F/S plants.
  Reproduces M. DE MORAES's binding peak to within 0.0001 % across all
  28 stages, closing a persistent ~108 m³/s gap that distorted hydro
  operation. Legacy `_compute_max_turbined_simple` preserved as
  fallback when `hidr.dat` columns are missing.

- **PIMENTAL → BELO MONTE diversion** (13 000 m³/s nameplate)
  restored. Without the explicit channel the Cobre LP had nowhere to
  route the upstream water NEWAVE accounts for via the fictitious-plant
  cascade, producing spurious spillage and downstream starvation.
- **NC plant support** in converters.
- **Operational-slack visibility** in both `cobre-bridge dashboard`
  and `cobre-bridge compare results`:
  - Plant-detail tab now plots `water_withdrawal_violation_{pos,neg}`
    and `inflow_nonnegativity_slack` with p10/p50/p90 bands, so the
    user can localize which plants/stages the LP had to relax under
    stochastic noise.
  - Plant-detail tab overlays NEWAVE `VIOL_POS/NEG_VRETIRUH` on the two
    withdrawal-slack panels (converted /2.63 from hm³ to m³/s),
    matching NEWAVE's sign convention.
  - Hydro Operation tab gains per-bus + SIN-total aggregates for all
    four paired hydro slacks (withdrawal + evaporation) plus the
    Cobre-only inflow non-negativity slack — driven by
    per-(entity_id, stage_id) frames since slacks don't go through the
    `ResultComparison` pipeline.
  - Dashed bound overlays for storage / generation / turbined / outflow
    on the compare-results tab (the dashboard tab already had them).
    Static values come from `hydros.json` via
    `read_cobre_hydro_metadata` and are shadowed per-stage by overrides
    from `constraints/hydro_bounds.parquet` via the new
    `read_cobre_hydro_per_stage_bounds` reader.
- **Risk-measure selection logging** in `convert_stages`. INFO log at
  the top names the selected mode (expectation / constant-CVaR /
  per-stage CVaR, from `dger.cvar`) and resolved alpha/lambda when CVaR
  is in play. Per-stage branch refactored into four explicit cases
  mirroring the log; output is bit-identical.
- **Spec doc** `docs/findings/cobre-anticipated-thermal-pre-horizon-
limitation.md` for the Cobre maintainers: self-contained spec of the
  "non-zero `past_anticipated_commitments` rejected" limitation, the
  bridge-side workaround, and the functional/acceptance requirements
  that would let cobre-bridge restore NEWAVE-parity GNL dispatch by
  flipping a single line.

### Changed

- **Default cut selection** now emits `method: "lml1"` with
  `memory_window: 0` (was `domination` + `domination_epsilon: 0.0`).
  Aligns with the `RowSelectionConfig` schema (`memory_window` is
  required for `lml1`).
- **Default inflow non-negativity** now `truncation_with_penalty`:
  clamp negative PAR(p) draws to zero before LP patching and keep the
  non-negativity slack columns as a backstop. Closes the exploit where
  the LP would otherwise route negative inflow noise through the
  withdrawal-neg slack (priced 1 R$/(m³/s) below the nonneg slack on
  the cobre-bridge calibration).
- **`inflow_nonnegativity_cost`** anchored to
  `water_withdrawal_violation_cost + 1` R$/m³/s (was 1.01 × max of
  flow-domain slacks, which made it cheaper than withdrawal and let
  the LP buy "free" water to dodge withdrawal violations).
- **D-regulation plant initial storage** anchored to
  `volume_referencia`, consistent with the collapsed
  `[vmin, vmax] = vref` bounds (NEWAVE freezes D reservoirs across
  stages).
- Stronger guard rails in deterministic mode.

### Fixed

- **Withdrawal slack sign labels** in `compare results` HTML. Cobre's
  `water_withdrawal_violation_pos/neg` columns use the inverse sign
  convention of NEWAVE's `VIOL_POS/NEG_VRETIRUH`; the NEWAVE → Cobre
  column mapping and display labels are swapped so each "Pos" / "Neg"
  panel pairs the columns that mean the same physical violation.
  Evaporation slacks already shared NEWAVE's convention and are left
  as-is.
- **Slack p10/p90 emission**. `read_cobre_hydro_percentiles`'
  `flow_cols` list now includes the three operational slacks added
  above, so the percentile frame ships
  `water_withdrawal_violation_{pos,neg}_m3s_{p10,p90}` and
  `inflow_nonnegativity_slack_m3s_{p10,p90}` columns. Without them,
  `_build_interactive_detail_html` emitted only the bare `Cobre Mean`
  trace per slack chart and the `x unified` hover had nothing to lock
  onto, breaking the band+P10/P90 tooltip every other flow variable
  enjoys.
- **P10–P90 unified-hover tooltip**: every band-trace site (5 in
  `charts.py` plus the interactive plant-detail JS) now sets
  `hoverinfo: "skip"` on the closing-polygon trace, and the visible
  p10/p90 lines are renamed to "Cobre P10" / "Cobre P90". Previously
  the unified hover showed the literal text `Cobre P10–P90` instead of
  values at the cursor x.
- **`past_anticipated_commitments` rejected by Cobre validator**.
  `convert_initial_conditions` now zeroes the `values_mw` array
  (length still `lead_stages`, as Cobre requires) and emits a WARNING
  naming the `adterm.dat` code and the MW values being dropped — so
  the user knows exactly what pre-horizon NEWAVE dispatch is not being
  honoured. `read_anticipated_dispatch` keeps computing the true
  block-weighted MW so the warning is informative; the zero-out
  policy lives at the conversion site, not in the reader. Flipping a
  single line restores genuine values when Cobre lifts the limitation.
- Miscellaneous converter fixes uncovered during the `pmo_set_24`
  stochastic-case investigation.

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

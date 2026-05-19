# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

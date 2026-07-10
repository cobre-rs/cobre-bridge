# Design: `NE` hydro support — dead-volume filling & generating-unit expansion

Status: **Draft for review** (pre-implementation)
Author: conversion team
Scope: `cobre-bridge convert newave` only
Target solver: **cobre `develop` branch** (filling schema is unreleased; see §9)

---

## 1. Motivation

NEWAVE classifies every hydro plant in `confhd.dat` by a two-letter status in
the `usina_existente` column:

| Status | Meaning                                                                                                     | Bridge today      |
| ------ | ----------------------------------------------------------------------------------------------------------- | ----------------- |
| `EX`   | Existing / in operation                                                                                     | Converted         |
| `NC`   | Future, _not considered_ (ignored by NEWAVE)                                                                | Dropped           |
| `NE`   | Future, _will be built_ — has a dead-volume filling schedule and gradual generating-unit entry (`exph.dat`) | **Dropped today** |

Until now every `NE` plant in our reference cases was manually rewritten to
`NC` so both NEWAVE and the bridge ignored it identically. cobre's `develop`
branch now models **dead-volume filling** and **entry/exit stages** as
first-class concepts, so we can convert `NE` plants faithfully instead of
discarding them.

The example case `example/newave_rodada_2001` was added to exercise this: it
contains exactly one `NE` plant, **309 JURUENA**.

---

## 2. NEWAVE inputs

### 2.1 `confhd.dat` — the `NE` status

Read via inewave `Confhd().usinas`; the status lives in column
`usina_existente` (values `EX` / `NC` / `NE`). `NE` plants carry the same
columns as any other (`codigo_usina`, `posto`, `codigo_usina_jusante`, `ree`,
`volume_inicial_percentual`, history years).

### 2.2 `exph.dat` — the filling + expansion schedule

Read via inewave `Exph().expansoes` (a single tidy DataFrame). Rows are grouped
per plant: the **first row** of a plant carries the _filling_ schedule, the
**following rows** carry one _generating-unit entry_ each.

| Column                     | Meaning                                                                | Used for                                 |
| -------------------------- | ---------------------------------------------------------------------- | ---------------------------------------- |
| `codigo_usina`             | plant code                                                             | key                                      |
| `data_inicio_enchimento`   | filling start (`MM/YYYY`)                                              | filling `start_stage_id`                 |
| `duracao_enchimento`       | filling duration (months)                                              | filling window length → `entry_stage_id` |
| `volume_morto`             | **fraction of the dead volume already impounded at filling start** (%) | `filling_storage` seed                   |
| `data_entrada_operacao`    | unit entry date (`MM/YYYY`)                                            | capacity-ramp stage                      |
| `potencia_instalada`       | power of the entering unit (MWmed)                                     | (sanity check vs `hidr`)                 |
| `maquina_entrada`          | entering machine number                                                | online-unit accounting                   |
| `conjunto_maquina_entrada` | machine group (`conjunto`) of the entering unit                        | online-unit accounting                   |

The dead-volume **magnitude** is _not_ in `exph.dat`; it comes from `hidr.dat`
(`volume_minimo` — see §4.2).

### 2.3 The example, raw

`confhd.dat` (only `NE` row shown):

```
  309 JURUENA       226     0    7   0.00   NE      1     1931     2022
```

`exph.dat`:

```
 309 JURUENA      10/2024       1      0.0
                                             1/2025   25.0   1  1
                                             1/2025   25.0   2  1
```

`hidr.dat` (plant 309): `volume_minimo = volume_maximo = 2.93 hm³`
(run-of-river — no useful storage), `numero_conjuntos_maquinas = 1`,
`maquinas_conjunto_1 = 2`, `potencia_nominal_conjunto_1 ≈ 25 MW`,
`vazao_nominal_conjunto_1 = 83 m³/s`, `queda_nominal_conjunto_1 = 34.3 m`.

`dger.dat`: study starts **Sep 2024**, 3-year (36-stage) monthly horizon.

---

## 3. Cobre target contract (`develop`)

A filling hydro traverses **three phases**, keyed off two **0-based, inclusive**
stage indices on the `Hydro` entity:

```
PreFilling: stage < start_stage_id          dam absent; storage frozen at 0;
                                             inflow short-circuited downstream
Filling:    start_stage_id ≤ stage < entry  impounds toward min_storage_hm3;
                                             generation AND turbined forced to 0;
                                             evaporation kept; soft per-stage
                                             target floor anchored backward
                                             from (entry_stage_id − 1) = min_storage
Operating:  stage ≥ entry_stage_id           normal hydro
```

Relevant `hydros.json` fields (cobre-io `RawHydro` / `RawFillingConfig`):

```jsonc
{
  "id": 0,
  "name": "JURUENA",
  "bus_id": ...,
  "downstream_id": null,
  "entry_stage_id": 2,          // i32|null, 0-based inclusive; null ⇒ from stage 0
  "exit_stage_id": null,        // i32|null, 0-based inclusive; null ⇒ never exits
  "reservoir": { "min_storage_hm3": 2.93, "max_storage_hm3": 2.93 },
  "generation": { "model": "...", "max_turbined_m3s": 166.0, "max_generation_mw": 50.0, ... },
  "filling": {
    "start_stage_id": 1,        // i32, 0-based inclusive; Filling begins here
    "filling_min_rate_m3s": 1.09 // per-stage MINIMUM accumulation floor [m³/s], default 0
  }
}
```

`initial_conditions.json` — a filling hydro is seeded via **`filling_storage`**,
**not** `storage` (cobre validation: exactly one of the two per hydro). cobre-io's
`RawHydroStorage` keys **both** `storage` and `filling_storage` on `value_hm3`:

```jsonc
{ "filling_storage": [{ "hydro_id": 0, "value_hm3": 0.0 }] }
```

Stage-varying capacity is expressed via `constraints/hydro_bounds.parquet`
(cobre-io `HydroBoundsRow`), which supports per-`(hydro_id, stage_id)`
`max_turbined_m3s`, `max_generation_mw`, `min_generation_mw`, plus
`filling_min_rate_m3s` overrides. Capacity ramp is **not** a modelled
rate-of-change constraint — it is just a series of per-stage ceilings.

---

## 4. The mapping — two independent subsystems

> **Key decision (from review):** dead-volume _filling_ and generating-unit
> _commissioning_ are **separate** mechanisms and must not be conflated.
> `entry_stage_id` marks the end of **filling** (when the reservoir becomes an
> operating node), **not** the unit-entry date. Unit commissioning is expressed
> _entirely_ through per-stage capacity bounds.

### 4.1 Subsystem A — dead-volume filling (from the exph _filling_ row)

| cobre field                      | NEWAVE source                                   | Formula                                                          |
| -------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| `filling.start_stage_id`         | `data_inicio_enchimento`                        | `stage_id(data_inicio_enchimento)` (clamp to 0 if ≤ study start) |
| `entry_stage_id`                 | `data_inicio_enchimento` + `duracao_enchimento` | `stage_id(data_inicio) + duracao_enchimento`                     |
| `reservoir.min_storage_hm3`      | `hidr.volume_minimo`                            | dead volume = the floor the reservoir must reach                 |
| `filling.filling_min_rate_m3s`   | dead vol, `volume_morto`, window ζ              | `remaining / Σ_{t∈[start,entry)} ζ_t` (see below)                |
| `filling_storage.value_hm3` (IC) | `volume_morto`                                  | `(volume_morto/100) × min_storage_hm3`                           |

where:

- `remaining = min_storage_hm3 − filling_storage.value_hm3 = min_storage_hm3 × (1 − volume_morto/100)`
- `ζ_t` (hm³ per m³/s for stage `t`) `= _month_hours(year_t, month_t) × 3600 / 1e6`
  (same calendar-hours basis as `temporal._month_hours`)

A constant `filling_min_rate_m3s = remaining / Σ ζ_t` makes the soft target
trajectory reach `min_storage_hm3` exactly at `entry_stage_id − 1`, i.e. a
**linear fill over `duracao_enchimento` months** — matching NEWAVE. The precise
per-stage anchor index in cobre (`ζ_{t}` vs `ζ_{t+1}`) is verified against
`cobre-sddp` `layout.rs` at implementation; for a single-stage window the target
is pinned to `min_storage_hm3` regardless of rate.

### 4.2 Subsystem B — generating-unit expansion (from the exph _unit_ rows)

The plant **exists as an operating reservoir node** from `entry_stage_id`
(Subsystem A), but its turbine/generation capacity is whatever **units are
online** at each stage. cobre already forces capacity to 0 during Filling/
PreFilling; we nonetheless **export** explicit 0/reduced caps over the **full
pre-operating window** so the exported bounds are plottable (see "Override rows"
below).

For each operating stage `s` and each machine group `c`:

```
online_machines(c, s) = # machines of group c whose data_entrada_operacao maps
                         to a stage ≤ s  (clamped so no unit is online before
                         entry_stage_id — water can't be turbined before fill)
```

Then per-stage caps are computed **with the existing operating-head logic, but
with the reduced machine counts** (reusing `_compute_max_turbined_head_corrected`
/ `_compute_max_turbined_rated` over the online subset):

- `max_turbined_m3s[s] = Σ_c online_machines(c,s) · q̂(c)` (head-corrected engolimento)
- `max_generation_mw[s] = Σ_c online_machines(c,s) · p_nom(c)`
- `min_generation_mw[s] = 0` while capacity is below full (cannot force generation)

Override rows are emitted for the **full pre-operating window** `[0,
all-units-online stage)`. `online_machines` clamps every unit's online stage up
to `entry_stage_id`, so the PreFilling/Filling stages `[0, entry_stage_id)` get
explicit `(0, 0)` caps, the ramp stages `[entry_stage_id, all-units-online)` get
reduced caps, and from the all-units-online stage the base `hydros.json` caps
(full count) apply, so no further rows are needed. The pre-entry 0-cap rows are
**for observability only** — a bounds plot reads the parquet directly and would
otherwise show NaN for the filling/pre-entry stages. They match cobre's internal
PreFilling/Filling forcing (capacity is already 0 there); cobre's `hydro_bounds`
reader is a **sparse override table with no stage-window validation**, so a
`max=0` row during PreFilling/Filling is inert to its internal forcing and the
**simulation result is unchanged**. (`max_generation_mw` is a **new column** in
our `hydro_bounds.parquet`; today we emit only `min_generation_mw`.)

---

## 5. Worked example — JURUENA (309)

Study start Sep 2024 = stage 0. `stage_id(m/Y) = (Y−2024)·12 + (m−9)`.

| Event                                            | Date     | Stage |
| ------------------------------------------------ | -------- | ----- |
| Study start                                      | Sep 2024 | 0     |
| Filling start (`data_inicio_enchimento`)         | Oct 2024 | 1     |
| Filling complete = `entry_stage_id` (`+1` month) | Nov 2024 | 2     |
| Units enter (`data_entrada_operacao`, ×2)        | Jan 2025 | 4     |

Emitted:

```jsonc
// hydros.json
"entry_stage_id": 2,
"exit_stage_id": null,
"reservoir": { "min_storage_hm3": 2.93, "max_storage_hm3": 2.93 },
"generation": { "max_turbined_m3s": 166.0, "max_generation_mw": 50.0, ... },  // full 2 units
"filling": { "start_stage_id": 1, "filling_min_rate_m3s": 2.93 / 2.6784 ≈ 1.09 }
// ζ(Oct 2024, 31d) = 744·3600/1e6 = 2.6784

// initial_conditions.json
"filling_storage": [ { "hydro_id": <id>, "value_hm3": 0.0 } ]   // volume_morto = 0%
```

```
# hydro_bounds.parquet (full pre-operating window [0, full_online) = [0, 4))
hydro_id stage_id max_turbined_m3s max_generation_mw min_generation_mw
<id>     0        0.0              0.0               0.0     # PreFilling, 0 units online
<id>     1        0.0              0.0               0.0     # Filling, 0 units online
<id>     2        0.0              0.0               0.0     # operating-but-idle, 0 units
<id>     3        0.0              0.0               0.0
# stage 4+ : 2 units online ⇒ base caps apply, no override needed
# (stages 0–1 mirror cobre's internal PreFilling/Filling forcing — exported for
#  plottability; cobre ignores them as a sparse override, result unchanged)
```

Resulting phases: PreFilling = stage 0 (dam absent, inflow → downstream),
Filling = stage 1 (impound 2.93 hm³, no generation), Operating-but-idle =
stages 2–3 (reservoir exists, 0 capacity, spills inflow), Full operation =
stage 4+.

---

## 6. Converter touch points

Admitting `NE` plants makes them real cascade/inflow nodes, so the change
reaches beyond `hydro.py`:

| #   | File                                          | Today                                                         | Change                                                                                                                                  |
| --- | --------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `newave_files.py`, `case.py`                  | `exph.dat` not discovered/loaded                              | Discover `exph.dat` via `arquivos.dat`; add inewave `Exph` to `CaseContext`                                                             |
| 2   | `plants.py`                                   | active set = `EX` only (`existing_hydros`)                    | Admit `NE` plants that have an `exph` filling row (new helper, e.g. `filling_hydros`; fold into `active_hydros`)                        |
| 3   | `id_map.py`                                   | enumerates `EX` active hydros in declaration order            | Include `NE` plants in confhd declaration order (dense IDs shift — acceptable, 2001 is a brand-new case with no prior converted output) |
| 4   | `converters/fict_cascade.py`                  | `status != "EX"` ⇒ "absent" pass-through                      | Reclassify `NE`-with-`exph` as a **real generating node** in the cascade walk                                                           |
| 5   | `converters/stochastic.py`                    | inflow `posto` map built for `EX` only                        | Include `NE` plants' `posto` (they receive inflow during filling/operation)                                                             |
| 6   | `converters/hydro.py::convert_hydros`         | `entry_stage_id`/`exit_stage_id` hard-coded `None`; full caps | Set `entry_stage_id` (= fill-complete stage); emit `filling { start_stage_id, filling_min_rate_m3s }` for `NE` plants                   |
| 7   | `converters/hydro.py::convert_storage_bounds` | per-`(hydro,stage)` bounds; no `max_generation_mw` column     | Emit ramp-window `max_turbined_m3s` / `max_generation_mw` (and `min_generation_mw=0`) overrides; **add `max_generation_mw` column**     |
| 8   | `initial_conditions.py`                       | seeds every active hydro into `storage`                       | Route `NE` plants to **`filling_storage`** (and exclude them from `storage`)                                                            |
| 9   | `pipeline.py`                                 | wiring                                                        | Thread `exph` through; pass online-unit/filling schedules to the hydro + IC converters                                                  |
| 10  | `diagnostics` / `ui`                          | —                                                             | Emit a `Diagnostic` per `NE` plant summarising its filling window + unit ramp                                                           |
| 11  | tests + example                               | —                                                             | Unit + pipeline tests against `newave_rodada_2001`; golden snapshot                                                                     |

---

## 7. Decisions (from review Q&A)

1. **`entry_stage_id` = filling-complete stage**, not the unit-entry date.
   Dead-volume filling and capacity commissioning are independent (§4).
2. **`volume_morto` % = fraction of the dead volume already impounded at filling
   start.** Seed `filling_storage = (volume_morto/100) × min_storage_hm3`
   (JURUENA: 0 % ⇒ empty).
3. **Filling rate honours NEWAVE's declared start + duration**: a constant
   `filling_min_rate_m3s` that fills `remaining` linearly across the
   `duracao_enchimento` window (§4.1).
4. **Process:** this design doc first, then a formal `/plan` master plan.

---

## 8. Edge cases to handle

- **`duracao_enchimento = 0`** (or no dead volume): no Filling phase →
  `start_stage_id == entry_stage_id`. Omit `filling`; the plant simply enters at
  that stage with a 0-capacity ramp. Seed via `storage = 0` vs `filling_storage`
  — TBD against cobre IC rules.
- **`data_inicio_enchimento` ≤ study start**: clamp `start_stage_id` to 0
  (then `start_stage_id == 0` ⇒ no PreFilling; seed is the stage-0 filling
  storage). If filling completed before study start the plant would be `EX`, not
  `NE`.
- **Unit `data_entrada_operacao` < `entry_stage_id`**: clamp the unit's online
  stage up to `entry_stage_id` — no turbining before the reservoir is filled.
- **`entry_stage_id` ≥ horizon**: plant fills but never operates within the
  study; still emitted (cobre handles a never-operating filling plant). The
  pre-operating window clamps to `[0, total_stages)`, so every in-study stage
  carries an explicit `0`-cap row.
- **Fictitious detection** (`plants.fictitious_codes`) requires `EX`; `NE`
  plants are never fictitious — no interaction, but re-confirm the filter ignores
  them.
- **`volume_morto > 0` with `start_stage_id > 0`**: confirm whether cobre's
  `filling_storage` IC seeds at stage 0 or at `start_stage_id` (PreFilling
  "seeds empty pit (0)"). Moot for JURUENA (0 %); verify before shipping a
  non-zero seed.

---

## 9. Validation / cobre-python lag

**Resolved as of cobre 0.10.0.** The filling schema (`filling`,
`filling_storage`, `entry/exit` semantics) shipped in cobre 0.9.1, and cobre
0.10.0 additionally made `operational_start_date` a required field on every
system entity. A converted case therefore now requires **cobre >= 0.10.0**
(tracked as `MIN_COBRE_VERSION` in `cli.py`, single source of truth).

`convert newave --validate` validates against an installed `cobre-python >= 0.10`.
When the installed `cobre-python` predates `MIN_COBRE_VERSION`, validation is
**skipped** gracefully (an informational note on stderr, `skipped_reason:
"cobre-python-too-old"` under `--json`) rather than producing a false failure.

---

## 10. Out of scope

- Thermal / other-entity commissioning (NEWAVE `expt.dat` etc.) — cobre's
  entry/exit is generic, but this feature converts **hydro `NE`** only.
- Hydro **decommissioning** (`exit_stage_id`) — not declared by `confhd`/`exph`
  in this flow; left `None`.
- Changing how `EX` or `NC` plants are converted.

---

## 11. Open questions

1. cobre `filling_storage` IC seed timing when `start_stage_id > 0` (§8).
2. Exact per-stage ζ anchor index in cobre's filling-target row (affects the
   `filling_min_rate_m3s` formula for multi-stage windows) — verify against
   `cobre-sddp/src/lp/builder/layout.rs`.
3. Minimum cobre / cobre-python version to record in the conversion manifest and
   the `--validate` gate (§9).

---

## 12. Test strategy

- **Unit:** stage-math helper (date → stage_id); filling-rate formula; online-
  unit accumulation per `(conjunto, stage)`; reduced-count bound computation.
- **Converter:** `convert_hydros` emits correct `filling` + `entry_stage_id` for
  JURUENA; `convert_storage_bounds` emits the 0-capacity rows for the full
  pre-operating window (stages 0–3); `initial_conditions` routes JURUENA to
  `filling_storage`.
- **Pipeline / golden:** full `convert newave example/newave_rodada_2001` →
  snapshot `hydros.json`, `initial_conditions.json`, `hydro_bounds.parquet` for
  plant 309.
- **Negative / regression:** `newave_rodada_2000_completo` (no `NE` plants)
  output is byte-identical to today — `NE` support must not perturb `EX`-only
  cases.

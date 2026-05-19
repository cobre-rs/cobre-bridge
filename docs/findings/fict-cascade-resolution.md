# FICT-plant cascade resolution: EARM and ρ_acum divergence vs NEWAVE

**Audience:** future cobre-bridge maintainers (and reviewers of the fix).
**Status:** Fixed in cobre-bridge `src/cobre_bridge/converters/fict_cascade.py`.
**Severity:** Was producing ~10% lower system EARM than NEWAVE on
NEWAVE-individualized cases with 7+ fictitious plants. Dispatch policy was
not affected — only the per-plant `accumulated_productivity_mw_per_m3s` used
for energy accounting and (indirectly) MAX_PRODTACUM-derived penalty costs.
**Date:** 2026-05-18.

---

## TL;DR

NEWAVE encodes a plant's energy-cascade through "fictitious" plants
(`FICT.<NAME>`) whose role is purely topological. Cobre-bridge filters
fictitious plants out of the LP (correctly — they have no turbine, no
reservoir, and zero `produtibilidade_especifica`) but the pre-fix code also
silently dropped the cascade _link_ they provide. Real plants whose
downstream chain went through a FICT (e.g. `TRES MARIAS → FICT.TRES MA →
SOBRADINHO → ...`) ended up wired as `downstream_id=None` in `hydros.json`,
collapsing their cobre-side `ρ_acum` from the full cascade sum down to
their own ρ_eq.

The fix introduces `cobre_bridge.converters.fict_cascade.resolve_cascade`,
which:

1. Walks any FICT chain between a real plant and the next real plant.
2. Returns the next real plant's NEWAVE code (or `None` for true sea
   sinks).
3. Sums the ρ_eq of every FICT plant traversed so callers can fold the
   contribution back into the upstream real plant's effective ρ_eq.

Both the `hydros.json` `downstream_id` wiring and the productivity helpers
(`compute_base_productivities`, `compute_per_stage_own_productivities`,
`convert_hydro_energy_productivity`) consume the resolver, so the LP
topology and the cascade-summed productivities stay in lockstep.

On the bundled NEWAVE case, the fix moves cobre-side ρ_acum to within
**0.004 MW/(m³/s)** of NEWAVE's `produtibilidade_acumulada_calculo_earm`
for all 67 matched plants (was off by up to 2.77 MW/(m³/s) on TRES MARIAS,
2.77 on SERRA MESA, etc.). The remaining noise is from cobre using
`produtibilidade_equivalente_volmin_volmax` while NEWAVE uses the same
metric for EARM accounting — the residual is numerical, not structural.

---

## How the cascade is encoded in NEWAVE

For each plant `<NAME>`, `confhd.dat` carries:

- `codigo_usina_jusante` — the physical-water downstream plant code, or
  `0` for sea sinks.
- `nome_usina` — the plant name (right-padded / truncated to 12 chars).

When a real plant `<NAME>` does **not** physically flow to a downstream
reservoir (because the downstream is run-of-river, or part of a different
subsystem, or simply not represented in this case), NEWAVE adds a sibling
fictitious plant in confhd called `FICT.<truncated NAME>`:

```text
FICT name           = "FICT." + (real_name[:7] right-stripped)
```

The 7-char limit comes from confhd's 12-character `nome_usina` column.
Examples from the bundled case:

| Real plant  | FICT entry   | FICT's `codigo_usina_jusante` |
| ----------- | ------------ | ----------------------------- |
| TRES MARIAS | FICT.TRES MA | SOBRADINHO                    |
| QUEIMADO    | FICT.QUEIMAD | SOBRADINHO                    |
| MAUA        | FICT.MAUA    | CAPIVARA                      |
| IRAPE       | FICT.IRAPE   | ITAPEBI                       |
| LAJEADO     | FICT.LAJEADO | ESTREITO TOC                  |
| PEIXE ANGIC | FICT.PEIXE A | FICT.LAJEADO (→ ESTREITO TOC) |
| SERRA MESA  | FICT.SERRA M | FICT.CANA BR → ... → LAJEADO  |

The convention is: every FICT plant has `produtibilidade_especifica = 0`
and zero installed turbine/generation, so it contributes only topology and
(in the general case) its volume-integrated ρ_eq, which is typically zero
because ρ_esp is zero.

A real plant `<NAME>`'s _effective_ downstream for energy accounting is:

- The plant pointed to by `codigo_usina_jusante` if that's a real plant.
- Otherwise, the next real plant reached by walking `FICT.<NAME>`'s
  `codigo_usina_jusante` chain (which may itself traverse multiple FICTs).
- `None` if the chain terminates at a sea sink.

NEWAVE's `pmo.dat::produtibilidades_equivalentes` reports
`produtibilidade_acumulada_calculo_earm` per real plant as the cascade
sum over this effective chain. The pre-fix cobre-bridge collapsed the
chain at the first FICT, producing an artificially low value.

---

## The pre-fix bug

`src/cobre_bridge/converters/hydro.py:506-516` (before the fix):

```python
downstream_id: int | None = None
jusante_raw = row.get("codigo_usina_jusante")
if (
    jusante_raw is not None
    and not _is_na(jusante_raw)
    and int(jusante_raw) != 0
):
    try:
        downstream_id = id_map.hydro_id(int(jusante_raw))
    except KeyError:
        pass            # <-- silently drops FICT downstreams
```

`id_map.hydro_id()` only knows real plants (FICTs are filtered at
`pipeline.py:_build_id_map`), so any FICT code raised `KeyError` and the
plant was wired as terminal.

The same pattern lived in `src/cobre_bridge/converters/constraints.py
::_build_hydro_downstream_map`, which is the source of cascade topology
for `compute_accumulated_productivities` (penalty conversion) and
`compute_per_stage_acc_productivities` (VminOP RHS). The MAX_PRODTACUM_SIN
scalar that drives `_EVAPORATION_MULT * MAX_DEFICIT * ρ_max_acum` would
therefore _also_ be affected, though in the bundled case G.P. SOUZA
happens to have no FICT in its chain so MAX was already correct.

---

## How the fix is structured

### `cobre_bridge.converters.fict_cascade`

A new module owns all the cascade-resolution logic. It exposes:

```python
@dataclass(frozen=True)
class FictCascadeResolution:
    downstream_code: int | None      # next real plant, or None for sea sink
    fict_chain: tuple[int, ...]       # FICT codes traversed along the path
    fict_rho_sum: float               # cumulative ρ_eq of those FICTs

def resolve_cascade(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame,
) -> dict[int, FictCascadeResolution]:
    ...
```

`resolve_cascade` applies three resolution rules:

1. `codigo_usina_jusante` points to a real plant → use it directly.
2. `codigo_usina_jusante` points to a FICT plant → walk that FICT's
   `codigo_usina_jusante` chain through any further FICTs until a real
   plant or terminal (0) is reached. Accumulate the ρ_eq of every FICT
   traversed.
3. `codigo_usina_jusante` is 0 (physically terminal) **and** a
   name-matched FICT exists (using the 7-char truncation rule) → use that
   FICT as the implicit downstream and apply rule 2 from there.

### Name-match ambiguity

Several real plants share the same 7-char prefix in the bundled case:

| 7-char prefix | Real plants                        |
| ------------- | ---------------------------------- |
| `BARRA B`     | BARRA BRAUNA, BARRA BONITA         |
| `CANOAS`      | CANOAS I, CANOAS II                |
| `CAPIM B`     | CAPIM BRANC1, CAPIM BRANC2         |
| `CORUMBA`     | CORUMBA I, CORUMBA III, CORUMBA IV |
| `ESTREIT`     | ESTREITO, ESTREITO TOC             |
| `ITIQUIR`     | ITIQUIRA I, ITIQUIRA II            |
| `STA CLA`     | STA CLARA MG, STA CLARA PR         |
| `STO ANT`     | STO ANTONIO, STO ANT JARI          |

None of these had a FICT counterpart in this case. When a _future_ case
ships both an ambiguous prefix and a matching FICT, the resolver:

- Detects the ambiguity (`real_by_key[key]` has > 1 entry).
- Logs a warning naming the candidates.
- Refuses to attribute the FICT to either real plant, leaving both as
  terminal (`downstream_code = None`).

This is the conservative choice — better a known-broken EARM on a single
plant than silently misattributing the cascade to the wrong real plant.
The warning surfaces the problem so a human can fix the FICT mapping
manually (or extend the resolver with a deterministic tiebreaker if a
robust convention is identified).

### Wiring on the cobre-bridge side

Three points consume the resolver:

- `convert_hydros` (`src/cobre_bridge/converters/hydro.py`) reads the
  resolution and writes the effective real-plant downstream code into
  `hydros.json::downstream_id`.
- `convert_hydro_energy_productivity`,
  `compute_per_stage_own_productivities`, and
  `compute_base_productivities` (same file) fold the resolution's
  `fict_rho_sum` into each real plant's emitted `ρ_eq` so that cobre's
  cascade sum at solve time reproduces NEWAVE's
  `produtibilidade_acumulada_calculo_earm`.
- `_build_hydro_downstream_map` and `compute_accumulated_productivities`
  (`src/cobre_bridge/converters/constraints.py`) consume the same
  resolution so cobre-bridge's internal cascade computations (penalty
  conversion, VminOP RHS) stay consistent with the LP topology emitted
  for cobre.

### A note on per-stage productivity overrides

A plant with CFUGA/CMONT overrides emits one row per stage in
`hydro_energy_productivity.parquet` (no per-hydro default `stage_id IS
NULL` row). The FICT `ρ_eq` contribution is added to the **base**
productivity _before_ the per-stage adjustments are applied
(`_per_stage_productivities` only modifies what's passed in), so the
fold-in propagates uniformly across all stages.

---

## Quantitative verification

On the bundled NEWAVE case (155 real plants, 10 FICT plants), comparing
cobre's `accumulated_productivity_mw_per_m3s` at stage 0 against NEWAVE's
`produtibilidade_acumulada_calculo_earm`:

|                            | Before fix |  After fix |
| -------------------------- | ---------: | ---------: |
| Plants matched             |         67 |         67 |
| Plants with `\|Δ\|` > 0.01 |          7 |      **0** |
| Max `\|Δ\|`                |       2.77 |  **0.004** |
| Mean `\|Δ\|`               |       0.19 | **0.0002** |

Per-plant rewires emitted by the converter:

```
hydro 13  (TRES MARIAS):  None -> SOBRADINHO
hydro 14  (QUEIMADO):     None -> SOBRADINHO
hydro 28  (LAJEADO):      None -> ESTREITO TOC
hydro 65  (IRAPE):        None -> ITAPEBI
hydro 106 (MAUA):         None -> CAPIVARA
```

Other plants whose ρ_acum changed because they sit upstream of these
five (e.g. RETIRO BAIXO → TRES MARIAS, SAO SALVADOR → PEIXE ANGIC →
LAJEADO → ESTREITO TOC → TUCURUI) pick up the corrected chain
automatically through cobre's downstream-walking accumulation.

In this case all 10 FICT plants have `produtibilidade_especifica = 0`, so
`fict_rho_sum = 0` everywhere and the fold-in is structurally a no-op.
That keeps the diff small and isolates the fix to pure topology. The
fold-in path is exercised by the unit tests with a synthetic non-zero
FICT productivity.

---

## Regression tests

`tests/test_entity_conversion.py::TestConvertHydrosDownstreamFict`:

- `test_downstream_to_fict_is_none` (pre-existing) — when confhd points
  to a FICT plant that does not exist anywhere in the dataframe, the
  resolver returns `None` (graceful fallback).
- `test_terminal_plant_with_matching_fict_resolves_through_chain` (new)
  — verifies a real plant with `jusante=0` whose name is matched by a
  FICT in confhd is rewired to the FICT's downstream real plant.

The full suite remains green at 835 tests after the fix.

---

## Operational warnings

The resolver emits two warnings worth knowing about:

1. **Ambiguous prefix with matching FICT**:

   ```
   FICT-cascade resolution for '<plant>' (code N) is ambiguous: its
   7-char truncation '<key>' is also shared by [...], and a matching
   FICT plant exists. Leaving downstream as terminal to avoid wrong
   attribution.
   ```

   Action: inspect confhd, decide which real plant the FICT actually
   represents, either rename one of the conflicting reals to break the
   prefix collision or extend the resolver with a tiebreaker.

2. **Orphan FICT plant**:

   ```
   FICT plants not matched to any real plant by 7-char prefix: [...].
   These plants contribute no cascade attribution to a real plant.
   ```

   Action: indicates a FICT plant whose name does not follow the
   `FICT.<truncated_real_name>` convention. Either the case uses a
   non-standard FICT name (manually crafted), or there's a typo in
   confhd. In either situation that FICT's downstream is silently
   ignored; if it lies on a meaningful cascade the user must rewire by
   hand.

Both warnings are scoped to `cobre_bridge.converters.fict_cascade` so
they can be filtered with the standard `logging` configuration.

---

## What this fix does not address

- **Cobre-side schema**: cobre still has no first-class representation
  of a FICT-style topology bridge. If a future case has FICT plants
  with `produtibilidade_especifica > 0` and the productivity computation
  is non-trivial (e.g. depends on volume-area geometry on the FICT),
  the fold-in remains a numerical approximation at the FICT's own
  reference volume.

- **Reverse-direction lookups**: the resolver only walks
  `codigo_usina_jusante` forward. If a future case encodes a FICT chain
  via upstream pointers in some other table, this resolver will miss
  it.

- **Cycles**: the resolver guards against `codigo_usina_jusante` cycles
  defensively (a self-revisit terminates the walk) but does not warn
  about them. A cycle in confhd would surface in NEWAVE first; this is
  not the responsibility of cobre-bridge.

If any of these limits matter for a future case, the resolver is the
single point of extension — both `hydro.py` and `constraints.py` already
consume its `FictCascadeResolution` records.

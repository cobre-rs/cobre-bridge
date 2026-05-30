# NEWAVE ↔ Cobre FCF Cut Comparison — Working Notes

Investigation notes + tooling for comparing the **future-cost-function (FCF)
cuts** that NEWAVE and Cobre build, to localize the systematic FCF gap (Cobre's
cost-to-go sits ~0.2–0.4 bi R$ below NEWAVE's, seeded at stage 0) and the related
reservoir storage-drawdown divergence. Picks up the broader NEWAVE-vs-Cobre
comparison (see `.claude/skills/compare-newave-cobre/known-divergences.md`).

Status: tooling works end-to-end on the example case; units pinned; alignment
established for the **study** period. Post-study alignment and a couple of
artifacts remain open (below).

## Tooling

| Script                        | Purpose                                                                                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `newave_cut_investigation.py` | Read NEWAVE `cortes.dat` cut coefficients. `read_newave_cut_coefficients(case, tipo_estagio, estagio)`; `newave_water_values(...)` → per-UHE water value (R$/hm³). |
| `cobre_cut_investigation.py`  | Read Cobre `output/policy/cuts/stage_NNN.bin` via `flatc`. `decode_stage_cuts(case, stage)`; `cobre_water_values(...)` → per-reservoir water value (R$/hm³).       |
| `compare_cuts.py`             | Align stages, join reservoirs by name, print side-by-side (R$/hm³) with Δ.                                                                                         |

Readers share filter params: `iteration`, `forward_pass`, `reduce`
(first/last/mean/median), and `select` (`oldest` | `newest` | `first_real`).
**No implicit averaging** — `reduce=None` requires a single cut after filtering.

Run: `.venv/bin/python compare_cuts.py` (edit `COBRE_STAGE` / `SELECT` at top).
Requires `flatc` (1.11 OK) and the Cobre schema at
`/home/rjalves/git/cobre/crates/cobre-io/schemas/policy.fbs` (`SCHEMA_PATH`).

## How the cuts are read (mechanics & gotchas)

### NEWAVE — `cortes.dat` (+ `cortesh.dat` header)

- Binary **linked list**: start at the most recent cut; each record points to the
  previous cut of the same stage.
- Params **derived from `cortesh`** (don't hardcode): `tamanho_registro =
cortesh.tamanho_corte` (17168 here); `numero_total_cortes = max(
ultimo_registro_cortes_estagio.indice_ultimo_corte)` (3150); per-stage
  `indice_ultimo_corte` from that table; REE order from `ree.dat`
  (`[1,6,7,5,10,12,2,11,3,4,8,9]`); UHE/submarket codes from
  `cortesh.dados_uhes` / `dados_submercados`.
- **Cross-checks**: `tamanho_registro × numero_total_cortes == file size`
  (`3150 × 17168 = 54,079,200`); column count `== 5 + gnl + (n_rees|n_uhes)·(1 +
ordem_parp)`.
- ⚠️ **inewave layout switch** (`modelos/cortes.py`): it picks the layout by
  `codigos_rees` **first** — pass `codigos_rees=[]` for **individualized** cuts
  (`pi_varm_uhe`, `pi_qafl_uhe_lag`), else you get the REE-aggregated layout
  (`pi_earm_ree`) and **garbage** on individualized stages. This case is
  individualized (185 cols REE vs **2018 cols** individualized = 5 + 24 gnl + 153
  `pi_varm_uhe` + 1836 `pi_qafl` (153×12)).
- ⚠️ Several `cortesh` scalar fields **read back 0** in this version and are
  unreliable: `tamanho_registro_individualizado/agregado`, `tipo_agregacao_caso`,
  `lag_maximo_gnl`, `considera_afluencia_anual`, `estagio_individualizado_*`. Use
  `tamanho_corte` + the cross-checks instead.

### Cobre — `output/policy/cuts/stage_NNN.bin`

- **FlatBuffers** (root `StageCuts`, schema `policy.fbs`). Decode to JSON with
  `flatc -t --strict-json --raw-binary --root-type StageCuts`. No Python
  `flatbuffers` dependency needed.
- Each `Cut`: `intercept` + positional `coefficients[state_dimension]`. Index →
  reservoir via `output/training/dictionaries/state_dictionary.json`
  (`state_variables[i] = {entity_id, "hydro", "storage", "hm3"}`).
- `stage_s.bin` = the cost-to-go that **stage s** accesses; the terminal stage's
  file is **empty** (`stage_063` here). `policy/metadata.json`:
  `state_dimension=152`, `completed_iterations=50`, `forward_passes=1`,
  `final_lower_bound = 2.3613e10`.

### Units (PINNED)

NEWAVE cut values are in **10³ R$**, Cobre in **10⁶ R$**. The readers normalize to
**R$** (`NEWAVE_MONETARY_UNIT_RS = 1e3`, `COBRE_MONETARY_UNIT_RS = 1e6`), so water
values compare directly in **R$/hm³**.

### Stage alignment (study period)

NEWAVE indexes stage 1 = January. Case starts **September** → September is the
first stage; the cost-to-go it accesses is NEWAVE **`estudo` stage 10** (October),
which is why NEWAVE's cuts start at estudo-10. Cobre is 0-based (September = stage
0), and `stage_s` holds the cost-to-go stage `s` accesses. Hence:

> **Cobre stage `s` ↔ NEWAVE (`estudo`, `s + 10`)**, for `s` in 0..26.

NEWAVE cut-bearing stages: `estudo` 10–36 + `pos` 1–36 (63). Cobre: stages 0–62
(63; stage 63 empty). **Post-study (`pos`) alignment is NOT done yet** — NEWAVE may
REE-aggregate post stages (different cut layout).

### Iteration / cut selection

- **NEWAVE** stores an **iteration-0 zero placeholder** as its oldest cut
  (`indice_corte=1, iteracao_construcao=0, indice_forward=0, rhs=0`, all coeffs 0
  = initial FCF). Its first **real** cut is `indice_corte=2, iteracao=2,
forward=1`. Iteration labels skip 1 (`{0, 2, 3, …, 50}`); `indice_forward ∈
{0,1}`.
- **Cobre** has no placeholder: iterations `{1..50}`, `forward_pass_index=0`; first
  cut = iteration 1.
- Use `select="first_real"` (NEWAVE skips the placeholder via `forward≥1` + oldest;
  Cobre = oldest) or `select="newest"` (converged, iter 50 both).
- **Intercepts are NOT directly comparable** — each is `α − β'·x̂`, tied to its own
  generating state x̂. Compare the **gradients** (water values) or the FCF **value**
  at a common state. (Cobre's converged intercept 2.36e10 ≈ its `lower_bound` ✓.)

## Differences found

1. **State space: 153 (NEWAVE) vs 152 (Cobre) — JURUENA.**
   `confhd usina_existente`: 162 `EX`, 3 `NC`, 1 `NE`. Cobre = 162 `EX` − 10
   fictitious (`FICT.*`) = 152. NEWAVE's cut state keeps **JURUENA** (code 309,
   `usina_existente="NE"`) → 153. Converter drops it (`EX`-only filter).
   **Harmless**: Juruena's NEWAVE water value ≈ −1e-4 (≈0, non-operating). The same
   plant 309 triggers the "MODIF.DAT unrecognised record (DefaultRegister)" log.
   → _Expected / correct exclusion; just a state-dimension mismatch to remember._

2. **Units reconciled.** After ×10³ / ×10⁶, most reservoirs land at comparable
   O(0.1–10) R$/hm³ on both sides — the scale gap was units.

3. **Três Marias artifact (open).** NEWAVE `pi_varm` for TRES MARIAS ≈
   **−1.9e8 R$/hm³**, _dimensionally impossible_ (× its storage ≫ the whole FCF),
   present in **both** first and converged cuts → suspected **per-reservoir parse /
   encoding artifact on the NEWAVE side**, not a real water value. Distorts any
   comparison touching it.

4. **First-cut noise (expected).** The first real cut is dominated by the few
   reservoirs binding in that single forward-pass scenario, and the two solvers
   sample different states → different reservoirs blow up (NEWAVE:
   Três Marias/Guarapiranga/Sinop; Cobre: Caconde/Irapé/Queimado). Not a verdict;
   use the 1-iteration runs for a deterministic first cut.

5. **Converged-cut sign disagreements (lead).** Several reservoirs have
   opposite-sign water values: e.g. HENRY BORDEN (+0.77 NW / −0.48 CB), VOLTA
   GRANDE (+0.54 / −0.12). Worth understanding.

6. **FCF value** matches in scale: Cobre `lower_bound` 23.6 bi ≈ NEWAVE September
   cost-to-go 23.8 bi — the ~0.2–0.4 bi gap is the divergence under investigation.

## Open items / next steps

- [ ] **1-iteration runs** of both models → deterministic single cut/stage; the
      cleanest version of this comparison (tool is ready, use `select="first_real"`
      or `iteration` filter).
- [ ] **Três Marias artifact** — isolate the NEWAVE individualized-parse behaviour
      for code(s) like Três Marias (column alignment? special encoding?).
- [ ] **Post-study (`pos`) alignment** — extend `newave_stage_for` and handle
      possible REE-aggregation of post stages (`AGREGADO_EM_REE`).
- [ ] **Sign disagreements** — investigate the reservoirs where NEWAVE and Cobre
      disagree on the sign of the water value.
- [ ] Tie back to the **storage-drawdown** divergence (Cobre's near-flat FCF
      gradient ⇒ less incentive to store).

## Related

- `.claude/skills/compare-newave-cobre/` — the comparison skill (methodology,
  `known-divergences.md`).
- Memory: `inewave-cortes-reading-gotchas`, `bounds-comparator-reimplements-conversion`.

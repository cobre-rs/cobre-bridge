# Anticipated thermals: pre-horizon `values_mw` rejected by cobre validator

**Audience:** cobre maintainers (feature branch `feat/anticipated-thermals`).
**Status in cobre-bridge:** Worked around in
`src/cobre_bridge/converters/initial_conditions.py` (commit `68bfa927`):
the bridge writes `values_mw = [0.0] * lead_stages` and logs a WARNING
naming the NEWAVE MW values being dropped.
**Status in cobre:** Validation-only restriction; the LP semantics that
motivate the restriction are still in place.
**Severity for cobre-bridge:** Loss of fidelity to NEWAVE on cases with
non-trivial `adterm.dat` — NEWAVE's pre-horizon GNL dispatch is silently
zeroed in cobre's LP. The example case `example/newave_rodada` has one
real non-zero commitment (`ST.CRUZ NOVA`, code 86, 204.5647 MW at lag 1)
that cannot be honored.
**Date:** 2026-05-22.

---

## TL;DR

`cobre-io::validation::semantic::thermal::check_committed_value_bounds`
rejects every non-zero `past_anticipated_commitments.values_mw[j]` with
`BusinessRuleViolation`. The error message names this a "current version"
limitation and the rationale is documented in
`cobre-core::initial_conditions::AnticipatedCommitmentHistory`: the LP's
fishing-constraint activation predicate is `false` at every stage before
the first matured delivery, and the ring-buffer shift overwrites slot 0
with the LP's own decision before any constraint can read a seeded
value, so a non-zero seed would be silently dropped.

The field is kept in the data model for forward compatibility — a
future cobre release may re-index slots or adjust the activation
predicate to allow pre-horizon seeding.

From the cobre-bridge side this means **any real NEWAVE GNL commitment
that should be paid on stages 1..K of the study is lost**. The bridge
emits zeros to keep the case loadable, but the operational outcome on
those K stages will differ from NEWAVE — fewer pre-committed MW means
more headroom for the LP to either avoid the cost or dispatch
differently.

---

## How the bridge produces the offending input today

`example/newave_rodada/adterm.dat` is a representative NEWAVE GNL
configuration: three thermals (LINHARES, ST.CRUZ NOVA, P. SERGIPE I)
each with `lag ∈ {1, 2}` and per-`patamar` MW commitments. The bridge
reads it via `inewave.newave.Adterm` and aggregates the per-block MWs
into one stage-mean MW per delivery stage using the block-duration
weighting:

```
MW_eq = Σ_b f_b · MW_b
```

where `f_b` is the patamar fraction at the delivery stage (from
`patamar.dat`). This preserves the total committed MWh under cobre's
constant-MW-per-stage convention.

For ST.CRUZ NOVA (lag 1) the inputs are
`MW = (227.86, 238.37, 173.51)` against September 2024 block fractions
`f = (0.2333, 0.2833, 0.4834)`, giving `MW_eq = 204.5647 MW` — a real
commitment representing the pre-existing GNL contract that NEWAVE
honors at stage 0 of its study.

Without the workaround the bridge wrote that 204.5647 into
`past_anticipated_commitments[thermal_id=35].values_mw[0]` and the
cobre validator produced:

```
error: constraint violation: [BusinessRuleViolation]
initial_conditions.json (thermals[id=35].anticipated_config):
Thermal 35: past_anticipated_commitments.values_mw[0] = 204.564693
is non-zero; pre-horizon commitments are not supported in the
current version and would be silently dropped. Set all values_mw
entries to 0.0
```

---

## What we'd like from cobre

We're not asking for an immediate fix — the workaround keeps the case
loading. But for any future work that re-introduces pre-horizon seeding,
here is what would unlock NEWAVE parity on GNL cases:

### Functional requirement

Honor a seeded `past_anticipated_commitments[j].values_mw[k]` as a
**hard MW commitment delivered at study stage `k`** (0-based), for
`k ∈ [0, lead_stages)`. Equivalently: for each anticipated thermal,
the LP must dispatch exactly `values_mw[k]` MW (constant across blocks)
at delivery stage `k`, paid using the cost-per-MWh convention already
used for decision-stage commitments (ticket-022: cost paid entirely at
decision stage; for pre-horizon seeds, the cost is implicit because the
decision happened before the study began, so the commitment is paid
at full price but **does not contribute to the objective function** —
the LP just receives it as a fixed dispatch).

### Concrete acceptance criteria

1. Given an anticipated thermal with `lead_stages = K` and
   `past_anticipated_commitments.values_mw = [v_0, …, v_{K-1}]` with at
   least one `v_k > 0`, the validator MUST NOT emit
   `BusinessRuleViolation`. The existing bounds check
   (`v_k ∈ [min_mw, max_mw]`) MUST remain.
2. At study stage `k` (0-based, for `k < K`), the LP MUST enforce
   `sum_b generation[block_b] = v_k * total_hours_at_stage_k`
   (equivalent fishing equality, same shape as the decision-stage one
   but with the RHS coming from the seeded value rather than an LP
   variable).
3. The committed value MUST NOT appear in the objective function —
   the decision was made externally and its cost is sunk.
4. The validator MUST keep rejecting any seed with `j >= K` (length
   mismatch); the length-equality invariant is already enforced.
5. Forward-pass output (`simulation.thermal_results`) MUST report the
   seeded MW at stage `k` so downstream comparison tools (e.g.
   cobre-bridge's `compare results`) can verify parity with NEWAVE's
   `MEDIAS-USIT::GTERM` on stages 0..K-1.

### Implementation hint (from cobre-bridge's investigation)

The current ring-buffer shift in
`cobre-sddp/src/noise.rs::shift_anticipated_state` (line 249-293, the
`shift_anticipated_state` function) writes the LP decision into slot
`K_p - 1` and shifts older slots down. The seeded value would need to
land in the correct slot **before** the first forward pass solves —
specifically, slot `K_p - 1 - k` for `values_mw[k]` (so that after
`k+1` shifts it reaches slot 0 and becomes a constant on the right
side of the matured fishing constraint at stage `k`).

The activation predicate at
`StageIndexer::anticipated_decision_active_at_stage` (line ~1422 in
`indexer.rs`) gates whether a decision _column_ exists at a given
stage; the matured fishing row at the delivery stage is governed by a
separate predicate that today reads only LP-decision-driven slot
values. Either predicate (or a third one) needs to honor seeded
slots by treating them as fixed RHS terms rather than variables.

### What the bridge will do once it lands

`cobre_bridge/converters/initial_conditions.py` has a single line that
zeroes the values:

```python
"values_mw": [0.0] * dispatch.lead_stages,
```

Flipping that to `list(dispatch.values_mw)` restores genuine pass-through.
The reader
(`cobre_bridge/converters/anticipated.py::read_anticipated_dispatch`)
already returns the true block-weighted MWs, so no other change is
needed on the bridge side.

---

## How to reproduce the rejection locally (today)

1. Check out a clean cobre-bridge tree.
2. In a copy of `example/newave_rodada`, edit `dger.dat` to set
   `despacho_antecipado_gnl = 1` (the example ships with `0` so the
   converter writes no anticipated thermals).
3. Run `cobre-bridge convert newave <case> <out>` — the converter
   currently emits zeros (workaround active) and logs a WARNING per
   thermal naming the dropped MWs. To reproduce the _rejection_, swap
   the converter's line above to `list(dispatch.values_mw)` and re-run
   `cobre load <out>` (or whatever the cobre validator entrypoint is).
4. The error reproduces with the message quoted in the TL;DR.

---

## Open questions for the cobre team

- Is the slot-indexing fix the right direction, or is there a simpler
  path (e.g. emitting an _additional_ fishing row at the matured stage
  with the seeded RHS, leaving the existing predicate-gated machinery
  alone)?
- Is the cost-sunk decision a confirmed design call, or should
  pre-horizon seeds be billed at `cost_per_mwh` like fresh decisions
  (and just bypass the discount? — both interpretations are physically
  defensible).
- Once seeding is supported, does the cobre simulation output schema
  need a new column to flag "MW came from initial conditions, not from
  the LP"? Comparison tooling on the bridge side would benefit.

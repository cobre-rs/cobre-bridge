# NEWAVE ↔ Cobre investigation scripts

Ad-hoc, exploratory scripts used to dig into divergences between a NEWAVE run
and the Cobre case produced from it. They back the analyses in the
`compare-newave-cobre` skill (`known-divergences.md`).

> ⚠️ **Throwaway/experimental.** These are _not_ part of the `cobre_bridge`
> package, are not covered by tests, and are not run in CI. They hard-code
> `example/...` case paths and are meant to be read and re-run by hand.

## TODO — restructure later

These overlap heavily with `src/cobre_bridge/comparators/` and should eventually
be folded into proper, tested comparison routines (parametrised by case path,
no repo-root assumptions) rather than living as loose root scripts. Until then,
keep them here. See the `project_cobre_excess_root_cause` /
`project_forward_penalty_validation` memories for the findings they produced.

## Running

Run from the **repo root** (so `example/...` paths and cross-imports resolve)
with the dev env installed (`pip install -e ".[dev]"`):

```bash
python investigations/<script>.py [args]
```

## Scripts

| Script                          | Purpose                                                                                                                                                                                             |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `forward_penalty_experiment.py` | Validate converted penalties against NEWAVE `forward.dat`. Also the canonical `load_forward()` decoder (pass the 151 `EX`/non-FICT hydro plants; per-patamar values are MWmed contributions → sum). |
| `compare_sem_intercambio.py`    | Per-bus energy-balance comparison (NEWAVE vs Cobre) on the no-exchange case. Imports `forward_penalty_experiment`.                                                                                  |
| `cobre_cut_investigation.py`    | Decode Cobre FCF cut coefficients (`output/policy/cuts/stage_NNN.bin`).                                                                                                                             |
| `newave_cut_investigation.py`   | Decode NEWAVE FCF cut coefficients (`cortes.dat`).                                                                                                                                                  |
| `compare_cuts.py`               | Side-by-side NEWAVE vs Cobre FCF cut / water-value comparison. Imports the two `*_cut_investigation` scripts.                                                                                       |
| `cortese_investigation.py`      | Decode NEWAVE FCF visited-states file (`cortese.dat`).                                                                                                                                              |
| `compare_states.py`             | Compare NEWAVE visited storage states against Cobre's. Imports `cobre_cut_investigation`.                                                                                                           |

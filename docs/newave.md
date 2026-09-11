# The NEWAVE track

`convert newave`, `check newave`, and `compare newave` work on a NEWAVE case
directory. This page describes what the converter reads, what it changes on
the way to a Cobre case, and how to read a comparison. Flags and their help
text are in the [CLI reference](cli.md).

## Inputs

A case is discovered the way NEWAVE itself does it: `caso.dat` names the file
index (`arquivos.dat` by default) and the index names every other data file.
File-name lookups are case-insensitive. The two binary registries the index
does not list, `hidr.dat` and `vazoes.dat`, are found by scanning the
directory.

Many inputs are optional (`modif.dat`, `cvar.dat`, `curva.dat`, `agrint.dat`,
and others). `check newave` prints one line per optional input that is absent
and states that the conversion proceeds without it. A required input that is
missing or unreadable makes `check` exit 2, and `convert` fails before writing
anything. Run `check newave` first when a case comes from an unfamiliar
source.

The field-by-field map from deck files to the converted case, including what
the converter does not convert yet, is the [NEWAVE data map](newave-data-map.md)
(in Portuguese).

## What the converter changes

Cobre's input format differs from NEWAVE's in ways the converter has to
resolve. Every step is deterministic: converting the same case twice produces
the same output.

- **Entity ids.** NEWAVE identifies plants, subsystems, and interchanges by
  arbitrary 1-based codes; Cobre uses dense 0-based ids. The converter sorts
  the source codes and assigns 0-based ids in that order, consistently across
  every output file. `compare newave` rebuilds the same mapping from the
  source case, so results trace back to the source codes.
- **Which plants exist.** Only hydros marked as existing in `confhd.dat`
  become Cobre entities. NEWAVE's fictitious accounting plants are removed.
  They are identified structurally, as a zero-productivity plant sharing its
  inflow gauge with a generating plant, rather than by the `FICT.` name
  prefix, so a cascade reduced to a subset of plants still classifies
  correctly. Cascade links that ran through a removed plant are rewired to the
  next real plant downstream, preserving the water-balance topology.
- **Horizon.** The study horizon comes from `dger.dat`: start month and year,
  study years, post-study years. Every per-stage table is sized to it. Data
  NEWAVE provides only for the study years (loads, block factors, some bounds)
  is extended into the post-study years by repeating the last study year's
  seasonal values.
- **Constraints.** The minimum-storage security curve (`curva.dat`), electric
  constraints (`restricao-eletrica.csv`), and interchange group limits
  (`agrint.dat`) all become Cobre generic constraints: linear expressions over
  storage, generation, and exchange variables with per-stage bounds. A term
  whose entity is absent from the converted case, such as an interchange line
  missing from a reduced system, is dropped with a diagnostic naming it.
- **Risk.** The CVaR setting in `dger.dat` selects expectation, constant CVaR,
  or per-stage CVaR from `cvar.dat`. If `dger.dat` asks for CVaR and
  `cvar.dat` is absent, the converter falls back to expectation and says so.
- **Penalties.** NEWAVE's flow-domain penalties (`penalid.dat`) are converted
  to Cobre's cost basis with the same system-mean productivity NEWAVE applies,
  so a violation is priced the same way in both models.
- **Stochastic data.** Historical inflows (`vazoes.dat`, `vazpast.dat`), load
  (`sistema.dat`, `c_adic.dat`), and block factors (`patamar.dat`) are written
  under `scenarios/`, stage-varying tables as Parquet.

Anything the converter cannot carry over faithfully is reported rather than
dropped silently: `convert newave` renders diagnostics as grouped panels,
`--diagnostics-json` saves them, and `--json` includes them in the verdict.

## Comparing results

`compare newave NEWAVE_DIR COBRE_OUTPUT_DIR` needs, on the NEWAVE side, the
`MEDIAS-*.CSV` result files and `pmo.dat` directly in `NEWAVE_DIR`, and on the
Cobre side the `output/` directory `cobre run` produced. It aligns entities
through the conversion's id mapping, aggregates Cobre's scenarios to NEWAVE's
published level, and reports per-variable agreement as a symmetric percentage
error and the share of points within tolerance. It also re-evaluates the
converted generic constraints against both models' operation. With
`--format html` it writes a multi-tab report.

Before treating a difference as a converter defect, confirm the two runs are
comparable:

- The security curve uses fixed penalization in `curva.dat`. The bridge models
  the curve as a per-stage slack at the fixed cost and does not reproduce
  NEWAVE's iterative penalization mode.
- NEWAVE's final simulation has preventive rationing enabled in `dger.dat`.
  With it disabled, NEWAVE drains reservoirs to avoid deficit while Cobre
  follows the converted policy, and the two diverge by construction.
- The hydro production function is linear (constant productivity). A
  head-dependent run is a different production model.
- You know whether the final simulation is deterministic or stochastic. A
  single historical series makes the inflow model irrelevant on both sides.
- No input `.dat` file is newer than the NEWAVE outputs. Otherwise the
  converted case and the published results come from different decks.

Then read the report cost-first: the cost breakdown says which component
differs, the per-stage costs say where, and the system and per-bus operation
tabs localize it. Load and non-controllable generation are converted directly
from the inputs and are rarely the cause.

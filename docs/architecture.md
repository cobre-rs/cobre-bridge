# Architecture

`cobre-bridge`'s source lives under `src/cobre_bridge/`, split into eight
top-level packages layered by an enforced import direction, plus a twin-track
spine shared by the `newave/` and `decomp/` conversion packages. This page
maps that structure for a reader with no local checkout of the repository.

## Package map

| Package | Path | Role |
| --- | --- | --- |
| `core` | `src/cobre_bridge/core/` | Cross-track shared foundation — diagnostics, errors, units, productivity, generic-constraint format and builder, calendar and physics kernels. Imports only itself. |
| `cobre` | `src/cobre_bridge/cobre/` | Cobre-side I/O, schemas, and version compatibility (`MIN_COBRE_VERSION`). |
| `newave` | `src/cobre_bridge/newave/` | The NEWAVE conversion track: spine modules, track-specific infrastructure, and a `converters/` subpackage. |
| `decomp` | `src/cobre_bridge/decomp/` | The DECOMP conversion track: spine modules, track-specific infrastructure, a `converters/` subpackage, and a `fcf/` subpackage for boundary-FCF import. |
| `comparators` | `src/cobre_bridge/comparators/` | NEWAVE-vs-Cobre comparison, twin-tracked under `comparators/newave/` and `comparators/decomp/`, with shared chart code under `comparators/charts/`. |
| `ui` | `src/cobre_bridge/ui/` | Rich terminal rendering and HTML rendering — the only package that constructs a Rich `Console` or imports Rich, with HTML rendering under `ui/html/`. |
| `dashboard` | `src/cobre_bridge/dashboard/` | The interactive Plotly dashboard, with one module per tab under `dashboard/tabs/`. |
| `cli` | `src/cobre_bridge/cli/` | Typer CLI wiring. The top of the stack — no other package imports it. |

Several packages carry a subpackage that refines this map:
`src/cobre_bridge/newave/converters/`, `src/cobre_bridge/decomp/converters/`,
`src/cobre_bridge/decomp/fcf/`, `src/cobre_bridge/comparators/newave/`,
`src/cobre_bridge/comparators/decomp/`, `src/cobre_bridge/comparators/charts/`,
`src/cobre_bridge/ui/html/`, and `src/cobre_bridge/dashboard/tabs/`.

## Import direction

Imports are layered: a package may reach into itself and into the packages
below it, never sideways or up. Readable summary:

| Package | May import |
| --- | --- |
| `core` | `core` |
| `cobre` | `core`, `cobre` |
| `ui` | `core`, `cobre`, `ui` |
| `newave` | `core`, `cobre`, `newave` |
| `decomp` | `core`, `cobre`, `decomp` |
| `comparators` | `core`, `cobre`, `comparators`, and `ui`'s theme/HTML-rendering modules (not its Rich console) |
| `dashboard` | `core`, `cobre`, `dashboard`, and `ui`'s theme/HTML-rendering modules (not its Rich console) |
| `cli` | everything below it — `core`, `cobre`, `ui`, `newave`, `decomp`, `comparators`, `dashboard`, `cli` |

`comparators`'s reach into a conversion track is scoped to its twin-tracked
subpackages: `comparators/newave/` may additionally import `newave`, and
`comparators/decomp/` may additionally import `decomp` — modules directly
under `comparators/` import neither track.

This table is a readability aid, not the enforced rule. `tests/test_package_boundaries.py`
is the authoritative, enforced source of the import direction — its
`_ALLOWED` map and `test_import_direction` fail the build on any import edge
this table does not capture precisely. Two invariants matter most to a reader
working in this tree:

- `core` imports only itself — it has no outward dependency on any other
  package.
- Nothing imports `cli` — checked with no allowlist by
  `test_nothing_imports_cli` in the same file, since `cli` sits at the top of
  the stack.

## Twin-track spine

`newave/` and `decomp/` are two conversions of the same shape. Each exposes
five spine modules at its package top — `case`, `files`, `id_map`, `pipeline`,
and `preflight` — and each carries a `converters/` subpackage
(`src/cobre_bridge/newave/converters/`, `src/cobre_bridge/decomp/converters/`);
`decomp/` additionally carries `src/cobre_bridge/decomp/fcf/` for
boundary-FCF import. `tests/test_package_boundaries.py`'s
`test_track_packages_are_structurally_symmetric` enforces both facts.

That guard does not require every module at a track's top to be one of the
five spine stems. The rule it enforces is narrower: no converter-role module
may sit loose at a track's top level — it belongs under `converters/`, or
`fcf/` for `decomp/` — while shared infrastructure helpers legitimately stay
loose on both tracks alongside the spine. Read "structurally symmetric" as
this rule — no converter-role module loose; shared infra helpers stay loose
on both tracks — not as "nothing else loose."

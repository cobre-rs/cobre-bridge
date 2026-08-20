"""Tests for the boundary FCF importer's config-patch orchestration
(``fcf/__init__.py::import_boundary_fcf``/``_patch_policy_boundary``).

**TRACKED COBRE-GAP C8** (see
``~/git/cobre/plans/conversion-found-improvements.md`` and the code comment
at ``fcf/__init__.py::_patch_policy_boundary``): cobre resolves
``policy.boundary.path`` against the run's ``--output`` directory, not
``case_dir``, so every ``cobre run`` invocation that touches the boundary
must pass ``--output`` equal to the case dir it runs against — never the
default and never a distinct scratch output dir. cobre itself is not
touched; this is a workaround, tracked until C8 is fixed upstream.

Every test below exercises the importer's orchestration (config-patch,
C8-warning, empty-storage-manifest guard) with every cobre/deck seam
monkeypatched — no binary, no deck, and no installed cobre wheel needed.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from cobre_bridge.decomp.fcf import _patch_policy_boundary, import_boundary_fcf
from cobre_bridge.decomp.fcf.mapper import MappingResult
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cut_record,
    make_manifest,
    make_mapped_cut,
    make_slot,
)

if TYPE_CHECKING:
    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts


def _mock_deck_and_cut_seams(
    monkeypatch: pytest.MonkeyPatch, fake_cuts: BoundaryCuts
) -> None:
    """Monkeypatch the deck-discovery + cut-reader seams shared by every
    binary-free ``import_boundary_fcf`` orchestration test below.

    Stubs ``discover_decomp_files``/``Dadger``/``DecompIdMap``/``Cortesh``
    to minimal stand-ins (their return values are opaque placeholders — only
    ``read_cortes`` needs a shape any downstream code inspects) and
    ``read_cortes`` to return ``fake_cuts`` verbatim. Also stubs the
    ``sys.modules['cobre']`` entry so ``import_boundary_fcf``'s own
    unconditional ``import cobre`` (needed for ``cobre.__version__``)
    resolves without the optional cobre-python wheel installed — the
    epic-01 `"stub sys.modules['cobre']"` pattern
    (``tests/test_decomp_fcf_bootstrap.py``), not a module attribute patch.
    """
    monkeypatch.setitem(sys.modules, "cobre", SimpleNamespace(__version__="0.13.0"))
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.discover_decomp_files",
        lambda _deck_dir: SimpleNamespace(dadger=Path("dadger.rv0"), dadgnl=None),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.Dadger",
        SimpleNamespace(read=lambda _path: object()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.DecompIdMap",
        SimpleNamespace(from_dadger=lambda _dadger: object()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.Cortesh",
        SimpleNamespace(read=lambda _path: object()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.read_cortes",
        lambda *_args, **_kwargs: fake_cuts,
    )
    # The coupling-stage per-block-hours read (case_dir/stages.json) is a
    # case-reading seam like the deck ones above; these binary-free
    # orchestration cases carry only a minimal config.json, so stub it to a
    # single 648 h block rather than author a full stages.json. `import_
    # boundary_fcf` derives its scalar `cost_unit_hours` as the sum of these
    # (ticket-001), so a one-element `[648.0]` preserves the prior 648 h; these
    # storage/C8 cases place no live GNL ring, so the per-block length is never
    # validated against `n_patamares`.
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf._final_stage_block_hours",
        lambda _case_dir: [648.0],
    )
    monkeypatch.setattr("cobre_bridge.decomp.fcf.ensure_writer_binding", lambda: None)


def test_patch_policy_boundary_preserves_other_sections(tmp_path: Path) -> None:
    """AC 1 — `_patch_policy_boundary` sets `policy.boundary` and leaves
    every other top-level section byte-for-byte unchanged, writing the file
    in `decomp/pipeline.py`'s `_write_json` style (`indent=2`,
    `ensure_ascii=False`, single trailing newline).

    Guards the raw serialized text against an independently constructed
    expected string, not just `json.loads`-parsed equality — parsed
    equality is indent-agnostic, so it would pass under any indentation and
    never actually exercise `indent=2`. `simulation.scenario_label` carries
    a raw non-ASCII character (not a `\\uXXXX` escape) so `ensure_ascii=False`
    is genuinely distinguished: under `ensure_ascii=True` that character
    would still round-trip through `json.loads` equality, but the raw text
    on disk would differ.
    """
    config_path = tmp_path / "config.json"
    other_sections = {
        "state_space": {"hydro_storage": {"n_plants": 3}},
        "training": {"stopping_rules": [{"type": "iteration_limit", "limit": 500}]},
        "simulation": {"num_openings": 10, "scenario_label": "cenário"},
    }
    config_path.write_text(
        json.dumps(other_sections, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _patch_policy_boundary(config_path, source_stage=10)

    patched_text = config_path.read_text(encoding="utf-8")
    patched = json.loads(patched_text)

    assert patched["policy"]["boundary"] == {"path": "boundary", "source_stage": 10}
    for key, value in other_sections.items():
        assert patched[key] == value

    expected_config = {
        **other_sections,
        "policy": {"boundary": {"path": "boundary", "source_stage": 10}},
    }
    expected_text = json.dumps(expected_config, indent=2, ensure_ascii=False) + "\n"
    assert patched_text == expected_text, (
        "patched config.json's raw serialized text diverges from the "
        "expected indent=2 / ensure_ascii=False / single-trailing-newline "
        "style"
    )
    assert "cenário" in patched_text, (
        "raw non-ASCII character was escaped on write — ensure_ascii=False not honored"
    )


def test_import_boundary_fcf_logs_c8_workaround(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Requirement 0 / AC 2, AC 3 — the C8 cobre-gap workaround is logged,
    not silently absorbed, and the orchestration's own return value +
    config patch are correct.

    Needs no cobre binary, no real deck, and no installed cobre wheel: every
    seam that would touch any of the three (`bootstrap_terminal_manifest`,
    the deck/cut readers, the checkpoint writer, and `import_boundary_fcf`'s
    own internal `import cobre`) is monkeypatched to a minimal stand-in — a
    "monkeypatched-shape unit path" — isolating the orchestration under
    test: that `import_boundary_fcf` returns `case_dir / "boundary"`,
    patches `config.json`'s `policy.boundary.source_stage` to the reader's
    boundary stage, emits a `WARNING` with the actionable run-with-`--output`
    guidance right after patching `policy.boundary`, and that the source
    carries the matching `TRACKED COBRE-GAP WORKAROUND (C8)` code comment at
    the patch site. The internal C8 marker lives in that comment, not in the
    end-user-facing log message.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text("{}\n", encoding="utf-8")

    # A real (if minimal) `BoundaryCuts`/`MappingResult` pair, not a bare
    # `SimpleNamespace` — `_emit_import_diagnostics` (called between
    # `map_boundary_cuts` and `write_boundary_checkpoint`) reads
    # `cuts.records` (via `summarize_cut_families`) and `mapping.dropped`, so
    # the fakes must carry those attributes with a shape that survives the
    # real triage pass, not just the fields this test's own assertions read.
    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=10,
    )
    _mock_deck_and_cut_seams(monkeypatch, fake_cuts)
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest([make_slot(0, 0, 0)]),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(
            cuts=(make_mapped_cut(coefficients=(1.5,)),), dropped=()
        ),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.build_stage_cuts_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    with caplog.at_level(logging.WARNING):
        boundary_dir = import_boundary_fcf(
            case_dir,
            tmp_path / "deck" / "cortesh.dat",
            tmp_path / "deck" / "cortes-010.dat",
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
        )

    assert boundary_dir == case_dir / "boundary"

    patched_config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    assert (
        patched_config["policy"]["boundary"]["source_stage"] == fake_cuts.boundary_stage
    )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # The workaround is surfaced as an actionable WARNING (the run-with-
    # `--output` constraint) — not silently absorbed. The internal "(C8)"
    # tracking marker lives in the code comment (asserted below), never in the
    # end-user-facing message, so this checks the actionable content only.
    assert any(
        "--output" in r.message and "cobre run" in r.message for r in warnings
    ), f"no actionable run-with-output WARNING found: {caplog.text}"
    # The message must not leak the internal tracking code to end users.
    assert not any("TRACKED COBRE-GAP" in r.message for r in warnings), (
        f"C8 tracking marker leaked into a user-facing log message: {caplog.text}"
    )

    source_path = inspect.getsourcefile(import_boundary_fcf)
    assert source_path is not None
    source_text = Path(source_path).read_text(encoding="utf-8")
    assert "# TRACKED COBRE-GAP WORKAROUND (C8" in source_text


def test_import_boundary_fcf_rejects_storageless_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 4 — a bootstrapped manifest with no `HydroStorage` slot at all is
    a terminal-manifest read bug, not a legitimate empty-storage case.

    `map_boundary_cuts` is deliberately left un-mocked here (unlike the C8
    test above) — its own read-bug guard must fire for real, and
    `import_boundary_fcf` must let the resulting `ValueError` propagate
    verbatim rather than swallow it.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text("{}\n", encoding="utf-8")

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=10,
    )
    _mock_deck_and_cut_seams(monkeypatch, fake_cuts)
    # Only a `HydroInflowLag` (entity_type 1) slot — no `HydroStorage`
    # (entity_type 0) slot at all — triggers `map_boundary_cuts`'s own
    # read-bug guard.
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest([make_slot(1, 0, 0)]),
    )

    with pytest.raises(ValueError, match="HydroStorage"):
        import_boundary_fcf(
            case_dir,
            tmp_path / "deck" / "cortesh.dat",
            tmp_path / "deck" / "cortes-010.dat",
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
        )

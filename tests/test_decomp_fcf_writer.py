"""Tests for the boundary-checkpoint writer (``fcf/writer.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import cobre
import pytest

from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
from cobre_bridge.decomp.fcf.mapper import MappedCut, MappingResult
from cobre_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)

# The writer binding is expected to be present in this repo's venv (see
# fcf/bootstrap.py::ensure_writer_binding); guard the one test that exercises
# the real cobre call so the suite still degrades gracefully if a future
# venv predates the binding.
_HAS_WRITER_BINDING = hasattr(cobre, "write_policy_checkpoint")


def _mapped_cut(
    *,
    coefficients: tuple[float, ...],
    intercept: float = 1.0,
    cut_id: int = 1,
    iteration: int = 1,
    forward_pass_index: int = 0,
    is_active: bool = True,
) -> MappedCut:
    """One hand-authored `MappedCut`."""
    return MappedCut(
        intercept=intercept,
        coefficients=coefficients,
        cut_id=cut_id,
        iteration=iteration,
        forward_pass_index=forward_pass_index,
        is_active=is_active,
    )


def _slot(entity_type: int, entity_id: int, subindex: int) -> dict[str, object]:
    """One hand-authored terminal-manifest slot dict (mirrors mapper tests)."""
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "subindex": subindex,
        "was_active": True,
        "delivery_anchor": -1,
    }


def _manifest(state_dimension: int) -> TerminalManifest:
    """A synthetic terminal manifest of `state_dimension` `HydroStorage` slots."""
    slots = tuple(_slot(0, i, 0) for i in range(state_dimension))
    return TerminalManifest(entity_manifest=slots, state_dimension=state_dimension)


def _valid_metadata(*, state_dimension: int, num_stages: int = 1) -> dict[str, object]:
    return build_metadata(
        state_dimension=state_dimension,
        num_stages=num_stages,
        cost_scale_factor=1.0,
        completed_iterations=1,
        final_lower_bound=0.0,
        max_iterations=1,
        forward_passes=1,
        warm_start_cuts=0,
        rng_seed=0,
        created_at="2026-08-03T00:00:00Z",
        cobre_version="0.13.0",
    )


def test_build_stage_cuts_payload_shape() -> None:
    manifest = _manifest(4)
    active_cut = _mapped_cut(
        coefficients=(1.0, 2.0, 3.0, 4.0),
        cut_id=101,
        iteration=5,
        forward_pass_index=2,
        is_active=True,
    )
    inactive_cut = _mapped_cut(
        coefficients=(5.0, 6.0, 7.0, 8.0),
        cut_id=202,
        iteration=6,
        forward_pass_index=3,
        is_active=False,
    )
    mapping = MappingResult(cuts=(active_cut, inactive_cut), dropped=())

    payload = build_stage_cuts_payload(mapping, manifest, stage_id=10)

    assert payload["stage_id"] == 10
    assert payload["state_dimension"] == 4
    assert payload["entity_manifest"] == list(manifest.entity_manifest)
    assert len(payload["cuts"]) == 2
    for cut in payload["cuts"]:
        assert len(cut["coefficients"]) == 4

    first, second = payload["cuts"]
    assert first["cut_id"] == active_cut.cut_id
    assert first["iteration"] == active_cut.iteration
    assert first["forward_pass_index"] == active_cut.forward_pass_index
    assert first["is_active"] is True
    assert second["cut_id"] == inactive_cut.cut_id
    assert second["iteration"] == inactive_cut.iteration
    assert second["forward_pass_index"] == inactive_cut.forward_pass_index
    assert second["is_active"] is False

    assert payload["active_cut_indices"] == [0]
    assert payload["populated_count"] == 2


def test_build_metadata_refuses_none_cost_scale_factor() -> None:
    with pytest.raises(RuntimeError, match="cost_scale_factor"):
        build_metadata(
            state_dimension=4,
            num_stages=1,
            cost_scale_factor=None,
            completed_iterations=1,
            final_lower_bound=0.0,
            max_iterations=1,
            forward_passes=1,
            warm_start_cuts=0,
            rng_seed=0,
            created_at="2026-08-03T00:00:00Z",
            cobre_version="0.13.0",
        )


@pytest.mark.skipif(
    not _HAS_WRITER_BINDING,
    reason="installed cobre wheel lacks write_policy_checkpoint",
)
def test_write_boundary_checkpoint_creates_files(tmp_path: Path) -> None:
    manifest = _manifest(2)
    mapping = MappingResult(
        cuts=(_mapped_cut(coefficients=(1.0, 2.0), cut_id=101, is_active=True),),
        dropped=(),
    )
    payload = build_stage_cuts_payload(mapping, manifest, stage_id=10)
    metadata = _valid_metadata(state_dimension=2)

    boundary_dir = tmp_path / "boundary"
    write_boundary_checkpoint(boundary_dir, payload, metadata)

    assert (boundary_dir / "metadata.json").exists()
    assert (boundary_dir / "cuts" / "stage_010.bin").exists()
    assert (boundary_dir / "basis").is_dir()

    on_disk_metadata = json.loads((boundary_dir / "metadata.json").read_text())
    assert isinstance(on_disk_metadata, dict)
    assert on_disk_metadata["cost_scale_factor"] is not None

    reloaded = cobre.results.load_policy(
        boundary_dir.parent, policy_subdir=boundary_dir.name
    )
    assert reloaded["metadata"]["cost_scale_factor"] is not None
    reloaded_stage = reloaded["stage_cuts"][0]
    assert reloaded_stage["state_dimension"] == 2
    assert len(reloaded_stage["cuts"]) == 1


def test_build_payload_rejects_wrong_coefficient_length() -> None:
    manifest = _manifest(4)
    mapping = MappingResult(
        cuts=(_mapped_cut(coefficients=(1.0, 2.0, 3.0)),),  # length 3 != 4
        dropped=(),
    )

    with pytest.raises(ValueError, match="state_dimension"):
        build_stage_cuts_payload(mapping, manifest, stage_id=10)

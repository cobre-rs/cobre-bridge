"""End-to-end schema-load gate for the DECOMP conversion track.

Tier-2 (needs the ``cobre-python`` wheel, no solver binary): runs the real
``convert_decomp_case`` against the committed ``tests/decks/decomp_mini/``
deck and then the real ``cobre.io.validate`` on the produced case, so a
schema-shape regression that makes converted output unloadable fails CI
instead of passing silently.
"""

from __future__ import annotations

from pathlib import Path

from cobre_bridge.decomp.pipeline import convert_decomp_case
from tests.conftest import requires_cobre_python


@requires_cobre_python
def test_decomp_mini_deck_converts_and_validates(
    decomp_mini_deck: Path, tmp_path: Path
) -> None:
    import cobre.io  # tier-2: guarded import inside the body, never module scope

    dst = tmp_path / "converted"
    convert_decomp_case(decomp_mini_deck, dst, force=True)

    result = cobre.io.validate(str(dst))
    assert result["valid"] is True
    assert result["errors"] == []

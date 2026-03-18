"""Basic smoke tests for the cobre-bridge package structure."""

import cobre_bridge


def test_version_string_is_set() -> None:
    """Package version should be a non-empty string."""
    assert isinstance(cobre_bridge.__version__, str)
    assert len(cobre_bridge.__version__) > 0

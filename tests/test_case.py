"""Tests for ``NewaveCase`` — the parse-once, cached NEWAVE case object."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cobre_bridge.case import NewaveCase
from tests.conftest import make_case, make_nw_files


def test_required_reader_parses_once_and_caches(tmp_path: Path) -> None:
    case = NewaveCase(files=make_nw_files(tmp_path))
    with patch("cobre_bridge.case.Dger") as mock_dger:
        sentinel = object()
        mock_dger.read.return_value = sentinel
        first = case.dger
        second = case.dger
    assert first is sentinel
    assert second is sentinel
    mock_dger.read.assert_called_once_with(str(tmp_path / "dger.dat"))


def test_optional_reader_is_none_when_file_absent(tmp_path: Path) -> None:
    case = NewaveCase(files=make_nw_files(tmp_path))  # modif defaults to None
    with patch("cobre_bridge.case.Modif") as mock_modif:
        assert case.modif is None
    mock_modif.read.assert_not_called()


def test_optional_reader_parses_when_file_present(tmp_path: Path) -> None:
    modif_path = tmp_path / "modif.dat"
    case = NewaveCase(files=make_nw_files(tmp_path, modif=modif_path))
    with patch("cobre_bridge.case.Modif") as mock_modif:
        sentinel = object()
        mock_modif.read.return_value = sentinel
        assert case.modif is sentinel
        assert case.modif is sentinel  # cached
    mock_modif.read.assert_called_once_with(str(modif_path))


def test_from_directory_binds_files_without_parsing(tmp_path: Path) -> None:
    files = make_nw_files(tmp_path)
    with patch(
        "cobre_bridge.case.NewaveFiles.from_directory", return_value=files
    ) as mock_from_dir:
        case = NewaveCase.from_directory(tmp_path)
    assert case.files is files
    mock_from_dir.assert_called_once_with(tmp_path)


def test_make_case_prefills_slots_without_io(tmp_path: Path) -> None:
    sentinel = object()
    case = make_case(tmp_path, dger=sentinel)
    with patch("cobre_bridge.case.Dger") as mock_dger:
        assert case.dger is sentinel
    mock_dger.read.assert_not_called()


def test_make_case_forces_optional_to_none(tmp_path: Path) -> None:
    case = make_case(tmp_path, cvar=None)
    with patch("cobre_bridge.case.Cvar") as mock_cvar:
        assert case.cvar is None
    mock_cvar.read.assert_not_called()


def test_vazoes_is_not_cached_on_case(tmp_path: Path) -> None:
    # The large single-use inflow file stays a path on case.files, not a reader.
    case = NewaveCase(files=make_nw_files(tmp_path))
    assert case.files.vazoes == tmp_path / "vazoes.dat"
    assert not hasattr(NewaveCase, "vazoes")
    with pytest.raises(AttributeError):
        _ = case.vazoes  # type: ignore[attr-defined]

"""Tests for ``NewaveFiles`` discovery — required-file failures as typed errors.

The discovery chain (``caso.dat`` → ``Arquivos`` → required files) now raises
:class:`~cobre_bridge.errors.SourceFileError` (a :class:`FileNotFoundError`
subclass) carrying ``path``/``field`` location detail, instead of a bare
``FileNotFoundError``. These tests pin the type, the instance-check that keeps
``except FileNotFoundError`` handlers working, and the location attributes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cobre_bridge.errors import SourceFileError
from cobre_bridge.newave_files import NewaveFiles, _resolve_required


def test_missing_caso_raises_source_file_error(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError) as exc_info:
        NewaveFiles.from_directory(tmp_path)
    assert isinstance(exc_info.value, FileNotFoundError)


def test_source_file_error_carries_path_and_field(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError) as exc_info:
        NewaveFiles.from_directory(tmp_path)
    exc = exc_info.value
    assert exc.path == str(tmp_path)
    assert exc.field == "caso.dat"


def test_resolve_required_missing_carries_path_and_field(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError) as exc_info:
        _resolve_required(tmp_path, "hidr.dat")
    exc = exc_info.value
    assert isinstance(exc, FileNotFoundError)
    assert exc.path == str(tmp_path)
    assert exc.field == "hidr.dat"


def test_missing_caso_message_is_preserved(tmp_path: Path) -> None:
    with pytest.raises(SourceFileError, match=f"caso.dat not found in {tmp_path}"):
        NewaveFiles.from_directory(tmp_path)


def test_newave_files_discovers_exph() -> None:
    files = NewaveFiles.from_directory(Path("example/newave_rodada_2001"))
    assert files.exph is not None
    assert files.exph.name.lower() == "exph.dat"

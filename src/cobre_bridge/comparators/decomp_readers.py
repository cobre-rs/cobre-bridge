"""DECOMP output readers for results comparison.

Reads the ``dec_oper_*.csv`` operation tables and the convergence report
from a DECOMP case directory via ``idecomp``, returning Polars
frames with the source's native column names (1-based ``estagio``,
node/scenario indices as written). Alignment onto Cobre entity ids and
stage indices happens in the comparison layer, not here.

Every reader treats an empty parse as an error: silent-empty DataFrames
are idecomp's characteristic failure mode on unexpected syntax, and a
comparison built on an empty frame would report a false zero-vs-zero
match.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

import pandas as pd
import polars as pl
from idecomp.decomp import DecOperInterc, DecOperSist, DecOperUsih, DecOperUsit, Relato

from cobre_bridge.comparators.newave_readers import _find_case_insensitive

_LOG = logging.getLogger(__name__)

_RELATO_PATTERN = re.compile(r"^relato\.rv\d+$", re.IGNORECASE)


class _TableFile(Protocol):
    """The idecomp reader surface the operation readers rely on."""

    tabela: pd.DataFrame | None


class _TableReader(Protocol):
    @staticmethod
    def read(path: str) -> _TableFile: ...


def _read_dec_oper(
    case_dir: Path,
    filename: str,
    reader_cls: _TableReader,
) -> pl.DataFrame:
    """Read one ``dec_oper_*`` table, rejecting missing or empty parses."""
    path = _find_case_insensitive(case_dir, filename)
    if path is None:
        raise FileNotFoundError(f"{filename} not found in {case_dir}")
    table = reader_cls.read(str(path)).tabela
    if table is None or table.empty:
        raise ValueError(
            f"{path} parsed empty; the run's outputs look incomplete or the "
            "file syntax is unsupported"
        )
    _LOG.debug("Read %s: %d rows", path.name, len(table))
    return pl.from_pandas(table)


def read_dec_oper_sist(case_dir: Path) -> pl.DataFrame:
    """Per-(stage, node, scenario, block, submarket) system operation.

    Includes demand, generation by family (hydro/thermal/anticipated/wind/
    small plants), exchanges, the Itaipu 50/60 Hz split, deficit, stored
    energy, and the marginal cost (``cmo``).
    """
    return _read_dec_oper(case_dir, "dec_oper_sist.csv", DecOperSist)


def read_dec_oper_usih(case_dir: Path) -> pl.DataFrame:
    """Per-hydro operation: storage, flows (natural/incremental/turbined/
    spilled/withdrawn/evaporated), generation, and available capacity
    (``potencia_disponivel_MW`` — the availability-rule oracle)."""
    return _read_dec_oper(case_dir, "dec_oper_usih.csv", DecOperUsih)


def read_dec_oper_usit(case_dir: Path) -> pl.DataFrame:
    """Per-thermal operation: generation with its effective min/max bounds
    and incremental cost."""
    return _read_dec_oper(case_dir, "dec_oper_usit.csv", DecOperUsit)


def read_dec_oper_interc(case_dir: Path) -> pl.DataFrame:
    """Per-exchange operation: origin/destination flows, losses, and the
    effective capacity."""
    return _read_dec_oper(case_dir, "dec_oper_interc.csv", DecOperInterc)


def _find_relato(case_dir: Path) -> Path | None:
    """Locate the revision-suffixed general report (``relato.rvN``)."""
    try:
        for entry in sorted(case_dir.iterdir()):
            if entry.is_file() and _RELATO_PATTERN.match(entry.name):
                return entry
    except OSError:
        pass
    return None


def read_relato_convergence(case_dir: Path) -> pl.DataFrame:
    """Read the convergence table (``iteracao``, ``zinf``, ``zsup``,
    ``gap_percentual``, …) from the general report."""
    path = _find_relato(case_dir)
    if path is None:
        raise FileNotFoundError(f"no relato.rvN found in {case_dir}")
    table = Relato.read(str(path)).convergencia
    if table is None or table.empty:
        raise ValueError(f"{path} has no convergence table")
    return pl.from_pandas(table)

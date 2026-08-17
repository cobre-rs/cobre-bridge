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


def _resolve_result_file(case_dir: Path, filename: str) -> Path | None:
    """Resolve *filename* case-insensitively, preferring ``saidas/`` over
    the deck root.

    DECOMP ships its full result export under the deck's ``saidas/``
    subfolder; the deck root carries only a curated subset. A file present
    in both locations is taken from ``saidas/``. A missing ``saidas/``
    directory is a silent miss, not an error.
    """
    for base in (case_dir / "saidas", case_dir):
        if base.is_dir():
            hit = _find_case_insensitive(base, filename)
            if hit is not None:
                return hit
    return None


def _read_dec_oper(
    case_dir: Path,
    filename: str,
    reader_cls: _TableReader,
) -> pl.DataFrame:
    """Read one ``dec_oper_*`` table, rejecting missing or empty parses."""
    path = _resolve_result_file(case_dir, filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename} not found in {case_dir} or its saidas/ subfolder"
        )
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


def _resolve_relato(case_dir: Path) -> Path | None:
    """Locate the revision-suffixed general report (``relato.rvN``),
    preferring ``saidas/`` over the deck root (same precedence as
    `_resolve_result_file`)."""
    for base in (case_dir / "saidas", case_dir):
        if not base.is_dir():
            continue
        try:
            for entry in sorted(base.iterdir()):
                if entry.is_file() and _RELATO_PATTERN.match(entry.name):
                    return entry
        except OSError:
            pass
    return None


def _read_relato_table(case_dir: Path, attr: str) -> pl.DataFrame:
    """Read one named pandas table off the general report (``relato.rvN``).

    Resolves the relato file via `_resolve_relato` (saidas-first), reads it
    via ``Relato.read``, and pulls the table at ``getattr(relato, attr)``.
    Shared by every ``relato``-backed reader (convergence, energy balance,
    and — in later tickets — costs and membership).
    """
    path = _resolve_relato(case_dir)
    if path is None:
        raise FileNotFoundError(
            f"no relato.rvN found in {case_dir} or its saidas/ subfolder"
        )
    table = getattr(Relato.read(str(path)), attr)
    if table is None or table.empty:
        raise ValueError(f"{path} has no {attr} table")
    return pl.from_pandas(table)


def read_relato_convergence(case_dir: Path) -> pl.DataFrame:
    """Read the convergence table (``iteracao``, ``zinf``, ``zsup``,
    ``gap_percentual``, …) from the general report."""
    return _read_relato_table(case_dir, "convergencia")


def read_relato_balance(case_dir: Path) -> pl.DataFrame:
    """Read the per-submarket energy balance table (demand, generation by
    source, purchase/sale, ENA, and EARM in/out) from the general report."""
    return _read_relato_table(case_dir, "balanco_energetico")

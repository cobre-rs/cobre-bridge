"""Decode NEWAVE FCF *visited-states* file (``cortese.dat``).

Companion to ``newave_cut_investigation.py`` (which reads ``cortes.dat``). NEWAVE
writes the forward-pass trial points where each cut was generated into
``cortese.dat`` when ``dger.dat`` line 90 ``ESTADOS GER. CORTES = 0`` *and* the run
does ``>= 2`` iterations.

Binary layout — **fully cracked** by matching the binary against a labelled
``estados.rel`` (parsed with ``inewave.nwlistcf.Estados``); ``fl[13:164]`` matched
``VARM`` 151/151 exactly.

Per **record** (each ``cortese.dat`` record shares the index of the ``cortes.dat``
cut generated at that trial point):

* Record size = ``file_size / n_slots`` (``tamanho_estado`` is 0 in the header).
* **3×int32 header** ``[_, SIMc, ITEf]`` (the first int is NOT a usable linked-list
  pointer — see below; ``SIMc`` = simulation/series index, ``ITEf`` = forward
  index), then N float64:

    ====================  ===========================================================
    ``float[0]``          ``FUNC.OBJ`` — objective value at the trial point
    ``float[1 : 1+R]``    ``EARM`` per REE (R = ``numero_rees``) — stored energy
    ``float[1+R : 1+R+U]``  ``VARM`` per UHE (U, ``dados_uhes`` order) — **useful**
                          storage volume (absolute − ``vmin``), hm³; run-of-river = 0
    ``float[1+R+U : ]``   ``MX_SAR`` + padding (the cut's per-UHE inflow lags ``VAF``
                          are not stored — the binary holds the compact state)
    ====================  ===========================================================

**Enumerating every visited state for a stage.** ``cortese.dat``'s own header is not
a navigable linked list, but it shares record indices with ``cortes.dat``, whose
first int32 IS the per-stage ``indice_proximo`` chain. So
:func:`read_cortese_stage` follows the **``cortes.dat``** linked list from the
stage's ``ultimo_registro_cortes_estagio`` to collect all records (every iteration ×
forward pass), reading state content from ``cortese.dat`` at those indices and the
construction iteration from the ``cortes.dat`` header.

⚠️ NEWAVE ``VARM`` is **useful volume** (absolute − ``vmin``, with VOLMIN/VMINT MODIF
overrides). Cobre stores **absolute** hm³ → convert Cobre to useful before comparing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from inewave.newave.cortesh import Cortesh

# int32 header sizes.
_CORTESE_HEADER_BYTES = 3 * 4  # [_, SIMc, ITEf]
_CORTES_HEADER_BYTES = 4 * 4  # [indice_proximo, iteracao, forward, desativacao]
_MAX_CHAIN = 100_000  # cycle/runaway guard


@dataclass(frozen=True)
class CorteseState:
    """One visited forward-pass trial point at a cut-generation node."""

    record: int  # 0-based physical record index (shared with cortes.dat)
    tipo_estagio: str  # "estudo" | "pos"
    estagio: int  # NEWAVE stage (1-based)
    iteracao: int  # construction iteration (from cortes.dat header)
    forward: int  # ITEf — forward-pass index (cortese header)
    simulacao: int  # SIMc — simulation/series index (cortese header)
    objective: float  # FUNC.OBJ — float[0]
    earm_ree: np.ndarray  # EARM per REE (stored energy)
    varm: pd.Series  # VARM per UHE code (useful volume hm³), dados_uhes order
    trailing: np.ndarray  # MX_SAR + padding


def _infer_record_size(file_size: int, cortesh: Cortesh) -> int:
    """Infer the ``cortese.dat`` record size (``tamanho_estado`` is 0 in practice)."""
    tam = int(getattr(cortesh, "tamanho_estado", 0) or 0)
    if tam > 0:
        return tam
    n_slots = int(cortesh.ultimo_registro_cortes_estagio["indice_ultimo_corte"].max())
    if n_slots <= 0 or file_size % n_slots != 0:
        raise ValueError(
            f"cannot infer cortese record size: file={file_size}, slots={n_slots}"
        )
    rec = file_size // n_slots
    if (rec - _CORTESE_HEADER_BYTES) % 8 != 0:
        raise ValueError(f"inferred record {rec} not header(12)+8k aligned")
    return rec


class _CorteseReader:
    """Holds the parsed files + geometry needed to materialise a state record."""

    def __init__(self, case_dir: Path) -> None:
        self.cortesh = Cortesh.read(str(case_dir / "cortesh.dat"))
        self.cortese = (case_dir / "cortese.dat").read_bytes()
        self.cortes = (case_dir / "cortes.dat").read_bytes()
        self.uhe_codes = self.cortesh.dados_uhes["codigo_usina"].tolist()
        n_ree = int(self.cortesh.numero_rees)
        self.varm_start = 1 + n_ree
        self.varm_end = self.varm_start + len(self.uhe_codes)
        self.rec_size = _infer_record_size(len(self.cortese), self.cortesh)
        self.n_floats = (self.rec_size - _CORTESE_HEADER_BYTES) // 8
        self.cortes_rec_size = int(self.cortesh.tamanho_corte)

    def _cortes_next(self, rec: int) -> int:
        """``indice_proximo`` (next record in the stage chain) from cortes.dat."""
        off = rec * self.cortes_rec_size
        return int(np.frombuffer(self.cortes[off : off + 4], dtype="<i4")[0])

    def _cortes_iteracao(self, rec: int) -> int:
        off = rec * self.cortes_rec_size
        hdr = self.cortes[off : off + _CORTES_HEADER_BYTES]
        return int(np.frombuffer(hdr, dtype="<i4")[1])

    def materialise(self, rec: int, tipo_estagio: str, estagio: int) -> CorteseState:
        base = rec * self.rec_size
        hdr = np.frombuffer(
            self.cortese[base : base + _CORTESE_HEADER_BYTES], dtype="<i4"
        )
        floats = np.frombuffer(
            self.cortese[
                base + _CORTESE_HEADER_BYTES : base
                + _CORTESE_HEADER_BYTES
                + 8 * self.n_floats
            ],
            dtype="<f8",
        )
        return CorteseState(
            record=rec,
            tipo_estagio=tipo_estagio,
            estagio=estagio,
            iteracao=self._cortes_iteracao(rec),
            forward=int(hdr[2]),
            simulacao=int(hdr[1]),
            objective=float(floats[0]),
            earm_ree=floats[1 : self.varm_start].copy(),
            varm=pd.Series(
                floats[self.varm_start : self.varm_end].copy(),
                index=self.uhe_codes,
                name="varm",
            ),
            trailing=floats[self.varm_end :].copy(),
        )


def read_cortese_stage(
    case_dir: Path | str,
    tipo_estagio: str = "estudo",
    estagio: int = 10,
) -> list[CorteseState]:
    """All visited states for one stage — every iteration × forward pass.

    Mirrors the ``cortes.dat`` reader: pick a stage, get the trial points from all
    iterations and forward passes that generated a cut there. Returns them ordered
    most-recent-first (the ``cortes.dat`` linked-list order), i.e. the latest
    iteration's nodes come first.
    """
    case_dir = Path(case_dir)
    r = _CorteseReader(case_dir)
    ult = r.cortesh.ultimo_registro_cortes_estagio
    sel = ult[(ult["tipo_estagio"] == tipo_estagio) & (ult["estagio"] == estagio)]
    if sel.empty:
        raise ValueError(f"no cuts for {tipo_estagio} stage {estagio}")
    idx = int(sel["indice_ultimo_corte"].iloc[0])

    out: list[CorteseState] = []
    seen: set[int] = set()
    while idx > 0 and idx not in seen and len(out) < _MAX_CHAIN:
        seen.add(idx)
        rec = idx - 1
        out.append(r.materialise(rec, tipo_estagio, estagio))
        idx = r._cortes_next(rec)
    return out


def read_cortese(case_dir: Path | str) -> list[CorteseState]:
    """Every visited state across all stages (every iteration × forward)."""
    case_dir = Path(case_dir)
    cortesh = Cortesh.read(str(case_dir / "cortesh.dat"))
    ult = cortesh.ultimo_registro_cortes_estagio
    out: list[CorteseState] = []
    for row in ult.itertuples(index=False):
        if int(row.indice_ultimo_corte) <= 0:
            continue
        out.extend(
            read_cortese_stage(case_dir, str(row.tipo_estagio), int(row.estagio))
        )
    return out


def _main() -> None:
    states = read_cortese_stage("./example/newave_rodada", "estudo", 10)
    print(f"estudo stage 10: {len(states)} visited states")
    for s in states:
        nz = int((s.varm.abs() > 1e-6).sum())
        print(
            f"  rec {s.record:>3} iter={s.iteracao} fwd={s.forward} "
            f"simc={s.simulacao}  obj={s.objective:,.0f}  "
            f"VARM nz={nz}/{s.varm.size} max={s.varm.max():.0f}"
        )


if __name__ == "__main__":
    _main()

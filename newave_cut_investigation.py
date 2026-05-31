"""Read NEWAVE FCF cut coefficients (``cortes.dat``).

The Benders/SDDP cuts define NEWAVE's future-cost function (FCF). ``cortes.dat`` is
a binary linked list; ``cortesh.dat`` is the header with the metadata to read it.

inewave selects the cut layout by ``codigos_rees`` FIRST (``modelos/cortes.py``):
non-empty → REE-aggregated (``pi_earm_ree``); ``codigos_rees=[]`` → individualized
(``pi_varm_uhe`` + ``pi_qafl_uhe_lag``). ⚠️ Passing BOTH silently forces the REE
layout and yields garbage on an individualized stage — so we pass
``codigos_rees=[]`` for individualized cuts. Read parameters are DERIVED from
``cortesh`` and cross-checked against the file size.

Units: NEWAVE cut values are in **10³ R$**; ``newave_water_values`` normalizes to
R$ (R$/hm³ for the water-value gradients) via ``NEWAVE_MONETARY_UNIT_RS``.

Importable API:
    read_newave_cut_coefficients(case_dir, tipo_estagio, estagio) -> (df, codigos_uhes)
    newave_water_values(case_dir, tipo_estagio, estagio, *, iteration, forward_pass,
                        reduce) -> DataFrame[codigo_usina, nome, water_value_newave]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from inewave.newave.confhd import Confhd
from inewave.newave.cortes import Cortes
from inewave.newave.cortesh import Cortesh
from inewave.newave.ree import Ree

ORDEM_MAXIMA_PARP = 12
AGREGADO_EM_REE = False
NEWAVE_MONETARY_UNIT_RS = 1.0e3  # NEWAVE cut values are in 10³ R$ -> ×1e3 = R$


def _reduce_rows(array: np.ndarray, reduce: str) -> np.ndarray:
    """Reduce a (n_cuts, …) array along axis 0 to a single cut's values."""
    if reduce == "first":
        return array[0]
    if reduce == "last":
        return array[-1]
    if reduce == "mean":
        return array.mean(axis=0)
    if reduce == "median":
        return np.median(array, axis=0)
    raise ValueError(f"unknown reduce={reduce!r} (use first|last|mean|median)")


def read_newave_cut_coefficients(
    case_dir: Path | str,
    tipo_estagio: str = "estudo",
    estagio: int = 10,
) -> tuple[pd.DataFrame, list[int]]:
    """Read the cuts for one (tipo_estagio, estagio); returns (df, codigos_uhes)."""
    case_dir = Path(case_dir)
    cortesh = Cortesh.read(str(case_dir / "cortesh.dat"))
    cortes_path = case_dir / "cortes.dat"

    tamanho_registro = int(cortesh.tamanho_corte)
    # Trust the cortesh header: it is the authority for THIS file's layout.
    # ⚠️ Do NOT fall back to ``or 2`` — when GNL anticipation is off
    # (``despacho_antecipado_gnl=0`` -> ``lag_maximo_gnl=0``, no ``pi_gnl``
    # columns), forcing 2 reserves 2 phantom GNL lags and MISALIGNS the whole
    # ``pi_varm`` block, re-attributing storage water values to the wrong UHE
    # codes and fabricating a spurious ``pi_gnl`` dual (was read as ~39% of the
    # cut). A legitimate 0 must be passed through as 0.
    lag_maximo_gnl = int(cortesh.lag_maximo_gnl or 0)
    ultimo = cortesh.ultimo_registro_cortes_estagio
    numero_total_cortes = int(ultimo["indice_ultimo_corte"].max())

    codigos_rees = Ree.read(str(case_dir / "ree.dat")).rees["codigo"].tolist()
    codigos_uhes = cortesh.dados_uhes["codigo_usina"].tolist()
    nsub = int(cortesh.numero_submercados)
    codigos_sub = cortesh.dados_submercados["codigo_submercado"].tolist()[:nsub]

    file_size = cortes_path.stat().st_size
    if tamanho_registro * numero_total_cortes != file_size:
        raise ValueError(
            f"tamanho_registro*numero_total_cortes "
            f"({tamanho_registro * numero_total_cortes}) != cortes.dat ({file_size})"
        )

    sel = ultimo[
        (ultimo["tipo_estagio"] == tipo_estagio) & (ultimo["estagio"] == estagio)
    ]
    if sel.empty or int(sel["indice_ultimo_corte"].iloc[0]) == 0:
        raise ValueError(f"no cuts for {tipo_estagio} stage {estagio}")
    indice_ultimo_corte = int(sel["indice_ultimo_corte"].iloc[0])

    arq = Cortes.read(
        str(cortes_path),
        tamanho_registro=tamanho_registro,
        indice_ultimo_corte=indice_ultimo_corte,
        numero_total_cortes=numero_total_cortes,
        codigos_rees=codigos_rees if AGREGADO_EM_REE else [],
        codigos_uhes=[] if AGREGADO_EM_REE else codigos_uhes,
        codigos_submercados=codigos_sub,
        ordem_maxima_parp=ORDEM_MAXIMA_PARP,
        lag_maximo_gnl=lag_maximo_gnl,
    )
    return arq.cortes, codigos_uhes


def uhe_code_to_name(case_dir: Path | str) -> dict[int, str]:
    """Map NEWAVE UHE code -> plant name (from confhd.dat)."""
    usinas = Confhd.read(str(Path(case_dir) / "confhd.dat")).usinas
    if "nome_usina" in usinas.columns:
        return dict(zip(usinas["codigo_usina"], usinas["nome_usina"]))
    return {}


def newave_water_values(
    case_dir: Path | str,
    tipo_estagio: str = "estudo",
    estagio: int = 10,
    *,
    iteration: int | None = None,
    forward_pass: int | None = None,
    reduce: str | None = None,
    select: str | None = None,
) -> pd.DataFrame:
    """Per-UHE water value (``pi_varm_uhe``, R$/hm³) for selected cut(s).

    ``select`` ('oldest'|'newest'|'first_real') picks one cut by construction order
    (``indice_corte``). 'first_real' skips NEWAVE's iteration-0 zero placeholder
    (its first stored cut). Applied after any iteration/forward_pass filter.

    Filter the cut pool with ``iteration`` (``iteracao_construcao``) and/or
    ``forward_pass`` (``indice_forward``). If more than one cut remains, pass
    ``reduce`` (first|last|mean|median); with ``reduce=None`` exactly one cut must
    remain (no implicit averaging). Returns ``[codigo_usina, nome,
    water_value_newave]``; ``.attrs`` carries ``rhs`` and ``n_cuts``.
    """
    df, codigos_uhes = read_newave_cut_coefficients(case_dir, tipo_estagio, estagio)
    sel = df
    if iteration is not None:
        sel = sel[sel["iteracao_construcao"] == iteration]
    if forward_pass is not None:
        sel = sel[sel["indice_forward"] == forward_pass]
    if len(sel) == 0:
        raise ValueError(
            f"no NEWAVE cut for iteration={iteration}, forward_pass={forward_pass} "
            f"at {tipo_estagio} stage {estagio}"
        )
    if select == "oldest":  # first cut built (min indice_corte; NEWAVE placeholder)
        sel = sel.loc[[sel["indice_corte"].idxmin()]]
    elif select == "newest":  # last cut built = latest iteration
        sel = sel.loc[[sel["indice_corte"].idxmax()]]
    elif select == "first_real":  # skip the iteration-0 / forward-0 zero placeholder
        real = sel[sel["indice_forward"] >= 1]
        if real.empty:
            raise ValueError("no real NEWAVE cut (indice_forward>=1) found")
        sel = real.loc[[real["indice_corte"].idxmin()]]
    elif select is not None:
        raise ValueError(f"unknown select={select!r} (use oldest|newest|first_real)")
    if len(sel) > 1 and reduce is None:
        raise ValueError(
            f"{len(sel)} NEWAVE cuts match; pass select=, reduce=, or filter "
            "further (iteration / forward_pass)"
        )

    used = [u for u in codigos_uhes if f"pi_varm_uhe{u}" in df.columns]
    cols = [f"pi_varm_uhe{u}" for u in used]
    coef = _reduce_rows(sel[cols].to_numpy(dtype=float), reduce or "first")
    rhs = float(_reduce_rows(sel["rhs"].to_numpy(dtype=float), reduce or "first"))

    names = uhe_code_to_name(case_dir)
    out = pd.DataFrame(
        {
            "codigo_usina": used,
            "nome": [names.get(u, "") for u in used],
            "water_value_newave": coef * NEWAVE_MONETARY_UNIT_RS,
        }
    )
    out.attrs["rhs"] = rhs * NEWAVE_MONETARY_UNIT_RS
    out.attrs["n_cuts"] = int(len(sel))
    return out


def _main() -> None:
    case_dir = Path("./example/newave_rodada")
    wv = newave_water_values(case_dir, "estudo", 10, select="oldest")
    print(
        f"NEWAVE estudo stage 10, primeiro corte: {wv.attrs['n_cuts']} corte, "
        f"rhs={wv.attrs['rhs']:.4e} R$"
    )
    top = wv.reindex(wv["water_value_newave"].abs().sort_values(ascending=False).index)
    print("top 12 |pi_varm_uhe| (R$/hm³):")
    for _, r in top.head(12).iterrows():
        print(
            f"  UHE {int(r['codigo_usina']):>3} {r['nome']:<13} "
            f"{r['water_value_newave']:>16.2f}"
        )


if __name__ == "__main__":
    _main()

"""Resolve NEWAVE fictitious-plant cascade chains for real plants.

NEWAVE's ``confhd.dat`` encodes the physical water-balance cascade via
``codigo_usina_jusante``.  For some real plants this link is 0 (terminal)
or points to a fictitious plant whose role is purely topological — a
``FICT.<NAME>`` entity used only for energy-cascade accounting.

Cobre-bridge filters fictitious plants out of the LP (they have no
turbine, no reservoir, ``produtibilidade_especifica = 0``) but it must
preserve the cascade connectivity they provide so that:

- Real-plant ``downstream_id`` in ``hydros.json`` points to the next
  real plant in the cascade.
- Accumulated cascade productivities (used by penalty conversion and
  ``EARM = (V − V_min) · ρ_acum`` accounting) correctly include any
  FICT plant's own productivity along the chain.

This module is the single source of truth for that resolution.  Both
``converters.hydro.convert_hydros`` and
``converters.constraints._build_hydro_downstream_map`` consume it so the
LP topology and the cascade accounting agree.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from cobre_bridge.productivity import compute_productivity

logger = logging.getLogger(__name__)

# NEWAVE truncates plant names to a 12-character column in confhd.dat, so
# a fictitious plant's name is ``"FICT." + real_name[:7]`` after right-strip.
# Anything shorter than 7 chars is kept verbatim.  This is the source of the
# canonical FICT ↔ real-plant name mapping.
_FICT_NAME_PREFIX = "FICT."
_FICT_NAME_SUFFIX_LEN = 7


def _real_to_fict_key(name: str) -> str:
    """Return the FICT-suffix key produced by NEWAVE's 12-char truncation."""
    return name.strip()[:_FICT_NAME_SUFFIX_LEN].rstrip()


def _fict_to_key(fict_name: str) -> str:
    """Return the FICT-suffix key from a ``FICT.<X>`` plant name."""
    return fict_name.strip().removeprefix(_FICT_NAME_PREFIX).rstrip()


@dataclass(frozen=True)
class FictCascadeResolution:
    """Resolved cascade for a single real plant.

    Attributes
    ----------
    downstream_code:
        NEWAVE code of the next real plant in the cascade, or ``None`` when
        the chain terminates at the sea.
    fict_chain:
        Ordered list of FICT-plant NEWAVE codes traversed between the real
        plant and ``downstream_code``.  Empty when no FICT plants lie on
        the path.
    fict_rho_sum:
        Cumulative ``ρ_eq`` of the FICT plants in ``fict_chain``.  This
        must be folded into the upstream real plant's effective ``ρ_eq``
        so that cobre's cascade sum reproduces NEWAVE's
        ``produtibilidade_acumulada_calculo_earm``.
    """

    downstream_code: int | None
    fict_chain: tuple[int, ...]
    fict_rho_sum: float


def resolve_cascade(
    confhd_df: pd.DataFrame,
    cadastro: pd.DataFrame,
) -> dict[int, FictCascadeResolution]:
    """Resolve the effective cascade downstream of every real (existing) plant.

    Resolution rules, applied per real plant ``P``:

    1. If ``P.codigo_usina_jusante`` is non-zero and points to a real
       plant ``R``, use ``R`` as the downstream.  No FICT contribution.
    2. If ``P.codigo_usina_jusante`` is non-zero and points to a FICT
       plant ``F``, walk ``F``'s ``codigo_usina_jusante`` chain through
       any further FICT plants until a real plant or terminal (0) is
       reached.  Accumulate the ``ρ_eq`` of every FICT plant traversed.
    3. If ``P.codigo_usina_jusante`` is 0 (physically terminal) but
       ``FICT.<NAME>`` exists in confhd (by name match against ``P``'s
       ``nome_usina``), treat ``FICT.<NAME>`` as the implicit downstream
       and walk its chain as in rule 2.  This is NEWAVE's convention for
       energy-cascade accounting on plants that do not physically flow
       to a downstream reservoir.
    4. Otherwise the cascade terminates: ``downstream_code = None``.

    Parameters
    ----------
    confhd_df:
        ``Confhd.usinas`` — must contain at least ``codigo_usina``,
        ``nome_usina``, ``codigo_usina_jusante``, ``usina_existente``.
    cadastro:
        ``Hidr.cadastro`` with MODIF.DAT overrides already applied.  Used
        to compute the FICT plants' ``ρ_eq``.  FICT plants whose code is
        absent from ``cadastro`` contribute zero productivity.

    Returns
    -------
    dict[int, FictCascadeResolution]
        One entry per real existing plant in ``confhd_df``.
    """
    # Index every row (EX, NE, NC) so the cascade walker can step through
    # plants that exist in confhd.dat but are absent from the LP.  ``NE``
    # ("não existirá") and ``NC`` ("não construída") plants are out-of-LP
    # — they contribute no productivity and no LP variable — but their
    # ``codigo_usina_jusante`` link is still the authoritative topology.
    # Treating them as opaque terminals (the previous behavior) silently
    # disconnects any real plant whose downstream is NE/NC, leaving the
    # upstream side without an outlet.  Walking through them mirrors the
    # FICT pass-through but skips the ρ_eq contribution since the plant
    # is not physically present.  The ``fict_by_key`` map uses the
    # FICT-suffix produced by NEWAVE's 12-char truncation rule so that
    # ``FICT.QUEIMAD`` resolves to ``QUEIMADO``, ``FICT.TRES MA`` to
    # ``TRES MARIAS``, etc.  Real plants that share the same truncated
    # 7-char prefix (e.g. ``CORUMBA I`` / ``CORUMBA III`` / ``CORUMBA IV``)
    # are recorded in ``real_by_key`` so we can detect ambiguous matches
    # and warn rather than picking arbitrarily.
    row_by_code: dict[int, pd.Series] = {}
    fict_by_key: dict[str, int] = {}
    real_codes: set[int] = set()
    absent_codes: set[int] = set()
    real_by_key: dict[str, list[int]] = defaultdict(list)
    for _, row in confhd_df.iterrows():
        code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        status = str(row["usina_existente"]).strip()
        row_by_code[code] = row
        if status != "EX":
            absent_codes.add(code)
            continue
        if name.startswith(_FICT_NAME_PREFIX):
            fict_by_key[_fict_to_key(name)] = code
        else:
            real_codes.add(code)
            real_by_key[_real_to_fict_key(name)].append(code)

    fict_rho_eq: dict[int, float] = {}
    for code, row in row_by_code.items():
        name = str(row["nome_usina"]).strip()
        if not name.startswith("FICT."):
            continue
        if code in absent_codes:
            # Out-of-LP plants contribute no productivity even when their
            # name carries the FICT prefix — they are simply absent.
            continue
        if code in cadastro.index:
            fict_rho_eq[code] = float(compute_productivity(cadastro.loc[code]))
        else:
            fict_rho_eq[code] = 0.0

    def _walk_chain(
        start_code: int,
    ) -> tuple[int | None, list[int], float]:
        """Walk transparently through FICT and absent (NE/NC) plants.

        Returns ``(real_downstream, fict_codes_traversed, rho_sum)``.
        Stops when reaching a real plant (returned), terminal (None), an
        unknown code, or when a cycle is detected (defensive — should not
        occur in well-formed NEWAVE cases).  FICT plants accumulate their
        ``ρ_eq`` into ``rho_sum`` and appear in ``fict_codes_traversed``;
        NE/NC plants are skipped silently as topological pass-throughs.
        """
        fict_codes: list[int] = []
        rho_sum = 0.0
        cur_code: int | None = start_code
        visited: set[int] = set()
        while cur_code is not None and cur_code != 0:
            if cur_code in visited:
                return None, fict_codes, rho_sum
            visited.add(cur_code)
            row = row_by_code.get(cur_code)
            if row is None:
                return None, fict_codes, rho_sum
            if cur_code in real_codes:
                return cur_code, fict_codes, rho_sum
            if cur_code not in absent_codes:
                # Real-status FICT plant — pick up its ρ_eq contribution.
                fict_codes.append(cur_code)
                rho_sum += fict_rho_eq.get(cur_code, 0.0)
            # NE/NC and FICT plants alike are walked through; only their
            # ρ_eq handling differs (NE/NC contributes nothing).
            ds_raw = row.get("codigo_usina_jusante")
            cur_code = int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0
        return None, fict_codes, rho_sum

    result: dict[int, FictCascadeResolution] = {}
    for code in real_codes:
        row = row_by_code[code]
        name = str(row["nome_usina"]).strip()
        ds_raw = row.get("codigo_usina_jusante")
        ds_code = int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0

        if ds_code != 0:
            if ds_code in real_codes:
                # Rule 1 — direct real downstream.
                result[code] = FictCascadeResolution(
                    downstream_code=ds_code,
                    fict_chain=(),
                    fict_rho_sum=0.0,
                )
                continue
            # Rule 2 — confhd points to a FICT or NE/NC plant.  Walk the
            # chain to find the next real (EX) plant downstream.
            real_ds, fict_codes, rho_sum = _walk_chain(ds_code)
            result[code] = FictCascadeResolution(
                downstream_code=real_ds,
                fict_chain=tuple(fict_codes),
                fict_rho_sum=rho_sum,
            )
            continue

        # Rule 3 — physically terminal.  Check for a name-matched FICT
        # plant that carries the energy-cascade link.  The match uses the
        # 7-char truncation rule (``_real_to_fict_key``) so that real
        # plants whose name exceeds 7 chars still resolve (e.g.
        # ``QUEIMADO`` ↔ ``FICT.QUEIMAD``).
        key = _real_to_fict_key(name)
        fict_code = fict_by_key.get(key)
        if fict_code is None:
            result[code] = FictCascadeResolution(
                downstream_code=None,
                fict_chain=(),
                fict_rho_sum=0.0,
            )
            continue
        # If a FICT plant exists for this prefix BUT the prefix is shared
        # by multiple real plants (e.g. ``CORUMBA I`` / ``CORUMBA III`` /
        # ``CORUMBA IV``), we cannot tell which one the FICT actually
        # corresponds to.  Refuse to link rather than silently
        # misattribute — better a known-broken EARM than a silently wrong
        # cascade attribution — and log a warning so the user can
        # investigate.
        ambiguous_siblings = [
            row_by_code[c]["nome_usina"] for c in real_by_key.get(key, []) if c != code
        ]
        if ambiguous_siblings:
            logger.warning(
                "FICT-cascade resolution for '%s' (code %d) is ambiguous: "
                "its 7-char truncation '%s' is also shared by %s, and a "
                "matching FICT plant exists. Leaving downstream as "
                "terminal to avoid wrong attribution.",
                name,
                code,
                key,
                [str(n).strip() for n in ambiguous_siblings],
            )
            result[code] = FictCascadeResolution(
                downstream_code=None,
                fict_chain=(),
                fict_rho_sum=0.0,
            )
            continue
        real_ds, fict_codes, rho_sum = _walk_chain(fict_code)
        result[code] = FictCascadeResolution(
            downstream_code=real_ds,
            fict_chain=tuple(fict_codes),
            fict_rho_sum=rho_sum,
        )

    # Log NE/NC plants that bridged a real-to-real cascade so the operator
    # can audit the bypassed mid-cascade links.  Top-of-cascade NE/NC
    # leaves are intentionally not reported — they have no upstream to
    # rewire and would only add noise.
    rewired_absent: dict[int, list[int]] = defaultdict(list)
    for upstream_code in result:
        ds_raw = row_by_code[upstream_code].get("codigo_usina_jusante")
        ds_seed = int(ds_raw) if ds_raw is not None and not pd.isna(ds_raw) else 0
        if ds_seed == 0 or ds_seed not in absent_codes:
            continue
        # The original confhd link landed on an NE/NC plant; the walker
        # found the next EX downstream (possibly through more NE/NC or
        # FICT hops).  Attribute every NE/NC along that path to this
        # upstream so a single log entry summarizes who got rewired.
        cur: int = ds_seed
        seen: set[int] = set()
        while cur not in (0,) and cur not in seen and cur not in real_codes:
            seen.add(cur)
            if cur in absent_codes:
                rewired_absent[cur].append(upstream_code)
            row = row_by_code.get(cur)
            if row is None:
                break
            ds_inner = row.get("codigo_usina_jusante")
            cur = int(ds_inner) if ds_inner is not None and not pd.isna(ds_inner) else 0
    for absent_code, upstreams in rewired_absent.items():
        status = str(row_by_code[absent_code]["usina_existente"]).strip()
        absent_name = str(row_by_code[absent_code]["nome_usina"]).strip()
        upstream_names = [str(row_by_code[u]["nome_usina"]).strip() for u in upstreams]
        logger.info(
            "Bypassing %s plant '%s' (code %d): rewired %d upstream(s) %s "
            "to its downstream in the cascade.",
            status,
            absent_name,
            absent_code,
            len(upstreams),
            upstream_names,
        )

    # Warn about orphan FICT plants (no real plant name resolves to them)
    # so the user can investigate unconventional name encodings.
    resolved_fict_codes = {c for r in result.values() for c in r.fict_chain}
    unmatched: list[str] = []
    for key, fcode in fict_by_key.items():
        if fcode in resolved_fict_codes:
            continue
        if key not in real_by_key:
            unmatched.append(str(row_by_code[fcode]["nome_usina"]).strip())
    if unmatched:
        logger.warning(
            "FICT plants not matched to any real plant by 7-char prefix: %s. "
            "These plants contribute no cascade attribution to a real plant.",
            unmatched,
        )

    return result

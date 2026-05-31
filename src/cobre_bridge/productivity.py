"""NEWAVE hydro productivity functions (presentation-free domain module).

Three ways NEWAVE turns a plant's volume→cota polynomial and tailrace level into
a MW/(m³/s) productivity, all **pure functions over a ``Hidr.cadastro`` row**
(``hreg``):

- :func:`compute_productivity` — point ρ at a single reference volume (the LP's
  ``gen = ρ·Q`` coefficient); NEWAVE ``produtibilidade_altura_65``.
- :func:`equivalent_productivity` — PRODT, the mean head over ``[vmin, vmax]``
  (used to convert PENALID R$/MWh penalties to the flow domain).
- :func:`integrated_productivity` — the volume-integrated ρ over the useful
  range (NEWAVE's stored-energy / EARM convention, used by VminOP).

These were private helpers in ``hydro.py`` that the constraint and FICT-cascade
converters reached into across module boundaries; hosting them here gives them
public names and one home. ``hydro.py`` keeps ``_compute_productivity`` /
``_equivalent_productivity`` / ``_compute_integrated_productivity`` aliases for
its own callers.
"""

from __future__ import annotations

import logging

import pandas as pd

_LOG = logging.getLogger(__name__)


def compute_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
    useful_volume_override: float | None = None,
) -> float:
    """Compute constant point productivity in MW/(m^3/s) for a hydro plant.

    Reads polynomial coefficients ``a0_volume_cota`` through ``a4_volume_cota``
    from the plant's cadastro row to map storage volume (hm3) to upstream height
    (m).  Subtracts the tailrace level to obtain gross drop, applies the loss
    model defined by ``tipo_perda`` and ``perdas``, then multiplies by
    ``produtibilidade_especifica``.

    Reference-volume selection (in priority order):

    1. ``useful_volume_override`` — explicit useful volume (hm³ above
       ``volume_minimo``): ``V = volume_minimo + useful_volume_override``.
    2. Monthly-regulated plants (``tipo_regulacao == "M"``) → 65% of useful
       storage (``V = vmin + 0.65 × (vmax − vmin)``); matches NEWAVE's
       ``produtibilidade_altura_65`` convention.
    3. All other plant types → ``volume_referencia``.

    ``cmont_override`` short-circuits the upstream polynomial entirely.
    """
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]

    canal_fuga = (
        canal_fuga_override
        if canal_fuga_override is not None
        else float(hreg["canal_fuga_medio"])
    )

    if cmont_override is not None:
        # CMONT supplies the upstream level directly.
        net_drop = cmont_override - canal_fuga
    else:
        if all(c == 0.0 for c in coeffs):
            _LOG.warning(
                "All volume_cota coefficients are zero for plant; "
                "returning zero productivity.",
                extra={"plant": hreg.get("nome_usina", "unknown")},
            )
            return 0.0

        def _poly(v: float) -> float:
            """Evaluate h(v) = c0 + c1*v + c2*v^2 + c3*v^3 + c4*v^4."""
            return (
                coeffs[0]
                + coeffs[1] * v
                + coeffs[2] * v**2
                + coeffs[3] * v**3
                + coeffs[4] * v**4
            )

        vol_min = float(hreg["volume_minimo"])

        if useful_volume_override is not None:
            net_drop = _poly(vol_min + useful_volume_override) - canal_fuga
        else:
            tipo_regulacao = str(hreg["tipo_regulacao"]).strip()
            vol_max = float(hreg["volume_maximo"])
            if tipo_regulacao == "M":
                v_65 = vol_min + 0.65 * (vol_max - vol_min)
                net_drop = _poly(v_65) - canal_fuga
            else:
                vol_ref = float(hreg["volume_referencia"])
                net_drop = _poly(vol_ref) - canal_fuga

    # Apply loss model.
    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    if tipo_perda == 1:
        # Multiplicative factor (perdas is a percentage, e.g. 2.35 = 2.35%).
        adjusted_drop = net_drop * (1.0 - perdas / 100.0)
    elif tipo_perda == 2:
        # Additive meters: adjusted_drop = net_drop - perdas
        adjusted_drop = net_drop - perdas
    else:
        adjusted_drop = net_drop

    produtibilidade = float(hreg["produtibilidade_especifica"])
    return produtibilidade * adjusted_drop


def equivalent_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> float:
    """Compute NEWAVE's PRODT (``produtibilidade_equivalente_volmin_volmax``).

    PRODT is the *equivalent* productivity from the minimum to the maximum
    operative volume (pmo.dat: "PROD. EQUIVALENTE (DO VOL. MINIMO AO VOL.
    MAXIMO)"). Unlike :func:`compute_productivity`, which evaluates the head at a
    single reference volume, PRODT uses the **mean upstream head over the
    useful-volume range**, obtained analytically from the volume→cota
    polynomial::

        h_eq = (1 / (Vmax - Vmin)) ∫_{Vmin}^{Vmax} (a0 + a1 V + … + a4 V⁴) dV
             = Σ_i a_i (Vmax^{i+1} - Vmin^{i+1}) / ((i+1)(Vmax - Vmin))

    productivity = ``produtibilidade_especifica × (h_eq − canal_fuga − perdas)``
    with the same loss model as :func:`compute_productivity`.

    Run-of-river plants (``Vmax == Vmin``) fall back to the point head at
    ``Vmin``; all-zero polynomials return 0 (matches NEWAVE).

    ``canal_fuga_override`` / ``cmont_override`` carry MODIF.DAT CFUGA / CMONT
    temporal overrides: CFUGA replaces the mean tailrace; CMONT pins the upstream
    level to a constant (so the head is ``cmont − cfuga``, no integral).
    """
    canal_fuga = (
        canal_fuga_override
        if canal_fuga_override is not None
        else float(hreg["canal_fuga_medio"])
    )

    if cmont_override is not None:
        net_drop = cmont_override - canal_fuga
    else:
        coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
        if all(c == 0.0 for c in coeffs):
            return 0.0

        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        if vol_max - vol_min < 1e-9:
            h_eq = sum(coeffs[i] * vol_min**i for i in range(5))
        else:
            integral = sum(
                coeffs[i] * (vol_max ** (i + 1) - vol_min ** (i + 1)) / (i + 1)
                for i in range(5)
            )
            h_eq = integral / (vol_max - vol_min)

        net_drop = h_eq - canal_fuga

    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    if tipo_perda == 1:
        net_drop *= 1.0 - perdas / 100.0
    elif tipo_perda == 2:
        net_drop -= perdas

    return float(hreg["produtibilidade_especifica"]) * net_drop


def integrated_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> float:
    """ρ_esp × ((1/useful) × ∫_vmin^vmax h(V) dV − cf − perdas).

    Mirrors NEWAVE's ``produtibilidade_equivalente_volmin_volmax``: the
    productivity averaged over the full useful storage range, used by NEWAVE to
    convert reservoir volume to stored energy (EARM) and to evaluate VminOP
    constraints.  Different from the point productivity at v_65 that
    :func:`compute_productivity` returns and that the LP uses as the
    ``gen = ρ·Q`` coefficient.

    For a polynomial ``h(V) = a0 + a1·V + ... + a4·V⁴`` the integral has a closed
    form: ``F(V) = a0·V + a1·V²/2 + a2·V³/3 + a3·V⁴/4 + a4·V⁵/5``.  With
    ``cmont_override`` the upstream level is held constant, so the integrated
    drop collapses to ``cmont − cf``.

    Run-of-river plants (vmax == vmin) evaluate the polynomial at the single
    operating point — equivalent to the point productivity.
    """
    if cmont_override is not None:
        cf = (
            canal_fuga_override
            if canal_fuga_override is not None
            else float(hreg["canal_fuga_medio"])
        )
        net_drop = cmont_override - cf
    else:
        coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
        if all(c == 0.0 for c in coeffs):
            _LOG.warning(
                "All volume_cota coefficients are zero for plant; "
                "returning zero integrated productivity.",
                extra={"plant": hreg.get("nome_usina", "unknown")},
            )
            return 0.0

        vmin = float(hreg["volume_minimo"])
        vmax = float(hreg["volume_maximo"])
        cf = (
            canal_fuga_override
            if canal_fuga_override is not None
            else float(hreg["canal_fuga_medio"])
        )

        if vmax - vmin <= 0.0:
            # Run-of-river: integrate over the singleton {vmin}.
            avg_h = sum(coeffs[i] * vmin**i for i in range(5))
        else:

            def _antideriv(v: float) -> float:
                return (
                    coeffs[0] * v
                    + coeffs[1] * v**2 / 2.0
                    + coeffs[2] * v**3 / 3.0
                    + coeffs[3] * v**4 / 4.0
                    + coeffs[4] * v**5 / 5.0
                )

            avg_h = (_antideriv(vmax) - _antideriv(vmin)) / (vmax - vmin)

        net_drop = avg_h - cf

    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    if tipo_perda == 1:
        adjusted_drop = net_drop * (1.0 - perdas / 100.0)
    elif tipo_perda == 2:
        adjusted_drop = net_drop - perdas
    else:
        adjusted_drop = net_drop

    return float(hreg["produtibilidade_especifica"]) * adjusted_drop

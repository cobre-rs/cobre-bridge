"""Export presentation-grade NEWAVE-vs-Cobre charts for a slide deck.

Renders static, high-resolution figures that pitch Cobre as a drop-in
replacement for NEWAVE, reusing the same ``compare_results`` pipeline that
backs ``cobre-bridge compare results`` (no NEWAVE/Cobre re-parsing):

  Slide 1  ``cmo_by_bus.{png,pdf}``     2x2 submarket panels of marginal cost
                                        (CMO, R$/MWh): NEWAVE mean vs Cobre
                                        mean + Cobre P10-P90 band.
  Slide 2  ``thermal_system.{png,pdf}`` system thermal generation (MW).
  Slide 3  ``total_storage.{png,pdf}``  system total storage (hm³).
  Slide 4  ``hydro_system.{png,pdf}``   system hydro generation (MW).
  Slide 5  ``spillage_system.{png,pdf}``system total spillage (m³/s).

Slides 2-5 plot the system-wide total: NEWAVE vs Cobre mean (summed across
matched plants) plus a *true* per-scenario P10-P90 band (computed from the
Cobre simulation parquets, not a sum of per-plant percentiles).

Unlike the interactive HTML report this uses matplotlib for full typographic
control and exports both a 300-DPI PNG (drop into PowerPoint) and a vector PDF
(crisp at any projector size).

Requires matplotlib (not a package dependency)::

    VIRTUAL_ENV=.venv uv pip install matplotlib

Usage::

    python scripts/presentation_charts.py \\
        --newave example/newave \\
        --cobre  example/cobre/output \\
        --outdir example/presentation
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from cobre_bridge.case import NewaveCase
from cobre_bridge.cli import _load_lines_json
from cobre_bridge.comparators.alignment import build_entity_alignment
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    compare_results,
)

# --- Palette (consistent with the comparison HTML, ui/theme.py) -------------
C_NEWAVE = "#F5A623"  # amber — the published reference
C_COBRE = "#4A90B8"  # blue — the simulated replacement
C_BAND = (74 / 255, 144 / 255, 184 / 255, 0.16)  # light Cobre fill
_BUS_ORDER = ["SUDESTE", "SUL", "NORDESTE", "NORTE"]
_MONTHS = [
    "Jan",
    "Fev",
    "Mar",
    "Abr",
    "Mai",
    "Jun",
    "Jul",
    "Ago",
    "Set",
    "Out",
    "Nov",
    "Dez",
]

# pt-BR variable names used in the chart title ("{prefix} - {var}").
VAR_CMO_PT = "Custo Marginal de Operação (CMO)"
VAR_THERMAL_PT = "Geração Térmica"
VAR_STORAGE_PT = "Armazenamento Total"
VAR_HYDRO_PT = "Geração Hidráulica"
VAR_SPILLAGE_PT = "Vertimento Total"
_DEFAULT_TITLE_PREFIX = "PMO SET/24 3 Anos (Individualizado)"

# Per-plant slides are emitted for each of these hydro plants (matched
# case-insensitively against the NEWAVE plant name, which equals the Cobre
# hydro name).  Tune this list freely — unknown names are skipped with a
# warning.
PLANTS: list[str] = ["TUCURUI", "ITAIPU"]

# Hydro variables exported both system-wide and per-plant. Each entry is
# ``(column, var_pt, ylabel, flow, system_stem, plant_stem)`` where ``flow``
# marks a per-block flow (block-hours-weighted) vs. an end-of-stage stock,
# and the two stems name the output files.
_HYDRO_VARS: list[tuple[str, str, str, bool, str, str]] = [
    (
        "storage_final_hm3",
        VAR_STORAGE_PT,
        "Armazenamento  (hm³)",
        False,
        "total_storage",
        "storage",
    ),
    (
        "generation_mw",
        VAR_HYDRO_PT,
        "Geração hidráulica  (MW)",
        True,
        "hydro_system",
        "generation",
    ),
    (
        "spillage_m3s",
        VAR_SPILLAGE_PT,
        "Vertimento  (m³/s)",
        True,
        "spillage_system",
        "spillage",
    ),
]


def _setup_style() -> None:
    """Apply a clean, editorial matplotlib style for slide figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#E8EDF3",
            "grid.linewidth": 0.9,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "axes.labelcolor": "#334155",
            "text.color": "#0F172A",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# ---------------------------------------------------------------------------
# Data loading (reuses the compare-results pipeline)
# ---------------------------------------------------------------------------


def load_comparison(
    newave_dir: Path, cobre_output_dir: Path, tolerance: float = 1e-2
) -> tuple[list[ResultComparison], PercentileData]:
    """Run the NEWAVE-vs-Cobre comparison and return its raw results."""
    case = NewaveCase.from_directory(newave_dir)
    id_map = case.id_map
    alignment = build_entity_alignment(id_map, case, _load_lines_json(cobre_output_dir))
    return compare_results(
        case=case,
        id_map=id_map,
        alignment=alignment,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
    )


def _case_dir(cobre_output_dir: Path) -> Path:
    """Locate the Cobre case dir (holding ``stages.json``) from the output dir."""
    if (cobre_output_dir / "stages.json").exists():
        return cobre_output_dir
    return cobre_output_dir.parent


def _stage_labels(cobre_output_dir: Path) -> dict[int, str]:
    """Map ``stage_id`` -> ``"Mon/YY"`` from the case ``stages.json``."""
    path = _case_dir(cobre_output_dir) / "stages.json"
    labels: dict[int, str] = {}
    if not path.exists():
        return labels
    stages = json.loads(path.read_text(encoding="utf-8")).get("stages", [])
    for stage in stages:
        sid = int(stage["id"])
        start = stage.get("start_date")
        if start:
            d = date.fromisoformat(start)
            labels[sid] = f"{_MONTHS[d.month - 1]}/{d.year % 100:02d}"
        else:
            labels[sid] = str(sid)
    return labels


# ---------------------------------------------------------------------------
# Series extraction
# ---------------------------------------------------------------------------


def cmo_by_bus(
    results: list[ResultComparison], pct: PercentileData
) -> dict[str, dict[str, list[float]]]:
    """Per-bus spot-price (CMO) series: NEWAVE/Cobre means + P10-P90 band."""
    by_bus: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    cobre_id: dict[str, int] = {}
    for r in results:
        if r.entity_type == "bus" and r.variable == "spot_price":
            name = r.entity_name.upper()
            by_bus[name][r.stage] = (r.newave_value, r.cobre_value)
            cobre_id[name] = r.cobre_id

    band: dict[tuple[int, int], tuple[float | None, float | None]] = {}
    bp = pct.bus
    if bp is not None and not bp.is_empty() and "spot_price_p10" in bp.columns:
        for row in bp.iter_rows(named=True):
            band[(int(row["entity_id"]), int(row["stage_id"]))] = (
                row.get("spot_price_p10"),
                row.get("spot_price_p90"),
            )

    out: dict[str, dict[str, list[float]]] = {}
    ordered = [b for b in _BUS_ORDER if b in by_bus]
    ordered += [b for b in sorted(by_bus) if b not in ordered]
    for name in ordered:
        stages = sorted(by_bus[name])
        eid = cobre_id[name]
        out[name] = {
            "stages": [float(s) for s in stages],
            "nw": [by_bus[name][s][0] for s in stages],
            "cb": [by_bus[name][s][1] for s in stages],
            "p10": [
                float(band.get((eid, s), (None, None))[0] or "nan") for s in stages
            ],
            "p90": [
                float(band.get((eid, s), (None, None))[1] or "nan") for s in stages
            ],
        }
    return out


def system_lines(
    results: list[ResultComparison], entity_type: str, variable: str
) -> tuple[list[int], list[float], list[float]]:
    """System-total of *variable* per stage: (stages, NEWAVE, Cobre).

    Sums the matched per-entity ``ResultComparison`` values for the given
    ``entity_type`` / ``variable`` across every plant in each stage. Because
    both sides come from the matched comparison rows, the NEWAVE-side unit
    handling already baked into ``compare_results`` carries through — e.g.
    hydro storage already has each plant's ``min_storage_hm3`` added back.
    """
    nw: dict[int, float] = defaultdict(float)
    cb: dict[int, float] = defaultdict(float)
    for r in results:
        if r.entity_type == entity_type and r.variable == variable:
            nw[r.stage] += r.newave_value
            cb[r.stage] += r.cobre_value
    stages = sorted(nw)
    return stages, [nw[s] for s in stages], [cb[s] for s in stages]


def thermal_system_band(
    cobre_output_dir: Path, stages: list[int]
) -> dict[int, tuple[float, float]]:
    """True per-scenario P10-P90 of *system* thermal MW for each stage.

    Sums ``generation_mwh`` across blocks and thermals per (scenario, stage),
    divides by the stage's total block-hours to recover a block-hours-weighted
    stage-mean MW per scenario (matching how the comparison defines Cobre's
    thermal ``generation_mw``), then takes percentiles across scenarios.
    """
    case_dir = _case_dir(cobre_output_dir)
    stages_json = json.loads((case_dir / "stages.json").read_text(encoding="utf-8"))
    stage_hours = pl.DataFrame(
        {
            "stage_id": [int(s["id"]) for s in stages_json["stages"]],
            "stage_hours": [
                float(sum(b["hours"] for b in s["blocks"]))
                for s in stages_json["stages"]
            ],
        }
    )
    glob = str(cobre_output_dir / "simulation" / "thermals" / "**" / "*.parquet")
    per_scen = (
        pl.scan_parquet(glob, hive_partitioning=True)
        .group_by(["scenario_id", "stage_id"])
        .agg(pl.col("generation_mwh").sum().alias("e"))
        .collect()
        .join(stage_hours, on="stage_id")
        .with_columns((pl.col("e") / pl.col("stage_hours")).alias("mw"))
    )
    band = (
        per_scen.group_by("stage_id")
        .agg(
            p10=pl.col("mw").quantile(0.1),
            p90=pl.col("mw").quantile(0.9),
        )
        .sort("stage_id")
    )
    wanted = set(stages)
    return {
        int(r["stage_id"]): (float(r["p10"]), float(r["p90"]))
        for r in band.iter_rows(named=True)
        if int(r["stage_id"]) in wanted
    }


def hydro_system_band(
    cobre_output_dir: Path,
    stages: list[int],
    column: str,
    *,
    flow: bool,
    entity_ids: set[int] | None = None,
) -> dict[int, tuple[float, float]]:
    """True per-scenario P10-P90 of a *system-total* hydro variable per stage.

    Sums ``column`` across hydro plants within each scenario/stage, then
    takes percentiles across scenarios — the hydro analogue of
    :func:`thermal_system_band`:

      * ``flow=True``  (``generation_mw``, ``spillage_m3s``): a per-block flow,
        collapsed to a block-hours-weighted stage mean per scenario (matching
        how ``read_cobre_hydro_means`` defines the Cobre mean line).
      * ``flow=False`` (``storage_final_hm3``): an end-of-stage stock that is
        replicated across blocks, so block 0 is taken per (scenario, stage).

    ``entity_ids`` restricts the sum to those Cobre hydro IDs — pass the
    plants matched by the comparison so the band covers the *same* plants as
    the mean line.  This matters for storage in particular: NEWAVE reports
    *useful* volume (so run-of-river plants it never matches contribute zero),
    while Cobre's parquet carries *absolute* volume — summing every plant
    would add ~100k hm³ of unmatched dead storage and lift the band off the
    mean line.
    """
    glob = str(cobre_output_dir / "simulation" / "hydros" / "**" / "*.parquet")
    lf = pl.scan_parquet(glob, hive_partitioning=True)
    if entity_ids is not None:
        names = set(lf.collect_schema().names())
        id_col = "hydro_id" if "hydro_id" in names else "entity_id"
        lf = lf.filter(pl.col(id_col).is_in(list(entity_ids)))

    if flow:
        case_dir = _case_dir(cobre_output_dir)
        stages_json = json.loads((case_dir / "stages.json").read_text(encoding="utf-8"))
        block_hours = pl.DataFrame(
            {
                "stage_id": [
                    int(s["id"]) for s in stages_json["stages"] for _ in s["blocks"]
                ],
                "block_id": [
                    int(b["id"]) for s in stages_json["stages"] for b in s["blocks"]
                ],
                "hours": [
                    float(b["hours"])
                    for s in stages_json["stages"]
                    for b in s["blocks"]
                ],
            }
        )
        per_scen = (
            lf
            # Sum the variable across all plants within each block first.
            .group_by(["scenario_id", "stage_id", "block_id"])
            .agg(pl.col(column).sum().alias("v"))
            .collect()
            .join(block_hours, on=["stage_id", "block_id"])
            # Block-hours-weighted stage mean per scenario.
            .group_by(["scenario_id", "stage_id"])
            .agg(
                ((pl.col("v") * pl.col("hours")).sum() / pl.col("hours").sum()).alias(
                    "v"
                )
            )
        )
    else:
        per_scen = (
            lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col(column).sum().alias("v"))
            .collect()
        )

    band = (
        per_scen.group_by("stage_id")
        .agg(
            p10=pl.col("v").quantile(0.1),
            p90=pl.col("v").quantile(0.9),
        )
        .sort("stage_id")
    )
    wanted = set(stages)
    return {
        int(r["stage_id"]): (float(r["p10"]), float(r["p90"]))
        for r in band.iter_rows(named=True)
        if int(r["stage_id"]) in wanted
    }


def plant_lines(
    results: list[ResultComparison], plant_name: str, variable: str
) -> tuple[list[int], list[float], list[float], int | None]:
    """Single-plant ``variable`` series: (stages, NEWAVE, Cobre, cobre_id).

    Matches ``plant_name`` case-insensitively against ``entity_name``.
    ``cobre_id`` is ``None`` when the plant/variable pair is absent from the
    comparison (e.g. a run-of-river plant has no storage rows).
    """
    target = plant_name.strip().upper()
    nw: dict[int, float] = {}
    cb: dict[int, float] = {}
    cobre_id: int | None = None
    for r in results:
        if (
            r.entity_type == "hydro"
            and r.variable == variable
            and r.entity_name.strip().upper() == target
        ):
            nw[r.stage] = r.newave_value
            cb[r.stage] = r.cobre_value
            cobre_id = r.cobre_id
    stages = sorted(nw)
    return stages, [nw[s] for s in stages], [cb[s] for s in stages], cobre_id


def plant_band(
    pct: PercentileData, cobre_id: int, column: str, stages: list[int]
) -> dict[int, tuple[float, float]]:
    """P10-P90 band for one plant straight from the per-entity percentiles.

    For a single entity the per-(entity, stage) percentiles in ``pct.hydro``
    already are the true per-scenario band — no cross-entity summing (and so
    none of :func:`hydro_system_band`'s matched-plant caveats) applies.
    """
    hp = pct.hydro
    if hp is None or hp.is_empty():
        return {}
    p10c, p90c = f"{column}_p10", f"{column}_p90"
    if p10c not in hp.columns or p90c not in hp.columns:
        return {}
    sub = hp.filter(pl.col("entity_id") == cobre_id)
    wanted = set(stages)
    out: dict[int, tuple[float, float]] = {}
    for row in sub.iter_rows(named=True):
        s = int(row["stage_id"])
        if s not in wanted or row[p10c] is None or row[p90c] is None:
            continue
        out[s] = (float(row[p10c]), float(row[p90c]))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    """Drop the top/right spines for a clean, open look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _month_ticks(ax: plt.Axes, stages: list[float], labels: dict[int, str]) -> None:
    """Place ~8 month-labelled ticks along the stage axis."""
    if not stages:
        return
    step = max(1, round(len(stages) / 8))
    ticks = [s for i, s in enumerate(stages) if i % step == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels.get(int(s), str(int(s))) for s in ticks], rotation=0)


def plot_cmo_by_bus(
    data: dict[str, dict[str, list[float]]],
    labels: dict[int, str],
    out: Path,
    title_prefix: str,
) -> None:
    """Render the 2x2 CMO-by-bus slide and save PNG + PDF to *out* stem."""
    buses = list(data)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True)
    flat = axes.flatten()
    for ax, bus in zip(flat, buses, strict=False):
        d = data[bus]
        stages = d["stages"]
        _style_axes(ax)
        # Draw the blue (Cobre) layers first, then NEWAVE (amber) on top.
        ax.fill_between(
            stages,
            d["p10"],
            d["p90"],
            color=C_BAND,
            lw=0,
            label="Cobre P10–P90",
        )
        ax.plot(stages, d["cb"], color=C_COBRE, lw=2.4, label="Cobre", zorder=2)
        ax.plot(
            stages,
            d["nw"],
            color=C_NEWAVE,
            lw=2.4,
            ms=3.2,
            label="NEWAVE",
            zorder=3,
        )
        ax.set_title(bus, loc="left", color="#0F172A")
        _month_ticks(ax, stages, labels)
        ax.margins(x=0.02)
    for ax in flat[len(buses) :]:
        ax.set_visible(False)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("CMO  (R$/MWh)")
    axes[1, 0].set_xlabel("Estágio (mês)")
    if len(buses) > 1:
        axes[1, 1].set_xlabel("Estágio (mês)")

    handles, leg_labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=12,
    )
    fig.suptitle(
        f"{title_prefix} - {VAR_CMO_PT}",
        y=1.02,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out)


def plot_system_series(
    stages: list[int],
    nw: list[float],
    cb: list[float],
    band: dict[int, tuple[float, float]],
    labels: dict[int, str],
    out: Path,
    title_prefix: str,
    *,
    var_pt: str,
    ylabel: str,
) -> None:
    """Render a system-total NEWAVE-vs-Cobre slide and save PNG + PDF.

    ``var_pt`` is the pt-BR variable name placed after the title prefix and
    ``ylabel`` is the y-axis label (with unit). Used for thermal generation,
    total storage, hydro generation and total spillage.
    """
    xs = [float(s) for s in stages]
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    _style_axes(ax)
    # Draw the blue (Cobre) layers first, then NEWAVE (amber) on top.
    if band:
        p10 = [band.get(s, (float("nan"), float("nan")))[0] for s in stages]
        p90 = [band.get(s, (float("nan"), float("nan")))[1] for s in stages]
        ax.fill_between(xs, p10, p90, color=C_BAND, lw=0, label="Cobre P10–P90")
    ax.plot(xs, cb, color=C_COBRE, lw=2.6, label="Cobre", zorder=2)
    ax.plot(xs, nw, color=C_NEWAVE, lw=2.6, ms=3.6, label="NEWAVE", zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Estágio (mês)")
    ax.margins(x=0.01)
    _month_ticks(ax, xs, labels)
    ax.legend(loc="upper right", ncol=3, fontsize=12)

    fig.suptitle(
        f"{title_prefix} - {var_pt}",
        y=1.0,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out)


def _slug(name: str) -> str:
    """Filesystem-safe lowercase slug for a plant name (e.g. ``TUCURUÍ``)."""
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", norm.lower()).strip("_") or "plant"


def _save(fig: Figure, out: Path) -> None:
    """Save *fig* as both a 300-DPI PNG and a vector PDF at the *out* stem."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, facecolor="white")
        print(f"  wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--newave",
        type=Path,
        default=Path("example/newave_mai_26"),
        help="NEWAVE case dir (with saidas/).",
    )
    parser.add_argument(
        "--cobre",
        type=Path,
        default=Path("example/cobre_mai_26/output"),
        help="Cobre output dir (with simulation/, training/).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("example/presentation_mai_26"),
        help="Directory for the exported figures.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--title-prefix",
        default=_DEFAULT_TITLE_PREFIX,
        help='Prefix for the chart titles ("{prefix} - {variable}").',
    )
    parser.add_argument(
        "--plants",
        nargs="*",
        default=PLANTS,
        metavar="NAME",
        help=(
            "Hydro plant names to export per-plant slides for "
            f"(default: {', '.join(PLANTS)}). Pass none to skip per-plant slides."
        ),
    )
    args = parser.parse_args()

    _setup_style()
    print(f"Loading comparison: {args.newave} <-> {args.cobre}")
    results, pct = load_comparison(args.newave, args.cobre, args.tolerance)
    labels = _stage_labels(args.cobre)

    print("Slide 1 — CMO by bus:")
    plot_cmo_by_bus(
        cmo_by_bus(results, pct),
        labels,
        args.outdir / "cmo_by_bus",
        args.title_prefix,
    )

    print("Slide 2 — system thermal generation:")
    stages, nw, cb = system_lines(results, "thermal", "generation_mw")
    band = thermal_system_band(args.cobre, stages)
    plot_system_series(
        stages,
        nw,
        cb,
        band,
        labels,
        args.outdir / "thermal_system",
        args.title_prefix,
        var_pt=VAR_THERMAL_PT,
        ylabel="Geração térmica  (MW)",
    )

    # Slides 3-5: system-wide hydro aggregates. Each plots the matched-plant
    # NEWAVE/Cobre mean line plus a true per-scenario Cobre P10-P90 band.
    # ``flow`` distinguishes block-varying flows (gen, spillage) from the
    # end-of-stage storage stock.
    for column, var_pt, ylabel, flow, system_stem, _plant_stem in _HYDRO_VARS:
        print(f"System — {var_pt}:")
        stages, nw, cb = system_lines(results, "hydro", column)
        matched_ids = {
            r.cobre_id
            for r in results
            if r.entity_type == "hydro" and r.variable == column
        }
        band = hydro_system_band(
            args.cobre, stages, column, flow=flow, entity_ids=matched_ids
        )
        plot_system_series(
            stages,
            nw,
            cb,
            band,
            labels,
            args.outdir / system_stem,
            args.title_prefix,
            var_pt=var_pt,
            ylabel=ylabel,
        )

    # Per-plant slides: same three variables for each plant in ``--plants``.
    # The single-plant band comes straight from the per-entity percentiles.
    for plant in args.plants:
        print(f"Plant {plant}:")
        for column, var_pt, ylabel, _flow, _system_stem, plant_stem in _HYDRO_VARS:
            stages, nw, cb, cobre_id = plant_lines(results, plant, column)
            if cobre_id is None:
                print(f"  - {var_pt}: no '{column}' data (plant not matched); skipped")
                continue
            band = plant_band(pct, cobre_id, column, stages)
            plot_system_series(
                stages,
                nw,
                cb,
                band,
                labels,
                args.outdir / f"plant_{_slug(plant)}_{plant_stem}",
                f"{args.title_prefix} — {plant}",
                var_pt=var_pt,
                ylabel=ylabel,
            )

    print(f"Done. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()

"""Export presentation-grade NEWAVE-vs-Cobre charts for a slide deck.

Renders two static, high-resolution figures that pitch Cobre as a drop-in
replacement for NEWAVE, reusing the same ``compare_results`` pipeline that
backs ``cobre-bridge compare results`` (no NEWAVE/Cobre re-parsing):

  Slide 1  ``cmo_by_bus.{png,pdf}``   2x2 submarket panels of marginal cost
                                      (CMO, R$/MWh): NEWAVE mean vs Cobre
                                      mean + Cobre P10-P90 band.
  Slide 2  ``thermal_system.{png,pdf}`` system thermal generation (MW):
                                      NEWAVE vs Cobre mean + a *true*
                                      per-scenario P10-P90 band.

Unlike the interactive HTML report this uses matplotlib for full typographic
control and exports both a 300-DPI PNG (drop into PowerPoint) and a vector PDF
(crisp at any projector size).

Requires matplotlib (not a package dependency)::

    VIRTUAL_ENV=.venv uv pip install matplotlib

Usage::

    python scripts/presentation_charts.py \\
        --newave example/comparacao_prod/newave \\
        --cobre  example/comparacao_prod/cobre/output \\
        --outdir example/presentation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from cobre_bridge.cli import _load_lines_json
from cobre_bridge.comparators.alignment import build_entity_alignment
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    compare_results,
)
from cobre_bridge.id_map import build_id_map
from cobre_bridge.newave_files import NewaveFiles

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
_DEFAULT_TITLE_PREFIX = "PMO SET/24 3 Anos (Individualizado)"


def _setup_style() -> None:
    """Apply a clean, editorial matplotlib style for slide figures."""
    plt.rcParams.update({
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
    })


# ---------------------------------------------------------------------------
# Data loading (reuses the compare-results pipeline)
# ---------------------------------------------------------------------------


def load_comparison(
    newave_dir: Path, cobre_output_dir: Path, tolerance: float = 1e-2
) -> tuple[list[ResultComparison], PercentileData]:
    """Run the NEWAVE-vs-Cobre comparison and return its raw results."""
    nw_files = NewaveFiles.from_directory(newave_dir)
    id_map = build_id_map(nw_files)
    alignment = build_entity_alignment(
        id_map, nw_files, _load_lines_json(cobre_output_dir)
    )
    return compare_results(
        nw_files=nw_files,
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
                float(band.get((eid, s), (None, None))[0] or "nan")
                for s in stages
            ],
            "p90": [
                float(band.get((eid, s), (None, None))[1] or "nan")
                for s in stages
            ],
        }
    return out


def thermal_system_lines(
    results: list[ResultComparison],
) -> tuple[list[int], list[float], list[float]]:
    """System thermal generation per stage: (stages, NEWAVE, Cobre)."""
    nw: dict[int, float] = defaultdict(float)
    cb: dict[int, float] = defaultdict(float)
    for r in results:
        if r.entity_type == "thermal":
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
    stages_json = json.loads(
        (case_dir / "stages.json").read_text(encoding="utf-8")
    )
    stage_hours = pl.DataFrame({
        "stage_id": [int(s["id"]) for s in stages_json["stages"]],
        "stage_hours": [
            float(sum(b["hours"] for b in s["blocks"]))
            for s in stages_json["stages"]
        ],
    })
    glob = str(
        cobre_output_dir / "simulation" / "thermals" / "**" / "*.parquet"
    )
    per_scen = (
        pl
        .scan_parquet(glob, hive_partitioning=True)
        .group_by(["scenario_id", "stage_id"])
        .agg(pl.col("generation_mwh").sum().alias("e"))
        .collect()
        .join(stage_hours, on="stage_id")
        .with_columns((pl.col("e") / pl.col("stage_hours")).alias("mw"))
    )
    band = (
        per_scen
        .group_by("stage_id")
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    """Drop the top/right spines for a clean, open look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _month_ticks(
    ax: plt.Axes, stages: list[float], labels: dict[int, str]
) -> None:
    """Place ~8 month-labelled ticks along the stage axis."""
    if not stages:
        return
    step = max(1, round(len(stages) / 8))
    ticks = [s for i, s in enumerate(stages) if i % step == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [labels.get(int(s), str(int(s))) for s in ticks], rotation=0
    )


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


def plot_thermal_system(
    stages: list[int],
    nw: list[float],
    cb: list[float],
    band: dict[int, tuple[float, float]],
    labels: dict[int, str],
    out: Path,
    title_prefix: str,
) -> None:
    """Render the system-thermal-generation slide and save PNG + PDF."""
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
    ax.set_ylabel("Geração térmica  (MW)")
    ax.set_xlabel("Estágio (mês)")
    ax.margins(x=0.01)
    _month_ticks(ax, xs, labels)
    ax.legend(loc="upper right", ncol=3, fontsize=12)

    fig.suptitle(
        f"{title_prefix} - {VAR_THERMAL_PT}",
        y=1.0,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out)


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
        default=Path("example/comparacao_prod/newave"),
        help="NEWAVE case dir (with saidas/).",
    )
    parser.add_argument(
        "--cobre",
        type=Path,
        default=Path("example/comparacao_prod/cobre/output"),
        help="Cobre output dir (with simulation/, training/).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("example/presentation"),
        help="Directory for the exported figures.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--title-prefix",
        default=_DEFAULT_TITLE_PREFIX,
        help='Prefix for the chart titles ("{prefix} - {variable}").',
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
    stages, nw, cb = thermal_system_lines(results)
    band = thermal_system_band(args.cobre, stages)
    plot_thermal_system(
        stages,
        nw,
        cb,
        band,
        labels,
        args.outdir / "thermal_system",
        args.title_prefix,
    )

    print(f"Done. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()

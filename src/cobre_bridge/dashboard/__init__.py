"""Modular dashboard package for Cobre simulation results.

Entry point: ``build_dashboard(case_dir, output_path)`` loads data,
discovers renderable tabs, and writes a self-contained HTML file.
"""

from __future__ import annotations

from pathlib import Path

from cobre_bridge.dashboard.data import DashboardData
from cobre_bridge.dashboard.tabs import get_renderable_tabs
from cobre_bridge.ui.css import dashboard_css
from cobre_bridge.ui.html import build_html
from cobre_bridge.ui.js import TAB_SWITCH_JS


def build_dashboard(case_dir: Path, output_path: Path) -> None:
    """Build an interactive HTML dashboard from Cobre simulation results."""
    data = DashboardData.load(case_dir)

    renderable = get_renderable_tabs(data)
    tab_defs = [(tab_id, tab_label) for tab_id, tab_label, _ in renderable]
    tab_contents = {tab_id: html for tab_id, _, html in renderable}

    case_name = case_dir.resolve().name
    html = build_html(
        title=f"Cobre Simulation Dashboard \u2014 {case_name}",
        tab_defs=tab_defs,
        tab_contents=tab_contents,
        css=dashboard_css(),
        js=TAB_SWITCH_JS,
    )

    print(f"Writing dashboard to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"Done. File size: {size_kb:.0f} KB")

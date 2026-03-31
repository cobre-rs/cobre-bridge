"""HTML comparison report builder with Cobre brand styling.

Generates self-contained HTML files with plotly.js CDN for interactive
charts.  Follows the same design language as ``scripts/dashboard.py``
(tabbed navigation, metric cards, chart cards) but is independent of it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CSS — duplicated from dashboard.py with adaptations for comparison reports
# ---------------------------------------------------------------------------

CSS = (  # noqa: E501
    """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'IBM Plex Sans', 'Inter', 'Segoe UI', system-ui, sans-serif;
       background: #F0EDE8; color: #374151; }

header {
    background: linear-gradient(135deg, #0F1419 0%, #1A2028 100%);
    border-top: 3px solid #B87333;
    color: white;
    padding: 16px 32px;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

nav {
    background: #1A2028;
    padding: 0 24px;
    display: flex;
    gap: 4px;
    overflow-x: auto;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

nav button {
    background: none;
    border: none;
    color: #8B9298;
    padding: 14px 20px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: color 0.2s, border-color 0.2s;
    letter-spacing: 0.3px;
}

nav button:hover { color: #E8E6E3; border-bottom-color: #B87333; }
nav button.active { color: #B87333; border-bottom-color: #B87333; }

main { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }

.tab-content { display: none; }
.tab-content.active { display: block; }

.chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.chart-grid-single {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.chart-card {
    background: #FAFAF8;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    overflow: hidden;
    min-width: 0;
}

.chart-card .plotly-graph-div { width: 100% !important; }
.chart-card .js-plotly-plot { width: 100% !important; }
.chart-card .plot-container { width: 100% !important; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.metric-card {
    background: #FAFAF8;
    border-radius: 8px;
    padding: 20px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.8rem;
    color: #8B9298;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    margin: 24px 0 12px;
    padding-left: 10px;
    border-left: 4px solid #B87333;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.plant-selector {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}

.plant-selector label {
    font-weight: 600;
    font-size: 0.9rem;
    color: #374151;
}

.plant-selector select {
    padding: 8px 12px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 0.88rem;
    min-width: 280px;
    background: #FAFAF8;
}

.plant-info {
    display: flex;
    gap: 20px;
    font-size: 0.82rem;
    color: #8B9298;
    flex-wrap: wrap;
}

.plant-info span { white-space: nowrap; }
"""
)

JS = """
function showTab(tabId, btn) {  // noqa: E501
    document.querySelectorAll('.tab-content').forEach(
        el => el.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(
        el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
    window.dispatchEvent(new Event('resize'));
}
window.addEventListener('load', function() {
    setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
});
"""

# Color palette for comparison traces.
COLOR_COBRE = "#4A90B8"
COLOR_NEWAVE = "#F5A623"
COLOR_DIFF = "#DC4C4C"
COLOR_MATCH = "#4A8B6F"

# Tab definitions for comparison report.
COMPARISON_TABS = [
    ("tab-overview", "Overview"),
    ("tab-system", "System"),
    ("tab-balance", "Energy Balance"),
    ("tab-hydro", "Hydro Operation"),
    ("tab-hydro-detail", "Plant Details"),
    ("tab-thermal", "Thermal Operation"),
    ("tab-thermal-detail", "Thermal Details"),
    ("tab-productivity", "Productivity"),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def metric_card(value: str, label: str) -> str:
    """Return an HTML metric card."""
    return (
        '<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        "</div>"
    )


def section_title(title: str) -> str:
    """Return an HTML section title element."""
    return f'<div class="section-title">{title}</div>'


def wrap_chart(chart_html: str) -> str:
    """Wrap a plotly chart div in a card container."""
    return f'<div class="chart-card">{chart_html}</div>'


def chart_grid(charts: list[str], single: bool = False) -> str:
    """Wrap charts in a responsive grid."""
    cls = "chart-grid-single" if single else "chart-grid"
    return f'<div class="{cls}">{"".join(charts)}</div>'


def metrics_grid(cards: list[str]) -> str:
    """Wrap metric cards in a responsive grid."""
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Full HTML builder
# ---------------------------------------------------------------------------


def build_comparison_html(
    title: str,
    tab_contents: dict[str, str],
) -> str:
    """Build a complete self-contained HTML comparison report.

    Parameters
    ----------
    title:
        Report title shown in the header and ``<title>`` tag.
    tab_contents:
        Mapping of tab IDs to their HTML content.  Tab IDs must
        match those in ``COMPARISON_TABS``.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    nav_buttons: list[str] = []
    for i, (tab_id, tab_label) in enumerate(COMPARISON_TABS):
        active = ' class="active"' if i == 0 else ""
        nav_buttons.append(
            f"<button{active} onclick=\"showTab('{tab_id}', this)\">"
            f"{tab_label}</button>"
        )

    tab_sections: list[str] = []
    for i, (tab_id, _) in enumerate(COMPARISON_TABS):
        active = " active" if i == 0 else ""
        content = tab_contents.get(tab_id, "<p>No data available</p>")
        tab_sections.append(
            f'<section id="{tab_id}" class="tab-content{active}">'
            f"\n{content}\n</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
{CSS}
    </style>
</head>
<body>
    <header>{title}</header>
    <nav>
        {"".join(nav_buttons)}
    </nav>
    <main>
        {"".join(tab_sections)}
    </main>
    <script>
{JS}
    </script>
</body>
</html>"""

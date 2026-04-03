"""Shared CSS stylesheets for dashboard and comparator HTML reports.

The copper accent color (#B87333) is defined in theme.py but appears as
a literal hex value in the CSS strings to preserve byte-identical output.
"""

from __future__ import annotations

BASE_CSS: str = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'IBM Plex Sans', 'Inter', 'Segoe UI', system-ui, sans-serif; background: #F0EDE8; color: #374151; }

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
"""

DATA_TABLE_CSS: str = """
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: #FAFAF8;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.data-table th, .data-table td {
    padding: 10px 12px;
    text-align: right;
    border-bottom: 1px solid #E0E0E0;
}
.data-table th {
    background: #374151;
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
.data-table td:first-child, .data-table th:first-child { text-align: left; }
.data-table td:nth-child(2), .data-table th:nth-child(2) { text-align: left; }
.data-table tr:hover td { background: #F0EDE8; }
"""

PLANT_SELECTOR_CSS: str = """
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


def dashboard_css() -> str:
    """Compose CSS for the dashboard (base + data-table styles)."""
    return BASE_CSS + DATA_TABLE_CSS


def comparison_css() -> str:
    """Compose CSS for comparison reports (base + plant-selector styles)."""
    return BASE_CSS + PLANT_SELECTOR_CSS

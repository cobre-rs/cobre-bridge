"""HTML builder functions for UI fragments and complete tabbed documents."""

from __future__ import annotations


def wrap_chart(html: str) -> str:
    """Wrap an HTML fragment in a chart-card container div."""
    return f'<div class="chart-card">{html}</div>'


def section_title(text: str) -> str:
    """Create a section title element using the .section-title class."""
    return f'<div class="section-title">{text}</div>'


def metric_card(value: str, label: str) -> str:
    """Create a metric card with a prominent value and a descriptive label."""
    return (
        '<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        "</div>"
    )


def metrics_grid(cards: list[str]) -> str:
    """Wrap a list of metric card HTML fragments in a metrics-grid container."""
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


def chart_grid(charts: list[str], single: bool = False) -> str:
    """Wrap chart HTML fragments in a grid container.

    Args:
        charts: List of HTML fragments, one per chart.
        single: When True, uses the single-column grid class.
    """
    cls = "chart-grid-single" if single else "chart-grid"
    return f'<div class="{cls}">{"".join(charts)}</div>'


def build_html(
    title: str,
    tab_defs: list[tuple[str, str]],
    tab_contents: dict[str, str],
    css: str,
    js: str,
) -> str:
    """Assemble a complete HTML document with tabbed navigation.

    Args:
        title: Page title shown in <title> and <header>.
        tab_defs: Ordered list of (tab_id, tab_label) pairs defining the nav.
        tab_contents: Mapping from tab_id to the HTML content for that tab.
            If a tab_id from tab_defs is missing here, the section renders
            "<p>No data</p>".
        css: CSS string injected into a <style> block in <head>.
        js: JavaScript string injected into a <script> block at the end of
            <body>.

    Returns:
        A complete <!DOCTYPE html> document string.
    """
    nav_buttons: list[str] = []
    for i, (tab_id, tab_label) in enumerate(tab_defs):
        active_cls = ' class="active"' if i == 0 else ""
        nav_buttons.append(
            f"<button{active_cls} onclick=\"showTab('{tab_id}', this)\">{tab_label}</button>"
        )

    tab_sections: list[str] = []
    for i, (tab_id, _) in enumerate(tab_defs):
        active_cls = " active" if i == 0 else ""
        content = tab_contents.get(tab_id, "<p>No data</p>")
        tab_sections.append(
            f'<section id="{tab_id}" class="tab-content{active_cls}">\n{content}\n</section>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
{css}
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
{js}
    </script>
</body>
</html>"""

"""HTML builder functions for UI fragments and complete tabbed documents."""

from __future__ import annotations


def wrap_chart(html: str) -> str:
    """Wrap an HTML fragment in a chart-card container div with an expand button."""
    return (
        '<div class="chart-card">'
        '<button type="button" class="expand-btn" title="Expand">'
        '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5">'
        '<polyline points="1,5 1,1 5,1"/><polyline points="11,1 15,1 15,5"/>'
        '<polyline points="15,11 15,15 11,15"/><polyline points="5,15 1,15 1,11"/>'
        "</svg></button>"
        f"{html}</div>"
    )


def section_title(text: str) -> str:
    """Create a section title element using the .section-title class."""
    return f'<div class="section-title">{text}</div>'


def collapsible_section(title: str, content: str) -> str:
    """Wrap a section title and content in a collapsible container.

    The section title receives a ``data-collapsible="true"`` attribute and an
    inline SVG chevron icon.  JavaScript in ``TAB_SWITCH_JS`` handles the
    click-to-collapse behaviour by toggling ``.collapsed`` on the sibling
    ``.collapsible-content`` div.

    Args:
        title: Section heading text.
        content: HTML string for the section body.

    Returns:
        A ``<div class="collapsible-section">`` fragment containing a
        clickable title bar and the content wrapped in
        ``<div class="collapsible-content">``.
    """
    chevron = (
        '<svg class="chevron" width="10" height="10" viewBox="0 0 10 10">'
        '<polyline points="2,3 5,7 8,3" fill="none" stroke="currentColor" stroke-width="1.5"/>'
        "</svg>"
    )
    return (
        '<div class="collapsible-section">'
        f'<div class="section-title" data-collapsible="true">'
        f"{title}"
        f"{chevron}"
        "</div>"
        f'<div class="collapsible-content">{content}</div>'
        "</div>"
    )


def _sparkline_svg(
    values: list[float],
    color: str,
    width: int = 60,
    height: int = 20,
) -> str:
    """Render a tiny inline SVG sparkline polyline from a list of float values.

    Normalizes values to the 0..height range and spaces x-coordinates evenly
    across width. Returns an SVG element with class ``metric-sparkline``.
    """
    n = len(values)
    min_v = min(values)
    max_v = max(values)
    value_range = max_v - min_v
    # Avoid division by zero for flat series
    scale = height / value_range if value_range > 0 else 0.0
    pts_list: list[str] = []
    for i, v in enumerate(values):
        x = i / (n - 1) * width
        # SVG y-axis is top-down, so invert
        y = height - (v - min_v) * scale
        pts_list.append(f"{x:.1f},{y:.1f}")
    pts = " ".join(pts_list)
    return (
        f'<svg width="{width}" height="{height}" class="metric-sparkline">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f"</svg>"
    )


def metric_card(
    value: str,
    label: str,
    *,
    color: str | None = None,
    delta: str | None = None,
    delta_direction: str | None = None,
    sparkline_values: list[float] | None = None,
) -> str:
    """Create a metric card with a prominent value and a descriptive label.

    Args:
        value: The metric value to display (e.g. "1,234 GWh").
        label: The metric label (e.g. "Total Hydro Generation").
        color: Optional accent hex color — adds a ``border-top: 4px solid`` style.
        delta: Optional delta text (e.g. "+5.2%") shown between value and label.
        delta_direction: ``"up"`` or ``"down"`` controls triangle color; ignored
            when ``delta`` is ``None``.
        sparkline_values: Optional list of floats rendered as an inline SVG
            sparkline. Omitted when the list has fewer than 2 elements.
    """
    style = f' style="border-top: 4px solid {color};"' if color is not None else ""

    delta_html = ""
    if delta is not None:
        if delta_direction == "up":
            delta_class = "metric-delta metric-delta-up"
            arrow = "&#9650;"
        elif delta_direction == "down":
            delta_class = "metric-delta metric-delta-down"
            arrow = "&#9660;"
        else:
            delta_class = "metric-delta"
            arrow = ""
        arrow_span = f"<span>{arrow}</span>" if arrow else ""
        delta_html = f'<div class="{delta_class}">{arrow_span}{delta}</div>'

    sparkline_html = ""
    if sparkline_values is not None and len(sparkline_values) >= 2:
        sparkline_color = color if color is not None else "#8B9298"
        sparkline_html = (
            f'<div class="metric-sparkline">'
            f"{_sparkline_svg(sparkline_values, sparkline_color)}"
            f"</div>"
        )

    return (
        f'<div class="metric-card"{style}>'
        f'<div class="metric-value">{value}</div>'
        f"{delta_html}"
        f'<div class="metric-label">{label}</div>'
        f"{sparkline_html}"
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


def plant_explorer_table(
    table_id: str,
    search_id: str,
    columns: list[tuple[str, str]],
    rows_html: str,
) -> str:
    """Generate an HTML plant explorer table pane with search input and sortable header.

    Args:
        table_id: The ``id`` attribute applied to the ``<tbody>`` element.
        search_id: The ``id`` attribute applied to the search ``<input>`` element.
        columns: Ordered list of ``(header_text, sort_type)`` pairs. ``sort_type``
            must be ``"string"``, ``"number"``, or ``"none"``.
        rows_html: Pre-rendered ``<tr>`` elements to inject into the ``<tbody>``.

    Returns:
        An HTML fragment containing the search input, a ``<table>`` with a
        sortable ``<thead>``, and a ``<tbody>`` populated with *rows_html*.
    """
    header_cells: list[str] = []
    for col_index, (header_text, sort_type) in enumerate(columns):
        if sort_type != "none":
            header_cells.append(
                f'<th class="sortable" data-sort-asc="false"'
                f" onclick=\"sortTable('{table_id}', {col_index}, '{sort_type}')\">"
                f'{header_text}<span class="sort-arrow"></span></th>'
            )
        else:
            header_cells.append(f"<th>{header_text}</th>")

    return (
        f'<input type="search" id="{search_id}" class="explorer-search"'
        f' placeholder="Search...">'
        f'<table class="explorer-table">'
        f"<thead><tr>{''.join(header_cells)}</tr></thead>"
        f'<tbody id="{table_id}">{rows_html}</tbody>'
        f"</table>"
    )


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

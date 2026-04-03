"""Tab-switching JavaScript for dashboard and comparator HTML reports.

The showTab function signature is referenced in HTML onclick attributes —
do not rename tabId or btn parameters.
"""

from __future__ import annotations

TAB_SWITCH_JS: str = """
function showTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
    window.dispatchEvent(new Event('resize'));
}
// Plotly charts in the initial active tab render before layout settles.
// Fire a deferred resize so they recalculate to the correct container width.
window.addEventListener('load', function() { setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50); });
"""

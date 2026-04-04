"""Tab-switching JavaScript for dashboard and comparator HTML reports.

The showTab function signature is referenced in HTML onclick attributes —
do not rename tabId or btn parameters.
"""

from __future__ import annotations

TAB_SWITCH_JS: str = """
document.addEventListener('DOMContentLoaded', function() {
    var nav = document.querySelector('nav');
    if (!nav) return;
    var firstBtn = nav.querySelector('button');
    if (!firstBtn) return;

    var underline = document.createElement('span');
    underline.className = 'tab-underline';
    nav.appendChild(underline);

    function positionUnderline(btn) {
        underline.style.width = btn.offsetWidth + 'px';
        underline.style.transform = 'translateX(' + btn.offsetLeft + 'px)';
    }

    positionUnderline(firstBtn);
});

function showTab(tabId, btn) {
    var nav = document.querySelector('nav');
    var underline = nav ? nav.querySelector('.tab-underline') : null;

    // Collapse expanded cards in the currently active tab before switching.
    var prevTab = document.querySelector('.tab-content.active');
    if (prevTab) {
        // Remove stagger animation from cards in the outgoing tab.
        prevTab.querySelectorAll('.chart-card').forEach(function(card) {
            card.classList.remove('card-enter');
            card.style.animationDelay = '';
        });
        prevTab.querySelectorAll('.chart-card-expanded').forEach(function(card) {
            card.classList.remove('chart-card-expanded');
        });
        // Reset all collapsible sections in the outgoing tab to expanded.
        prevTab.querySelectorAll('.collapsible-content').forEach(function(el) {
            el.classList.remove('collapsed');
        });
        prevTab.querySelectorAll('.section-title[data-collapsible]').forEach(function(el) {
            el.classList.remove('collapsed-title');
            var chevron = el.querySelector('.chevron');
            if (chevron) { chevron.classList.remove('rotated'); }
        });
    }

    document.querySelectorAll('.tab-content').forEach(function(el) {
        el.classList.remove('active', 'tab-content-fade');
    });
    document.querySelectorAll('nav button').forEach(function(el) {
        el.classList.remove('active');
    });

    var activeTab = document.getElementById(tabId);
    if (activeTab) {
        activeTab.classList.add('active');
        // Trigger fade animation by forcing a reflow then re-adding the class.
        void activeTab.offsetWidth;
        activeTab.classList.add('tab-content-fade');

        // Staggered card entry animation: remove existing, force reflow, re-apply.
        var cards = activeTab.querySelectorAll('.chart-card');
        cards.forEach(function(card) {
            card.classList.remove('card-enter');
            card.style.animationDelay = '';
        });
        // Force reflow so removing the class takes effect before re-adding.
        void activeTab.offsetWidth;
        cards.forEach(function(card, i) {
            card.style.animationDelay = (i * 50) + 'ms';
            card.classList.add('card-enter');
        });

        // Reset collapsible sections in the incoming tab to expanded state.
        activeTab.querySelectorAll('.collapsible-content').forEach(function(el) {
            el.classList.remove('collapsed');
        });
        activeTab.querySelectorAll('.section-title[data-collapsible]').forEach(function(el) {
            el.classList.remove('collapsed-title');
            var chevron = el.querySelector('.chevron');
            if (chevron) { chevron.classList.remove('rotated'); }
        });
    }
    btn.classList.add('active');

    if (underline) {
        underline.style.width = btn.offsetWidth + 'px';
        underline.style.transform = 'translateX(' + btn.offsetLeft + 'px)';
    }

    window.dispatchEvent(new Event('resize'));
}

document.addEventListener('click', function(e) {
    var expandBtn = e.target.closest('.expand-btn');
    if (!expandBtn) return;
    var card = expandBtn.closest('.chart-card');
    if (!card) return;
    card.classList.toggle('chart-card-expanded');
    setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
}, false);

// Delegated collapsible section toggle: handles [data-collapsible] title clicks.
document.addEventListener('click', function(e) {
    var titleEl = e.target.closest('.section-title[data-collapsible]');
    if (!titleEl) return;
    var content = titleEl.nextElementSibling;
    if (!content || !content.classList.contains('collapsible-content')) return;
    content.classList.toggle('collapsed');
    titleEl.classList.toggle('collapsed-title');
    var chevron = titleEl.querySelector('.chevron');
    if (chevron) { chevron.classList.toggle('rotated'); }
}, false);

// Plotly charts in the initial active tab render before layout settles.
// Fire a deferred resize so they recalculate to the correct container width.
window.addEventListener('load', function() { setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50); });
"""

PLANT_EXPLORER_JS: str = """
// ---------------------------------------------------------------------------
// Plant Explorer: shared infrastructure for master-detail split-pane tables.
// All functions use ES5 var declarations for inline-script compatibility.
// ---------------------------------------------------------------------------

function initPlantExplorer(config) {
    // config: { tableId, searchInputId, detailContainerId, dataVar, labelsVar,
    //           renderDetail, columns }
    var searchInput = document.getElementById(config.searchInputId);
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterTable(config.tableId, config.searchInputId);
        });
    }

    var tbody = document.getElementById(config.tableId);
    if (tbody) {
        var rows = tbody.querySelectorAll('tr');
        rows.forEach(function(row) {
            row.addEventListener('click', function() {
                selectRow(config.tableId, config.detailContainerId, row,
                          config.dataVar, config.renderDetail);
            });
        });
        if (rows.length > 0) {
            selectRow(config.tableId, config.detailContainerId, rows[0],
                      config.dataVar, config.renderDetail);
        }
    }
}

function filterTable(tableId, searchInputId) {
    var input = document.getElementById(searchInputId);
    var query = input ? input.value.toLowerCase() : '';
    var tbody = document.getElementById(tableId);
    if (!tbody) { return; }
    var rows = tbody.querySelectorAll('tr');
    rows.forEach(function(row) {
        var name = (row.getAttribute('data-name') || '').toLowerCase();
        row.style.display = (query === '' || name.indexOf(query) !== -1) ? '' : 'none';
    });
}

function sortTable(tableId, colIndex, type) {
    if (type === 'none') { return; }
    var tbody = document.getElementById(tableId);
    if (!tbody) { return; }

    var table = tbody.closest('table');
    var th = table ? table.querySelectorAll('th')[colIndex] : null;
    var asc = th ? th.getAttribute('data-sort-asc') !== 'true' : true;

    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
    rows.sort(function(a, b) {
        var cellA = a.querySelectorAll('td')[colIndex];
        var cellB = b.querySelectorAll('td')[colIndex];
        var valA = cellA ? cellA.getAttribute('data-sort-value') || cellA.textContent || '' : '';
        var valB = cellB ? cellB.getAttribute('data-sort-value') || cellB.textContent || '' : '';
        if (type === 'number') {
            valA = parseFloat(valA) || 0;
            valB = parseFloat(valB) || 0;
            return asc ? valA - valB : valB - valA;
        }
        valA = valA.toLowerCase();
        valB = valB.toLowerCase();
        if (valA < valB) { return asc ? -1 : 1; }
        if (valA > valB) { return asc ? 1 : -1; }
        return 0;
    });

    rows.forEach(function(row) { tbody.appendChild(row); });

    if (th) {
        th.setAttribute('data-sort-asc', asc ? 'true' : 'false');
        if (table) {
            table.querySelectorAll('th').forEach(function(el) {
                var arrow = el.querySelector('.sort-arrow');
                if (arrow) { arrow.textContent = ''; }
            });
        }
        var arrow = th.querySelector('.sort-arrow');
        if (arrow) { arrow.textContent = asc ? ' \u25b2' : ' \u25bc'; }
    }
}

function selectRow(tableId, detailContainerId, rowElement, dataVar, renderDetail) {
    if (!rowElement) { return; }
    var tbody = document.getElementById(tableId);
    if (tbody) {
        tbody.querySelectorAll('tr').forEach(function(r) {
            r.classList.remove('explorer-row-selected');
        });
    }
    rowElement.classList.add('explorer-row-selected');

    var idx = rowElement.getAttribute('data-index');
    if (idx !== null && dataVar && renderDetail) {
        var data = window[dataVar];
        if (data && data[idx] !== undefined) {
            renderDetail(detailContainerId, data[idx]);
        }
    }
}

function plotlyBand(labels, p10, p90, color, name) {
    return {
        x: labels.concat(labels.slice().reverse()),
        y: p90.concat(p10.slice().reverse()),
        fill: 'toself',
        fillcolor: color,
        line: { color: 'transparent' },
        name: name,
        showlegend: true,
        type: 'scatter',
        mode: 'none',
        hoverinfo: 'skip'
    };
}

function plotlyLine(labels, y, color, name, width, dash) {
    return {
        x: labels,
        y: y,
        mode: 'lines',
        line: {
            color: color,
            width: width !== undefined ? width : 2,
            dash: dash || 'solid'
        },
        name: name,
        type: 'scatter'
    };
}

function plotlyRef(labels, values, color, name) {
    var yArr;
    if (Array.isArray(values)) {
        yArr = values;
    } else {
        yArr = labels.map(function() { return values; });
    }
    return {
        x: labels,
        y: yArr,
        mode: 'lines',
        line: { color: color, width: 1, dash: 'dot' },
        name: name,
        type: 'scatter'
    };
}

function plotlyLayout(overrides) {
    var defaults = {
        hovermode: 'x unified',
        margin: { l: 50, r: 20, t: 40, b: 50 },
        legend: { orientation: 'h', x: 0, y: -0.2 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };
    var result = {};
    for (var k in defaults) {
        if (defaults.hasOwnProperty(k)) { result[k] = defaults[k]; }
    }
    if (overrides) {
        for (var key in overrides) {
            if (overrides.hasOwnProperty(key)) { result[key] = overrides[key]; }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// syncHover: cross-chart hover synchronization.
// Attaches plotly_hover / plotly_unhover listeners to each chart div.
// A _syncLock guard prevents re-entrant cascades.
// ---------------------------------------------------------------------------

var _syncLock = false;

function syncHover(chartIds) {
    chartIds.forEach(function(sourceId) {
        var sourceDiv = document.getElementById(sourceId);
        if (!sourceDiv) { return; }

        sourceDiv.on('plotly_hover', function(eventData) {
            if (_syncLock) { return; }
            var pts = eventData && eventData.points;
            if (!pts || pts.length === 0) { return; }
            var pointIndex = pts[0].pointIndex;
            _syncLock = true;
            chartIds.forEach(function(targetId) {
                if (targetId === sourceId) { return; }
                var targetDiv = document.getElementById(targetId);
                if (!targetDiv) { return; }
                try {
                    Plotly.Fx.hover(targetDiv, [{ curveNumber: 0, pointNumber: pointIndex }]);
                } catch (e) { /* chart not yet rendered */ }
            });
            _syncLock = false;
        });

        sourceDiv.on('plotly_unhover', function() {
            if (_syncLock) { return; }
            _syncLock = true;
            chartIds.forEach(function(targetId) {
                if (targetId === sourceId) { return; }
                var targetDiv = document.getElementById(targetId);
                if (!targetDiv) { return; }
                try {
                    Plotly.Fx.hover(targetDiv, []);
                } catch (e) { /* chart not yet rendered */ }
            });
            _syncLock = false;
        });
    });
}

// ---------------------------------------------------------------------------
// initComparisonMode: checkbox-driven plant comparison overlay.
// config keys: tableId, dataVar, labelsVar, chartIds, renderComparison,
//              renderDetail (optional, for reverting to single-plant mode),
//              maxCompare (default 3).
// ---------------------------------------------------------------------------

var _compareSelected = [];
var _comparePalette = ['#2196F3', '#FF9800', '#4CAF50'];

function initComparisonMode(config) {
    var maxCompare = config.maxCompare !== undefined ? config.maxCompare : 3;
    var tbody = document.getElementById(config.tableId);
    if (!tbody) { return; }

    tbody.addEventListener('change', function(e) {
        var cb = e.target;
        if (!cb || cb.type !== 'checkbox' || !cb.classList.contains('compare-checkbox')) {
            return;
        }
        var plantId = cb.getAttribute('data-id');
        if (!plantId) { return; }

        if (cb.checked) {
            if (_compareSelected.indexOf(plantId) === -1) {
                if (_compareSelected.length >= maxCompare) {
                    // Uncheck the oldest selection.
                    var oldest = _compareSelected.shift();
                    var oldCb = tbody.querySelector(
                        '.compare-checkbox[data-id="' + oldest + '"]'
                    );
                    if (oldCb) { oldCb.checked = false; }
                }
                _compareSelected.push(plantId);
            }
        } else {
            var idx = _compareSelected.indexOf(plantId);
            if (idx !== -1) { _compareSelected.splice(idx, 1); }
        }

        // Update row border classes.
        tbody.querySelectorAll('tr').forEach(function(row) {
            row.classList.remove('compare-active-1', 'compare-active-2', 'compare-active-3');
        });
        _compareSelected.forEach(function(pid, i) {
            var rowCb = tbody.querySelector('.compare-checkbox[data-id="' + pid + '"]');
            var row = rowCb ? rowCb.closest('tr') : null;
            if (row) { row.classList.add('compare-active-' + (i + 1)); }
        });

        // Render comparison or revert to single-plant.
        var data = window[config.dataVar];
        var labels = window[config.labelsVar];
        if (_compareSelected.length === 0) {
            // Revert to the currently selected row.
            var selectedRow = tbody.querySelector('.explorer-row-selected');
            if (selectedRow && config.renderDetail && data) {
                var selIdx = selectedRow.getAttribute('data-index');
                if (selIdx !== null && data[selIdx] !== undefined) {
                    config.renderDetail(null, data[selIdx]);
                }
            }
        } else {
            var entries = [];
            _compareSelected.forEach(function(pid) {
                if (data && data[pid] !== undefined) {
                    entries.push(data[pid]);
                }
            });
            config.renderComparison(entries, labels);
        }
    }, false);
}
"""

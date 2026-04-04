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

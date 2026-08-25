"""Layout direction guard for `src/cobre_bridge`.

A filesystem-driven ``ast`` walk enforcing five package-boundary rules ahead
of the package-layout-symmetry migration: import direction (A), no private
name crossing a package boundary (B), no loose modules at the package root
(C), no module shadowing a stdlib name (D), and that nothing imports `cli`
(E) -- plus a census of the `TYPE_CHECKING`-only edges exempted from A and B.
Each rule with an allowlist shrinks it as later moves land; `__main__`
reprints the live sets so an allowlist is always updated by diff, not by
hand.

Tier-1: pure ``ast``, ``sys`` and ``pathlib`` -- no ``cobre`` import -- so
this collects and runs even in a cobre-free environment.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "cobre_bridge"

# `converters/` is the NEWAVE track's pre-move home -- it resolves to
# `newave` from day one so a `decomp -> converters` edge is already visible
# as a direction violation, not hidden behind its current directory name.
_DIR_TO_PACKAGE: dict[str, str] = {
    "core": "core",
    "cobre": "cobre",
    "ui": "ui",
    "newave": "newave",
    "decomp": "decomp",
    "comparators": "comparators",
    "dashboard": "dashboard",
    "cli": "cli",
    "converters": "newave",
}

_SUBPACKAGE_REFINEMENTS: dict[tuple[str, str], str] = {
    ("ui", "html"): "ui.html",
    ("comparators", "newave"): "comparators.newave",
    ("comparators", "decomp"): "comparators.decomp",
}

# AMENDMENT-1 (ratified): `ui.theme` is its own leaf column, importable by
# every presentation-consuming package, forbidden to core/cobre/newave/decomp.
_ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset({"core"}),
    "cobre": frozenset({"core", "cobre"}),
    "ui": frozenset({"core", "cobre", "ui", "ui.theme"}),
    "ui.theme": frozenset({"core"}),
    "ui.html": frozenset({"core", "ui", "ui.theme", "ui.html"}),
    "newave": frozenset({"core", "cobre", "newave"}),
    "decomp": frozenset({"core", "cobre", "decomp"}),
    "comparators": frozenset({"core", "cobre", "ui.theme", "ui.html", "comparators"}),
    "comparators.newave": frozenset(
        {
            "core",
            "cobre",
            "ui.theme",
            "ui.html",
            "newave",
            "comparators",
            "comparators.newave",
        }
    ),
    "comparators.decomp": frozenset(
        {
            "core",
            "cobre",
            "ui.theme",
            "ui.html",
            "decomp",
            "comparators",
            "comparators.decomp",
        }
    ),
    "dashboard": frozenset({"core", "cobre", "ui.theme", "ui.html", "dashboard"}),
    "cli": frozenset(
        {
            "core",
            "cobre",
            "ui",
            "ui.theme",
            "ui.html",
            "newave",
            "decomp",
            "comparators",
            "comparators.newave",
            "comparators.decomp",
            "dashboard",
            "cli",
        }
    ),
}


def _package_of(dotted: str) -> str:
    """Resolve `dotted` (relative to `cobre_bridge`) to an `_ALLOWED` key.

    Filesystem-driven so it stays correct as modules move: a loose file (no
    matching directory) falls back to the exempt pseudo-package `"<root>"`
    rather than raising on an unrecognised top-level name.
    """
    if dotted == "ui.theme":
        return "ui.theme"
    parts = dotted.split(".")
    first = parts[0]
    if not (_SRC / first).is_dir():
        return "<root>"
    package = _DIR_TO_PACKAGE.get(first)
    if package is None:
        return "<root>"
    if len(parts) >= 2 and (_SRC / first / parts[1]).is_dir():
        refined = _SUBPACKAGE_REFINEMENTS.get((first, parts[1]))
        if refined is not None:
            return refined
    return package


def _qualname(dotted: str, package: str) -> str:
    """Offender id for `dotted`: bare when it already names its own package,
    `package.dotted` when the `converters` -> `newave` remap would otherwise
    hide which package the module resolves to."""
    if package == "<root>" or dotted == package or dotted.startswith(package + "."):
        return dotted
    return f"{package}.{dotted}"


def _module_dotted(path: Path) -> str:
    parts = list(path.relative_to(_SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _type_checking_node_ids(tree: ast.Module) -> set[int]:
    """id() of every node inside an `if TYPE_CHECKING:` body, so `_imports`
    can classify each import without a second parse."""
    marked: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for stmt in node.body:
                marked.update(id(sub) for sub in ast.walk(stmt))
    return marked


def _imports(path: Path) -> list[tuple[str, list[str], bool]]:
    """Return `(target_dotted, imported_names, in_type_checking)` triples
    for every `cobre_bridge`-internal import in `path`.

    Lazy (function-body) imports are runtime imports -- ``ast.walk`` already
    descends into them, which is deliberate.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    type_checking_ids = _type_checking_node_ids(tree)
    triples: list[tuple[str, list[str], bool]] = []
    for node in ast.walk(tree):
        in_type_checking = id(node) in type_checking_ids
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.module == "cobre_bridge":
                for alias in node.names:
                    triples.append((alias.name, [alias.name], in_type_checking))
            elif node.module.startswith("cobre_bridge."):
                target = node.module.removeprefix("cobre_bridge.")
                names = [alias.name for alias in node.names]
                triples.append((target, names, in_type_checking))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cobre_bridge."):
                    target = alias.name.removeprefix("cobre_bridge.")
                    triples.append((target, [], in_type_checking))
    return triples


def _all_py_files() -> list[Path]:
    return sorted(p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _find_root_modules() -> frozenset[str]:
    return frozenset(p.stem for p in _SRC.glob("*.py") if p.name != "__init__.py")


def _find_shadowed_modules() -> frozenset[str]:
    return frozenset(
        "cobre_bridge." + _module_dotted(p)
        for p in _all_py_files()
        if p.name != "__init__.py" and p.stem in sys.stdlib_module_names
    )


def _scan_imports() -> tuple[
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str, str]],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str]],
]:
    """Single-pass walk of every import edge in the tree, returning
    `(direction_violations, private_violations, cli_violations,
    type_checking_census)`."""
    direction: set[tuple[str, str]] = set()
    private: set[tuple[str, str, str]] = set()
    cli_imports: set[tuple[str, str]] = set()
    type_checking: set[tuple[str, str]] = set()

    for path in _all_py_files():
        source_dotted = _module_dotted(path)
        source_package = _package_of(source_dotted)
        source_id = _qualname(source_dotted, source_package)

        for target_dotted, names, in_type_checking in _imports(path):
            target_package = _package_of(target_dotted)
            target_id = _qualname(target_dotted, target_package)
            cross_package = source_package != target_package
            root_involved = "<root>" in (source_package, target_package)

            # Rule E has no TYPE_CHECKING exemption and no root exemption.
            if target_package == "cli" and source_package != "cli":
                cli_imports.add((source_id, target_id))

            if in_type_checking:
                if cross_package and not root_involved:
                    type_checking.add((source_id, target_id))
                continue

            if root_involved:
                continue

            if target_package not in _ALLOWED.get(source_package, frozenset()):
                direction.add((source_id, target_id))

            if cross_package:
                for name in names:
                    if name.startswith("_"):
                        private.add((source_id, target_id, name))

    return (
        frozenset(direction),
        frozenset(private),
        frozenset(cli_imports),
        frozenset(type_checking),
    )


# Rule C. Shrinks to empty as loose modules move into a package; never widen.
_PENDING_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "cli",
        "cli_args",
        "cobre_validation",
        "config_resolution",
        "conversion_manifest",
        "logging_config",
        "verdict",
    }
)

# Rule A. Shrinks to empty as each move corrects its direction; never widen.
_PENDING_DIRECTION_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("comparators.charts._shared", "ui.html"),
        ("comparators.charts._shared", "ui.plotly_helpers"),
        ("comparators.charts.constraints", "ui.plotly_helpers"),
        ("comparators.charts.convergence", "ui.plotly_helpers"),
        ("comparators.charts.costs", "ui.plotly_helpers"),
        ("comparators.charts.fpha", "ui.html"),
        ("comparators.charts.fpha", "ui.plotly_helpers"),
        ("comparators.charts.hydro", "ui.plotly_helpers"),
        ("comparators.charts.network", "ui.plotly_helpers"),
        ("comparators.charts.performance", "ui.plotly_helpers"),
        ("comparators.charts.productivity", "ui.html"),
        ("comparators.charts.productivity", "ui.plotly_helpers"),
        ("comparators.charts.spillage", "ui.plotly_helpers"),
        ("comparators.charts.system", "ui.plotly_helpers"),
        ("comparators.charts.thermal", "ui.plotly_helpers"),
        ("comparators.decomp_results", "decomp.case"),
        ("comparators.decomp_results", "decomp.constraint_registers"),
        ("comparators.html_report", "ui.css"),
        ("comparators.html_report", "ui.html"),
        ("comparators.html_report", "ui.js"),
        ("comparators.newave_readers", "newave.converters.stochastic"),
        ("comparators.report", "ui.console"),
        ("comparators.report_builder", "ui.plotly_helpers"),
        ("comparators.results", "newave.converters.constraints"),
        ("comparators.results", "newave.converters.hydro"),
        # writer-relocation debt: _write_diagnostics_json's lazy import, retired
        # when the cli/ epic moves the writer out of core.diagnostics.
        ("core.diagnostics", "ui.console"),
        ("dashboard", "ui.css"),
        ("dashboard", "ui.html"),
        ("dashboard", "ui.js"),
        ("dashboard.chart_helpers", "ui.html"),
        ("dashboard.chart_helpers", "ui.plotly_helpers"),
        ("dashboard.tabs.constraints", "ui.html"),
        ("dashboard.tabs.constraints", "ui.plotly_helpers"),
        ("dashboard.tabs.constraints_utils", "ui.html"),
        ("dashboard.tabs.costs", "ui.html"),
        ("dashboard.tabs.costs", "ui.plotly_helpers"),
        ("dashboard.tabs.energy_balance", "ui.html"),
        ("dashboard.tabs.energy_balance", "ui.plotly_helpers"),
        ("dashboard.tabs.network", "ui.html"),
        ("dashboard.tabs.network", "ui.plotly_helpers"),
        ("dashboard.tabs.overview", "ui.html"),
        ("dashboard.tabs.overview", "ui.plotly_helpers"),
        ("dashboard.tabs.performance", "ui.html"),
        ("dashboard.tabs.performance", "ui.plotly_helpers"),
        ("dashboard.tabs.performance_charts", "ui.plotly_helpers"),
        ("dashboard.tabs.plants", "ui.html"),
        ("dashboard.tabs.plants", "ui.js"),
        ("dashboard.tabs.plants", "ui.plotly_helpers"),
        ("dashboard.tabs.stochastic", "ui.html"),
        ("dashboard.tabs.stochastic", "ui.js"),
        ("dashboard.tabs.stochastic", "ui.plotly_helpers"),
        ("dashboard.tabs.training", "ui.html"),
        ("dashboard.tabs.training", "ui.plotly_helpers"),
    }
)

# Rule B. Shrinks to empty as each name is promoted to a public home; never widen.
_PENDING_PRIVATE_EDGES: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("dashboard.tabs.plants", "ui.html", "_sparkline_svg"),
    }
)

# Rule D. Shrinks to empty as the shadowing file becomes a package; never widen.
_PENDING_SHADOWED_MODULES: frozenset[str] = frozenset(
    {
        "cobre_bridge.ui.html",
    }
)

# TYPE_CHECKING census: exact set, not an upper bound -- an unexpected edge
# fails this and so does a stale one, so touching one is a conscious act.
_TYPE_CHECKING_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("comparators.alignment", "newave.case"),
        ("comparators.alignment", "newave.id_map"),
        ("comparators.constraints_compare", "newave.id_map"),
        ("comparators.decomp_results", "decomp.case"),
        ("comparators.decomp_results", "decomp.constraint_registers"),
        ("comparators.decomp_results", "decomp.id_map"),
        ("comparators.results", "newave.case"),
        ("comparators.results", "newave.id_map"),
        ("core.provenance", "decomp.pipeline"),
        ("core.provenance", "newave.files"),
        ("decomp.constraints", "core.generic_constraint_builder"),
        ("ui.console", "comparators.verdict"),
        ("ui.console", "core.conversion"),
        ("ui.console", "core.preflight"),
    }
)


def test_import_direction() -> None:
    """Every cross-package runtime import's target package is in the
    source package's `_ALLOWED` set, or the pair is on the pending
    burn-down list."""
    direction, _private, _cli, _tc = _scan_imports()
    offenders = sorted(direction - _PENDING_DIRECTION_EDGES)
    assert offenders == []


def test_direction_allowlist_has_no_stale_entries() -> None:
    """A pending direction pair with no matching live violation must be
    removed in the same change that fixes it -- otherwise it would mask a
    regression that reintroduces the same edge."""
    direction, _private, _cli, _tc = _scan_imports()
    stale = sorted(_PENDING_DIRECTION_EDGES - direction)
    assert stale == []


def test_no_private_cross_package_imports() -> None:
    """No module imports an underscore-private name from a module in a
    different package, unless the triple is on the pending list."""
    _direction, private, _cli, _tc = _scan_imports()
    offenders = sorted(private - _PENDING_PRIVATE_EDGES)
    assert offenders == []


def test_private_allowlist_has_no_stale_entries() -> None:
    _direction, private, _cli, _tc = _scan_imports()
    stale = sorted(_PENDING_PRIVATE_EDGES - private)
    assert stale == []


def test_no_loose_root_modules() -> None:
    """Every module at `src/cobre_bridge/*.py` belongs to a package,
    unless its stem is still on the pending list."""
    offenders = sorted(_find_root_modules() - _PENDING_ROOT_MODULES)
    assert offenders == []


def test_root_allowlist_has_no_stale_entries() -> None:
    stale = sorted(_PENDING_ROOT_MODULES - _find_root_modules())
    assert stale == []


def test_no_stdlib_shadowing_modules() -> None:
    """No `.py` file's stem shadows a stdlib module name, unless its
    dotted path is on the pending list."""
    offenders = sorted(_find_shadowed_modules() - _PENDING_SHADOWED_MODULES)
    assert offenders == []


def test_shadowed_allowlist_has_no_stale_entries() -> None:
    stale = sorted(_PENDING_SHADOWED_MODULES - _find_shadowed_modules())
    assert stale == []


def test_nothing_imports_cli() -> None:
    """No module outside the `cli` package imports one inside it -- no
    allowlist, the live count is zero."""
    _direction, _private, cli_imports, _tc = _scan_imports()
    assert sorted(cli_imports) == []


def test_type_checking_edges_match_tree() -> None:
    """The cross-package `TYPE_CHECKING`-only edges, exact set: an
    unexpected edge fails this the same as a stale one, so widening or
    shrinking the census is always a conscious diff."""
    _direction, _private, _cli, type_checking = _scan_imports()
    assert type_checking == _TYPE_CHECKING_EDGES


def _print_literal(name: str, value: frozenset[object]) -> None:
    print(f"{name} = frozenset(")
    print("    {")
    for item in sorted(value, key=repr):
        print(f"        {item!r},")
    print("    }")
    print(")")
    print()


if __name__ == "__main__":
    _direction, _private, _cli, _type_checking = _scan_imports()
    _print_literal("_PENDING_ROOT_MODULES", _find_root_modules())
    _print_literal("_PENDING_DIRECTION_EDGES", _direction)
    _print_literal("_PENDING_PRIVATE_EDGES", _private)
    _print_literal("_PENDING_SHADOWED_MODULES", _find_shadowed_modules())
    _print_literal("_TYPE_CHECKING_EDGES", _type_checking)

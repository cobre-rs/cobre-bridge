"""Unit tests for the Rich rendering layer (``cobre_bridge.ui.console``)."""

from __future__ import annotations

import io

from rich.console import Console

from cobre_bridge.diagnostics import Diagnostic, DiagnosticTable, Severity
from cobre_bridge.pipeline import ConversionReport
from cobre_bridge.ui.console import (
    MAX_TABLE_ROWS,
    _progress_enabled,
    conversion_progress,
    render_conversion_summary,
    render_diagnostics,
    render_error,
    spinner,
)


def _console() -> tuple[Console, io.StringIO]:
    """A deterministic, colourless, fixed-width console writing to a buffer."""
    buf = io.StringIO()
    console = Console(file=buf, width=100, no_color=True, highlight=False, emoji=False)
    return console, buf


class TestConversionSummary:
    def test_summary_line_keeps_entity_counts(self) -> None:
        console, buf = _console()
        report = ConversionReport(
            hydro_count=10, thermal_count=5, bus_count=4, line_count=3, stage_count=60
        )
        render_conversion_summary(report, console=console)
        text = buf.getvalue()
        assert "10 hydros" in text
        assert "5 thermals" in text
        assert "60 stages" in text


class TestRenderError:
    def test_error_prefix_and_message_preserved(self) -> None:
        console, buf = _console()
        render_error("source directory 'x' does not exist", console=console)
        assert "Error: source directory 'x' does not exist" in buf.getvalue()


class TestRenderDiagnostics:
    def _gtmin(self) -> Diagnostic:
        return Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title="GTMIN exceeds capacity (1 plant)",
            summary="one plant affected",
            table=DiagnosticTable(
                columns=["Plant", "Code", "Stages", "GTMIN MW", "Cap MW"],
                rows=[["ANGRA 2", 13, "2-3", 481.3, 423.4]],
                justify=["left", "right", "left", "right", "right"],
            ),
            remediation="Check EXPT FCMAX/GTMIN and MANUTT for these plants.",
        )

    def test_empty_renders_nothing(self) -> None:
        console, buf = _console()
        render_diagnostics([], console=console)
        assert buf.getvalue() == ""

    def test_renders_title_table_cells_and_remediation(self) -> None:
        console, buf = _console()
        render_diagnostics([self._gtmin()], console=console)
        text = buf.getvalue()
        assert "GTMIN exceeds capacity" in text
        assert "ANGRA 2" in text  # resolved plant name, not just the code
        assert "2-3" in text  # stage range
        assert "481.3" in text  # GTMIN value
        assert "423.4" in text  # capacity value
        assert "Check EXPT FCMAX/GTMIN" in text  # remediation hint
        assert "Thermal bounds" in text  # category header

    def test_rollup_counts_warnings_and_notes(self) -> None:
        console, buf = _console()
        note = Diagnostic(
            code="n",
            severity=Severity.INFO,
            category="Entity exclusion",
            title="Note",
            summary="a note",
        )
        render_diagnostics([self._gtmin(), note], console=console)
        text = buf.getvalue()
        assert "1 warning(s)" in text
        assert "1 note(s)" in text

    def test_quiet_suppresses_info_but_keeps_warnings(self) -> None:
        console, buf = _console()
        note = Diagnostic(
            code="n",
            severity=Severity.INFO,
            category="Entity exclusion",
            title="Fictitious note",
            summary="excluded plants",
        )
        render_diagnostics([self._gtmin(), note], console=console, quiet=True)
        text = buf.getvalue()
        assert "GTMIN exceeds capacity" in text
        assert "Fictitious note" not in text

    def test_long_table_is_capped_with_overflow_caption(self) -> None:
        console, buf = _console()
        rows = [[f"P{i}", i] for i in range(MAX_TABLE_ROWS + 5)]
        diag = Diagnostic(
            code="many",
            severity=Severity.INFO,
            category="Entity exclusion",
            title="Many plants",
            summary="lots",
            table=DiagnosticTable(columns=["Plant", "Code"], rows=rows),
        )
        render_diagnostics([diag], console=console)
        text = buf.getvalue()
        assert "5 more" in text  # overflow summarised
        assert "P0" in text
        assert f"P{MAX_TABLE_ROWS + 4}" not in text  # last row not printed


class TestProgressGating:
    def test_enabled_only_on_interactive_non_verbose_non_quiet(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
        assert _progress_enabled(verbose=False, quiet=False) is True
        assert _progress_enabled(verbose=True, quiet=False) is False
        assert _progress_enabled(verbose=False, quiet=True) is False

    def test_disabled_off_tty(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
        assert _progress_enabled(verbose=False, quiet=False) is False

    def test_conversion_progress_noop_off_tty(self, capsys) -> None:
        # Under pytest stderr is captured (not a TTY): the bar is suppressed and the
        # yielded callback is a harmless no-op, so output stays deterministic.
        with conversion_progress(6) as step:
            step("Discovering files")
            step("Writing Parquet")
        assert capsys.readouterr().err == ""

    def test_spinner_noop_off_tty(self, capsys) -> None:
        with spinner("Comparing…"):
            pass
        assert capsys.readouterr().err == ""

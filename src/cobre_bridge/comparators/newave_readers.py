"""NEWAVE output file readers for results comparison.

Reads MEDIAS CSV files (hydro, thermal, system) from the ``saidas/``
directory and pmo.dat convergence/productivity data.

MEDIAS files are parsed directly with Polars since inewave v1.13 does
not provide dedicated reader classes for them.  pmo.dat is read via
``inewave.newave.Pmo``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import polars as pl

_LOG = logging.getLogger(__name__)


def _find_case_insensitive(directory: Path, filename: str) -> Path | None:
    """Find a file in *directory* ignoring case, or return None."""
    lower = filename.lower()
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() == lower:
                return entry
    except OSError:
        pass
    return None


def _find_saidas_dir(newave_dir: Path) -> Path | None:
    """Locate the ``saidas/`` subdirectory case-insensitively."""
    lower = "saidas"
    try:
        for entry in newave_dir.iterdir():
            if entry.is_dir() and entry.name.lower() == lower:
                return entry
    except OSError:
        pass
    return None


def _find_pmo(newave_dir: Path) -> Path | None:
    """Locate ``pmo.dat`` in the NEWAVE directory or its ``saidas/`` subdir."""
    # Try root first, then saidas/.
    result = _find_case_insensitive(newave_dir, "pmo.dat")
    if result is not None:
        return result
    saidas = _find_saidas_dir(newave_dir)
    if saidas is not None:
        return _find_case_insensitive(saidas, "pmo.dat")
    return None


# -------------------------------------------------------------------
# MEDIAS CSV readers
# -------------------------------------------------------------------

# MEDIAS CSV structure (semicolon-separated):
#   SÉRIE/CENÁRIO ; USINA ; PATAMAR ; EST001 ; EST002 ; ...
# We filter for PATAMAR=0 (full-stage average) and pivot from wide
# to long format: (newave_code, stage, value).


def _read_medias_csv(
    saidas_dir: Path,
    filename: str,
    variable_filter: str | None = None,
) -> pl.DataFrame:
    """Read a MEDIAS CSV and return long-format DataFrame.

    Returns DataFrame with columns:
    - ``newave_code`` (Int64): entity code (usina or submercado)
    - ``stage`` (Int64): 1-indexed stage number
    - ``variable`` (Utf8): variable name from CSV (if multi-variable)
    - ``value`` (Float64): stage mean value

    Returns an empty DataFrame with the correct schema if the file
    is not found.
    """
    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )

    path = _find_case_insensitive(saidas_dir, filename)
    if path is None:
        _LOG.warning("%s not found in %s", filename, saidas_dir)
        return empty

    # Try comma first (NEWAVE v29+ format), fall back to semicolon.
    for sep in (",", ";"):
        try:
            df = pl.read_csv(
                path,
                separator=sep,
                truncate_ragged_lines=True,
                infer_schema_length=100,
            )
            if len(df.columns) > 2:
                break
        except Exception:  # noqa: BLE001
            continue
    else:
        _LOG.warning("Failed to parse %s", path)
        return empty

    # Normalize column names: strip whitespace.
    df = df.rename({c: c.strip() for c in df.columns})

    # Identify entity column, variable column, patamar column, and stage
    # value columns.
    #
    # NEWAVE MEDIAS files come in two layouts:
    #   v29+:  USIH_ext, VAR, 3, 4, 5, ...   (comma-sep, no PATAMAR/SERIE)
    #   older: SERIE; USINA; PATAMAR; EST001; EST002; ...  (semicolon-sep)
    entity_col: str | None = None
    patamar_col: str | None = None
    variable_col: str | None = None
    stage_cols: list[str] = []

    for col in df.columns:
        upper = col.upper()
        # v29+ entity columns: USIH_ext, USIT_ext, SBM_ext, etc.
        if upper.endswith("_EXT") or upper in (
            "USINA",
            "CODIGO_USINA",
            "SUBMERCADO",
            "CODIGO_SUBMERCADO",
        ):
            entity_col = col
        elif upper in ("PATAMAR", "PAT"):
            patamar_col = col
        elif upper in ("SERIE", "CENARIO", "SÉRIE/CENÁRIO", "SERIE/CENARIO"):
            pass  # skip scenario column
        elif upper in ("VAR", "VARIAVEL"):
            variable_col = col
        elif upper.startswith("EST") or upper.startswith("ESTAGIO"):
            stage_cols.append(col)

    if entity_col is None:
        # Fallback: first column is usually the entity code.
        entity_col = df.columns[0]

    # If no EST* columns found, look for bare-integer column names (v29+).
    if not stage_cols:
        exclude = {entity_col, patamar_col, variable_col}
        for col in df.columns:
            if col in exclude:
                continue
            # Bare integer column names like "3", "4", ..., "60".
            stripped = col.strip()
            if stripped.isdigit():
                stage_cols.append(col)
                continue
            # Also accept numeric-typed columns as fallback.
            if df[col].dtype in (pl.Float64, pl.Int64, pl.Float32):
                stage_cols.append(col)

    if not stage_cols:
        _LOG.warning("No stage columns found in %s", filename)
        return empty

    # Filter patamar=0 (stage-level average) — only present in older format.
    if patamar_col is not None:
        df = df.filter(pl.col(patamar_col).cast(pl.Int64, strict=False) == 0)

    # Filter to mean scenario — only present in older format.
    serie_col: str | None = None
    for col in df.columns:
        upper = col.upper()
        if upper in ("SERIE", "CENARIO", "SÉRIE/CENÁRIO", "SERIE/CENARIO"):
            serie_col = col
            break

    if serie_col is not None:
        df = df.filter(pl.col(serie_col).cast(pl.Int64, strict=False) == 1)

    # Filter by variable if needed.
    if variable_filter is not None and variable_col is not None:
        df = df.filter(
            pl.col(variable_col).str.strip_chars().str.to_uppercase() == variable_filter
        )

    # Unpivot stage columns to long format.
    id_cols = [entity_col]
    if variable_col is not None:
        id_cols.append(variable_col)

    try:
        long_df = df.unpivot(
            on=stage_cols,
            index=id_cols,
            variable_name="stage_col",
            value_name="value",
        )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to unpivot %s", filename)
        return empty

    # Extract stage number from column name.
    # For bare-integer names ("3", "4") this extracts the number directly.
    # For EST001-style names it strips the prefix.
    long_df = long_df.with_columns(
        pl.col("stage_col")
        .str.replace_all(r"[^0-9]", "")
        .cast(pl.Int64, strict=False)
        .alias("stage")
    )

    # Build output.  Strip whitespace before casting — MEDIAS values
    # are whitespace-padded (e.g. "   1", "   525.55").
    result = long_df.select(
        pl.col(entity_col)
        .str.strip_chars()
        .cast(pl.Int64, strict=False)
        .alias("newave_code"),
        pl.col("stage"),
        (
            pl.col(variable_col).str.strip_chars().alias("variable")
            if variable_col is not None
            else pl.lit("value").alias("variable")
        ),
        pl.col("value").str.strip_chars().cast(pl.Float64, strict=False),
    ).drop_nulls(subset=["newave_code", "stage"])

    return result


def read_medias_hydro(saidas_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-USIH.CSV and return hydro results.

    Returns DataFrame with columns: ``newave_code``, ``stage``,
    ``variable``, ``value``.  Variables include VARMUH, GHIDUH,
    QTURUH, QVERTUH, QAFLUH, etc.

    Returns empty DataFrame if file not found.
    """
    return _read_medias_csv(saidas_dir, "MEDIAS-USIH.CSV")


def read_medias_thermal(saidas_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-USIT.CSV and return thermal results.

    Returns DataFrame with columns: ``newave_code``, ``stage``,
    ``variable``, ``value``.

    Returns empty DataFrame if file not found.
    """
    return _read_medias_csv(saidas_dir, "MEDIAS-USIT.CSV")


def read_medias_system(saidas_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-MERC.CSV and return system/market results.

    Returns DataFrame with columns: ``newave_code`` (submercado code),
    ``stage``, ``variable``, ``value``.

    Returns empty DataFrame if file not found.
    """
    return _read_medias_csv(saidas_dir, "MEDIAS-MERC.CSV")


# -------------------------------------------------------------------
# PMO readers
# -------------------------------------------------------------------


# Match one data row of the pmo.dat convergence table:
#   iter   LIM.INF   ZINF   LIM.SUP   ZSUP   DZINF   ZSUP_ITER   [tempo]
# Column semantics (NEWAVE manual):
#   - ZINF        = cuts-based lower bound at end of iteration (SDDP LB)
#   - ZSUP        = mean forward-simulation cost (SDDP UB)
#   - LIM.INF/SUP = lower/upper confidence-interval brackets on ZSUP; collapse
#                   to ZSUP when there is a single forward sample
#   - ZSUP_ITER   = current-iteration forward cost (per-sample, not aggregated)
# DZINF is "-" on the first iteration and numeric afterwards. The trailing
# wall-clock field is optional. We greedily consume six numeric columns
# (treating "-" as a missing DZINF) so that subsequent layout drift in
# pmo.dat (longer decimal widths, extra padding) does not break us.
_PMO_CONV_ROW = re.compile(
    r"^\s*(\d+)"  # iteration                                     -> group 1
    r"\s+(?:\d+\.\d+)"  # LIM.INF (not captured)
    r"\s+(\d+\.\d+)"  # ZINF (cuts-based lower bound)           -> group 2
    r"\s+(?:\d+\.\d+)"  # LIM.SUP (not captured)
    r"\s+(\d+\.\d+)"  # ZSUP  (mean forward cost = upper bound) -> group 3
    r"\s+(?:-|\d+\.\d+)"  # DZINF
    r"\s+(?:\d+\.\d+)"  # ZSUP_ITER (per-iter sample; not captured)
    r"(?:\s+\S+)?\s*$",  # optional tempo column
    re.MULTILINE,
)


def _parse_pmo_convergence_text(text: str) -> dict[int, tuple[float, float]]:
    """Extract the iteration table from raw pmo.dat text.

    Returns ``{iteration: (zinf, zsup)}`` keyed on iteration number,
    taking the LAST occurrence per iteration (NEWAVE writes two lines
    per iteration; only the second carries DZINF and ZSUP_ITER, which
    the regex requires, so this naturally selects the post-backward
    summary line).
    """
    out: dict[int, tuple[float, float]] = {}
    for match in _PMO_CONV_ROW.finditer(text):
        it = int(match.group(1))
        zinf = float(match.group(2))
        zsup = float(match.group(3))
        out[it] = (zinf, zsup)
    return out


def read_pmo_convergence(newave_dir: Path) -> pl.DataFrame:
    """Read pmo.dat convergence table.

    Returns DataFrame with columns: ``iteration`` (Int64),
    ``lower_bound`` (Float64), ``upper_bound_mean`` (Float64).

    The bound columns come from NEWAVE's ``ZINF`` (cuts-based lower
    bound) and ``ZSUP`` (mean forward-simulation cost) fields, parsed
    directly from the pmo.dat iteration table via regex. inewave's
    tabular parser misaligns the high-precision columns on real-world
    pmo.dat layouts (truncates ``LIM.INF`` to 3 decimals, drops the
    leading digit of ``LIM.SUP``, and frequently returns ``NaN`` for
    ``ZINF``), so we cannot use ``Pmo.convergencia`` here.

    Returns empty DataFrame if pmo.dat is not found or the iteration
    table cannot be located.
    """
    empty = pl.DataFrame(
        schema={
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        }
    )

    pmo_path = _find_pmo(newave_dir)
    if pmo_path is None:
        _LOG.warning("pmo.dat not found in %s", newave_dir)
        return empty

    try:
        text = pmo_path.read_text(encoding="latin-1")
    except OSError:
        _LOG.exception("Failed to read pmo.dat at %s", pmo_path)
        return empty

    rows = _parse_pmo_convergence_text(text)
    if not rows:
        _LOG.warning(
            "No convergence-table rows matched in %s; "
            "expected the NEWAVE 'ITER LIM.INF ZINF LIM.SUP ZSUP DZINF "
            "ZSUP_ITER' table",
            pmo_path,
        )
        return empty

    iters = sorted(rows)
    result = pl.DataFrame(
        {
            "iteration": iters,
            "lower_bound": [rows[i][0] for i in iters],
            "upper_bound_mean": [rows[i][1] for i in iters],
        },
        schema={
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        },
    )

    # NEWAVE pmo.dat exports convergence values in 10^6 R$.
    # Multiply by 1e6 to convert to R$ (matching Cobre convention).
    return result.with_columns(
        pl.col("lower_bound") * 1e6,
        pl.col("upper_bound_mean") * 1e6,
    )


# pmo.dat ``produtibilidades_equivalentes`` source columns → output names.
# These are the per-plant productivities NEWAVE's FPHA produces at different
# heads (altura min/65/max), the equivalent productivity between min/max
# volume, and the accumulated productivity used for EARM. Many run-of-river
# plants leave the altura_* / acumulada columns NaN — preserved as null so the
# per-comparison drop-NaN logic in the report keeps them out of each scatter
# without crashing.
_PMO_PROD_COLUMNS: dict[str, str] = {
    "produtibilidade_altura_minima": "altura_min",
    "produtibilidade_altura_65": "altura_65",
    "produtibilidade_altura_maxima": "altura_max",
    "produtibilidade_equivalente_volmin_volmax": "equivalent",
    "produtibilidade_acumulada_calculo_earm": "accumulated_earm",
}


def read_pmo_productivity_detail(newave_dir: Path) -> pl.DataFrame:
    """Read the per-plant productivity breakdown from pmo.dat.

    Pulls ``pmo.produtibilidades_equivalentes`` (filtered to the first
    configuration, ``configuracao == min``) and surfaces the head-dependent
    productivities NEWAVE's FPHA computes.

    Returns DataFrame with columns: ``plant_name`` (Utf8), ``altura_min``,
    ``altura_65``, ``altura_max``, ``equivalent``, ``accumulated_earm``
    (all Float64; NaN preserved as null so run-of-river plants with no
    head-dependence don't crash per-column comparisons).

    Returns an empty DataFrame with the correct schema if pmo.dat is not
    found or the productivity table is unavailable.
    """
    empty = pl.DataFrame(
        schema={
            "plant_name": pl.Utf8,
            "altura_min": pl.Float64,
            "altura_65": pl.Float64,
            "altura_max": pl.Float64,
            "equivalent": pl.Float64,
            "accumulated_earm": pl.Float64,
        }
    )

    pmo_path = _find_pmo(newave_dir)
    if pmo_path is None:
        _LOG.warning("pmo.dat not found in %s", newave_dir)
        return empty

    try:
        from inewave.newave import Pmo

        pmo = Pmo.read(str(pmo_path))
        prod_df = pmo.produtibilidades_equivalentes
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read productivities from pmo.dat")
        return empty

    if prod_df is None or prod_df.empty:
        return empty
    if "nome_usina" not in prod_df.columns:
        _LOG.warning(
            "pmo.dat productivities lack nome_usina column: %s",
            list(prod_df.columns),
        )
        return empty

    df = prod_df
    if "configuracao" in df.columns:
        df = df[df["configuracao"] == df["configuracao"].min()]

    source_cols = [c for c in _PMO_PROD_COLUMNS if c in df.columns]
    result = pl.from_pandas(df[["nome_usina", *source_cols]])
    result = result.rename(
        {"nome_usina": "plant_name", **{c: _PMO_PROD_COLUMNS[c] for c in source_cols}}
    )
    casts = [pl.col("plant_name").str.strip_chars()]
    casts += [
        pl.col(_PMO_PROD_COLUMNS[c]).cast(pl.Float64, strict=False) for c in source_cols
    ]
    result = result.with_columns(casts).drop_nulls(subset=["plant_name"])

    # Backfill any column the source frame lacked so the schema is stable.
    for out_name in _PMO_PROD_COLUMNS.values():
        if out_name not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(out_name))

    return result.select(
        "plant_name",
        "altura_min",
        "altura_65",
        "altura_max",
        "equivalent",
        "accumulated_earm",
    )


_INT_FILENAME_RE = re.compile(r"^int(\d{3})(\d{3})\.out$", re.IGNORECASE)


def read_nwlistop_intercambio(saidas_dir: Path) -> pl.DataFrame:
    """Read NWLISTOP ``intXXXYYY.out`` files into long-format flow data.

    NWLISTOP emits one file per directional submercado pair containing
    block-decomposed flow (``patamar``) plus an aggregated ``TOTAL``
    row whose value is the block-hours-weighted month average — the
    natural counterpart to the per-stage interchange we want to
    compare against Cobre's ``net_flow_mw``.

    Stage numbering follows the MEDIAS convention: 1 = first calendar
    month of the file (typically January of NEWAVE's first study
    year), 9 = September year 1, 21 = September year 2, etc. This
    matches MEDIAS column headers (which start at the first study
    month, e.g. 9 for a September-start study) so the existing
    ``_nw_stage_offset`` logic translates both consistently into
    Cobre's 0-based ``stage_id``.

    Returns columns ``from_submarket_code`` (Int64),
    ``to_submarket_code`` (Int64), ``from_name`` (Utf8),
    ``to_name`` (Utf8), ``stage`` (Int64), ``variable`` (Utf8, always
    "INTERC"), ``value`` (Float64).
    Returns empty DataFrame when no int*.out files are present.
    """
    empty = pl.DataFrame(
        schema={
            "from_submarket_code": pl.Int64,
            "to_submarket_code": pl.Int64,
            "from_name": pl.Utf8,
            "to_name": pl.Utf8,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )
    if not saidas_dir.is_dir():
        return empty

    matches: list[tuple[int, int, Path]] = []
    for path in saidas_dir.iterdir():
        m = _INT_FILENAME_RE.match(path.name)
        if m is None:
            continue
        matches.append((int(m.group(1)), int(m.group(2)), path))
    if not matches:
        return empty

    from inewave.nwlistop import Intercambio  # local import — heavy module

    rows: list[dict] = []
    for from_code, to_code, path in matches:
        try:
            obj = Intercambio.read(str(path))
        except Exception:  # noqa: BLE001
            _LOG.warning("Failed to parse %s", path)
            continue
        df = obj.valores
        if df is None or df.empty:
            continue
        total = df[df["patamar"] == "TOTAL"].copy()
        if total.empty:
            continue
        total["data"] = pd.to_datetime(total["data"])
        min_year = int(total["data"].dt.year.min())
        total["stage"] = (total["data"].dt.year - min_year) * 12 + total[
            "data"
        ].dt.month
        # Series cover disjoint years in NWLISTOP output; mean is a no-op when
        # only one series carries a given stage, and remains correct otherwise.
        agg = total.groupby("stage")["valor"].mean().reset_index()

        from_name = str(obj.submercado_de or "").strip()
        to_name = str(obj.submercado_para or "").strip()
        for _, r in agg.iterrows():
            rows.append(
                {
                    "from_submarket_code": from_code,
                    "to_submarket_code": to_code,
                    "from_name": from_name,
                    "to_name": to_name,
                    "stage": int(r["stage"]),
                    "variable": "INTERC",
                    "value": float(r["valor"]),
                }
            )
    if not rows:
        return empty
    return pl.from_dicts(rows).cast(
        {
            "from_submarket_code": pl.Int64,
            "to_submarket_code": pl.Int64,
            "stage": pl.Int64,
            "value": pl.Float64,
        }
    )


def read_medias_market(saidas_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-MERC.CSV with all variables in long format.

    Returns DataFrame with columns: ``newave_code``, ``stage``,
    ``variable``, ``value``.  Unlike ``read_medias_system`` which only
    returns CMO/DEFT, this returns all market variables for energy
    balance analysis.
    """
    return _read_medias_csv(saidas_dir, "MEDIAS-MERC.CSV")


def read_medias_sin(saidas_dir: Path) -> pl.DataFrame:
    """Read MEDIAS-SIN.CSV (system interconnected aggregate).

    Returns DataFrame with columns ``newave_code`` (always 0 for SIN),
    ``stage``, ``variable``, ``value``.  Useful variables include
    ``EARMF`` (stored energy final, MWmes), ``ENA`` (natural energy
    inflow, MWmes), ``EARMFP`` (percentage of max storage).
    """
    return _read_medias_csv(saidas_dir, "MEDIAS-SIN.CSV")


_MERCL_FILE_RE = re.compile(r"mercl(\d+)\.out$", re.IGNORECASE)
_MERCL_YEAR_RE = re.compile(r"ANO:\s*(\d{4})")
_MERCL_VALUE_RE = re.compile(r"-?\d+\.\d*")


def _parse_mercl_file(path: Path) -> dict[tuple[int, int], float]:
    """Parse an nwlistop ``mercl<NNN>.out`` file.

    Returns ``{(year, month): net_load_mwmed}``.  Each ``ANO: YYYY`` block is
    followed by a header row (bare column indices 1..12) and a value row of 12
    trailing-dot floats; calendar months outside the horizon are written ``0.``.
    """
    out: dict[tuple[int, int], float] = {}
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = _MERCL_YEAR_RE.search(line)
        if m is None:
            continue
        year = int(m.group(1))
        # The value row is the first following line with >= 12 *dotted* floats
        # (the header row carries bare indices 1..12, which have no dot).
        for j in range(i + 1, min(i + 4, len(lines))):
            nums = _MERCL_VALUE_RE.findall(lines[j])
            if len(nums) >= 12:
                for month, raw in enumerate(nums[:12], start=1):
                    out[(year, month)] = float(raw)
                break
    return out


def read_newave_net_load_nwlistop(newave_dir: Path) -> pl.DataFrame:
    """Read net load (MERCADO LIQUIDO) from per-submarket ``mercl*.out`` files.

    nwlistop writes one ``mercl<NNN>.out`` per submarket (``mercl001`` =
    submarket 1, ...) plus ``merclsin.out`` for the SIN total.  Unlike the
    ``sistema.dat`` reconstruction in :func:`read_newave_net_load`, these cover
    the **full horizon** (study + post-study), so the energy-balance charts and
    the comparison data do not drop after the study period.

    Returns the same schema as :func:`read_newave_net_load`, or an empty frame
    when no per-submarket ``mercl`` files are present.
    """
    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )
    saidas = _find_saidas_dir(newave_dir)
    if saidas is None:
        return empty
    # Per-submarket files only; merclsin.out (the SIN total) has no digits and
    # is excluded by the regex.
    files = sorted(p for p in saidas.iterdir() if _MERCL_FILE_RE.match(p.name))
    if not files:
        return empty

    parsed: dict[int, dict[tuple[int, int], float]] = {}
    for p in files:
        m = _MERCL_FILE_RE.match(p.name)
        if m is None:
            continue
        code = int(m.group(1))  # mercl001 -> submarket 1 (sequential codes)
        parsed[code] = _parse_mercl_file(p)
    parsed = {code: months for code, months in parsed.items() if months}
    if not parsed:
        return empty

    # MEDIAS stage numbering: stage = (year - base_year) * 12 + month, so the
    # study's first calendar month (e.g. September -> 9) keeps its MEDIAS index.
    base_year = min(year for months in parsed.values() for (year, _m) in months)
    rows: list[tuple[int, int, str, float]] = []
    for code, months in parsed.items():
        for (year, month), value in months.items():
            # Drop the pre-study calendar months NEWAVE writes as 0.
            if value == 0.0:
                continue
            stage = (year - base_year) * 12 + month
            rows.append((code, stage, "NET_LOAD", value))
    if not rows:
        return empty
    return pl.DataFrame(
        rows,
        schema=["newave_code", "stage", "variable", "value"],
        orient="row",
    ).cast({"newave_code": pl.Int64, "stage": pl.Int64, "value": pl.Float64})


def read_newave_net_load(newave_dir: Path) -> pl.DataFrame:
    """Read deterministic net load from ``sistema.dat`` and ``c_adic.dat``.

    Computes ``net_load = mercado_energia + c_adic - sum(geracao_usinas_nao_simuladas)``
    per submarket and date.  The C_ADIC contribution (must-take energy from
    Itaipu, ANDE, MMGD, etc.) is added so the newave load is comparable to
    cobre's simulation output, which already includes C_ADIC.

    Returns a DataFrame compatible with the ``nw_market`` schema used in
    the energy balance charts:

    - ``newave_code`` (Int64): NEWAVE submarket code
    - ``stage`` (Int64): stage number (aligned with MEDIAS column naming)
    - ``variable`` (Utf8): always ``"NET_LOAD"``
    - ``value`` (Float64): net load in MW·med

    Prefers the nwlistop ``mercl*.out`` files (full horizon) when present,
    falling back to this ``sistema.dat`` reconstruction (study period only,
    since post-study dates are written under a filtered-out sentinel year).
    """
    nwlistop = read_newave_net_load_nwlistop(newave_dir)
    if not nwlistop.is_empty():
        return nwlistop

    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )

    sistema_path = _find_case_insensitive(newave_dir, "sistema.dat")
    if sistema_path is None:
        _LOG.warning("sistema.dat not found in %s", newave_dir)
        return empty

    try:
        from inewave.newave import Sistema

        sistema = Sistema.read(str(sistema_path))
        load_df = sistema.mercado_energia
        ncs_df = sistema.geracao_usinas_nao_simuladas
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read sistema.dat for net load")
        return empty

    if load_df is None or load_df.empty:
        return empty

    # Filter to valid study dates (drop NaN load values and sentinel years).
    load_df = load_df.dropna(subset=["valor"])
    load_df = load_df[load_df["data"].dt.year < 9000]

    # Add C_ADIC must-take energy (Itaipu, ANDE, MMGD, etc.) to load.
    cadical_path = _find_case_insensitive(newave_dir, "c_adic.dat")
    if cadical_path is not None:
        try:
            from cobre_bridge.converters.stochastic import parse_cadical

            cadical = parse_cadical(cadical_path)
            load_df = load_df.copy()
            load_df["valor"] = load_df.apply(
                lambda row: (
                    row["valor"]
                    + cadical.get(
                        (
                            int(row["codigo_submercado"]),
                            int(row["data"].year),
                            int(row["data"].month),
                        ),
                        0.0,
                    )
                ),
                axis=1,
            )
        except Exception:  # noqa: BLE001
            _LOG.warning("Failed to parse c_adic.dat for net load adjustment")

    # NCS: sum across all source types per (submarket, date).
    ncs_total = None
    if ncs_df is not None and not ncs_df.empty:
        ncs_df = ncs_df.dropna(subset=["valor"])
        ncs_df = ncs_df[ncs_df["data"].dt.year < 9000]
        ncs_total = (
            ncs_df.groupby(["codigo_submercado", "data"])["valor"]
            .sum()
            .reset_index()
            .rename(columns={"valor": "ncs"})
        )

    # Compute net load = load - NCS.
    if ncs_total is not None:
        merged = load_df.merge(
            ncs_total,
            on=["codigo_submercado", "data"],
            how="left",
        )
        merged["ncs"] = merged["ncs"].fillna(0.0)
        merged["net_load"] = merged["valor"] - merged["ncs"]
    else:
        merged = load_df.copy()
        merged["net_load"] = merged["valor"]

    # Assign stage numbers: sequential from the first study month's number
    # (matching MEDIAS-MERC column naming convention).
    merged = merged.sort_values(["codigo_submercado", "data"])
    first_month = int(merged["data"].min().month)
    stage_map = {
        date: first_month + i for i, date in enumerate(sorted(merged["data"].unique()))
    }
    merged["stage"] = merged["data"].map(stage_map)

    result = pl.from_pandas(merged[["codigo_submercado", "stage", "net_load"]]).rename(
        {"codigo_submercado": "newave_code", "net_load": "value"}
    )
    result = result.with_columns(pl.lit("NET_LOAD").alias("variable"))
    result = result.cast(
        {"newave_code": pl.Int64, "stage": pl.Int64, "value": pl.Float64}
    )

    return result


def read_pmo_cost_breakdown(newave_dir: Path) -> dict[str, float]:
    """Read cost breakdown from pmo.dat ``custo_operacao_series_simuladas``.

    Returns ``{category: expected_value_in_R$}`` where values are
    converted from 10^6 R$ to R$.  Zero-cost categories are excluded.
    """
    pmo_path = _find_pmo(newave_dir)
    if pmo_path is None:
        return {}

    try:
        from inewave.newave import Pmo

        pmo = Pmo.read(str(pmo_path))
        df = pmo.custo_operacao_series_simuladas
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read cost breakdown from pmo.dat")
        return {}

    if df is None or df.empty:
        return {}

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row["parcela"]).strip()
        value = float(row["valor_esperado"]) * 1e6  # Convert 10^6 R$ to R$
        if abs(value) > 0.01:
            result[name] = value

    return result


# Per-iteration block in newave.tim:
#   Iteracao   N - Backward: HHHhMMminSSs
#                  Forward:  HHHhMMminSSs
#                  Total:    HHHhMMminSSs
# Whitespace between H/min/s is irregular; the regex tolerates 0+ spaces.
_TIM_ITER_RE = re.compile(
    r"^\s*Iteracao\s+(\d+)\s*-\s*Backward:\s*"
    r"(\d+)h\s*(\d+)min\s*(\d+)s\s*\n"
    r"\s*Forward:\s*(\d+)h\s*(\d+)min\s*(\d+)s\s*\n"
    r"\s*Total:\s*(\d+)h\s*(\d+)min\s*(\d+)s",
    re.MULTILINE,
)


def read_newave_tim_iterations(newave_dir: Path) -> pl.DataFrame:
    """Parse per-iteration timings from ``newave.tim``.

    Returns DataFrame with columns ``iteration`` (Int64),
    ``backward_seconds``, ``forward_seconds``, ``total_seconds``
    (Float64). Empty when the file is missing.

    Note: NEWAVE's first iteration row carries clock-initialisation
    garbage on the Backward / Forward fields (sometimes hundreds of
    hours). Values are returned as-is so the caller can decide how to
    surface them (e.g. annotate the chart but keep the data).
    """
    empty = pl.DataFrame(
        schema={
            "iteration": pl.Int64,
            "backward_seconds": pl.Float64,
            "forward_seconds": pl.Float64,
            "total_seconds": pl.Float64,
        }
    )
    tim_path = _find_case_insensitive(newave_dir, "newave.tim")
    if tim_path is None:
        _LOG.warning("newave.tim not found in %s", newave_dir)
        return empty
    try:
        text = tim_path.read_text(encoding="latin-1")
    except OSError:
        _LOG.exception("Failed to read newave.tim at %s", tim_path)
        return empty

    rows: list[dict[str, float | int]] = []
    for m in _TIM_ITER_RE.finditer(text):
        it = int(m.group(1))
        bw = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
        fw = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7))
        tt = int(m.group(8)) * 3600 + int(m.group(9)) * 60 + int(m.group(10))
        rows.append(
            {
                "iteration": it,
                "backward_seconds": float(bw),
                "forward_seconds": float(fw),
                "total_seconds": float(tt),
            }
        )
    if not rows:
        return empty
    return (
        pl.from_dicts(rows)
        .sort("iteration")
        .cast(
            {
                "iteration": pl.Int64,
                "backward_seconds": pl.Float64,
                "forward_seconds": pl.Float64,
                "total_seconds": pl.Float64,
            }
        )
    )


def read_newave_tim_stages(newave_dir: Path) -> dict[str, float]:
    """Read stage timings from ``newave.tim`` via ``inewave.Newavetim``.

    Returns ``{etapa_name: seconds}`` keyed on the Portuguese stage
    labels NEWAVE emits (e.g. ``"Tempo Total"``, ``"Calculo da
    Politica"``, ``"Simulacao Final"``). Empty dict on parse failure.
    """
    tim_path = _find_case_insensitive(newave_dir, "newave.tim")
    if tim_path is None:
        return {}
    try:
        from inewave.newave import Newavetim

        tim = Newavetim.read(str(tim_path))
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to parse newave.tim via inewave at %s", tim_path)
        return {}
    df = tim.tempos_etapas
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row["etapa"]).strip()
        tdelta = row["tempo"]
        out[name] = float(tdelta.total_seconds())
    return out

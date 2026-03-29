"""NEWAVE output file readers for results comparison.

Reads MEDIAS CSV files (hydro, thermal, system) from the ``saidas/``
directory and pmo.dat convergence/productivity data.

MEDIAS files are parsed directly with Polars since inewave v1.13 does
not provide dedicated reader classes for them.  pmo.dat is read via
``inewave.newave.Pmo``.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

    try:
        df = pl.read_csv(
            path,
            separator=";",
            truncate_ragged_lines=True,
            infer_schema_length=100,
        )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to parse %s", path)
        return empty

    # Normalize column names: strip whitespace.
    df = df.rename({c: c.strip() for c in df.columns})

    # Identify entity column, variable column, patamar column, and
    # stage value columns (EST001, EST002, ...).
    # Column names vary; common patterns: USINA, CODIGO_USINA, SUBMERCADO.
    entity_col: str | None = None
    patamar_col: str | None = None
    variable_col: str | None = None
    stage_cols: list[str] = []

    for col in df.columns:
        upper = col.upper()
        if upper in ("USINA", "CODIGO_USINA", "SUBMERCADO", "CODIGO_SUBMERCADO"):
            entity_col = col
        elif upper in ("PATAMAR", "PAT"):
            patamar_col = col
        elif upper in ("SERIE", "CENARIO", "SÉRIE/CENÁRIO", "SERIE/CENARIO"):
            pass  # skip scenario column
        elif upper.startswith("VARIAVEL") or upper == "VARIAVEL":
            variable_col = col
        elif upper.startswith("EST") or upper.startswith("ESTAGIO"):
            stage_cols.append(col)

    if entity_col is None:
        # Fallback: use second column as entity
        if len(df.columns) >= 2:
            entity_col = df.columns[1]
        else:
            _LOG.warning("Cannot identify entity column in %s", filename)
            return empty

    if not stage_cols:
        # Fallback: all numeric columns that aren't entity/patamar
        exclude = {entity_col, patamar_col, variable_col}
        for col in df.columns:
            if col not in exclude and df[col].dtype in (
                pl.Float64,
                pl.Int64,
                pl.Float32,
            ):
                stage_cols.append(col)

    if not stage_cols:
        _LOG.warning("No stage columns found in %s", filename)
        return empty

    # Filter patamar=0 (stage-level average).
    if patamar_col is not None:
        df = df.filter(pl.col(patamar_col).cast(pl.Int64, strict=False) == 0)

    # Filter to mean scenario (cenario="mean" or first row per entity).
    # MEDIAS files contain scenario mean in the first row group.
    # They have SERIE=1 for the mean row.
    serie_col: str | None = None
    for col in df.columns:
        upper = col.upper()
        if upper in ("SERIE", "CENARIO", "SÉRIE/CENÁRIO", "SERIE/CENARIO"):
            serie_col = col
            break

    if serie_col is not None:
        # Keep only mean row (usually SERIE=1 or the first occurrence).
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

    # Extract stage number from column name (e.g., "EST001" -> 1).
    long_df = long_df.with_columns(
        pl.col("stage_col")
        .str.replace_all(r"[^0-9]", "")
        .cast(pl.Int64, strict=False)
        .alias("stage")
    )

    # Build output.
    result = long_df.select(
        pl.col(entity_col).cast(pl.Int64, strict=False).alias("newave_code"),
        pl.col("stage"),
        (
            pl.col(variable_col).str.strip_chars().alias("variable")
            if variable_col is not None
            else pl.lit("value").alias("variable")
        ),
        pl.col("value").cast(pl.Float64, strict=False),
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


def read_pmo_convergence(newave_dir: Path) -> pl.DataFrame:
    """Read pmo.dat convergence table.

    Returns DataFrame with columns: ``iteration`` (Int64),
    ``lower_bound`` (Float64, ZINF), ``upper_bound_mean`` (Float64, ZSUP).

    Returns empty DataFrame if pmo.dat not found or convergence data
    unavailable.
    """
    empty = pl.DataFrame(
        schema={
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        }
    )

    pmo_path = _find_case_insensitive(newave_dir, "pmo.dat")
    if pmo_path is None:
        _LOG.warning("pmo.dat not found in %s", newave_dir)
        return empty

    try:
        from inewave.newave import Pmo

        pmo = Pmo.read(str(pmo_path))
        conv_df = pmo.convergencia
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read convergence from pmo.dat")
        return empty

    if conv_df is None or conv_df.empty:
        return empty

    # Identify columns: convergencia returns a pandas DataFrame.
    # Common columns: iteracao, zinf, zsup_medio (names may vary by version).
    col_map: dict[str, str] = {}
    for col in conv_df.columns:
        lower = str(col).lower().strip()
        if "iteracao" in lower or "iteration" in lower:
            col_map["iteration"] = col
        elif "zinf" in lower or "lower" in lower:
            col_map["lower_bound"] = col
        elif "zsup" in lower or "upper" in lower:
            col_map["upper_bound_mean"] = col

    if "iteration" not in col_map:
        # Use index as iteration number.
        conv_df = conv_df.reset_index()
        conv_df.columns = [str(c) for c in conv_df.columns]
        col_map["iteration"] = conv_df.columns[0]

    if "lower_bound" not in col_map or "upper_bound_mean" not in col_map:
        _LOG.warning(
            "Cannot identify ZINF/ZSUP columns in pmo.dat convergence: %s",
            list(conv_df.columns),
        )
        return empty

    result = pl.from_pandas(
        conv_df[
            [col_map["iteration"], col_map["lower_bound"], col_map["upper_bound_mean"]]
        ]
    )
    result = result.rename(
        {
            col_map["iteration"]: "iteration",
            col_map["lower_bound"]: "lower_bound",
            col_map["upper_bound_mean"]: "upper_bound_mean",
        }
    )
    result = result.cast(
        {
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        }
    )

    return result


def read_pmo_productivity(newave_dir: Path) -> pl.DataFrame:
    """Read pmo.dat productivity data.

    Returns DataFrame with columns: ``newave_code`` (Int64),
    ``productivity`` (Float64).

    Returns empty DataFrame if pmo.dat not found or productivity data
    unavailable.
    """
    empty = pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "productivity": pl.Float64,
        }
    )

    pmo_path = _find_case_insensitive(newave_dir, "pmo.dat")
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

    # Identify columns.
    col_map: dict[str, str] = {}
    for col in prod_df.columns:
        lower = str(col).lower().strip()
        if "codigo" in lower or "usina" in lower or "code" in lower:
            col_map["newave_code"] = col
        elif "prodt" in lower or "produtibilidade" in lower or "productivity" in lower:
            col_map["productivity"] = col

    if "newave_code" not in col_map or "productivity" not in col_map:
        _LOG.warning(
            "Cannot identify code/productivity columns in pmo.dat: %s",
            list(prod_df.columns),
        )
        return empty

    result = pl.from_pandas(prod_df[[col_map["newave_code"], col_map["productivity"]]])
    result = result.rename(
        {
            col_map["newave_code"]: "newave_code",
            col_map["productivity"]: "productivity",
        }
    )
    result = result.cast({"newave_code": pl.Int64, "productivity": pl.Float64})

    return result

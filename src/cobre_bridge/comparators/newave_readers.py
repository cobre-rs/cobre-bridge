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

    pmo_path = _find_pmo(newave_dir)
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

    # Identify columns by exact name first, then by substring.
    # inewave returns columns like: iteracao, limite_inferior_zinf, zinf,
    # limite_superior_zinf, zsup, delta_zinf, zsup_iteracao, tempo.
    # We want exactly "iteracao", "zinf", "zsup".
    cols = list(conv_df.columns)
    col_map: dict[str, str] = {}

    # Exact matches first (most reliable).
    exact = {c.lower().strip(): c for c in cols}
    if "iteracao" in exact:
        col_map["iteration"] = exact["iteracao"]
    if "zinf" in exact:
        col_map["lower_bound"] = exact["zinf"]
    if "zsup" in exact:
        col_map["upper_bound_mean"] = exact["zsup"]

    # Fallback: substring matching (for other inewave versions).
    if "iteration" not in col_map:
        for col in cols:
            lower = str(col).lower().strip()
            if lower == "iteration" or lower == "iteracao":
                col_map["iteration"] = col
                break

    if "lower_bound" not in col_map:
        for col in cols:
            if str(col).lower().strip() in ("zinf", "lower_bound"):
                col_map["lower_bound"] = col
                break

    if "upper_bound_mean" not in col_map:
        for col in cols:
            if str(col).lower().strip() in ("zsup", "zsup_medio", "upper_bound_mean"):
                col_map["upper_bound_mean"] = col
                break

    if "iteration" not in col_map:
        conv_df = conv_df.reset_index()
        conv_df.columns = [str(c) for c in conv_df.columns]
        col_map["iteration"] = conv_df.columns[0]

    if "lower_bound" not in col_map or "upper_bound_mean" not in col_map:
        _LOG.warning(
            "Cannot identify ZINF/ZSUP columns in pmo.dat convergence: %s",
            list(conv_df.columns),
        )
        return empty

    # Keep only the last row per iteration (inner iterations → take final).
    subset = conv_df[
        [col_map["iteration"], col_map["lower_bound"], col_map["upper_bound_mean"]]
    ].copy()
    subset.columns = ["iteration", "lower_bound", "upper_bound_mean"]
    subset = subset.groupby("iteration", as_index=False).last()

    result = pl.from_pandas(subset)
    result = result.cast(
        {
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        }
    )

    # NEWAVE pmo.dat exports convergence values in 10^6 R$.
    # Multiply by 1e6 to convert to R$ (matching Cobre convention).
    result = result.with_columns(
        pl.col("lower_bound") * 1e6,
        pl.col("upper_bound_mean") * 1e6,
    )

    return result


def read_pmo_productivity(newave_dir: Path) -> pl.DataFrame:
    """Read pmo.dat productivity data.

    Returns DataFrame with columns: ``plant_name`` (Utf8),
    ``productivity`` (Float64).

    The plant name is the NEWAVE plant name from ``nome_usina``.
    The preferred productivity column is
    ``produtibilidade_equivalente_volmin_volmax`` (equivalent
    productivity between min and max volume).

    Returns empty DataFrame if pmo.dat not found or productivity data
    unavailable.
    """
    empty = pl.DataFrame(
        schema={
            "plant_name": pl.Utf8,
            "productivity": pl.Float64,
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

    # Identify name column and best productivity column.
    name_col: str | None = None
    prod_col: str | None = None
    for col in prod_df.columns:
        lower = str(col).lower().strip()
        if "nome" in lower or lower == "usina":
            name_col = col
        # Prefer the volmin_volmax equivalent productivity.
        elif lower == "produtibilidade_equivalente_volmin_volmax":
            prod_col = col

    # Fallback: first column with "produtibilidade" in the name.
    if prod_col is None:
        for col in prod_df.columns:
            if "produtibilidade" in str(col).lower():
                prod_col = col
                break

    if name_col is None or prod_col is None:
        _LOG.warning(
            "Cannot identify name/productivity columns in pmo.dat: %s",
            list(prod_df.columns),
        )
        return empty

    result = pl.from_pandas(prod_df[[name_col, prod_col]])
    result = result.rename({name_col: "plant_name", prod_col: "productivity"})
    result = result.with_columns(
        pl.col("plant_name").str.strip_chars(),
        pl.col("productivity").cast(pl.Float64, strict=False),
    ).drop_nulls(subset=["plant_name", "productivity"])

    return result

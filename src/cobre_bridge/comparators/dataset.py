"""Canonical tidy/long comparison data model for the ANALYZE layer.

This module defines the single, serializable source of truth shared across the
console, HTML report, and export paths: :class:`ComparisonDataset`. It holds a
tidy/long value frame (one observation per row), a per-variable summary frame,
and a typed metadata side-table for non-tidy artifacts carried verbatim during
the strangler migration.

This is a leaf module: it must not import from any other comparator (notably
``results.py`` or ``bounds.py``) so that later epics can wire it in without
creating import cycles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs not installed
import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


class SchemaError(ValueError):
    """Raised when a :class:`ComparisonDataset` violates its column contract.

    Subclasses :class:`ValueError` so callers can catch it as a value problem.
    The message always names the offending column(s) or source value(s).
    """


#: Allowed values for the tidy ``source`` column.
VALID_SOURCES: frozenset[str] = frozenset({"newave", "cobre", "p10", "p50", "p90"})

#: Ordered schema of the canonical tidy/long value frame. Every later epic
#: consumes this exact column contract.
TIDY_SCHEMA: dict[str, type[pl.DataType]] = {
    "entity_type": pl.Utf8,
    "entity_id": pl.Int64,
    "entity_name": pl.Utf8,
    "bus": pl.Int64,
    "stage": pl.Int64,
    "block": pl.Int64,
    "variable": pl.Utf8,
    "source": pl.Utf8,
    "value": pl.Float64,
}

#: Wrapper key marking a metadata value that was a polars/pandas DataFrame.
#: A wrapped entry has the shape ``{_FRAME_SENTINEL: "<polars|pandas>",
#: "records": [...], "columns": {name: dtype_name}}``. The ``columns`` map
#: preserves the schema so empty frames keep their columns/dtypes on round-trip.
_FRAME_SENTINEL: str = "__frame__"

#: File names written/read by :meth:`ComparisonDataset.to_dir` /
#: :meth:`ComparisonDataset.from_dir`, in their canonical order.
_TIDY_FILE: str = "comparison.parquet"
_SUMMARY_FILE: str = "summary.parquet"
_METADATA_FILE: str = "metadata.json"

#: Ordered schema of the per-variable summary frame.
SUMMARY_SCHEMA: dict[str, type[pl.DataType]] = {
    "variable": pl.Utf8,
    "count": pl.Int64,
    "mean_abs_diff": pl.Float64,
    "max_abs_diff": pl.Float64,
    "mean_smape": pl.Float64,
    "max_smape": pl.Float64,
    "within_tol_rate": pl.Float64,
    "correlation": pl.Float64,
}


@dataclass
class ComparisonDataset:
    """Central tidy/long comparison dataset for the ANALYZE layer.

    Attributes:
        tidy: The long value frame conforming to :data:`TIDY_SCHEMA`. One row per
            (entity, stage, block, variable, source) observation.
        summary: Per-variable summary statistics conforming to
            :data:`SUMMARY_SCHEMA`.
        metadata: Side-table holding non-tidy artifacts carried verbatim during
            the strangler migration (e.g. entity-name dicts, ``line_meta``
            lists, cost dicts).
    """

    tidy: pl.DataFrame
    summary: pl.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> ComparisonDataset:
        """Return a valid empty dataset matching the schemas.

        Returns:
            A dataset whose ``tidy`` and ``summary`` frames are empty but typed
            per :data:`TIDY_SCHEMA` and :data:`SUMMARY_SCHEMA`, with an empty
            ``metadata`` dict.
        """
        return cls(
            tidy=pl.DataFrame(schema=TIDY_SCHEMA),
            summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
            metadata={},
        )

    def validate(self) -> None:
        """Validate the dataset against its column contract.

        Raises:
            SchemaError: If the ``tidy`` frame columns do not equal the ordered
                :data:`TIDY_SCHEMA` keys, if any ``tidy`` or ``summary`` column
                has a dtype differing from its schema, if any ``source`` value
                (when the frame is non-empty) is outside :data:`VALID_SOURCES`,
                or if the ``summary`` frame columns do not equal the ordered
                :data:`SUMMARY_SCHEMA` keys.
        """
        expected_tidy = list(TIDY_SCHEMA)
        actual_tidy = list(self.tidy.columns)
        if actual_tidy != expected_tidy:
            msg = f"tidy frame columns {actual_tidy} != expected {expected_tidy}"
            raise SchemaError(msg)
        _check_dtypes(self.tidy, TIDY_SCHEMA, "tidy")

        if self.tidy.height > 0:
            seen = set(self.tidy["source"].unique().to_list())
            invalid = seen - VALID_SOURCES
            if invalid:
                msg = (
                    f"tidy frame has invalid source values {sorted(invalid)}; "
                    f"allowed sources are {sorted(VALID_SOURCES)}"
                )
                raise SchemaError(msg)

        expected_summary = list(SUMMARY_SCHEMA)
        actual_summary = list(self.summary.columns)
        if actual_summary != expected_summary:
            msg = (
                f"summary frame columns {actual_summary} != expected {expected_summary}"
            )
            raise SchemaError(msg)
        _check_dtypes(self.summary, SUMMARY_SCHEMA, "summary")

    def to_dir(self, out_dir: Path) -> list[Path]:
        """Serialize this dataset to ``out_dir`` as Parquet + JSON.

        Validates first (fail fast), creates ``out_dir`` if needed, then writes
        the tidy frame to ``comparison.parquet``, the summary frame to
        ``summary.parquet``, and the JSON-native view of :attr:`metadata` to
        ``metadata.json``.

        Metadata serialization (see :func:`_metadata_to_json`): JSON-native
        values pass through; ``pl.DataFrame`` / ``pd.DataFrame`` values are
        wrapped as ``{"__frame__": "<polars|pandas>", "records": [...]}``; any
        other value raises :class:`TypeError` naming its key.

        Note:
            JSON object keys are always strings, so a metadata dict with
            non-string keys (e.g. ``{0: "ITAIPU"}``) round-trips with string
            keys (``{"0": "ITAIPU"}``). This coercion is intentional and not
            reversed on load.

        Args:
            out_dir: Destination directory; created with ``parents=True``.

        Returns:
            The three written paths in the order
            ``[comparison.parquet, summary.parquet, metadata.json]``.

        Raises:
            SchemaError: If :meth:`validate` fails.
            TypeError: If a metadata value is neither JSON-native nor a frame.
        """
        self.validate()
        out_dir.mkdir(parents=True, exist_ok=True)

        tidy_path = out_dir / _TIDY_FILE
        summary_path = out_dir / _SUMMARY_FILE
        metadata_path = out_dir / _METADATA_FILE

        self.tidy.write_parquet(tidy_path)
        self.summary.write_parquet(summary_path)
        metadata_view = _metadata_to_json(self.metadata)
        metadata_path.write_text(json.dumps(metadata_view, indent=2), encoding="utf-8")

        return [tidy_path, summary_path, metadata_path]

    @classmethod
    def from_dir(cls, in_dir: Path) -> ComparisonDataset:
        """Reconstruct a :class:`ComparisonDataset` written by :meth:`to_dir`.

        Reads ``comparison.parquet``, ``summary.parquet``, and ``metadata.json``
        from ``in_dir``, reconstructs frame-wrapped metadata values back to their
        original frame type, validates the result, and returns it.

        Note:
            JSON object keys are strings, so metadata dicts with originally
            non-string keys come back with string keys (see :meth:`to_dir`).

        Args:
            in_dir: Directory containing the three serialized files.

        Returns:
            The validated reconstructed dataset.

        Raises:
            FileNotFoundError: If any of the three files is absent; the message
                names the first missing path.
            SchemaError: If the reconstructed dataset fails :meth:`validate`.
            TypeError: If a wrapped frame entry declares an unknown frame type.
        """
        tidy_path = in_dir / _TIDY_FILE
        summary_path = in_dir / _SUMMARY_FILE
        metadata_path = in_dir / _METADATA_FILE

        for path in (tidy_path, summary_path, metadata_path):
            if not path.exists():
                msg = f"missing dataset file: {path}"
                raise FileNotFoundError(msg)

        tidy = pl.read_parquet(tidy_path)
        summary = pl.read_parquet(summary_path)
        metadata_view = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = _metadata_from_json(metadata_view)

        dataset = cls(tidy=tidy, summary=summary, metadata=metadata)
        dataset.validate()
        return dataset


def _metadata_to_json(meta: dict[str, object]) -> dict[str, object]:
    """Build the JSON-serializable view of a metadata dict.

    JSON-native values (``str, int, float, bool, None`` and ``list``/``dict``
    thereof) are passed through verbatim. ``pl.DataFrame`` and ``pd.DataFrame``
    values are wrapped as ``{_FRAME_SENTINEL: "<polars|pandas>", "records":
    [...], "columns": {name: dtype_name}}`` using the frame's native record
    export. The ``columns`` map carries the schema so that an *empty* frame
    (whose ``records`` list is ``[]``) keeps its column names and dtypes on
    round-trip instead of collapsing to a zero-column frame.

    Args:
        meta: The metadata side-table.

    Returns:
        A dict whose values are all JSON-serializable.

    Raises:
        TypeError: If any value is neither JSON-native nor a supported frame;
            the message names the offending key.
    """
    view: dict[str, object] = {}
    for key, value in meta.items():
        if isinstance(value, pl.DataFrame):
            columns = {name: str(dtype) for name, dtype in value.schema.items()}
            view[key] = {
                _FRAME_SENTINEL: "polars",
                "records": value.to_dicts(),
                "columns": columns,
            }
        elif isinstance(value, pd.DataFrame):
            columns = {str(name): str(dtype) for name, dtype in value.dtypes.items()}
            view[key] = {
                _FRAME_SENTINEL: "pandas",
                "records": value.to_dict(orient="records"),
                "columns": columns,
            }
        elif _is_json_native(value):
            view[key] = value
        else:
            type_name = type(value).__name__
            msg = f"metadata key {key!r} has non-serializable value of type {type_name}"
            raise TypeError(msg)
    return view


def _metadata_from_json(view: dict[str, object]) -> dict[str, object]:
    """Reconstruct a metadata dict from its JSON view (inverse of to-json).

    Entries shaped like ``{_FRAME_SENTINEL: "<polars|pandas>", "records":
    [...], "columns": {name: dtype_name}}`` are rebuilt into the corresponding
    frame type; all other entries pass through unchanged. When ``records`` is
    empty and a ``columns`` map is present, the frame is reconstructed *with* an
    explicit schema so that an empty frame keeps its column names and dtypes
    (see :func:`_metadata_to_json`).

    Args:
        view: The JSON-deserialized metadata view.

    Returns:
        The reconstructed metadata side-table.

    Raises:
        TypeError: If a wrapped entry is missing its ``records`` key, or declares
            an unknown frame type; the message names the offending key.
    """
    meta: dict[str, object] = {}
    for key, value in view.items():
        if isinstance(value, dict) and _FRAME_SENTINEL in value:
            kind = value[_FRAME_SENTINEL]
            if kind not in ("polars", "pandas"):
                msg = f"metadata key {key!r} has unknown frame type {kind!r}"
                raise TypeError(msg)
            if "records" not in value:
                msg = f"metadata key {key!r} frame wrapper is missing 'records' key"
                raise TypeError(msg)
            records = value["records"]
            if not isinstance(records, list):
                type_name = type(records).__name__
                msg = (
                    f"metadata key {key!r} frame 'records' must be a list, "
                    f"got {type_name}"
                )
                raise TypeError(msg)
            columns = value.get("columns")
            if kind == "polars":
                meta[key] = _build_polars_frame(records, columns)
            else:
                meta[key] = _build_pandas_frame(records, columns)
        else:
            meta[key] = value
    return meta


def _build_polars_frame(records: list[object], columns: object) -> pl.DataFrame:
    """Rebuild a polars frame from records, preserving schema when empty.

    Args:
        records: The ``records`` payload (a list of row dicts).
        columns: The ``columns`` map (column-name -> polars dtype-name), or
            ``None`` for legacy wrappers without a schema.

    Returns:
        The reconstructed polars frame. When ``records`` is empty and
        ``columns`` is a mapping, an explicitly-schemed empty frame is built so
        column names and dtypes survive; otherwise the records drive inference.
    """
    if not records and isinstance(columns, dict):
        schema = {name: getattr(pl, dtype_name) for name, dtype_name in columns.items()}
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records)


def _build_pandas_frame(records: list[object], columns: object) -> pd.DataFrame:
    """Rebuild a pandas frame from records, preserving schema when empty.

    Args:
        records: The ``records`` payload (a list of row dicts).
        columns: The ``columns`` map (column-name -> pandas dtype string), or
            ``None`` for legacy wrappers without a schema.

    Returns:
        The reconstructed pandas frame. When ``records`` is empty and
        ``columns`` is a mapping, an empty frame with those columns and dtypes
        is built; otherwise the records drive inference.
    """
    if not records and isinstance(columns, dict):
        empty_cols = {
            name: pd.Series([], dtype=dtype_name)
            for name, dtype_name in columns.items()
        }
        return pd.DataFrame(empty_cols)
    return pd.DataFrame(records)


def _is_json_native(value: object) -> bool:
    """Return whether ``value`` is composed only of JSON-native types.

    JSON-native means ``None``/``bool``/``int``/``float``/``str`` scalars, or a
    ``list`` of such, or a ``dict`` whose values are such (recursively). Dict
    keys are not constrained here because JSON coerces them to strings on dump.

    Args:
        value: The value to inspect.

    Returns:
        ``True`` if ``json.dumps`` would accept ``value`` without a custom
        encoder, ``False`` otherwise.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(_is_json_native(item) for item in value.values())
    return False


def _check_dtypes(
    frame: pl.DataFrame,
    schema: dict[str, type[pl.DataType]],
    label: str,
) -> None:
    """Verify each column dtype of ``frame`` matches its schema entry.

    The schema maps column name to a polars dtype *class* (e.g. ``pl.Int64``);
    the class is instantiated (``pl.Int64()``) before comparison because the
    frame's runtime schema holds dtype *instances*. Assumes column names/order
    have already been validated against ``schema``.

    Args:
        frame: The frame whose dtypes to check.
        schema: The column-name -> dtype-class contract.
        label: A human label for the frame (``"tidy"`` / ``"summary"``) used in
            the error message.

    Raises:
        SchemaError: If any column's dtype differs from its expected dtype; the
            message names the column, its actual dtype, and the expected dtype.
    """
    actual_schema = frame.schema
    for name, dtype_cls in schema.items():
        expected = dtype_cls()
        actual = actual_schema[name]
        if actual != expected:
            msg = (
                f"{label} frame column {name!r} has dtype {actual} "
                f"!= expected {expected}"
            )
            raise SchemaError(msg)

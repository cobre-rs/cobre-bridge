"""Tests for the DECOMP energy-contract reader and emitter (``CI``/``CE``)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import jsonschema
import pandas as pd
import pyarrow.parquet as pq
import pytest

from cobre_bridge.decomp.contracts import (
    _CONTRACT_BOUNDS_SCHEMA,
    _SCHEMA_URL,
    Contract,
    ContractStage,
    _signed_price,
    convert_contract_bounds,
    convert_energy_contracts,
    read_contracts,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import build_operative_calendar

_COBRE_SCHEMA = (
    Path.home() / "git" / "cobre" / "schemas" / "energy_contracts.schema.json"
)


class _StubDadger:
    """Minimal Dadger stand-in carrying ci/ce DataFrames."""

    def __init__(
        self,
        ci: pd.DataFrame | None = None,
        ce: pd.DataFrame | None = None,
    ) -> None:
        self._ci, self._ce = ci, ce

    def ci(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ci

    def ce(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ce


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _uniform_calendar():
    """A calendar whose per-stage blocks all carry equal hours."""
    hours = [[56.0, 56.0, 56.0]] * 2 + [[248.0, 248.0, 248.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _bus_id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))


def _contract_row(
    numero: int,
    estagio: int,
    *,
    nome: str = "CONTRATO",
    bus_code: int = 1,
    min_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_mw: tuple[float, float, float] = (0.0, 0.0, 0.0),
    custo: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fator_perdas: float | None = None,
) -> dict:
    row: dict = {
        "numero_contrato": numero,
        "codigo_submercado": bus_code,
        "nome_contrato": nome,
        "estagio": estagio,
        "fator_perdas": fator_perdas,
    }
    for k in range(3):
        row[f"limite_inferior_{k + 1}"] = min_mw[k]
        row[f"limite_superior_{k + 1}"] = max_mw[k]
        row[f"custo_{k + 1}"] = custo[k]
    return row


def test_read_contracts_forward_fills_and_assigns_dense_ids() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="IMPORTACAO",
                min_mw=(1.0, 1.0, 1.0),
                max_mw=(10.0, 10.0, 10.0),
                custo=(5.0, 5.0, 5.0),
            ),
            _contract_row(
                1,
                3,
                nome="IMPORTACAO",
                min_mw=(2.0, 2.0, 2.0),
                max_mw=(20.0, 20.0, 20.0),
                custo=(6.0, 6.0, 6.0),
            ),
        ]
    )
    ce = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="EXPORTACAO",
                min_mw=(0.5, 0.5, 0.5),
                max_mw=(8.0, 8.0, 8.0),
                custo=(4.0, 4.0, 4.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=ce)

    contracts = read_contracts(dadger, _calendar())

    assert len(contracts) == 2
    imported = next(c for c in contracts if c.kind == "import")
    exported = next(c for c in contracts if c.kind == "export")
    assert imported.id == 0
    assert exported.id == 1
    assert imported.numero == 1
    assert exported.numero == 1

    # Stage-2 (index 1) is absent in the declared data; it is forward-filled
    # from stage 1 (index 0). Stage 3 (index 2) carries its own declared
    # values, distinct from stage 1's.
    assert imported.stages[1] == imported.stages[0]
    assert imported.stages[2] != imported.stages[0]
    assert imported.stages[2].max_mw == [20.0, 20.0, 20.0]
    assert imported.stages[2].min_mw == [2.0, 2.0, 2.0]
    assert imported.stages[2].custo == [6.0, 6.0, 6.0]


def test_read_contracts_skips_blank_all_zero_placeholder() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                1,
                1,
                nome="",
                min_mw=(0.0, 0.0, 0.0),
                max_mw=(0.0, 0.0, 0.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=None)

    assert read_contracts(dadger, _calendar()) == []


def test_read_contracts_synthesizes_name_when_blank_but_nonzero() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                2,
                1,
                nome="",
                min_mw=(0.0, 0.0, 0.0),
                max_mw=(5.0, 0.0, 0.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=None)

    contracts = read_contracts(dadger, _calendar())

    assert len(contracts) == 1
    assert contracts[0].name == "CI 2"


def test_read_contracts_missing_stage_one_raises() -> None:
    ci = pd.DataFrame(
        [
            _contract_row(
                5,
                2,
                nome="SOMENTE ESTAGIO 2",
                max_mw=(10.0, 10.0, 10.0),
            )
        ]
    )
    dadger = _StubDadger(ci=ci, ce=None)

    with pytest.raises(ValueError, match=r"CI 5"):
        read_contracts(dadger, _calendar())


def test_signed_price_negates_export_only() -> None:
    assert _signed_price(100.0, "import") == 100.0
    assert _signed_price(100.0, "export") == -100.0


def test_convert_energy_contracts_import_base_shape() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[50.0, 50.0, 50.0],
                custo=[200.0, 200.0, 200.0],
                loss_factor=None,
            )
        ],
    )

    result = convert_energy_contracts(
        [contract], id_map, calendar, calendar[0].start_date
    )

    assert result["contracts"][0] == {
        "id": 0,
        "name": "IMPORTACAO",
        "operational_start_date": calendar[0].start_date.isoformat(),
        "bus_id": 0,
        "type": "import",
        "price_per_mwh": 200.0,
        "limits": {"min_mw": 0.0, "max_mw": 50.0},
    }
    assert "entry_stage_id" not in result["contracts"][0]
    assert "exit_stage_id" not in result["contracts"][0]


def test_convert_energy_contracts_export_price_is_negative() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contract = Contract(
        id=1,
        kind="export",
        numero=1,
        name="EXPORTACAO",
        bus_code=1,
        stages=[
            ContractStage(
                min_mw=[0.0, 0.0, 0.0],
                max_mw=[30.0, 30.0, 30.0],
                custo=[150.0, 150.0, 150.0],
                loss_factor=None,
            )
        ],
    )

    result = convert_energy_contracts(
        [contract], id_map, calendar, calendar[0].start_date
    )

    assert result["contracts"][0]["price_per_mwh"] == -150.0


@pytest.mark.skipif(
    not _COBRE_SCHEMA.exists(),
    reason="cobre schema not present (sibling checkout ~/git/cobre required)",
)
def test_convert_energy_contracts_validates_against_schema() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()
    contracts = [
        Contract(
            id=0,
            kind="import",
            numero=1,
            name="IMPORTACAO",
            bus_code=1,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[50.0, 50.0, 50.0],
                    custo=[200.0, 200.0, 200.0],
                    loss_factor=None,
                )
            ],
        ),
        Contract(
            id=1,
            kind="export",
            numero=1,
            name="EXPORTACAO",
            bus_code=2,
            stages=[
                ContractStage(
                    min_mw=[0.0, 0.0, 0.0],
                    max_mw=[30.0, 30.0, 30.0],
                    custo=[150.0, 150.0, 150.0],
                    loss_factor=None,
                )
            ],
        ),
    ]

    result = convert_energy_contracts(
        contracts, id_map, calendar, calendar[0].start_date
    )

    schema = json.loads(_COBRE_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def test_convert_energy_contracts_empty_is_total() -> None:
    calendar = _uniform_calendar()
    id_map = _bus_id_map()

    result = convert_energy_contracts([], id_map, calendar, calendar[0].start_date)

    assert result == {"$schema": _SCHEMA_URL, "contracts": []}


def test_contract_bounds_uniform_emits_base_rows_only() -> None:
    calendar = _uniform_calendar()
    stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[stage, stage, stage],
    )

    table = convert_contract_bounds([contract], calendar)

    assert table.num_rows == 3
    assert table.column("block_id").null_count == 3
    for row in table.to_pylist():
        assert row["min_mw"] == 0.0
        assert row["max_mw"] == 50.0
        assert row["price_per_mwh"] == 200.0


def test_contract_bounds_nonuniform_max_emits_per_block_rows() -> None:
    calendar = _uniform_calendar()
    nonuniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[40.0, 50.0, 60.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    uniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[nonuniform_stage, uniform_stage, uniform_stage],
    )

    table = convert_contract_bounds([contract], calendar)
    stage0 = [row for row in table.to_pylist() if row["stage_id"] == 0]

    assert len(stage0) == 4
    base = next(row for row in stage0 if row["block_id"] is None)
    assert base["max_mw"] == 50.0  # hours-weighted, uniform-hours calendar
    overrides = sorted(
        (row for row in stage0 if row["block_id"] is not None),
        key=lambda r: r["block_id"],
    )
    assert [o["block_id"] for o in overrides] == [0, 1, 2]
    assert [o["max_mw"] for o in overrides] == [40.0, 50.0, 60.0]


def test_contract_bounds_export_price_negated_per_block() -> None:
    calendar = _uniform_calendar()
    nonuniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[30.0, 30.0, 30.0],
        custo=[100.0, 200.0, 300.0],
        loss_factor=None,
    )
    uniform_stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[30.0, 30.0, 30.0],
        custo=[150.0, 150.0, 150.0],
        loss_factor=None,
    )
    contract = Contract(
        id=1,
        kind="export",
        numero=1,
        name="EXPORTACAO",
        bus_code=1,
        stages=[nonuniform_stage, uniform_stage, uniform_stage],
    )

    table = convert_contract_bounds([contract], calendar)
    overrides = sorted(
        (
            row
            for row in table.to_pylist()
            if row["stage_id"] == 0 and row["block_id"] is not None
        ),
        key=lambda r: r["block_id"],
    )

    assert [o["price_per_mwh"] for o in overrides] == [-100.0, -200.0, -300.0]


def test_contract_bounds_schema_roundtrips_through_parquet(tmp_path: Path) -> None:
    calendar = _uniform_calendar()
    stage = ContractStage(
        min_mw=[0.0, 0.0, 0.0],
        max_mw=[50.0, 50.0, 50.0],
        custo=[200.0, 200.0, 200.0],
        loss_factor=None,
    )
    contract = Contract(
        id=0,
        kind="import",
        numero=1,
        name="IMPORTACAO",
        bus_code=1,
        stages=[stage, stage, stage],
    )
    table = convert_contract_bounds([contract], calendar)
    out = tmp_path / "contract_bounds.parquet"

    pq.write_table(table, out)
    read_back = pq.read_table(out)

    assert read_back.schema.names == [
        "contract_id",
        "stage_id",
        "min_mw",
        "max_mw",
        "price_per_mwh",
        "block_id",
    ]
    assert [str(field_type) for field_type in read_back.schema.types] == [
        "int32",
        "int32",
        "double",
        "double",
        "double",
        "int32",
    ]


def test_contract_bounds_empty_is_total() -> None:
    calendar = _uniform_calendar()

    table = convert_contract_bounds([], calendar)

    assert table.num_rows == 0
    assert table.schema == _CONTRACT_BOUNDS_SCHEMA

"""Thermal entity converter: maps NEWAVE thermal plant data to Cobre thermal JSON."""

from __future__ import annotations

import logging

from inewave.newave import Clast, Conft, Term

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/thermals.schema.json"
)


def convert_thermals(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE thermal plant data to a Cobre ``thermals.json`` dict.

    Reads ``conft.dat``, ``clast.dat``, and ``term.dat`` from *nw_files*.
    Returns a dict with a ``"thermals"`` key containing a list of thermal
    entries sorted by Cobre 0-based ID.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built ID mapping for bus cross-references.
    """
    conft = Conft.read(str(nw_files.conft))
    clast = Clast.read(str(nw_files.clast))
    term = Term.read(str(nw_files.term))

    conft_df = conft.usinas
    clast_df = clast.usinas
    term_df = term.usinas

    # Build cost lookup: codigo_usina -> cost for indice_ano_estudo == 1.
    cost_map: dict[int, float] = {}
    if clast_df is not None:
        first_year = clast_df[clast_df["indice_ano_estudo"] == 1]
        for _, row in first_year.iterrows():
            cost_map[int(row["codigo_usina"])] = float(row["valor"])

    # Build term lookup: codigo_usina -> (capacity, max_factor, min_gen_month1).
    term_map: dict[int, dict[str, float]] = {}
    if term_df is not None:
        # Use the first month's geracao_minima (mes == 1).
        month1 = term_df[term_df["mes"] == 1]
        for _, row in month1.iterrows():
            code = int(row["codigo_usina"])
            cap = float(row["potencia_instalada"])
            max_factor = float(row["fator_capacidade_maximo"])
            gen_min = float(row["geracao_minima"])
            term_map[code] = {
                "capacity": cap,
                "max_factor": max_factor,
                "gen_min": gen_min,
            }

        # For plants that appear in term but not in month1, use any row.
        for _, row in term_df.iterrows():
            code = int(row["codigo_usina"])
            if code not in term_map:
                cap = float(row["potencia_instalada"])
                max_factor = float(row["fator_capacidade_maximo"])
                term_map[code] = {
                    "capacity": cap,
                    "max_factor": max_factor,
                    "gen_min": 0.0,
                }

    thermals: list[dict] = []
    for _, row in conft_df.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        submercado = int(row["submercado"])

        bus_id = id_map.bus_id(submercado)

        term_info = term_map.get(
            newave_code, {"capacity": 0.0, "max_factor": 1.0, "gen_min": 0.0}
        )
        capacity = term_info["capacity"]
        max_factor = term_info["max_factor"]
        gen_min = term_info["gen_min"]

        max_mw = capacity * max_factor
        cost = cost_map.get(newave_code, 0.0)

        thermal_entry: dict = {
            "id": id_map.thermal_id(newave_code),
            "name": name,
            "bus_id": bus_id,
            "cost_segments": [
                {
                    "capacity_mw": max_mw,
                    "cost_per_mwh": cost,
                }
            ],
            "generation": {
                "min_mw": gen_min,
                "max_mw": max_mw,
            },
            "gnl_config": None,
            "entry_stage_id": None,
            "exit_stage_id": None,
        }
        thermals.append(thermal_entry)

    thermals.sort(key=lambda t: t["id"])

    return {
        "$schema": _SCHEMA_URL,
        "thermals": thermals,
    }

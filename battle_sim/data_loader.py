"""Load pokered-derived JSON once per process, expose O(1) lookups."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_types() -> dict[str, int]:
    return json.loads((DATA_DIR / "type_ids.json").read_text())


@lru_cache(maxsize=1)
def load_moves() -> dict[str, dict]:
    return json.loads((DATA_DIR / "moves.json").read_text())


@lru_cache(maxsize=1)
def load_moves_by_id() -> dict[int, dict]:
    """Lookup a move by its 1..165 id.  Injects the name under key 'name'."""
    out: dict[int, dict] = {}
    for name, data in load_moves().items():
        d = dict(data)
        d["name"] = name
        out[d["id"]] = d
    return out


@lru_cache(maxsize=1)
def load_pokemon() -> dict[str, dict]:
    return json.loads((DATA_DIR / "pokemon.json").read_text())


@lru_cache(maxsize=1)
def load_type_chart() -> dict[tuple[int, int], float]:
    """Return {(atk_id, def_id): multiplier} with missing pairs defaulting to 1.0.

    pokered stores only non-neutral matchups; the engine must default to 1.0
    for absent keys.
    """
    raw = json.loads((DATA_DIR / "type_chart.json").read_text())
    mult_map = {"SUPER": 2.0, "NVE": 0.5, "NO_EFFECT": 0.0}
    return {(int(atk), int(dfn)): mult_map[m] for atk, dfn, m in raw}


def type_effectiveness(atk_type: int, def_t1: int, def_t2: int) -> float:
    """Combined matchup multiplier.  Same-type defenders aren't double-counted."""
    chart = load_type_chart()
    m1 = chart.get((atk_type, def_t1), 1.0)
    if def_t1 == def_t2:
        return m1
    m2 = chart.get((atk_type, def_t2), 1.0)
    return m1 * m2

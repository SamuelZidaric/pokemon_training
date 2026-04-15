"""Data-layer smoke tests: parser output is coherent and complete."""
from __future__ import annotations

import pytest

from battle_sim.data_loader import (
    load_moves,
    load_pokemon,
    load_type_chart,
    load_types,
    type_effectiveness,
)


def test_type_constants_present():
    t = load_types()
    assert t["NORMAL"] == 0x00
    assert t["FIRE"] == 0x14
    assert t["PSYCHIC_TYPE"] == 0x18
    assert t["DRAGON"] == 0x1A


def test_moves_coverage():
    moves = load_moves()
    assert len(moves) == 165
    # Spot checks
    assert moves["TACKLE"]["power"] == 35
    assert moves["TACKLE"]["pp"] == 35
    assert moves["TACKLE"]["accuracy_pct"] == 95
    assert moves["THUNDERBOLT"]["power"] == 95
    # Accuracy encoding: 100% → 255 (POUND is one of the 100%-accurate moves)
    assert moves["POUND"]["accuracy_pct"] == 100
    assert moves["POUND"]["accuracy"] == 255
    # STRUGGLE present
    assert moves["STRUGGLE"]["id"] == 165


def test_pokemon_coverage():
    mons = load_pokemon()
    assert len(mons) == 151
    bb = mons["BULBASAUR"]
    assert bb["hp"] == 45 and bb["atk"] == 49
    assert bb["type1"] == "GRASS" and bb["type2"] == "POISON"


def test_ghost_vs_psychic_is_zero():
    """Gen 1 cartridge data: Ghost → Psychic = NO_EFFECT.  The v2 env's
    hand-written chart has this as 2×; the sim follows pokered."""
    chart = load_type_chart()
    ghost = load_types()["GHOST"]
    psy = load_types()["PSYCHIC_TYPE"]
    assert chart[(ghost, psy)] == 0.0
    # Double-check via the combined helper (Gastly is Ghost/Poison)
    assert type_effectiveness(ghost, psy, psy) == 0.0


def test_type_effectiveness_basic():
    t = load_types()
    # Water vs Fire = 2x
    assert type_effectiveness(t["WATER"], t["FIRE"], t["FIRE"]) == 2.0
    # Water vs Fire/Rock = 4x  (Magcargo-style, just using types for the test)
    assert type_effectiveness(t["WATER"], t["FIRE"], t["ROCK"]) == 4.0
    # Electric vs Ground = 0x
    assert type_effectiveness(t["ELECTRIC"], t["GROUND"], t["GROUND"]) == 0.0
    # Gen 1: Poison is SE vs Bug
    assert type_effectiveness(t["POISON"], t["BUG"], t["BUG"]) == 2.0


def test_psyduck_vs_charizard_not_a_thing():
    """Sanity: neutral matchups default to 1.0 (chart doesn't list them)."""
    t = load_types()
    assert type_effectiveness(t["NORMAL"], t["FIRE"], t["FIRE"]) == 1.0

"""The transfer contract: obs shape, dtype, ranges, and drift vs v2.

This is THE test that must pass before anyone loads battle-expert weights
into the full-game agent.  Any failure here means the tactical-branch
embedding layout has drifted and loading is unsafe.
"""
from __future__ import annotations

import numpy as np
import pytest

from battle_sim.engine import BattleState
from battle_sim.entities import Pokemon
from battle_sim.obs import tactical_obs
from battle_sim.v2_contract import (
    NUM_POKEMON_TYPES,
    TACTICAL_OBS_SIZE,
    TYPE_ID_LIST,
    TYPE_ID_TO_INDEX,
)


# Authoritative v2 values — if v2/game_state.py changes these, this test
# must be updated in the SAME commit that updates the sim.
V2_TYPE_ID_LIST = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08,
    0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
]
V2_TACTICAL_OBS_SIZE = 22


def test_type_id_list_matches_v2():
    assert TYPE_ID_LIST == V2_TYPE_ID_LIST
    assert NUM_POKEMON_TYPES == 15
    assert len(TYPE_ID_TO_INDEX) == 15


def test_tactical_obs_size_matches_v2():
    assert TACTICAL_OBS_SIZE == V2_TACTICAL_OBS_SIZE


def test_obs_shape_and_dtype():
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH", "EMBER", "GROWL"])
    o = Pokemon.build("PIDGEY", 5, ["TACKLE", "SAND_ATTACK"])
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    assert vec.shape == (22,)
    assert vec.dtype == np.float32


def test_obs_ranges():
    """Every component must be in [-1, 1] to fit the env Box."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH", "EMBER", "GROWL"])
    o = Pokemon.build("PIDGEY", 5, ["TACKLE", "SAND_ATTACK"])
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    assert (vec >= -1.0).all()
    assert (vec <= 1.0).all()


def test_obs_specific_fields():
    """Spot-check the important indices against hand-computed values."""
    p = Pokemon.build("CHARMANDER", 10, ["EMBER", "SCRATCH"])
    o = Pokemon.build("BULBASAUR", 10, ["TACKLE"])
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)

    # idx 0: in_battle = 1
    assert vec[0] == 1.0
    # idx 1: battle_type = 1 (wild) → 0.5
    assert vec[1] == 0.5
    # idx 2: our HP fraction = 1.0 at start
    assert vec[2] == pytest.approx(1.0)
    # idx 3: opp HP fraction = 1.0 at start
    assert vec[3] == pytest.approx(1.0)
    # idx 4: opp level 10 / 100
    assert vec[4] == pytest.approx(0.1)
    # idx 5/6: Charmander (Fire) vs Bulbasaur (Grass/Poison):
    #   Fire vs Grass = 2x.  → signal = +1, bme = 2/4 = 0.5
    assert vec[5] == 1.0
    assert vec[6] == pytest.approx(0.5)


def test_battle_type_domain():
    """v2 expects battle_type ∈ {0, 0.5, 1.0}."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("PIDGEY", 5, ["TACKLE"])
    # trainer battle
    state = BattleState(player=p, opponent=o, battle_type=2)
    vec = tactical_obs(state)
    assert vec[1] == 1.0
    # wild
    state.battle_type = 1
    vec = tactical_obs(state)
    assert vec[1] == 0.5


def test_opp_idx_15_16_set():
    """Opponent type indices must be populated (non-zero for non-Normal types)."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("GEODUDE", 10, ["TACKLE"])  # Rock / Ground
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    # ROCK (0x05) → index 5; GROUND (0x04) → index 4
    assert vec[15] == pytest.approx(TYPE_ID_TO_INDEX[0x05] / 14)
    assert vec[16] == pytest.approx(TYPE_ID_TO_INDEX[0x04] / 14)

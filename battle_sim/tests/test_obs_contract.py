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
V2_TACTICAL_OBS_SIZE = 92  # v0.4 — 36 (v0.3) + 8 stages + 45 bench + 3 meta


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
    assert vec.shape == (92,)
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


def test_v03_status_onehot_and_stages():
    """v0.3 additions — indices 22..35.

    Layout:
      22..26 self status 1-hot [PAR, SLP, BRN, PSN, FRZ]
      27..31 opp  status 1-hot [PAR, SLP, BRN, PSN, FRZ]
      32     self.atk_stage / 6
      33     self.def_stage / 6
      34     opp.atk_stage / 6
      35     opp.def_stage / 6
    """
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    p.status = "BRN"          # slot 2 of self-status block
    o.status = "PAR"          # slot 0 of opp-status block
    o.atk_stage = -2
    o.def_stage = 3
    p.def_stage = -1
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)

    # self one-hot: BRN at 22+2 = 24
    assert list(vec[22:27]) == [0.0, 0.0, 1.0, 0.0, 0.0]
    # opp one-hot: PAR at 27+0 = 27
    assert list(vec[27:32]) == [1.0, 0.0, 0.0, 0.0, 0.0]
    # stages
    assert vec[32] == pytest.approx(0.0)      # p atk_stage 0
    assert vec[33] == pytest.approx(-1 / 6)   # p def_stage
    assert vec[34] == pytest.approx(-2 / 6)   # o atk_stage
    assert vec[35] == pytest.approx(3 / 6)    # o def_stage


def test_v03_status_ok_is_all_zeros():
    """Healthy mons contribute 0 to every status slot (sparse encoding)."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    assert list(vec[22:27]) == [0.0] * 5
    assert list(vec[27:32]) == [0.0] * 5


def test_v04_block_b_remaining_stages():
    """Block B (36..43): spe/spc/acc/eva × (self, opp), each /6."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    p.spd_stage = 2
    p.spc_stage = -1
    p.acc_stage = 0
    p.eva_stage = 3
    o.spd_stage = -2
    o.spc_stage = 1
    o.acc_stage = -3
    o.eva_stage = 0
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    assert vec[36] == pytest.approx(2 / 6)
    assert vec[37] == pytest.approx(-1 / 6)
    assert vec[38] == pytest.approx(0.0)
    assert vec[39] == pytest.approx(3 / 6)
    assert vec[40] == pytest.approx(-2 / 6)
    assert vec[41] == pytest.approx(1 / 6)
    assert vec[42] == pytest.approx(-3 / 6)
    assert vec[43] == pytest.approx(0.0)


def test_v04_block_c_bench_single_mon_is_all_zeros():
    """Block C (44..88): with a 1-mon team, all 5 bench slots zero-filled."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    assert list(vec[44:89]) == [0.0] * 45


def test_v04_block_c_bench_populated():
    """Bench slot 0 should carry hp_frac, level, status, eff rollups."""
    active = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    benched = Pokemon.build("SQUIRTLE", 12, ["TACKLE", "BUBBLE"])
    benched.status = "PAR"
    o = Pokemon.build("BELLSPROUT", 10, ["VINE_WHIP"])  # Grass/Poison
    state = BattleState(
        player_party=[active, benched], opp_party=[o], battle_type=1,
    )
    vec = tactical_obs(state)
    slot0 = vec[44:53]  # 9 dims
    # +0 hp_frac = 1.0 fresh mon
    assert slot0[0] == pytest.approx(1.0)
    # +1 level 12/100
    assert slot0[1] == pytest.approx(0.12)
    # +2..+6 status: PAR at index 0
    assert list(slot0[2:7]) == [1.0, 0.0, 0.0, 0.0, 0.0]
    # +7 off-eff: Squirtle(Water) vs Bellsprout(Grass/Poison).  Combined
    # Water→Grass(0.5) × Water→Poison(1.0) = 0.5.  /4 = 0.125.
    assert slot0[7] == pytest.approx(0.125)
    # +8 def-eff: max over opp types — Grass→Water=2x, Poison→Water=1x.
    # Squirtle is single-Water so only one defender type counts.  Max = 2.0.
    assert slot0[8] == pytest.approx(2.0 / 4.0)


def test_v04_block_d_meta():
    """Block D (89..91): active_slot_index/5, self_alive/6, opp_remaining/6."""
    a = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    b = Pokemon.build("SQUIRTLE", 10, ["TACKLE"])
    c = Pokemon.build("BULBASAUR", 10, ["TACKLE"])
    b.hp = 0  # fainted
    o1 = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    o2 = Pokemon.build("RATTATA", 10, ["TACKLE"])
    state = BattleState(
        player_party=[a, b, c], opp_party=[o1, o2], player_active=0, battle_type=1,
    )
    vec = tactical_obs(state)
    assert vec[89] == pytest.approx(0 / 5)      # active slot 0
    assert vec[90] == pytest.approx(2 / 6)      # 2 of 3 alive (b fainted)
    assert vec[91] == pytest.approx(2 / 6)      # 2 opp remaining


def test_v04_bench_ordering_skips_active():
    """When active is mid-party, bench enumerates other members in order."""
    a = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    b = Pokemon.build("SQUIRTLE", 10, ["TACKLE"])
    c = Pokemon.build("BULBASAUR", 10, ["TACKLE"])
    o = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    state = BattleState(
        player_party=[a, b, c], opp_party=[o], player_active=1, battle_type=1,
    )
    vec = tactical_obs(state)
    # With active=1, bench display order = [party[0]=Charm, party[2]=Bulba].
    # slot 0 of bench = Charmander (level 10/100)
    assert vec[44 + 1] == pytest.approx(0.10)
    # slot 1 of bench = Bulbasaur (level 10/100)
    assert vec[44 + 9 + 1] == pytest.approx(0.10)


def test_v04_bench_rollup_uses_species_types_not_moves():
    """The off/def eff rollup must use attacker SPECIES types only — PyBoy
    can't read opp hidden moves, so the sim can't either for this signal."""
    # Charmander benched, with a HYPER_BEAM-like fake (doesn't matter for
    # rollup).  Opp is Geodude (Rock/Ground).
    active = Pokemon.build("SQUIRTLE", 10, ["TACKLE"])
    bench = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])  # Fire type
    o = Pokemon.build("GEODUDE", 10, ["TACKLE"])          # Rock/Ground
    state = BattleState(
        player_party=[active, bench], opp_party=[o], battle_type=1,
    )
    vec = tactical_obs(state)
    # off-eff: Fire→Rock=0.5 × Fire→Ground=1.0 = 0.5 combined.  /4 = 0.125.
    # (Combined, not max — see data_loader.type_effectiveness.)
    assert vec[44 + 7] == pytest.approx(0.125)


def test_opp_idx_15_16_set():
    """Opponent type indices must be populated (non-zero for non-Normal types)."""
    p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
    o = Pokemon.build("GEODUDE", 10, ["TACKLE"])  # Rock / Ground
    state = BattleState(player=p, opponent=o, battle_type=1)
    vec = tactical_obs(state)
    # ROCK (0x05) → index 5; GROUND (0x04) → index 4
    assert vec[15] == pytest.approx(TYPE_ID_TO_INDEX[0x05] / 14)
    assert vec[16] == pytest.approx(TYPE_ID_TO_INDEX[0x04] / 14)

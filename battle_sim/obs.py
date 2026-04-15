"""Tactical observation encoder — mirrors v2/game_state.py:tactical_obs().

The 92-float layout is THE transfer contract (see TRANSFER_CONTRACT.md).
If any index changes, the battle-expert weights can't load into the
full-game PokemonNet's tactical branch.

Critical: this encoder reproduces the v0.1 "own types as move-type proxy"
simplification (v2/game_state.py:best_move_effectiveness).  The sim knows
real move types but MUST NOT use them for the type-advantage fields until
v2 is upgraded in lockstep.

v0.4 layout blocks:
  A [ 0..35] — v0.3 active-mon tactical (UNCHANGED for contract compat).
  B [36..43] — remaining stat stages (spe/spc/acc/eva × self/opp) / 6.
  C [44..88] — 5 bench slots × 9 dims (hp, level, status-onehot,
               off-eff rollup, def-eff rollup).  Rollups use species types
               only — NO hidden moveset access, so PyBoy-side can emit
               identical values from RAM alone.
  D [89..91] — active_slot_index/5, self_alive_count/6, opp_remaining/6.
"""
from __future__ import annotations

import numpy as np

from .data_loader import type_effectiveness
from .engine import BattleState
from .entities import Pokemon
from .v2_contract import (
    BENCH_SLOT_DIMS,
    MAX_BENCH_SIZE,
    NUM_POKEMON_TYPES,
    STATUS_ONEHOT_INDEX,
    STATUS_ONEHOT_SIZE,
    TACTICAL_BLOCK_A_END,
    TACTICAL_BLOCK_B_END,
    TACTICAL_BLOCK_C_END,
    TACTICAL_BLOCK_D_END,
    TACTICAL_OBS_SIZE,
    TYPE_ID_TO_INDEX,
)


def _status_onehot(status: str) -> list[float]:
    """5-dim one-hot for status.  OK = all zeros.  Domain matches v2."""
    v = [0.0] * STATUS_ONEHOT_SIZE
    idx = STATUS_ONEHOT_INDEX.get(status)
    if idx is not None:
        v[idx] = 1.0
    return v


def _type_adv_signal(eff: float) -> float:
    if eff >= 2.0:
        return 1.0
    if eff <= 0.5:   # includes 0.0 and 0.25
        return -1.0
    return 0.0


def _best_move_effectiveness_proxy(attacker: Pokemon, defender: Pokemon) -> float:
    """v2-parity: uses attacker's OWN TYPES as the move-type proxy, not real move types."""
    eff1 = type_effectiveness(attacker.type1_id, defender.type1_id, defender.type2_id)
    eff2 = type_effectiveness(attacker.type2_id, defender.type1_id, defender.type2_id)
    return max(eff1, eff2)


def _bench_slot_vec(mon: Pokemon | None, opp_active: Pokemon | None) -> list[float]:
    """Encode one bench slot.  mon=None → empty slot, all zeros."""
    if mon is None:
        return [0.0] * BENCH_SLOT_DIMS
    hp_frac = mon.hp / mon.max_hp if mon.max_hp else 0.0
    lvl = mon.level / 100.0
    st = _status_onehot(mon.status)
    if opp_active is None:
        off_eff = 0.0
        def_eff = 0.0
    else:
        # Both rollups use species types only — visible to PyBoy from RAM.
        off_eff = _best_move_effectiveness_proxy(mon, opp_active)
        def_eff = _best_move_effectiveness_proxy(opp_active, mon)
    return [hp_frac, lvl, *st, min(off_eff, 4.0) / 4.0, min(def_eff, 4.0) / 4.0]


def tactical_obs(state: BattleState) -> np.ndarray:
    """Emit the 92-float tactical vector.

    Always assumes in_battle=True (the sim wraps a battle that is always
    live).  All scalings match [game_state.py:tactical_obs()].
    """
    p = state.player
    o = state.opponent

    # ========== Block A — v0.3 active-mon tactical (UNCHANGED) ==========
    in_b = 1.0
    b_type = state.battle_type / 2.0

    our_hp = p.hp / p.max_hp if p.max_hp else 1.0
    opp_hp = o.hp / o.max_hp if o.max_hp else 0.0
    opp_lvl = o.level / 100.0

    bme = _best_move_effectiveness_proxy(p, o)
    ta = _type_adv_signal(bme)
    bme_n = min(bme, 4.0) / 4.0

    pp = [min(m.pp, 40) / 40.0 for m in p.moves]
    while len(pp) < 4:
        pp.append(0.0)

    # Active-mon "party fainted" / party-size in v0.3 semantics — for the
    # active slot only, so party-aware info lives in Block D instead.
    fainted = (1.0 if p.fainted else 0.0) / 6.0
    party_size = len(state.player_party) / 6.0

    lt1 = TYPE_ID_TO_INDEX.get(p.type1_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)
    lt2 = TYPE_ID_TO_INDEX.get(p.type2_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)
    ot1 = TYPE_ID_TO_INDEX.get(o.type1_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)
    ot2 = TYPE_ID_TO_INDEX.get(o.type2_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)

    moves = [min(m.id, 165) / 165.0 for m in p.moves]
    while len(moves) < 4:
        moves.append(0.0)

    box = 0.0

    p_st = _status_onehot(p.status)
    o_st = _status_onehot(o.status)
    p_atk_s = p.atk_stage / 6.0
    p_def_s = p.def_stage / 6.0
    o_atk_s = o.atk_stage / 6.0
    o_def_s = o.def_stage / 6.0

    block_a = [
        in_b, b_type, our_hp, opp_hp, opp_lvl,
        ta, bme_n,
        *pp,
        fainted, party_size,
        lt1, lt2, ot1, ot2,
        *moves,
        box,
        *p_st, *o_st,
        p_atk_s, p_def_s, o_atk_s, o_def_s,
    ]
    assert len(block_a) == TACTICAL_BLOCK_A_END

    # ========== Block B — remaining stat stages (36..43) ==========
    block_b = [
        p.spd_stage / 6.0,
        p.spc_stage / 6.0,
        p.acc_stage / 6.0,
        p.eva_stage / 6.0,
        o.spd_stage / 6.0,
        o.spc_stage / 6.0,
        o.acc_stage / 6.0,
        o.eva_stage / 6.0,
    ]
    assert len(block_b) == TACTICAL_BLOCK_B_END - TACTICAL_BLOCK_A_END

    # ========== Block C — 5 bench slots × 9 dims (44..88) ==========
    # Bench = the 5 non-active party members, in original-team order.
    # If active is party[2], bench is [party[0], party[1], party[3],
    # party[4], party[5]].  Short teams pad with None (all-zeros slot).
    # This preserves team order for the policy while hiding the active
    # (which lives in Block A).  The switch-action semantics work because
    # action 4+k maps to original party slot k+1, which we can compute from
    # the bench position — the engine knows the mapping independently.
    bench_mons: list[Pokemon | None] = []
    for pos in range(len(state.player_party)):
        if pos == state.player_active:
            continue
        bench_mons.append(state.player_party[pos])
    while len(bench_mons) < MAX_BENCH_SIZE:
        bench_mons.append(None)

    block_c: list[float] = []
    for mon in bench_mons:
        block_c.extend(_bench_slot_vec(mon, o))
    assert len(block_c) == TACTICAL_BLOCK_C_END - TACTICAL_BLOCK_B_END

    # ========== Block D — battle-global meta (89..91) ==========
    active_idx = state.player_active / 5.0
    self_alive = sum(1 for m in state.player_party if not m.fainted) / 6.0
    opp_remaining = sum(1 for m in state.opp_party if not m.fainted) / 6.0
    block_d = [active_idx, self_alive, opp_remaining]
    assert len(block_d) == TACTICAL_BLOCK_D_END - TACTICAL_BLOCK_C_END

    vec = np.array(block_a + block_b + block_c + block_d, dtype=np.float32)
    assert vec.shape == (TACTICAL_OBS_SIZE,), \
        f"tactical obs shape {vec.shape} != {TACTICAL_OBS_SIZE}"
    return vec

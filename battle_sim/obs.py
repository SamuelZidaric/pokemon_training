"""Tactical observation encoder — mirrors v2/game_state.py:tactical_obs().

The 22-float layout is THE transfer contract.  If any index changes, the
battle-expert weights can't load into the full-game PokemonNet's tactical
branch.  See SPEC.md Part 1 for the authoritative table.

Critical: this encoder reproduces the v0.1 "own types as move-type proxy"
simplification (v2/game_state.py:best_move_effectiveness).  The sim knows
real move types but MUST NOT use them for the type-advantage fields until
v2 is upgraded in lockstep.
"""
from __future__ import annotations

import numpy as np

from .data_loader import type_effectiveness
from .engine import BattleState
from .entities import Pokemon
from .v2_contract import NUM_POKEMON_TYPES, TACTICAL_OBS_SIZE, TYPE_ID_TO_INDEX


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


def tactical_obs(state: BattleState) -> np.ndarray:
    """Emit the 22-float tactical vector.

    Always assumes in_battle=True (the sim wraps a battle that is always
    live).  All scalings match [game_state.py:tactical_obs()].
    """
    p = state.player
    o = state.opponent

    # [0] in_battle
    in_b = 1.0
    # [1] battle_type / 2.0  ∈ {0, 0.5, 1.0}
    b_type = state.battle_type / 2.0

    # [2] our HP fraction (party-wide in v2; in 1v1 sim that's the active mon)
    our_hp = p.hp / p.max_hp if p.max_hp else 1.0
    # [3] opponent HP fraction
    opp_hp = o.hp / o.max_hp if o.max_hp else 0.0
    # [4] opponent level / 100
    opp_lvl = o.level / 100.0

    # [5] type advantage signal; [6] best-move effectiveness (capped /4)
    bme = _best_move_effectiveness_proxy(p, o)
    ta = _type_adv_signal(bme)
    bme_n = min(bme, 4.0) / 4.0

    # [7..10] lead PP, capped at 40, /40
    pp = [min(m.pp, 40) / 40.0 for m in p.moves]
    while len(pp) < 4:
        pp.append(0.0)

    # [11] party fainted (in 1v1: 1 if faint, else 0) / 6
    fainted = (1.0 if p.fainted else 0.0) / 6.0
    # [12] party size / 6
    p_size = 1.0 / 6.0

    # [13..14] lead type indices / 14
    lt1 = TYPE_ID_TO_INDEX.get(p.type1_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)
    lt2 = TYPE_ID_TO_INDEX.get(p.type2_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)
    # [15..16] opp type indices / 14
    ot1 = TYPE_ID_TO_INDEX.get(o.type1_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)
    ot2 = TYPE_ID_TO_INDEX.get(o.type2_id, 0) / max(NUM_POKEMON_TYPES - 1, 1)

    # [17..20] lead moves, capped at 165, /165
    moves = [min(m.id, 165) / 165.0 for m in p.moves]
    while len(moves) < 4:
        moves.append(0.0)

    # [21] box count / 20 — zero in the battle sim (no PC)
    box = 0.0

    vec = np.array([
        in_b, b_type, our_hp, opp_hp, opp_lvl,
        ta, bme_n,
        *pp,
        fainted, p_size,
        lt1, lt2, ot1, ot2,
        *moves,
        box,
    ], dtype=np.float32)
    assert vec.shape == (TACTICAL_OBS_SIZE,), \
        f"tactical obs shape {vec.shape} != {TACTICAL_OBS_SIZE}"
    return vec

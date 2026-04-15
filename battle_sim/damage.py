"""Pure Gen 1 damage, hit, and crit functions.

All three are deliberately stateless — they consume a Pokemon, a Move, and
optional RNG draws, and return a damage integer (or bool for hit/crit).
Keeps the engine testable in isolation from the turn loop.
"""
from __future__ import annotations

from .data_loader import type_effectiveness
from .entities import Move, Pokemon
from .rng import BattleRNG


# Gen 1 physical vs special is decided by move TYPE, not move category.
PHYSICAL_TYPES = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08}


def is_physical(move: Move) -> bool:
    return move.type_id in PHYSICAL_TYPES


def crit_threshold(attacker: Pokemon, high_crit: bool = False) -> int:
    """Gen 1 crit threshold: base_speed // 2, or * 4 for high-crit moves.

    High-crit moves are SLASH, KARATE_CHOP, CRABHAMMER, RAZOR_LEAF — callers
    pass ``high_crit=True`` for these.  The result is capped at 255.
    """
    t = attacker.base_spd // 2
    if high_crit:
        t *= 4
    return min(t, 255)


HIGH_CRIT_MOVES = {"SLASH", "KARATE_CHOP", "CRABHAMMER", "RAZOR_LEAF"}


def compute_damage(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    rng: BattleRNG,
    force_crit: bool | None = None,
    force_roll: int | None = None,
) -> tuple[int, bool]:
    """Return (damage, crit).  Damage is 0 if move misses or is immune.

    Hit check is the caller's job so that the miss animation can be
    distinguished from zero-damage crits.  This function assumes the hit
    already landed.

    Gen 1 formula (integer math, truncation after every step):

        base = (((2*L*crit // 5) + 2) * power * A // D) // 50 + 2

    Crits *double the level term*.  Then STAB, then type_mult, then roll.
    """
    if move.power == 0:
        return 0, False

    # Crit decision
    if force_crit is not None:
        crit = force_crit
    else:
        crit = rng.crit_check(crit_threshold(attacker, move.name in HIGH_CRIT_MOVES))

    phys = is_physical(move)
    if crit:
        # Crit in Gen 1: doubles level, and critically, uses UN-staged stats
        # (bypasses both attacker buffs and defender buffs).
        level_term = 2 * attacker.level
        atk = attacker.atk if phys else attacker.spc
        dfn = defender.df if phys else defender.spc
    else:
        level_term = attacker.level
        atk = attacker.effective_atk(phys)
        dfn = defender.effective_def(phys)

    # Burn halves Attack for physical moves (Gen 1 — applies even through
    # crit's stat-stage bypass; the halving is a status multiplier, not a stage).
    if phys:
        from .effects import status_atk_multiplier
        num, den = status_atk_multiplier(attacker)
        atk = atk * num // den

    if dfn <= 0:
        dfn = 1

    base = ((2 * level_term) // 5 + 2) * move.power * atk // dfn // 50 + 2

    # STAB — 1.5 via integer math
    if move.type_id == attacker.type1_id or move.type_id == attacker.type2_id:
        base = base * 3 // 2

    # Type matchup
    mult = type_effectiveness(move.type_id, defender.type1_id, defender.type2_id)
    if mult == 0.0:
        return 0, crit
    # Avoid float drift: express as num/den
    if mult == 0.5:
        base = base // 2
    elif mult == 0.25:
        base = base // 4
    elif mult == 2.0:
        base = base * 2
    elif mult == 4.0:
        base = base * 4
    # mult == 1.0 → no change

    # Damage roll
    roll = force_roll if force_roll is not None else rng.damage_roll()
    base = base * roll // 255

    return max(1, base), crit


def accuracy_check(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    rng: BattleRNG,
) -> bool:
    """Gen 1 accuracy: strict-less-than byte compare, 1/256 miss on 100%.

    We fold accuracy/evasion stages here (Gen 1 multiplies them into the
    accuracy byte before the comparison).
    """
    from .entities import STAGE_MULT
    acc_n, acc_d = STAGE_MULT[attacker.acc_stage]
    # Evasion is the inverse direction: +stage LOWERS hit rate.
    eva_n, eva_d = STAGE_MULT[-defender.eva_stage]
    eff = move.accuracy * acc_n // acc_d * eva_n // eva_d
    eff = min(eff, 255)
    return rng.hit_check(eff)

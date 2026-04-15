"""Gen 1 move effects — stat stages, status, secondary effects.

Dispatches on ``move.effect`` (string constant from pokered's
``constants/move_effect_constants.asm``).  Only the subset needed for the
early-game opponent pool is implemented in v0.2; unknown effects fall
through to no-op so the sim still runs.

Design: effect functions mutate the battle in place and return a list of
event-log strings.  They're called from the engine AFTER the damage step
(so secondary effects see whether the move hit) and BEFORE the faint
check.
"""
from __future__ import annotations

from .entities import STAGE_MULT, Pokemon
from .rng import BattleRNG


# ---------------------------------------------------------------------------
# Stat-stage helpers
# ---------------------------------------------------------------------------

def _clamp_stage(stage: int) -> int:
    return max(-6, min(6, stage))


def _apply_stage(mon: Pokemon, stat: str, delta: int) -> list[str]:
    """Mutate one stat stage on mon by ``delta``, clamped to [-6, +6]."""
    attr = f"{stat}_stage"
    old = getattr(mon, attr)
    new = _clamp_stage(old + delta)
    setattr(mon, attr, new)
    if new == old:
        arrow = "won't go any" + (" higher" if delta > 0 else " lower")
        return [f"{mon.species}'s {stat} {arrow}"]
    verb = "rose" if delta > 0 else "fell"
    qual = "sharply " if abs(delta) >= 2 else ""
    return [f"{mon.species}'s {stat} {qual}{verb}"]


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

# Gen 1 type immunities to specific status.
STATUS_TYPE_IMMUNE: dict[str, set[int]] = {
    # Poison: Poison-type immune
    "PSN": {0x03},
    # Burn: Fire-type immune
    "BRN": {0x14},
    # Freeze: Ice-type immune
    "FRZ": {0x19},
    # Paralysis: no type immunity in Gen 1 (Electric IS affected)
    "PAR": set(),
    # Sleep: no type immunity
    "SLP": set(),
}


def _can_apply_status(target: Pokemon, status: str) -> bool:
    if target.status != "OK":
        return False
    if target.type1_id in STATUS_TYPE_IMMUNE.get(status, set()):
        return False
    if target.type2_id in STATUS_TYPE_IMMUNE.get(status, set()):
        return False
    return True


def apply_status(target: Pokemon, status: str, rng: BattleRNG) -> list[str]:
    """Apply a major status with Gen 1 immunity rules."""
    if not _can_apply_status(target, status):
        return [f"{target.species} is unaffected by {status}"]
    target.status = status
    if status == "SLP":
        # 1-7 turns (Gen 1 rolls 1..7; we use 1..7 inclusive)
        target.sleep_turns = 1 + (rng.next_byte() % 7)
        return [f"{target.species} fell asleep for {target.sleep_turns} turns"]
    if status == "PAR":
        return [f"{target.species} was paralyzed"]
    if status == "BRN":
        return [f"{target.species} was burned"]
    if status == "PSN":
        return [f"{target.species} was poisoned"]
    if status == "FRZ":
        return [f"{target.species} was frozen"]
    return []


# ---------------------------------------------------------------------------
# Effect dispatch
# ---------------------------------------------------------------------------

# Effects that DON'T do damage themselves — invoked only if the move landed
# (accuracy check succeeded).  Damage-dealing moves have already resolved
# damage by the time we get here; we only apply the secondary.
#
# For status/stat-stage moves (power=0), damage was 0 — this dispatch is
# where the actual effect happens.

def apply_move_effect(
    move_effect: str,
    attacker: Pokemon,
    defender: Pokemon,
    rng: BattleRNG,
    move_hit: bool,
    did_damage: bool,
) -> list[str]:
    """Apply a move's effect post-damage.  Returns event-log lines."""
    # No-effect moves — most common
    if move_effect in ("NO_ADDITIONAL_EFFECT", "RECOIL_EFFECT",
                       "TELEPORT_EFFECT", "SPLASH_EFFECT", "MIMIC_EFFECT",
                       "RAGE_EFFECT", "BIDE_EFFECT"):
        return []

    if not move_hit:
        # Whiffed — status/stat-stage moves need the hit check too
        return []

    # --- Primary status (100% chance moves like Thunder Wave, Sleep Powder)
    if move_effect == "SLEEP_EFFECT":
        return apply_status(defender, "SLP", rng)
    if move_effect == "PARALYZE_EFFECT":
        return apply_status(defender, "PAR", rng)
    if move_effect == "POISON_EFFECT":
        return apply_status(defender, "PSN", rng)
    if move_effect == "TOXIC_EFFECT":   # same bucket in v0.2
        return apply_status(defender, "PSN", rng)
    if move_effect == "CONFUSION_EFFECT":
        if defender.confusion_turns == 0:
            defender.confusion_turns = 2 + (rng.next_byte() % 4)  # 2..5
            return [f"{defender.species} became confused"]
        return []

    # --- Stat-stage moves on the USER (primary effect — always triggers on hit)
    user_buffs = {
        "ATTACK_UP1_EFFECT":    ("atk",  +1),
        "ATTACK_UP2_EFFECT":    ("atk",  +2),
        "DEFENSE_UP1_EFFECT":   ("def",  +1),
        "DEFENSE_UP2_EFFECT":   ("def",  +2),
        "SPEED_UP2_EFFECT":     ("spd",  +2),
        "SPECIAL_UP1_EFFECT":   ("spc",  +1),
        "SPECIAL_UP2_EFFECT":   ("spc",  +2),
        "EVASION_UP1_EFFECT":   ("eva",  +1),
        "ACCURACY_UP1_EFFECT":  ("acc",  +1),
    }
    if move_effect in user_buffs:
        stat, delta = user_buffs[move_effect]
        return _apply_stage(attacker, stat, delta)

    # --- Stat-stage moves on the OPPONENT (primary — always on hit)
    opp_debuffs = {
        "ATTACK_DOWN1_EFFECT":   ("atk", -1),
        "ATTACK_DOWN2_EFFECT":   ("atk", -2),
        "DEFENSE_DOWN1_EFFECT":  ("def", -1),
        "DEFENSE_DOWN2_EFFECT":  ("def", -2),
        "SPEED_DOWN1_EFFECT":    ("spd", -1),
        "SPECIAL_DOWN1_EFFECT":  ("spc", -1),
        "ACCURACY_DOWN1_EFFECT": ("acc", -1),
        "EVASION_DOWN1_EFFECT":  ("eva", -1),
    }
    if move_effect in opp_debuffs:
        stat, delta = opp_debuffs[move_effect]
        return _apply_stage(defender, stat, delta)

    # --- Secondary-chance status (side-effect on damaging moves)
    # Gen 1: ~10% chance for most "side effect 1", ~30% for "side effect 2"
    if not did_damage:
        return []

    chance_10 = rng.next_byte() < 26     # 10.2%
    chance_30 = rng.next_byte() < 77     # 30.1%

    if move_effect in ("BURN_SIDE_EFFECT1", "FIRE_SIDE_EFFECT1"):
        if chance_10:
            return apply_status(defender, "BRN", rng)
    if move_effect == "BURN_SIDE_EFFECT2":
        if chance_30:
            return apply_status(defender, "BRN", rng)
    if move_effect in ("FREEZE_SIDE_EFFECT", "FREEZE_SIDE_EFFECT_LOWER_SPEED"):
        if chance_10:
            return apply_status(defender, "FRZ", rng)
    if move_effect == "PARALYZE_SIDE_EFFECT1":
        if chance_10:
            return apply_status(defender, "PAR", rng)
    if move_effect == "PARALYZE_SIDE_EFFECT2":
        if chance_30:
            return apply_status(defender, "PAR", rng)
    if move_effect == "POISON_SIDE_EFFECT1":
        if chance_10:
            return apply_status(defender, "PSN", rng)
    if move_effect == "POISON_SIDE_EFFECT2":
        if chance_30:
            return apply_status(defender, "PSN", rng)
    if move_effect == "FLINCH_SIDE_EFFECT1":
        if chance_10:
            defender.flinched = True
            return [f"{defender.species} flinched"]
    if move_effect == "FLINCH_SIDE_EFFECT2":
        if chance_30:
            defender.flinched = True
            return [f"{defender.species} flinched"]

    # --- Secondary stat debuffs on damaging moves
    secondary_debuffs = {
        "ATTACK_DOWN_SIDE_EFFECT":   ("atk", -1),
        "DEFENSE_DOWN_SIDE_EFFECT":  ("def", -1),
        "SPEED_DOWN_SIDE_EFFECT":    ("spd", -1),
        "SPECIAL_DOWN_SIDE_EFFECT":  ("spc", -1),
    }
    if move_effect in secondary_debuffs and did_damage:
        if chance_10:
            stat, delta = secondary_debuffs[move_effect]
            return _apply_stage(defender, stat, delta)

    # Unknown → no-op (intentional: the engine keeps running with other moves)
    return []


# ---------------------------------------------------------------------------
# Turn-start action gating (sleep, paralysis skip, freeze, flinch, confusion)
# ---------------------------------------------------------------------------

def check_action_allowed(
    actor: Pokemon, rng: BattleRNG
) -> tuple[bool, list[str]]:
    """Decide whether the actor can act this turn.

    Returns ``(can_act, log_lines)``.  If can_act is False, the actor's
    move for this turn is skipped.  This is called BEFORE move selection
    is consumed (PP is not decremented on a skipped turn).

    Also applies end-of-pre-turn cleanup:
      - decrements sleep_turns (wakes at 0 next turn start)
      - clears flinched flag (flinch is single-turn)
    """
    log: list[str] = []

    # Flinch clears after skipping
    if actor.flinched:
        actor.flinched = False
        log.append(f"{actor.species} flinched and couldn't move")
        return False, log

    # Freeze — Gen 1: no self-thaw, stuck forever (until hit by Fire move,
    # which we don't model in v0.2)
    if actor.status == "FRZ":
        log.append(f"{actor.species} is frozen solid")
        return False, log

    # Sleep — count down first, then decide
    if actor.status == "SLP":
        if actor.sleep_turns > 0:
            actor.sleep_turns -= 1
            log.append(f"{actor.species} is fast asleep ({actor.sleep_turns} left)")
            if actor.sleep_turns == 0:
                # Gen 1: the wake-up turn is also a skipped action
                actor.status = "OK"
                log.append(f"{actor.species} woke up!")
            return False, log

    # Paralysis — 25% chance of fully paralyzed this turn
    if actor.status == "PAR":
        if rng.next_byte() < 64:    # 25.0%
            log.append(f"{actor.species} is fully paralyzed")
            return False, log

    # Confusion — 50% chance to hurt self, 50% to act normally.  In v0.2 we
    # simplify: with confusion, 50% skip (not self-hit).  Full self-hit in v0.3.
    if actor.confusion_turns > 0:
        actor.confusion_turns -= 1
        if actor.confusion_turns == 0:
            log.append(f"{actor.species} snapped out of confusion")
        elif rng.next_byte() < 128:
            log.append(f"{actor.species} is confused and hurt itself")
            # Self-hit: typeless 40-power physical on self
            self_dmg = max(1, actor.atk // 4)
            actor.hp = max(0, actor.hp - self_dmg)
            log.append(f"{actor.species} took {self_dmg} confusion damage")
            return False, log

    return True, log


# ---------------------------------------------------------------------------
# End-of-turn residual damage (burn, poison)
# ---------------------------------------------------------------------------

def end_of_turn_residuals(mon: Pokemon) -> list[str]:
    """Apply end-of-turn burn/poison tick.  1/16 max HP, floor 1."""
    if mon.fainted or mon.status not in ("BRN", "PSN"):
        return []
    tick = max(1, mon.max_hp // 16)
    mon.hp = max(0, mon.hp - tick)
    label = "burn" if mon.status == "BRN" else "poison"
    return [f"{mon.species} took {tick} {label} damage → {mon.hp}/{mon.max_hp}"]


# ---------------------------------------------------------------------------
# Stat multipliers from status (applied in damage.py)
# ---------------------------------------------------------------------------

def status_atk_multiplier(mon: Pokemon) -> tuple[int, int]:
    """Burn halves Attack in Gen 1 (physical moves only — caller checks)."""
    if mon.status == "BRN":
        return (1, 2)
    return (1, 1)


def status_spd_multiplier(mon: Pokemon) -> tuple[int, int]:
    """Paralysis quarters Speed in Gen 1."""
    if mon.status == "PAR":
        return (1, 4)
    return (1, 1)

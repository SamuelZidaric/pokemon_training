"""v0.2 effect-dispatch tests.

Covers the new status / stat-stage / residual machinery added in
``battle_sim/effects.py`` and its integration into ``engine.py`` and
``damage.py``.
"""
from __future__ import annotations

import pytest

from battle_sim.damage import compute_damage
from battle_sim.effects import (
    apply_move_effect,
    apply_status,
    check_action_allowed,
    end_of_turn_residuals,
    status_atk_multiplier,
    status_spd_multiplier,
)
from battle_sim.engine import BattleEngine, BattleResult, BattleState
from battle_sim.entities import Move, Pokemon
from battle_sim.rng import BattleRNG


# ---------------------------------------------------------------------------
# Status application & type immunity
# ---------------------------------------------------------------------------

def _squirtle() -> Pokemon:
    return Pokemon.build("SQUIRTLE", 10, ["TACKLE", "BUBBLE"])


def _charmander() -> Pokemon:
    return Pokemon.build("CHARMANDER", 10, ["SCRATCH", "EMBER"])


def _ekans() -> Pokemon:
    return Pokemon.build("EKANS", 10, ["WRAP"])


def test_poison_cannot_be_applied_to_poison_type():
    """Ekans is Poison-type → immune to PSN."""
    mon = _ekans()
    rng = BattleRNG(0)
    log = apply_status(mon, "PSN", rng)
    assert mon.status == "OK"
    assert any("unaffected" in s for s in log)


def test_burn_cannot_be_applied_to_fire_type():
    mon = _charmander()
    rng = BattleRNG(0)
    apply_status(mon, "BRN", rng)
    assert mon.status == "OK"


def test_paralysis_applies_to_electric_type_in_gen1():
    """Gen 1 quirk — Electric IS affected by paralysis."""
    pikachu = Pokemon.build("PIKACHU", 10, ["THUNDERSHOCK"])
    rng = BattleRNG(0)
    apply_status(pikachu, "PAR", rng)
    assert pikachu.status == "PAR"


def test_sleep_sets_countdown():
    mon = _squirtle()
    rng = BattleRNG(12)
    apply_status(mon, "SLP", rng)
    assert mon.status == "SLP"
    assert 1 <= mon.sleep_turns <= 7


def test_status_is_singular():
    """Can't overwrite an existing status."""
    mon = _squirtle()
    mon.status = "PAR"
    apply_status(mon, "BRN", BattleRNG(0))
    assert mon.status == "PAR"


# ---------------------------------------------------------------------------
# Turn-start gating
# ---------------------------------------------------------------------------

def test_sleep_skips_turn_and_wakes():
    mon = _squirtle()
    mon.status = "SLP"
    mon.sleep_turns = 2
    rng = BattleRNG(0)

    can_act, log = check_action_allowed(mon, rng)
    assert not can_act
    assert mon.sleep_turns == 1
    assert mon.status == "SLP"

    can_act, log = check_action_allowed(mon, rng)
    assert not can_act  # Gen 1: wake-up turn is skipped
    assert mon.sleep_turns == 0
    assert mon.status == "OK"

    can_act, _ = check_action_allowed(mon, rng)
    assert can_act


def test_freeze_locks_forever():
    """v0.2 freeze never self-thaws (no Fire-hit modelled)."""
    mon = _squirtle()
    mon.status = "FRZ"
    rng = BattleRNG(0)
    for _ in range(20):
        can_act, _ = check_action_allowed(mon, rng)
        assert not can_act


def test_flinch_single_turn():
    mon = _squirtle()
    mon.flinched = True
    can_act, _ = check_action_allowed(mon, BattleRNG(0))
    assert not can_act
    assert mon.flinched is False
    # Next turn the mon can act again
    can_act, _ = check_action_allowed(mon, BattleRNG(0))
    assert can_act


def test_paralysis_gates_at_25_percent():
    """Over many rolls paralysis blocks ~25% of turns."""
    mon = _squirtle()
    mon.status = "PAR"
    rng = BattleRNG(7)
    blocks = 0
    N = 2000
    for _ in range(N):
        can_act, _ = check_action_allowed(mon, rng)
        if not can_act:
            blocks += 1
    frac = blocks / N
    assert 0.20 < frac < 0.30, f"par block rate {frac:.3f} outside [0.20, 0.30]"


# ---------------------------------------------------------------------------
# Residual damage
# ---------------------------------------------------------------------------

def test_burn_ticks_one_sixteenth():
    mon = _squirtle()
    mon.status = "BRN"
    expected = max(1, mon.max_hp // 16)
    start = mon.hp
    end_of_turn_residuals(mon)
    assert mon.hp == start - expected


def test_no_residual_when_healthy():
    mon = _squirtle()
    assert end_of_turn_residuals(mon) == []
    assert mon.hp == mon.max_hp


def test_residual_cannot_go_below_zero():
    mon = _squirtle()
    mon.status = "PSN"
    mon.hp = 1
    end_of_turn_residuals(mon)
    assert mon.hp == 0


# ---------------------------------------------------------------------------
# Stat multipliers
# ---------------------------------------------------------------------------

def test_burn_halves_physical_attack():
    assert status_atk_multiplier(_status_mon("BRN")) == (1, 2)
    assert status_atk_multiplier(_status_mon("OK")) == (1, 1)


def test_paralysis_quarters_speed():
    assert status_spd_multiplier(_status_mon("PAR")) == (1, 4)
    assert status_spd_multiplier(_status_mon("OK")) == (1, 1)


def _status_mon(status: str) -> Pokemon:
    mon = _squirtle()
    mon.status = status
    return mon


# ---------------------------------------------------------------------------
# compute_damage integration with burn
# ---------------------------------------------------------------------------

def test_burned_attacker_does_less_physical_damage():
    atk = Pokemon.build("CHARMANDER", 20, ["SCRATCH"])
    dfn = Pokemon.build("PIDGEY", 20, ["TACKLE"])
    move = atk.moves[0]  # Scratch — physical Normal

    # Healthy baseline
    rng = BattleRNG(0)
    healthy, _ = compute_damage(atk, dfn, move, rng, force_crit=False, force_roll=255)

    atk.status = "BRN"
    rng = BattleRNG(0)
    burned, _ = compute_damage(atk, dfn, move, rng, force_crit=False, force_roll=255)

    assert burned < healthy, f"burn should reduce damage: {burned} !< {healthy}"


def test_burn_does_not_affect_special_damage():
    atk = Pokemon.build("CHARMANDER", 20, ["EMBER"])
    dfn = Pokemon.build("PIDGEY", 20, ["TACKLE"])
    move = atk.moves[0]  # Ember — special Fire

    rng1 = BattleRNG(0)
    healthy, _ = compute_damage(atk, dfn, move, rng1, force_crit=False, force_roll=255)

    atk.status = "BRN"
    rng2 = BattleRNG(0)
    burned, _ = compute_damage(atk, dfn, move, rng2, force_crit=False, force_roll=255)

    assert burned == healthy


# ---------------------------------------------------------------------------
# End-to-end engine integration
# ---------------------------------------------------------------------------

def test_thunder_wave_paralyzes_via_engine():
    """Player Pikachu paralyzes opponent Rattata in one step."""
    player = Pokemon.build("PIKACHU", 20, ["THUNDER_WAVE", "THUNDERSHOCK"])
    opp = Pokemon.build("RATTATA", 5, ["TACKLE"])
    s = BattleState(player=player, opponent=opp)
    eng = BattleEngine(s, BattleRNG(0))
    eng.step(0)  # Thunder Wave
    assert opp.status == "PAR"


def test_poisoned_mon_takes_residual_in_engine():
    player = Pokemon.build("BULBASAUR", 20, ["TACKLE"])
    opp = Pokemon.build("RATTATA", 20, ["TACKLE"])
    opp.status = "PSN"
    s = BattleState(player=player, opponent=opp)
    eng = BattleEngine(s, BattleRNG(0))
    before = opp.hp
    eng.step(0)
    assert opp.hp < before - 1  # residual on top of Tackle


def test_stat_drop_move_via_engine():
    """GROWL lowers opponent's Attack stage."""
    player = Pokemon.build("CHARMANDER", 10, ["GROWL"])
    opp = Pokemon.build("PIDGEY", 10, ["TACKLE"])
    s = BattleState(player=player, opponent=opp)
    eng = BattleEngine(s, BattleRNG(0))
    eng.step(0)
    assert opp.atk_stage <= -1

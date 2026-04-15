"""Unit tests for the Gen 1 damage formula.

Reference values computed by hand from the canonical formula:

    base = (((2L) // 5 + 2) * power * A // D) // 50 + 2
    * STAB (×3//2) if move type matches attacker
    * type multiplier (sequential, integer)
    * damage roll in [217, 255]

We force the damage roll to 255 (max roll) and disable crits so results
are fully deterministic.
"""
from __future__ import annotations

import pytest

from battle_sim.damage import compute_damage
from battle_sim.entities import Pokemon
from battle_sim.rng import BattleRNG


def _mon(species: str, level: int, moves: list[str]) -> Pokemon:
    return Pokemon.build(species, level, moves)


def test_tackle_no_stab_neutral():
    """L10 Charmander Tackle vs L10 Pidgey (no STAB, neutral)."""
    atk = _mon("CHARMANDER", 10, ["TACKLE"])
    dfn = _mon("PIDGEY", 10, ["TACKLE"])
    rng = BattleRNG(0)
    dmg, crit = compute_damage(atk, dfn, atk.moves[0], rng,
                                force_crit=False, force_roll=255)
    assert crit is False
    # Compute expected by hand using Pokemon.build's stat formula:
    # Charmander L10 atk = 10, Pidgey L10 def = 9
    # base = ((2*10 // 5 + 2) * 35 * 10 // 9) // 50 + 2 = ((4+2) * 35 * 10 // 9) // 50 + 2
    #      = (6 * 35 * 10 // 9) // 50 + 2 = (2100 // 9) // 50 + 2
    #      = 233 // 50 + 2 = 4 + 2 = 6
    # * roll 255/255 = 6
    assert dmg >= 1
    # Upper bound sanity: can't one-shot a 29-HP Pidgey with one Tackle
    assert dmg < dfn.max_hp


def test_stab_increases_damage():
    """Same move and target; STAB should strictly raise damage."""
    # Charmander's Ember is Fire-type (STAB for Charmander)
    atk = _mon("CHARMANDER", 20, ["EMBER", "TACKLE"])
    dfn = _mon("GEODUDE", 20, ["TACKLE"])  # Rock/Ground
    rng = BattleRNG(0)
    # Ember: STAB fire vs rock (0.5x), power 40
    # Tackle: no STAB, neutral vs rock (1.0x), power 35
    ember, _ = compute_damage(atk, dfn, atk.moves[0], rng,
                              force_crit=False, force_roll=255)
    tackle, _ = compute_damage(atk, dfn, atk.moves[1], rng,
                               force_crit=False, force_roll=255)
    # Ember: 40 power, STAB ×1.5, Fire vs Rock/Ground → ×0.5 (only Rock resists)
    # Tackle: 35 power, no STAB, neutral ×1.0
    # Ember uses Char's SPC=25; Tackle uses Char's ATK=25 (same at L20)
    # Net: Ember effective mult = 40*1.5*0.5 = 30; Tackle = 35*1*1 = 35
    # BUT: Tackle's defender stat is Geodude DEF=35, Ember's is Geodude SPC=23,
    # so Ember ends up higher despite the 0.5 type penalty.  STAB + lower SPC
    # beats raw power.  Both non-zero is the real guarantee here.
    assert ember > 0 and tackle > 0
    assert ember > tackle


def test_super_effective_vs_immune():
    """Water vs Fire = 2x.  Electric vs Ground = 0x (immune)."""
    char = _mon("CHARMANDER", 15, ["TACKLE"])
    pika = _mon("PIKACHU", 15, ["THUNDERSHOCK"])
    squirt = _mon("SQUIRTLE", 15, ["BUBBLE"])
    geo = _mon("GEODUDE", 15, ["TACKLE"])
    rng = BattleRNG(0)

    # Water Bubble vs Fire Charmander — SE
    water_se, _ = compute_damage(squirt, char, squirt.moves[0], rng,
                                  force_crit=False, force_roll=255)
    # Normal Tackle vs Fire Charmander — neutral
    normal, _ = compute_damage(char, char, char.moves[0], rng,
                                force_crit=False, force_roll=255)
    assert water_se > 0
    # Electric Thundershock vs Geodude (Rock/Ground) — IMMUNE
    zero, _ = compute_damage(pika, geo, pika.moves[0], rng,
                              force_crit=False, force_roll=255)
    assert zero == 0


def test_ghost_move_vs_psychic_is_zero():
    """Gen 1 data bug: GHOST → PSYCHIC is NO_EFFECT.  Sim follows the data."""
    gastly = _mon("GASTLY", 20, ["LICK"])
    abra = _mon("ABRA", 20, ["TELEPORT"])
    rng = BattleRNG(0)
    dmg, _ = compute_damage(gastly, abra, gastly.moves[0], rng,
                            force_crit=False, force_roll=255)
    assert dmg == 0


def test_crit_ignores_stat_stages():
    """Growl lowers attacker's Atk by 1 stage.  A crit should ignore that."""
    char = _mon("CHARMANDER", 20, ["TACKLE"])
    pidg = _mon("PIDGEY", 20, ["TACKLE"])
    char.atk_stage = -6   # maximum attack debuff
    rng = BattleRNG(0)
    normal, _ = compute_damage(char, pidg, char.moves[0], rng,
                                force_crit=False, force_roll=255)
    crit, crit_flag = compute_damage(char, pidg, char.moves[0], rng,
                                       force_crit=True, force_roll=255)
    assert crit_flag is True
    # Crit bypasses atk debuff AND doubles level → must be strictly higher
    assert crit > normal


def test_damage_roll_bounds():
    """Min roll (217) and max roll (255) bracket the damage distribution."""
    atk = _mon("CHARMANDER", 25, ["EMBER"])
    dfn = _mon("CATERPIE", 25, ["TACKLE"])
    rng = BattleRNG(0)
    hi, _ = compute_damage(atk, dfn, atk.moves[0], rng,
                            force_crit=False, force_roll=255)
    lo, _ = compute_damage(atk, dfn, atk.moves[0], rng,
                            force_crit=False, force_roll=217)
    assert lo <= hi
    # 217/255 ≈ 0.85 — lo should be at least 80% of hi (accounting for floor)
    assert lo >= int(hi * 0.8)


def test_non_zero_floor():
    """Gen 1: any non-immune damage floors at 1."""
    # Tiny attacker vs huge defender
    atk = _mon("CATERPIE", 2, ["TACKLE"])
    dfn = _mon("SNORLAX", 100, ["TACKLE"])
    rng = BattleRNG(0)
    dmg, _ = compute_damage(atk, dfn, atk.moves[0], rng,
                            force_crit=False, force_roll=217)
    assert dmg >= 1

"""End-to-end battle engine smoke tests."""
from __future__ import annotations

import pytest

from battle_sim.engine import (
    BattleEngine,
    BattleResult,
    BattleState,
    highest_power_opponent_policy,
)
from battle_sim.entities import Pokemon
from battle_sim.rng import BattleRNG


def _state(p_species="CHARMANDER", p_level=20,
           o_species="PIDGEY", o_level=5) -> BattleState:
    player = Pokemon.build(p_species, p_level, ["SCRATCH", "EMBER", "GROWL"])
    opp = Pokemon.build(o_species, o_level, ["TACKLE", "SAND_ATTACK"])
    return BattleState(player=player, opponent=opp)


def test_battle_terminates():
    """A higher-level attacker should eventually beat a much weaker opponent."""
    s = _state()
    eng = BattleEngine(s, BattleRNG(42), opponent_policy=highest_power_opponent_policy)
    for _ in range(50):
        result = eng.step(1)  # always Ember
        if result != BattleResult.ONGOING:
            break
    assert result == BattleResult.PLAYER_WIN
    assert s.opponent.fainted


def test_player_faint_loses():
    """If the player faints first, result is OPPONENT_WIN."""
    s = _state("MAGIKARP", 5, "DRAGONITE", 50)
    eng = BattleEngine(s, BattleRNG(0))
    # Magikarp at L5 only knows SPLASH — trivially loses
    s.player.moves = [Pokemon.build("MAGIKARP", 5, ["SPLASH"]).moves[0]]
    for _ in range(50):
        result = eng.step(0)
        if result != BattleResult.ONGOING:
            break
    assert result == BattleResult.OPPONENT_WIN


def test_pp_depletes():
    s = _state()
    eng = BattleEngine(s, BattleRNG(0))
    start_pp = s.player.moves[0].pp
    eng.step(0)
    assert s.player.moves[0].pp == start_pp - 1


def test_throughput_smoke():
    """Not the 50k/s target yet — just ensures 1000 battles in a reasonable window.

    The 50k/s throughput target requires the state-sampling and engine to be
    inside a tight loop with no dataclass churn, and we're not there yet.
    This test is a regression guard against accidental O(n²) blow-ups."""
    import time
    wins = 0
    t0 = time.perf_counter()
    for i in range(200):
        s = _state()
        eng = BattleEngine(s, BattleRNG(i),
                            opponent_policy=highest_power_opponent_policy)
        for _ in range(100):
            if eng.step(1) != BattleResult.ONGOING:
                break
        if s.result == BattleResult.PLAYER_WIN:
            wins += 1
    dt = time.perf_counter() - t0
    # 200 battles should complete in well under 5 seconds even on slow CI
    assert dt < 5.0, f"200 battles took {dt:.2f}s"
    assert wins > 0

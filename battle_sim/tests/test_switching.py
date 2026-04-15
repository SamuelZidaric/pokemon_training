"""v0.4 switching + invalid-action rejection tests.

Covers:
- Valid switch swaps active, resets volatile status, logs the event.
- Switch to fainted bench slot → rejected, penalty applied, opp still acts.
- Switch to out-of-range bench slot → rejected.
- Forced auto-switch on faint picks lowest-index living party member.
- Bench display-order action semantics (action 4 = k-th non-active mon).
- Invalid-move-slot (0-PP or out-of-range) path also triggers penalty.

See TRANSFER_CONTRACT.md §3 for the full rejection protocol.
"""
from __future__ import annotations

import pytest

from battle_sim.engine import BattleEngine, BattleResult, BattleState
from battle_sim.entities import Pokemon
from battle_sim.env import PokemonBattleEnv
from battle_sim.rng import BattleRNG
from battle_sim.v2_contract import INVALID_ACTION_PENALTY


def _dummy_state(party_size: int = 3, opp_size: int = 1) -> BattleState:
    pp = [
        Pokemon.build("CHARMANDER", 10, ["SCRATCH", "GROWL"]),
        Pokemon.build("SQUIRTLE",   12, ["TACKLE", "BUBBLE"]),
        Pokemon.build("BULBASAUR",  11, ["TACKLE", "VINE_WHIP"]),
    ][:party_size]
    op = [Pokemon.build("PIDGEY", 8, ["TACKLE", "SAND_ATTACK"])] * opp_size
    # Duplicate-reference guard: rebuild opp to avoid shared state
    op = [Pokemon.build("PIDGEY", 8, ["TACKLE", "SAND_ATTACK"]) for _ in range(opp_size)]
    return BattleState(player_party=pp, opp_party=op, battle_type=1)


def _opp_noop(state, rng):
    """Opp policy that always picks move slot 0 (deterministic)."""
    return 0


def test_valid_switch_changes_active():
    s = _dummy_state(party_size=3)
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    # action 4 = bench display pos 0 = party[1] (Squirtle)
    eng.step(4)
    assert s.player_active == 1
    assert not s.last_action_was_invalid
    assert any("switched" in e for e in s.last_events)


def test_switch_action_4_selects_bench_display_position_0():
    """When active=0, bench display = [1, 2]; action 4 → party[1]."""
    s = _dummy_state(party_size=3)
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(4)
    assert s.player_active == 1


def test_switch_action_5_selects_bench_display_position_1():
    """When active=0, bench display = [1, 2]; action 5 → party[2]."""
    s = _dummy_state(party_size=3)
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(5)
    assert s.player_active == 2


def test_switch_back_to_slot_0_works_after_leaving():
    """The display-order mapping supports switching back to party[0]."""
    s = _dummy_state(party_size=3)
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(4)   # active 0 → 1
    assert s.player_active == 1
    # Now active=1, bench display = [party[0], party[2]].  action 4 → party[0].
    eng.step(4)
    assert s.player_active == 0


def test_switch_to_fainted_is_rejected_and_penalized():
    s = _dummy_state(party_size=3)
    s.player_party[1].hp = 0  # Squirtle fainted
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    # action 4 = bench display pos 0 = party[1] (the fainted one)
    eng.step(4)
    assert s.player_active == 0  # no switch happened
    assert s.last_action_was_invalid
    assert any("fainted" in e.lower() for e in s.last_events)


def test_switch_out_of_range_bench_is_rejected():
    s = _dummy_state(party_size=2)  # bench has only 1 slot (pos 0)
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    # action 5 = bench pos 1, but bench has only pos 0.
    eng.step(5)
    assert s.player_active == 0
    assert s.last_action_was_invalid


def test_invalid_move_slot_is_rejected_and_penalized():
    """Move slot past end of moveset → rejected with same penalty path."""
    s = _dummy_state(party_size=1)
    # Charmander has 2 moves (slots 0, 1).  Action 2 → no slot 2.
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(2)
    assert s.last_action_was_invalid
    # Turn still advanced (opponent acted).
    assert s.turn == 1


def test_invalid_move_zero_pp_is_rejected():
    s = _dummy_state(party_size=1)
    s.player_party[0].moves[0].pp = 0
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(0)
    assert s.last_action_was_invalid


def test_switch_clears_volatile_status():
    """Switching out clears confusion and flinch on the outgoing mon."""
    s = _dummy_state(party_size=2)
    s.player_party[0].confusion_turns = 3
    s.player_party[0].flinched = True
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(4)
    # Outgoing mon (party[0]) should have its volatiles cleared.
    assert s.player_party[0].confusion_turns == 0
    assert s.player_party[0].flinched is False


def test_forced_switch_on_faint_picks_lowest_living():
    """When active KOs, lowest-index non-fainted member auto-takes the field."""
    s = _dummy_state(party_size=3)
    # Set up a situation where after opp's move, player's active faints.
    s.player_party[0].hp = 1
    s.player_party[1].hp = 0   # fainted — should be skipped
    # party[2] is healthy; will be auto-sent.
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    # Player uses move slot 0.  Opponent (Pidgey L8) hits with TACKLE against
    # a 1-hp Charmander — will faint.
    eng.step(0)
    # Active should have been force-switched to party[2], skipping fainted [1].
    assert s.player_active == 2
    # Result not resolved yet because party[2] is alive.
    assert s.result == BattleResult.ONGOING


def test_battle_ends_when_all_party_fainted():
    s = _dummy_state(party_size=2)
    s.player_party[0].hp = 1
    s.player_party[1].hp = 0
    eng = BattleEngine(s, BattleRNG(0), opponent_policy=_opp_noop)
    eng.step(0)
    assert s.result == BattleResult.OPPONENT_WIN


def test_env_invalid_action_penalty_applied():
    """The env wraps engine rejection into a -0.05 reward delta."""
    # Build a deterministic single-mon scenario with a 0-PP move.
    def sampler(rng):
        p = Pokemon.build("CHARMANDER", 10, ["SCRATCH"])
        p.moves[0].pp = 0
        o = Pokemon.build("PIDGEY", 8, ["TACKLE"])
        return [p], [o]
    env = PokemonBattleEnv(sample_teams=sampler, opponent_policy=_opp_noop, seed=0)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(0)
    assert info["invalid_action"] is True
    # Reward should include at least the invalid-action penalty.
    assert reward <= INVALID_ACTION_PENALTY + 0.0


def test_valid_action_does_not_apply_invalid_penalty():
    env = PokemonBattleEnv(seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["invalid_action"] is False

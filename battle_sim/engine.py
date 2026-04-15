"""1v1 battle state machine (v0.1).

Out of scope for v0.1: status conditions, switching, items, multi-hit moves,
fixed-damage moves, confusion, Substitute.  These are v0.2+.

Action space (semantic, Discrete(9)):
    0..3   — use move at that slot
    4..8   — switch to party slot 1..5 (no-op in 1v1)
    (Run is absent; sim is gated to non-escapable contexts.)

The engine is the only module that advances turn state.  obs.py reads it,
env.py wraps it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .damage import accuracy_check, compute_damage
from .entities import Pokemon
from .rng import BattleRNG


class BattleResult(IntEnum):
    ONGOING = 0
    PLAYER_WIN = 1
    OPPONENT_WIN = 2


@dataclass
class BattleState:
    player: Pokemon
    opponent: Pokemon
    turn: int = 0
    result: BattleResult = BattleResult.ONGOING
    # One trainer-ish variant flag for obs idx 1 (0=none, 1=wild, 2=trainer)
    battle_type: int = 1
    # Last-turn log for debugging / tests
    last_events: list[str] = field(default_factory=list)


class BattleEngine:
    """Advance a 1v1 battle one player-action at a time.

    The opponent policy is injected (callable taking BattleState →
    action int).  Default is ``random_opponent_policy``.
    """

    def __init__(
        self,
        state: BattleState,
        rng: BattleRNG,
        opponent_policy=None,
    ) -> None:
        self.state = state
        self.rng = rng
        self.opponent_policy = opponent_policy or random_opponent_policy

    # -- action handlers ---------------------------------------------------

    def _use_move(
        self, attacker: Pokemon, defender: Pokemon, move_slot: int
    ) -> list[str]:
        """Attempt to use a move; returns event log lines."""
        log: list[str] = []
        if move_slot < 0 or move_slot >= len(attacker.moves):
            log.append(f"{attacker.species} has no move in slot {move_slot}")
            return log
        move = attacker.moves[move_slot]
        if move.pp <= 0:
            log.append(f"{attacker.species}'s {move.name} has no PP")
            return log
        move.pp -= 1

        if not accuracy_check(attacker, defender, move, self.rng):
            log.append(f"{attacker.species}'s {move.name} missed")
            return log

        dmg, crit = compute_damage(attacker, defender, move, self.rng)
        defender.hp = max(0, defender.hp - dmg)
        tag = " (crit)" if crit else ""
        log.append(f"{attacker.species} used {move.name}: {dmg} dmg{tag} → "
                   f"{defender.species} HP {defender.hp}/{defender.max_hp}")
        return log

    # -- turn execution ----------------------------------------------------

    def step(self, player_action: int) -> BattleResult:
        """Execute one turn given a player action (0..8).

        Returns the updated BattleResult.
        """
        if self.state.result != BattleResult.ONGOING:
            return self.state.result

        s = self.state
        s.last_events = []
        opp_action = self.opponent_policy(s, self.rng)

        # v0.1: only move actions (0..3) are meaningful; switches are no-ops
        player_move = player_action if 0 <= player_action <= 3 else None
        opp_move = opp_action if 0 <= opp_action <= 3 else None

        # Determine turn order — Gen 1 uses effective speed; ties broken by
        # the engine's internal coin flip.
        p_spd = s.player.spd
        o_spd = s.opponent.spd
        if p_spd > o_spd:
            first, second = "player", "opponent"
        elif p_spd < o_spd:
            first, second = "opponent", "player"
        else:
            first, second = ("player", "opponent") if self.rng.next_byte() < 128 \
                            else ("opponent", "player")

        for actor in (first, second):
            if s.player.fainted or s.opponent.fainted:
                break
            if actor == "player" and player_move is not None:
                s.last_events.extend(
                    self._use_move(s.player, s.opponent, player_move)
                )
            elif actor == "opponent" and opp_move is not None:
                s.last_events.extend(
                    self._use_move(s.opponent, s.player, opp_move)
                )

        # Resolve result
        if s.opponent.fainted and s.player.fainted:
            s.result = BattleResult.OPPONENT_WIN  # pessimistic; Gen 1 edge case
        elif s.opponent.fainted:
            s.result = BattleResult.PLAYER_WIN
        elif s.player.fainted:
            s.result = BattleResult.OPPONENT_WIN

        s.turn += 1
        return s.result


# ---------------------------------------------------------------------------
# Built-in opponent policies
# ---------------------------------------------------------------------------

def random_opponent_policy(state: BattleState, rng: BattleRNG) -> int:
    """Pick a uniformly random move with PP > 0.  Falls back to slot 0."""
    opp = state.opponent
    valid = [i for i, m in enumerate(opp.moves) if m.pp > 0]
    if not valid:
        return 0
    return valid[rng.next_byte() % len(valid)]


def highest_power_opponent_policy(state: BattleState, rng: BattleRNG) -> int:
    """Always pick highest-power available move (greedy baseline)."""
    opp = state.opponent
    best_i, best_pow = 0, -1
    for i, m in enumerate(opp.moves):
        if m.pp > 0 and m.power > best_pow:
            best_i, best_pow = i, m.power
    return best_i

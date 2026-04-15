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
from .effects import (
    apply_move_effect,
    check_action_allowed,
    end_of_turn_residuals,
    status_spd_multiplier,
)
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

        hit = accuracy_check(attacker, defender, move, self.rng)
        if not hit:
            log.append(f"{attacker.species}'s {move.name} missed")
            # Miss still triggers the effect dispatch with move_hit=False so
            # that status moves don't fire on whiffs — apply_move_effect
            # short-circuits if move_hit is False.
            log.extend(apply_move_effect(
                move.effect, attacker, defender, self.rng,
                move_hit=False, did_damage=False,
            ))
            return log

        dmg, crit = compute_damage(attacker, defender, move, self.rng)
        defender.hp = max(0, defender.hp - dmg)

        # v0.3: Fire-type damaging hits thaw a frozen defender before the
        # rest of the turn resolves.  pokered: any Fire move that hits
        # clears FRZ (even 0-damage if it still landed, but we gate on
        # power > 0 since status Fire moves don't exist in Gen 1).
        FIRE_TYPE_ID = 0x14
        if (defender.status == "FRZ"
                and move.type_id == FIRE_TYPE_ID and move.power > 0):
            defender.status = "OK"
            log.append(f"{defender.species} thawed out")
        tag = " (crit)" if crit else ""
        if dmg > 0:
            log.append(f"{attacker.species} used {move.name}: {dmg} dmg{tag} → "
                       f"{defender.species} HP {defender.hp}/{defender.max_hp}")
        else:
            log.append(f"{attacker.species} used {move.name}")

        log.extend(apply_move_effect(
            move.effect, attacker, defender, self.rng,
            move_hit=True, did_damage=(dmg > 0),
        ))
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
        # the engine's internal coin flip.  Paralysis quarters speed.
        p_n, p_d = status_spd_multiplier(s.player)
        o_n, o_d = status_spd_multiplier(s.opponent)
        p_spd = s.player.spd * p_n // p_d
        o_spd = s.opponent.spd * o_n // o_d
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
            actor_mon = s.player if actor == "player" else s.opponent
            defender_mon = s.opponent if actor == "player" else s.player
            move_slot = player_move if actor == "player" else opp_move

            # Turn-start gating — sleep/freeze/paralysis/flinch/confusion.
            # If blocked, skip the move entirely (no PP consumed).
            can_act, gate_log = check_action_allowed(actor_mon, self.rng)
            s.last_events.extend(gate_log)
            if not can_act:
                continue

            if move_slot is not None:
                s.last_events.extend(
                    self._use_move(actor_mon, defender_mon, move_slot)
                )

        # End-of-turn residuals (burn/poison tick) — only if both survived
        if not s.player.fainted:
            s.last_events.extend(end_of_turn_residuals(s.player))
        if not s.opponent.fainted:
            s.last_events.extend(end_of_turn_residuals(s.opponent))

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

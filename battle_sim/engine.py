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
    """Battle state for 6v6 (v0.4) with backward-compat constructor.

    v0.4 semantics:
    - `player_party` and `opp_party` are lists of 1..6 Pokemon in team order.
    - `player_active` / `opp_active` index into their parties.
    - `player` / `opponent` are @property accessors returning the active mon
      so v0.1-era code keeps working.  Direct `player=`/`opponent=` kwargs
      are still accepted and construct a 1-mon party internally.
    """
    player_party: list[Pokemon] = field(default_factory=list)
    opp_party: list[Pokemon] = field(default_factory=list)
    player_active: int = 0
    opp_active: int = 0
    turn: int = 0
    result: BattleResult = BattleResult.ONGOING
    # One trainer-ish variant flag for obs idx 1 (0=none, 1=wild, 2=trainer)
    battle_type: int = 1
    # Last-turn log for debugging / tests
    last_events: list[str] = field(default_factory=list)
    # v0.4: set by engine when the atomic action was rejected (faint switch,
    # 0-PP move, out-of-range slot).  Env reads this to apply -0.05 penalty.
    last_action_was_invalid: bool = False

    def __init__(
        self,
        player: Pokemon | None = None,
        opponent: Pokemon | None = None,
        player_party: list[Pokemon] | None = None,
        opp_party: list[Pokemon] | None = None,
        player_active: int = 0,
        opp_active: int = 0,
        turn: int = 0,
        result: BattleResult = BattleResult.ONGOING,
        battle_type: int = 1,
        last_events: list[str] | None = None,
        last_action_was_invalid: bool = False,
    ) -> None:
        # Accept either player= (single mon, legacy) or player_party= (v0.4).
        if player_party is None:
            if player is None:
                raise ValueError("BattleState needs player_party or player")
            player_party = [player]
        if opp_party is None:
            if opponent is None:
                raise ValueError("BattleState needs opp_party or opponent")
            opp_party = [opponent]
        self.player_party = player_party
        self.opp_party = opp_party
        self.player_active = player_active
        self.opp_active = opp_active
        self.turn = turn
        self.result = result
        self.battle_type = battle_type
        self.last_events = last_events if last_events is not None else []
        self.last_action_was_invalid = last_action_was_invalid

    @property
    def player(self) -> Pokemon:
        return self.player_party[self.player_active]

    @property
    def opponent(self) -> Pokemon:
        return self.opp_party[self.opp_active]


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

    # -- switch logic (v0.4) ----------------------------------------------

    def _bench_display_order(self, side: str) -> list[int]:
        """Return list of party indices that are NOT currently active, in
        original-team order.  Matches obs.py's Block C bench ordering so the
        policy sees the same positions it controls via action indices.
        """
        party = self.state.player_party if side == "player" else self.state.opp_party
        active = self.state.player_active if side == "player" else self.state.opp_active
        return [i for i in range(len(party)) if i != active]

    def _resolve_bench_target(self, side: str, bench_pos: int) -> tuple[int | None, str]:
        """Map bench_pos (0..4 from action space) to a real party index.
        Returns (party_index, reason); party_index is None if invalid.
        """
        bench = self._bench_display_order(side)
        if bench_pos < 0 or bench_pos >= len(bench):
            return None, "out-of-range"
        party_idx = bench[bench_pos]
        party = self.state.player_party if side == "player" else self.state.opp_party
        if party[party_idx].fainted:
            return None, "fainted"
        return party_idx, "ok"

    def _validate_switch(self, side: str, party_idx: int) -> tuple[bool, str]:
        """Validate a resolved party index (post _resolve_bench_target)."""
        party = self.state.player_party if side == "player" else self.state.opp_party
        active = self.state.player_active if side == "player" else self.state.opp_active
        if party_idx < 0 or party_idx >= len(party):
            return False, "out-of-range"
        if party_idx == active:
            return False, "already-active"
        if party[party_idx].fainted:
            return False, "fainted"
        return True, "ok"

    def _apply_switch(self, side: str, target_slot: int) -> list[str]:
        """Perform the switch; caller must validate first.  Resets volatile
        status on the outgoing mon (confusion, flinch) per Gen 1."""
        party = self.state.player_party if side == "player" else self.state.opp_party
        active = self.state.player_active if side == "player" else self.state.opp_active
        outgoing = party[active]
        outgoing.confusion_turns = 0
        outgoing.flinched = False
        if side == "player":
            self.state.player_active = target_slot
        else:
            self.state.opp_active = target_slot
        return [f"{side} switched {outgoing.species} → {party[target_slot].species}"]

    def _force_switch_if_fainted(self, side: str) -> list[str]:
        """If the side's active mon has fainted, auto-send the lowest-index
        non-fainted party member.  Returns [] if no switch needed or no
        living member remains."""
        party = self.state.player_party if side == "player" else self.state.opp_party
        active = self.state.player_active if side == "player" else self.state.opp_active
        if not party[active].fainted:
            return []
        for i, mon in enumerate(party):
            if not mon.fainted:
                if side == "player":
                    self.state.player_active = i
                else:
                    self.state.opp_active = i
                return [f"{side} auto-sent {mon.species}"]
        return []

    def _side_wiped(self, side: str) -> bool:
        party = self.state.player_party if side == "player" else self.state.opp_party
        return all(m.fainted for m in party)

    # -- turn execution ----------------------------------------------------

    def step(self, player_action: int) -> BattleResult:
        """Execute one turn given a player action (0..8).

        v0.4: switches (4..8) are handled BEFORE any move.  Invalid atomic
        actions (out-of-range, fainted-target, already-active, 0-PP move)
        set ``state.last_action_was_invalid`` and burn the player's turn
        — the opponent still acts, applying the implicit "wasted turn" cost,
        plus env layers a -0.05 penalty via INVALID_ACTION_PENALTY.

        Returns the updated BattleResult.
        """
        s = self.state
        if s.result != BattleResult.ONGOING:
            return s.result

        s.last_events = []
        s.last_action_was_invalid = False
        opp_action = self.opponent_policy(s, self.rng)

        # --- classify player action --------------------------------------
        player_switch_slot: int | None = None
        player_move: int | None = None
        if 0 <= player_action <= 3:
            # Validate move slot: must exist and have PP.
            mv = s.player.moves
            if player_action >= len(mv) or mv[player_action].pp <= 0:
                s.last_action_was_invalid = True
                s.last_events.append(
                    f"player: invalid move slot {player_action} (rejected)"
                )
            else:
                player_move = player_action
        elif 4 <= player_action <= 8:
            bench_pos = player_action - 4  # action 4 → bench[0], ..., 8 → bench[4]
            party_idx, reason = self._resolve_bench_target("player", bench_pos)
            if party_idx is None:
                s.last_action_was_invalid = True
                s.last_events.append(
                    f"player: invalid switch bench_pos {bench_pos} ({reason}, rejected)"
                )
            else:
                player_switch_slot = party_idx
        else:
            s.last_action_was_invalid = True
            s.last_events.append(f"player: action {player_action} out of range")

        # Opp action: sim opp policies only emit 0..3 today, but validate.
        opp_switch_slot: int | None = None
        opp_move: int | None = None
        if 0 <= opp_action <= 3:
            if opp_action < len(s.opponent.moves) and s.opponent.moves[opp_action].pp > 0:
                opp_move = opp_action
        elif 4 <= opp_action <= 8:
            bench_pos = opp_action - 4
            party_idx, _ = self._resolve_bench_target("opponent", bench_pos)
            if party_idx is not None:
                opp_switch_slot = party_idx

        # --- Phase 1: switches resolve first, both sides ------------------
        # In Gen 1 switches always precede attacks; speed order only applies
        # within the attack phase.
        if player_switch_slot is not None:
            s.last_events.extend(self._apply_switch("player", player_switch_slot))
        if opp_switch_slot is not None:
            s.last_events.extend(self._apply_switch("opponent", opp_switch_slot))

        # --- Phase 2: attacks resolve in speed order ---------------------
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
            if move_slot is None:
                continue

            # Turn-start gating — sleep/freeze/paralysis/flinch/confusion.
            can_act, gate_log = check_action_allowed(actor_mon, self.rng)
            s.last_events.extend(gate_log)
            if not can_act:
                continue

            s.last_events.extend(
                self._use_move(actor_mon, defender_mon, move_slot)
            )

        # --- End-of-turn residuals -------------------------------------
        if not s.player.fainted:
            s.last_events.extend(end_of_turn_residuals(s.player))
        if not s.opponent.fainted:
            s.last_events.extend(end_of_turn_residuals(s.opponent))

        # --- Forced switch-on-faint (auto-send lowest living) -----------
        s.last_events.extend(self._force_switch_if_fainted("player"))
        s.last_events.extend(self._force_switch_if_fainted("opponent"))

        # --- Resolve result --------------------------------------------
        p_wiped = self._side_wiped("player")
        o_wiped = self._side_wiped("opponent")
        if p_wiped and o_wiped:
            s.result = BattleResult.OPPONENT_WIN  # pessimistic edge case
        elif o_wiped:
            s.result = BattleResult.PLAYER_WIN
        elif p_wiped:
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

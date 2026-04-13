"""Episode milestone tracker for Pokemon Red RL.

Tracks the step at which key game milestones are first achieved
within an episode.  This data feeds into TensorBoard / W&B so you
can measure *when* the agent reaches each milestone, not just *if*.

Usage::

    tracker = MilestoneTracker()

    # each step:
    tracker.update(step_count, game_state)

    # at episode end:
    tracker.summary()  # {"first_pokemon_step": 120, "rival_1_step": None, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState


@dataclass
class Milestone:
    """A single trackable milestone."""

    name: str
    description: str
    achieved_at_step: Optional[int] = None

    @property
    def achieved(self) -> bool:
        return self.achieved_at_step is not None

    def mark(self, step: int) -> bool:
        """Mark as achieved. Returns True if this is the first time."""
        if self.achieved:
            return False
        self.achieved_at_step = step
        return True


class MilestoneTracker:
    """Tracks game progress milestones within an episode."""

    def __init__(self) -> None:
        self._milestones: dict[str, Milestone] = {}
        self._prev_badge_count: int = 0
        self._prev_party_size: int = 0
        self._prev_map_id: int = -1
        self._battles_won: int = 0
        self._prev_in_battle: bool = False
        self._prev_opponent_hp: float = 1.0

        # Register default milestones
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            ("got_starter", "Obtained starter Pokemon"),
            ("first_battle_won", "Won first battle"),
            ("caught_pokemon", "Caught a wild Pokemon (party size > 1)"),
            ("reached_viridian", "Reached Viridian City (map 1)"),
            ("reached_pewter", "Reached Pewter City (map 2)"),
            ("badge_1", "Earned Boulder Badge"),
            ("reached_cerulean", "Reached Cerulean City (map 3)"),
            ("badge_2", "Earned Cascade Badge"),
            ("reached_vermilion", "Reached Vermilion City (map 5)"),
            ("badge_3", "Earned Thunder Badge"),
            ("party_full", "Party has 6 Pokemon"),
            ("first_evolution", "A Pokemon evolved (level jump detected)"),
            ("badge_4", "Earned Rainbow Badge"),
            ("badge_5", "Earned Soul Badge"),
            ("badge_6", "Earned Marsh Badge"),
            ("badge_7", "Earned Volcano Badge"),
            ("badge_8", "Earned Earth Badge"),
        ]
        for name, desc in defaults:
            self._milestones[name] = Milestone(name=name, description=desc)

    def reset(self) -> None:
        """Reset all milestones for a new episode."""
        for m in self._milestones.values():
            m.achieved_at_step = None
        self._prev_badge_count = 0
        self._prev_party_size = 0
        self._prev_map_id = -1
        self._battles_won = 0
        self._prev_in_battle = False
        self._prev_opponent_hp = 1.0

    def update(self, step: int, game: GameState) -> list[str]:
        """Check all milestones against current game state.

        Returns list of milestone names newly achieved this step.
        """
        newly_achieved = []

        # Party size milestones
        party_size = game.party_size
        if party_size >= 1 and self._prev_party_size == 0:
            if self._mark("got_starter", step):
                newly_achieved.append("got_starter")
        if party_size > 1 and self._prev_party_size <= 1:
            if self._mark("caught_pokemon", step):
                newly_achieved.append("caught_pokemon")
        if party_size >= 6:
            if self._mark("party_full", step):
                newly_achieved.append("party_full")
        self._prev_party_size = party_size

        # Battle won detection
        in_battle = game.in_battle
        if self._prev_in_battle and not in_battle and self._prev_opponent_hp <= 0:
            self._battles_won += 1
            if self._battles_won == 1:
                if self._mark("first_battle_won", step):
                    newly_achieved.append("first_battle_won")
        self._prev_in_battle = in_battle
        self._prev_opponent_hp = game.opponent_hp_fraction if in_battle else 1.0

        # Badge milestones
        badge_count = game.badge_count
        badge_map = {
            1: "badge_1", 2: "badge_2", 3: "badge_3", 4: "badge_4",
            5: "badge_5", 6: "badge_6", 7: "badge_7", 8: "badge_8",
        }
        if badge_count > self._prev_badge_count:
            for b in range(self._prev_badge_count + 1, badge_count + 1):
                if b in badge_map:
                    if self._mark(badge_map[b], step):
                        newly_achieved.append(badge_map[b])
        self._prev_badge_count = badge_count

        # Map-based milestones
        map_id = game.map_id
        map_milestones = {
            1: "reached_viridian",
            2: "reached_pewter",
            3: "reached_cerulean",
            5: "reached_vermilion",
        }
        if map_id != self._prev_map_id and map_id in map_milestones:
            if self._mark(map_milestones[map_id], step):
                newly_achieved.append(map_milestones[map_id])
        self._prev_map_id = map_id

        return newly_achieved

    def _mark(self, name: str, step: int) -> bool:
        if name in self._milestones:
            return self._milestones[name].mark(step)
        return False

    def summary(self) -> dict[str, Optional[int]]:
        """Return {milestone_name: step_achieved_or_None}."""
        return {
            name: m.achieved_at_step
            for name, m in self._milestones.items()
        }

    def achieved_milestones(self) -> dict[str, int]:
        """Return only milestones that were achieved, with their steps."""
        return {
            name: m.achieved_at_step
            for name, m in self._milestones.items()
            if m.achieved
        }

    def achievement_count(self) -> int:
        return sum(1 for m in self._milestones.values() if m.achieved)

"""Curriculum learning via save state rotation.

Instead of always starting from init.state (Pallet Town), the curriculum
system can rotate through progressively harder save states.  This lets
the agent practice late-game mechanics without mastering the early game
first every episode.

Usage in training script::

    from curriculum import CurriculumScheduler

    scheduler = CurriculumScheduler([
        CurriculumStage("early",  "../init.state",              weight=1.0),
        CurriculumStage("brock",  "../has_pokedex_nballs.state", weight=0.5),
        CurriculumStage("mtmoon", "../post_brock.state",         weight=0.3),
    ])

    # In make_env:
    env_config["init_state"] = scheduler.sample()

    # After each training iteration:
    scheduler.update(mean_reward)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CurriculumStage:
    """A single curriculum stage tied to a save state file."""

    name: str
    state_path: str
    weight: float = 1.0
    min_reward_to_unlock: float = 0.0
    unlocked: bool = True

    def exists(self) -> bool:
        return Path(self.state_path).is_file()


class CurriculumScheduler:
    """Samples save states weighted by curriculum stage weights.

    Stages can be gated behind a minimum reward threshold so harder
    stages only unlock once the agent demonstrates basic competence.
    """

    def __init__(
        self,
        stages: list[CurriculumStage],
        auto_advance: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("At least one curriculum stage is required")
        self.stages = stages
        self.auto_advance = auto_advance
        self._best_reward: float = 0.0

    def sample(self) -> str:
        """Return a save state path, weighted by stage weights."""
        available = [s for s in self.stages if s.unlocked and s.exists()]
        if not available:
            # fall back to first stage regardless
            return self.stages[0].state_path

        weights = [s.weight for s in available]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = random.choices(available, weights=weights, k=1)[0]
        return chosen.state_path

    def update(self, mean_reward: float) -> list[str]:
        """Call after each training iteration to potentially unlock stages.

        Returns list of newly unlocked stage names.
        """
        self._best_reward = max(self._best_reward, mean_reward)
        newly_unlocked = []

        if self.auto_advance:
            for stage in self.stages:
                if not stage.unlocked and self._best_reward >= stage.min_reward_to_unlock:
                    if stage.exists():
                        stage.unlocked = True
                        newly_unlocked.append(stage.name)

        return newly_unlocked

    @property
    def unlocked_stages(self) -> list[CurriculumStage]:
        return [s for s in self.stages if s.unlocked]

    @property
    def locked_stages(self) -> list[CurriculumStage]:
        return [s for s in self.stages if not s.unlocked]

    def status(self) -> dict[str, bool]:
        return {s.name: s.unlocked for s in self.stages}


def create_battle_curriculum(gym_state_path: str) -> CurriculumScheduler:
    """Tactical sandbox — loads a single save state repeatedly.

    Used for the "Hyperbolic Time Chamber" approach: the agent fights
    the same gym leader thousands of times in rapid succession.

    Parameters
    ----------
    gym_state_path : str
        Path to a .state file positioned right before a gym battle.
    """
    return CurriculumScheduler([
        CurriculumStage(
            name="gym_battle",
            state_path=gym_state_path,
            weight=1.0,
            unlocked=True,
        ),
    ])


def create_gym_gauntlet_curriculum() -> CurriculumScheduler:
    """Progressive gym gauntlet — trains against each gym in sequence.

    Each gym unlocks when the agent demonstrates competence at the
    previous one (measured by reward threshold).
    """
    return CurriculumScheduler([
        CurriculumStage(
            name="brock",
            state_path="../pre_brock.state",
            weight=1.0,
            min_reward_to_unlock=0,
            unlocked=True,
        ),
        CurriculumStage(
            name="misty",
            state_path="../pre_misty.state",
            weight=1.0,
            min_reward_to_unlock=100,
            unlocked=False,
        ),
        CurriculumStage(
            name="surge",
            state_path="../pre_surge.state",
            weight=1.0,
            min_reward_to_unlock=200,
            unlocked=False,
        ),
        CurriculumStage(
            name="erika",
            state_path="../pre_erika.state",
            weight=1.0,
            min_reward_to_unlock=300,
            unlocked=False,
        ),
        CurriculumStage(
            name="koga",
            state_path="../pre_koga.state",
            weight=1.0,
            min_reward_to_unlock=400,
            unlocked=False,
        ),
        CurriculumStage(
            name="sabrina",
            state_path="../pre_sabrina.state",
            weight=1.0,
            min_reward_to_unlock=500,
            unlocked=False,
        ),
        CurriculumStage(
            name="blaine",
            state_path="../pre_blaine.state",
            weight=1.0,
            min_reward_to_unlock=600,
            unlocked=False,
        ),
        CurriculumStage(
            name="giovanni",
            state_path="../pre_giovanni.state",
            weight=1.0,
            min_reward_to_unlock=700,
            unlocked=False,
        ),
    ])


def create_default_curriculum() -> CurriculumScheduler:
    """A sensible default curriculum for Pokemon Red.

    Stages unlock progressively as the agent improves.
    Paths assume save states are in the parent directory.
    """
    return CurriculumScheduler([
        CurriculumStage(
            name="pallet_town",
            state_path="../init.state",
            weight=1.0,
            min_reward_to_unlock=0,
            unlocked=True,
        ),
        CurriculumStage(
            name="has_pokedex",
            state_path="../has_pokedex.state",
            weight=0.8,
            min_reward_to_unlock=50,
            unlocked=False,
        ),
        CurriculumStage(
            name="has_pokeballs",
            state_path="../has_pokedex_nballs.state",
            weight=0.6,
            min_reward_to_unlock=100,
            unlocked=False,
        ),
    ])

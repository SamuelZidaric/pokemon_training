"""Composable reward functions for Pokemon Red RL.

Each reward function takes a ``RewardContext`` and returns a float.
Functions are registered in a ``RewardSystem`` which computes the
weighted sum each step.  This makes it trivial to add, remove, or
re-weight individual reward signals without touching the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

RewardFn = Callable[["RewardContext"], float]


@dataclass
class RewardContext:
    """Snapshot of everything a reward function might need.

    Populated by the environment at each step so reward functions
    stay decoupled from the env internals.
    """

    # exploration
    seen_coords_count: int = 0
    current_coord_visits: int = 0

    # events / progress
    event_flag_sum: int = 0
    base_event_flags: int = 0
    has_museum_ticket: bool = False

    # party
    hp_fraction: float = 1.0
    last_hp_fraction: float = 1.0
    party_size: int = 0
    last_party_size: int = 0
    levels_sum: int = 0

    # badges
    badge_count: int = 0

    # healing / death tracking (accumulated over episode)
    total_healing_reward: float = 0.0
    died_count: int = 0

    # opponent
    max_opponent_level: int = 0

    # battle state (for battle-aware rewards)
    in_battle: bool = False
    opponent_hp_fraction: float = 1.0
    prev_opponent_hp_fraction: float = 1.0
    battles_won: int = 0

    # PC softlock detection
    is_box_full: bool = False
    pokemon_in_box: int = 0

    # tactical: type effectiveness (set by env when in battle)
    # +1 = super effective lead move exists, 0 = neutral, -1 = resisted
    type_advantage: float = 0.0
    used_super_effective: bool = False

    # survival
    hp_loss_this_step: float = 0.0
    party_fainted_count: int = 0


@dataclass
class RewardComponent:
    """A named, weighted reward function."""

    name: str
    fn: RewardFn
    weight: float = 1.0


class RewardSystem:
    """Registry of weighted reward components.

    Usage::

        rs = RewardSystem()
        rs.add("explore", explore_reward, weight=0.1)
        rs.add("badges", badge_reward, weight=10.0)
        scores = rs.compute(ctx)  # {"explore": 3.2, "badges": 10.0, ...}
    """

    def __init__(self) -> None:
        self._components: list[RewardComponent] = []

    def add(self, name: str, fn: RewardFn, weight: float = 1.0) -> None:
        self._components.append(RewardComponent(name=name, fn=fn, weight=weight))

    def compute(self, ctx: RewardContext) -> dict[str, float]:
        return {c.name: c.weight * c.fn(ctx) for c in self._components}

    @property
    def names(self) -> list[str]:
        return [c.name for c in self._components]


# ---------------------------------------------------------------------------
# Built-in reward functions
# ---------------------------------------------------------------------------

def event_reward(ctx: RewardContext) -> float:
    """Reward for triggering new game event flags."""
    raw = ctx.event_flag_sum - ctx.base_event_flags - int(ctx.has_museum_ticket)
    return float(max(raw, 0))


def explore_reward(ctx: RewardContext) -> float:
    """Reward proportional to unique coordinates visited."""
    return float(ctx.seen_coords_count)


def badge_reward(ctx: RewardContext) -> float:
    """Reward per gym badge earned."""
    return float(ctx.badge_count)


def healing_reward(ctx: RewardContext) -> float:
    """Accumulated reward for healing (set externally per step)."""
    return ctx.total_healing_reward


def stuck_penalty(ctx: RewardContext) -> float:
    """Penalty when lingering too long on the same coordinate."""
    return -1.0 if ctx.current_coord_visits >= 600 else 0.0


def level_reward(ctx: RewardContext) -> float:
    """Scaled reward for party level sum, with diminishing returns past threshold."""
    min_poke_level = 2
    starter_bonus = 4
    raw = max(ctx.levels_sum - min_poke_level * max(ctx.party_size, 1) - starter_bonus, 0)
    threshold = 22
    scale = 4
    if raw < threshold:
        return float(raw)
    return float((raw - threshold) / scale + threshold)


def battle_win_reward(ctx: RewardContext) -> float:
    """Reward for winning battles (accumulated count)."""
    return float(ctx.battles_won)


def opponent_damage_reward(ctx: RewardContext) -> float:
    """Reward for dealing damage to opponent during battle."""
    if not ctx.in_battle:
        return 0.0
    damage_dealt = ctx.prev_opponent_hp_fraction - ctx.opponent_hp_fraction
    return max(damage_dealt, 0.0)


def pc_box_full_penalty(ctx: RewardContext) -> float:
    """Penalty when the current PC box is full (softlock risk)."""
    if ctx.is_box_full:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def create_default_reward_system(
    reward_scale: float = 1.0,
    explore_weight: float = 1.0,
) -> RewardSystem:
    """Replicate the original v2 reward weights."""
    rs = RewardSystem()
    rs.add("event", event_reward, weight=reward_scale * 4)
    rs.add("heal", healing_reward, weight=reward_scale * 10)
    rs.add("badge", badge_reward, weight=reward_scale * 10)
    rs.add("explore", explore_reward, weight=reward_scale * explore_weight * 0.1)
    rs.add("stuck", stuck_penalty, weight=reward_scale * 0.05)
    return rs


def create_enhanced_reward_system(
    reward_scale: float = 1.0,
    explore_weight: float = 1.0,
) -> RewardSystem:
    """Enhanced reward system with battle awareness and softlock prevention."""
    rs = RewardSystem()
    rs.add("event", event_reward, weight=reward_scale * 4)
    rs.add("heal", healing_reward, weight=reward_scale * 10)
    rs.add("badge", badge_reward, weight=reward_scale * 10)
    rs.add("explore", explore_reward, weight=reward_scale * explore_weight * 0.1)
    rs.add("stuck", stuck_penalty, weight=reward_scale * 0.05)
    rs.add("level", level_reward, weight=reward_scale * 1.0)
    rs.add("battle_win", battle_win_reward, weight=reward_scale * 5)
    rs.add("opponent_dmg", opponent_damage_reward, weight=reward_scale * 2)
    rs.add("pc_full", pc_box_full_penalty, weight=reward_scale * 1.0)
    return rs


# ---------------------------------------------------------------------------
# Tactical reward functions
# ---------------------------------------------------------------------------

def hp_loss_penalty(ctx: RewardContext) -> float:
    """Penalty proportional to HP lost this step — teaches survival."""
    return -ctx.hp_loss_this_step


def super_effective_bonus(ctx: RewardContext) -> float:
    """Bonus when a super-effective type matchup is exploited."""
    if not ctx.in_battle:
        return 0.0
    return max(ctx.type_advantage, 0.0)


def party_fainted_penalty(ctx: RewardContext) -> float:
    """Penalty per fainted party member — teaches team preservation."""
    return -float(ctx.party_fainted_count)


def efficiency_reward(ctx: RewardContext) -> float:
    """Reward for winning battles while keeping HP high."""
    if not ctx.in_battle:
        return 0.0
    damage_dealt = ctx.prev_opponent_hp_fraction - ctx.opponent_hp_fraction
    if damage_dealt <= 0:
        return 0.0
    # Bonus scales with how much HP the agent still has
    return max(damage_dealt, 0.0) * ctx.hp_fraction


# ---------------------------------------------------------------------------
# Tactical presets
# ---------------------------------------------------------------------------

def create_battle_focused_system(reward_scale: float = 1.0) -> RewardSystem:
    """Hyperbolic Time Chamber preset — maximises battle performance.

    Designed for use with a save state right before a gym leader.
    Exploration and event rewards are minimised since the agent
    shouldn't be wandering.
    """
    rs = RewardSystem()
    rs.add("battle_win", battle_win_reward, weight=reward_scale * 20)
    rs.add("opponent_dmg", opponent_damage_reward, weight=reward_scale * 10)
    rs.add("super_eff", super_effective_bonus, weight=reward_scale * 5)
    rs.add("efficiency", efficiency_reward, weight=reward_scale * 8)
    rs.add("hp_loss", hp_loss_penalty, weight=reward_scale * 3)
    rs.add("fainted", party_fainted_penalty, weight=reward_scale * 5)
    rs.add("badge", badge_reward, weight=reward_scale * 50)
    rs.add("heal", healing_reward, weight=reward_scale * 5)
    # Minimal exploration — agent should focus on the battle
    rs.add("explore", explore_reward, weight=reward_scale * 0.01)
    return rs


def create_survival_system(reward_scale: float = 1.0) -> RewardSystem:
    """Survival preset — agent learns to preserve HP and use healing.

    Good for teaching the agent to manage resources through
    dungeons like Mt. Moon or Rock Tunnel.
    """
    rs = RewardSystem()
    rs.add("event", event_reward, weight=reward_scale * 4)
    rs.add("explore", explore_reward, weight=reward_scale * 0.1)
    rs.add("heal", healing_reward, weight=reward_scale * 20)
    rs.add("hp_loss", hp_loss_penalty, weight=reward_scale * 8)
    rs.add("fainted", party_fainted_penalty, weight=reward_scale * 10)
    rs.add("battle_win", battle_win_reward, weight=reward_scale * 5)
    rs.add("badge", badge_reward, weight=reward_scale * 10)
    rs.add("pc_full", pc_box_full_penalty, weight=reward_scale * 2)
    return rs


def create_speedrun_system(reward_scale: float = 1.0) -> RewardSystem:
    """Speedrun preset — maximises game progress per step.

    Heavily weights events and badges, penalises lingering.
    Battles are rewarded only insofar as they unlock progress.
    """
    rs = RewardSystem()
    rs.add("event", event_reward, weight=reward_scale * 10)
    rs.add("badge", badge_reward, weight=reward_scale * 30)
    rs.add("explore", explore_reward, weight=reward_scale * 0.15)
    rs.add("stuck", stuck_penalty, weight=reward_scale * 0.2)
    rs.add("battle_win", battle_win_reward, weight=reward_scale * 3)
    rs.add("opponent_dmg", opponent_damage_reward, weight=reward_scale * 1)
    rs.add("efficiency", efficiency_reward, weight=reward_scale * 5)
    rs.add("heal", healing_reward, weight=reward_scale * 5)
    return rs

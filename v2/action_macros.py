"""Action macros for Pokemon Red RL.

Extends the base 7-action space with higher-level macro actions that
execute multiple frames.  This helps the agent clear text boxes and
navigate menus faster without needing to learn "spam A for N frames."

Usage::

    from action_macros import MacroActionWrapper

    env = RedGymEnv(config)
    env = MacroActionWrapper(env, macros=DEFAULT_MACROS)
    # action_space is now Discrete(7 + len(macros))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class ActionMacro:
    """A sequence of base actions to execute as one "macro" action."""

    name: str
    actions: list[int]
    description: str = ""

    @property
    def length(self) -> int:
        return len(self.actions)


# Default macros that help with common Pokemon Red interactions.
# Action indices: 0=Down, 1=Left, 2=Right, 3=Up, 4=A, 5=B, 6=Start
DEFAULT_MACROS = [
    ActionMacro(
        name="spam_a",
        actions=[4] * 5,
        description="Press A 5 times — clears text boxes and confirmations",
    ),
    ActionMacro(
        name="spam_b",
        actions=[5] * 5,
        description="Press B 5 times — exits menus and cancels",
    ),
    ActionMacro(
        name="walk_down_3",
        actions=[0] * 3,
        description="Walk down 3 steps",
    ),
    ActionMacro(
        name="walk_up_3",
        actions=[3] * 3,
        description="Walk up 3 steps",
    ),
    ActionMacro(
        name="walk_left_3",
        actions=[1] * 3,
        description="Walk left 3 steps",
    ),
    ActionMacro(
        name="walk_right_3",
        actions=[2] * 3,
        description="Walk right 3 steps",
    ),
]


class MacroActionWrapper(gym.Wrapper):
    """Extends the action space with macro actions.

    Base actions 0..N-1 pass through directly.
    Actions N..N+M-1 trigger macro sequences that execute multiple
    base actions in a single ``step()`` call.

    The returned observation/reward/done are from the *last* step
    in the macro.  Rewards are summed across all steps.
    """

    def __init__(
        self,
        env: gym.Env,
        macros: Optional[list[ActionMacro]] = None,
    ) -> None:
        super().__init__(env)
        self.macros = macros or DEFAULT_MACROS
        self._base_n = env.action_space.n
        self.action_space = spaces.Discrete(self._base_n + len(self.macros))

    def step(self, action: int):
        if action < self._base_n:
            return self.env.step(action)

        macro_idx = action - self._base_n
        macro = self.macros[macro_idx]

        total_reward = 0.0
        obs = None
        terminated = False
        truncated = False
        info = {}

        for base_action in macro.actions:
            obs, reward, terminated, truncated, info = self.env.step(base_action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

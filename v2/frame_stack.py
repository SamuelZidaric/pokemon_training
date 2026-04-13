"""Dictionary-aware frame stacking wrapper for Gymnasium.

Gymnasium's built-in ``FrameStack`` only works with ``Box`` observation
spaces.  Pokemon Red uses a ``Dict`` observation space with mixed types
(Box, MultiBinary, MultiDiscrete).  This wrapper handles all of them.

Design decisions
----------------
- **Visual keys** (configurable, default ``["screens"]``) get stacked
  along a new trailing axis, giving the CNN temporal context.
- **Scalar / vector keys** (health, level, badges, etc.) get stacked
  into a rolling buffer so the policy sees the last N values.
- **Passthrough keys** (events, map) are large and mostly static — they
  are passed through without stacking to save memory.

Usage::

    from frame_stack import DictFrameStack

    env = RedGymEnv(config)
    env = DictFrameStack(env, n_stack=4)
    # observation_space is updated automatically
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DictFrameStack(gym.Wrapper):
    """Frame-stacking wrapper for Dict observation spaces.

    Parameters
    ----------
    env : gym.Env
        Environment with a ``Dict`` observation space.
    n_stack : int
        Number of frames to stack.
    stack_keys : list[str] | None
        Which observation keys to stack. If None, stacks all ``Box``
        keys. Keys not in this list are passed through as-is.
    passthrough_keys : list[str] | None
        Keys to explicitly pass through without stacking, even if they
        are Box spaces. Useful for large, mostly-static observations
        like the event flags or exploration map.
    """

    def __init__(
        self,
        env: gym.Env,
        n_stack: int = 4,
        stack_keys: Optional[list[str]] = None,
        passthrough_keys: Optional[list[str]] = None,
    ) -> None:
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                f"DictFrameStack requires a Dict observation space, "
                f"got {type(env.observation_space)}"
            )

        self.n_stack = n_stack
        self.passthrough_keys = set(passthrough_keys or [])

        # Determine which keys to stack
        if stack_keys is not None:
            self.stack_keys = set(stack_keys)
        else:
            # Auto-detect: stack all Box spaces that aren't passthrough
            self.stack_keys = {
                k
                for k, v in env.observation_space.spaces.items()
                if isinstance(v, spaces.Box) and k not in self.passthrough_keys
            }

        # Build the new observation space
        new_spaces = {}
        for key, space in env.observation_space.spaces.items():
            if key in self.stack_keys:
                new_spaces[key] = self._expand_space(key, space)
            else:
                new_spaces[key] = space

        self.observation_space = spaces.Dict(new_spaces)

        # Frame buffers — one deque per stacked key
        self._buffers: dict[str, deque] = {}

    def _expand_space(self, key: str, space: spaces.Space) -> spaces.Space:
        """Create the stacked version of a space."""
        if isinstance(space, spaces.Box):
            # Stack along a new trailing axis
            new_shape = (*space.shape, self.n_stack)
            low = np.repeat(
                space.low[..., np.newaxis], self.n_stack, axis=-1
            )
            high = np.repeat(
                space.high[..., np.newaxis], self.n_stack, axis=-1
            )
            return spaces.Box(low=low, high=high, dtype=space.dtype)

        elif isinstance(space, spaces.MultiBinary):
            # Stack into (original_n, n_stack)
            n = space.n if isinstance(space.n, int) else np.prod(space.n)
            return spaces.Box(
                low=0, high=1, shape=(n, self.n_stack), dtype=np.int8
            )

        elif isinstance(space, spaces.MultiDiscrete):
            # Stack into (original_n, n_stack)
            return spaces.Box(
                low=0,
                high=max(space.nvec),
                shape=(len(space.nvec), self.n_stack),
                dtype=np.int64,
            )

        else:
            # Unsupported space type — pass through
            return space

    def _init_buffers(self, obs: dict[str, np.ndarray]) -> None:
        """Initialize frame buffers with copies of the first observation."""
        self._buffers = {}
        for key in self.stack_keys:
            if key in obs:
                buf = deque(maxlen=self.n_stack)
                for _ in range(self.n_stack):
                    buf.append(obs[key].copy())
                self._buffers[key] = buf

    def _stack_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Push new observation into buffers and return stacked result."""
        result = {}
        for key, value in obs.items():
            if key in self.stack_keys and key in self._buffers:
                self._buffers[key].append(value.copy())
                # Stack along a new trailing axis
                result[key] = np.stack(list(self._buffers[key]), axis=-1)
            else:
                result[key] = value
        return result

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._init_buffers(obs)
        return self._stack_obs(obs), info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._stack_obs(obs), reward, terminated, truncated, info

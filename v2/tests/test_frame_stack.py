"""Tests for DictFrameStack wrapper."""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from frame_stack import DictFrameStack


class DictObsEnv(gym.Env):
    """Minimal env with a Dict observation space for testing."""

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8),
            "health": spaces.Box(low=0, high=1, shape=(1,)),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary(16),
            "actions": spaces.MultiDiscrete([4, 4, 4]),
        })
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        self._step += 1
        return self._obs(), 1.0, False, self._step >= 100, {}

    def _obs(self):
        return {
            "screen": np.full((72, 80, 1), self._step % 256, dtype=np.uint8),
            "health": np.array([0.5 + self._step * 0.01]),
            "badges": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "events": np.zeros(16, dtype=np.int8),
            "actions": np.array([1, 2, 3], dtype=np.int64),
        }


class TestDictFrameStackInit:
    def test_requires_dict_space(self):
        env = gym.make("CartPole-v1")
        with pytest.raises(TypeError, match="Dict"):
            DictFrameStack(env)

    def test_default_stacks_all_box_keys(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=4)
        # screen and health are Box — should be stacked
        assert "screen" in wrapped.stack_keys
        assert "health" in wrapped.stack_keys

    def test_custom_stack_keys(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=4, stack_keys=["screen"])
        assert wrapped.stack_keys == {"screen"}

    def test_passthrough_keys(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(
            env, n_stack=4, passthrough_keys=["events"]
        )
        assert "events" not in wrapped.stack_keys


class TestDictFrameStackSpaces:
    def test_box_space_expanded(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=4, stack_keys=["screen"])
        space = wrapped.observation_space["screen"]
        # Original: (72, 80, 1), stacked: (72, 80, 1, 4)
        assert space.shape == (72, 80, 1, 4)
        assert space.dtype == np.uint8

    def test_health_space_expanded(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=3, stack_keys=["health"])
        space = wrapped.observation_space["health"]
        assert space.shape == (1, 3)

    def test_passthrough_space_unchanged(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(
            env, n_stack=4,
            stack_keys=["screen"],
            passthrough_keys=["events"],
        )
        # events should be unchanged
        assert wrapped.observation_space["events"] == env.observation_space["events"]


class TestDictFrameStackBehavior:
    def test_reset_fills_buffer(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=4, stack_keys=["screen"])
        obs, info = wrapped.reset()
        # All 4 frames should be the same (copies of first obs)
        screen = obs["screen"]
        assert screen.shape == (72, 80, 1, 4)
        # All stack slots should be equal after reset
        for i in range(4):
            np.testing.assert_array_equal(screen[..., 0], screen[..., i])

    def test_step_shifts_buffer(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=4, stack_keys=["screen"])
        obs0, _ = wrapped.reset()  # step=0, all frames = 0
        obs1, _, _, _, _ = wrapped.step(0)  # step=1

        screen = obs1["screen"]
        # Oldest 3 frames should be 0, newest should be 1
        assert screen[0, 0, 0, -1] == 1  # most recent
        assert screen[0, 0, 0, 0] == 0   # oldest

    def test_multiple_steps_rolling(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=3, stack_keys=["screen"])
        wrapped.reset()
        # Take 5 steps
        for _ in range(5):
            obs, _, _, _, _ = wrapped.step(0)

        screen = obs["screen"]
        # After 5 steps, buffer should contain steps 3, 4, 5
        assert screen[0, 0, 0, 0] == 3  # oldest
        assert screen[0, 0, 0, 1] == 4
        assert screen[0, 0, 0, 2] == 5  # newest

    def test_health_stacking(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=3, stack_keys=["health"])
        wrapped.reset()

        for _ in range(3):
            obs, _, _, _, _ = wrapped.step(0)

        health = obs["health"]
        assert health.shape == (1, 3)
        # Health values should be different across the stack
        # (0.5 + step*0.01 for steps 1, 2, 3)
        assert health[0, 0] < health[0, 1] < health[0, 2]

    def test_non_stacked_keys_passthrough(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(
            env, n_stack=4,
            stack_keys=["screen"],
            passthrough_keys=["events"],
        )
        obs, _ = wrapped.reset()
        # events should be plain (16,), not stacked
        assert obs["events"].shape == (16,)

    def test_reset_clears_buffer(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=3, stack_keys=["screen"])
        wrapped.reset()
        # Take several steps
        for _ in range(10):
            wrapped.step(0)

        # Reset should clear buffer
        obs, _ = wrapped.reset()
        screen = obs["screen"]
        # All frames should be identical after reset
        for i in range(3):
            np.testing.assert_array_equal(screen[..., 0], screen[..., i])

    def test_n_stack_1_is_identity(self):
        """With n_stack=1, output shape should just add a trailing dim."""
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=1, stack_keys=["screen"])
        obs, _ = wrapped.reset()
        assert obs["screen"].shape == (72, 80, 1, 1)

    def test_large_n_stack(self):
        env = DictObsEnv()
        wrapped = DictFrameStack(env, n_stack=8, stack_keys=["screen"])
        obs, _ = wrapped.reset()
        assert obs["screen"].shape == (72, 80, 1, 8)
        for _ in range(20):
            obs, _, _, _, _ = wrapped.step(0)
        assert obs["screen"].shape == (72, 80, 1, 8)

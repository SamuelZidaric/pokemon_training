"""Tests for action macros."""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from action_macros import ActionMacro, MacroActionWrapper, DEFAULT_MACROS

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FakeEnv(gym.Env):
    """Minimal env for testing macro wrapper."""

    def __init__(self):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,))
        self.actions_received = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.actions_received = []
        self.step_count = 0
        return np.array([0.0]), {}

    def step(self, action):
        self.actions_received.append(action)
        self.step_count += 1
        return np.array([1.0]), 1.0, False, False, {}


class TestActionMacro:
    def test_length(self):
        m = ActionMacro("test", [0, 1, 2])
        assert m.length == 3

    def test_default_macros_exist(self):
        assert len(DEFAULT_MACROS) >= 4
        assert DEFAULT_MACROS[0].name == "spam_a"


class TestMacroActionWrapper:
    def test_base_action_passthrough(self):
        env = FakeEnv()
        wrapped = MacroActionWrapper(env)
        wrapped.reset()
        wrapped.step(4)  # A button
        assert env.actions_received == [4]

    def test_macro_executes_sequence(self):
        env = FakeEnv()
        wrapped = MacroActionWrapper(env)
        wrapped.reset()
        # Action 7 = first macro = spam_a = [4,4,4,4,4]
        wrapped.step(7)
        assert env.actions_received == [4, 4, 4, 4, 4]

    def test_macro_sums_rewards(self):
        env = FakeEnv()
        wrapped = MacroActionWrapper(env)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(7)  # 5 steps * 1.0 each
        assert reward == 5.0

    def test_action_space_expanded(self):
        env = FakeEnv()
        wrapped = MacroActionWrapper(env)
        assert wrapped.action_space.n == 7 + len(DEFAULT_MACROS)

    def test_custom_macros(self):
        env = FakeEnv()
        macros = [ActionMacro("test", [0, 1, 2])]
        wrapped = MacroActionWrapper(env, macros=macros)
        assert wrapped.action_space.n == 8
        wrapped.reset()
        wrapped.step(7)
        assert env.actions_received == [0, 1, 2]

    def test_macro_stops_on_truncated(self):
        class EarlyDoneEnv(FakeEnv):
            def step(self, action):
                self.actions_received.append(action)
                self.step_count += 1
                truncated = self.step_count >= 2
                return np.array([1.0]), 1.0, False, truncated, {}

        env = EarlyDoneEnv()
        wrapped = MacroActionWrapper(env)
        wrapped.reset()
        _, reward, _, truncated, _ = wrapped.step(7)  # spam_a
        assert truncated is True
        assert len(env.actions_received) == 2
        assert reward == 2.0

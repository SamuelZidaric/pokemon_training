"""Smoke test for the Gymnasium env wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from battle_sim.env import PokemonBattleEnv


def test_env_reset_and_step():
    env = PokemonBattleEnv(seed=0)
    obs, info = env.reset(seed=0)
    # Observation space sanity
    for key, space in env.observation_space.spaces.items():
        assert key in obs, f"missing {key}"
        assert obs[key].shape == space.shape, \
            f"{key} shape {obs[key].shape} != {space.shape}"
    obs, reward, term, trunc, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_env_episode_terminates():
    """An env rollout must terminate within max_turns."""
    env = PokemonBattleEnv(seed=0, max_turns=200)
    env.reset(seed=0)
    for _ in range(300):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert term or trunc


def test_env_determinism():
    """Same seed → same first obs and same reward trajectory."""
    env1 = PokemonBattleEnv(seed=7)
    env2 = PokemonBattleEnv(seed=7)
    o1, _ = env1.reset(seed=7)
    o2, _ = env2.reset(seed=7)
    np.testing.assert_array_equal(o1["tactical"], o2["tactical"])

    seq1, seq2 = [], []
    for _ in range(10):
        _, r1, t1, *_ = env1.step(0)
        _, r2, t2, *_ = env2.step(0)
        seq1.append(r1); seq2.append(r2)
        if t1 or t2:
            break
    assert seq1 == seq2

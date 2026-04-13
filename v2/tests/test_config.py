"""Tests for EnvConfig dataclass."""

import pytest
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EnvConfig


class TestEnvConfig:
    def test_defaults(self):
        cfg = EnvConfig()
        assert cfg.headless is True
        assert cfg.action_freq == 24
        assert cfg.max_steps == 2048 * 80
        assert cfg.reward_scale == 1.0
        assert isinstance(cfg.session_path, Path)

    def test_from_dict_filters_unknown_keys(self):
        d = {
            "headless": False,
            "action_freq": 12,
            "unknown_key": "should be ignored",
            "gb_path": "test.gb",
            "session_path": "my_runs",
        }
        cfg = EnvConfig.from_dict(d)
        assert cfg.headless is False
        assert cfg.action_freq == 12
        assert cfg.gb_path == "test.gb"
        assert cfg.session_path == Path("my_runs")

    def test_from_dict_preserves_defaults_for_missing_keys(self):
        cfg = EnvConfig.from_dict({"headless": True})
        assert cfg.max_steps == 2048 * 80
        assert cfg.explore_weight == 1.0

    def test_string_session_path_converted(self):
        cfg = EnvConfig(session_path="some/path")
        assert isinstance(cfg.session_path, Path)
        assert cfg.session_path == Path("some/path")

    def test_invalid_action_freq_raises(self):
        with pytest.raises(ValueError, match="action_freq"):
            EnvConfig(action_freq=0)

    def test_invalid_max_steps_raises(self):
        with pytest.raises(ValueError, match="max_steps"):
            EnvConfig(max_steps=-1)

    def test_invalid_reward_scale_raises(self):
        with pytest.raises(ValueError, match="reward_scale"):
            EnvConfig(reward_scale=0)

    def test_instance_id_auto_generated(self):
        cfg1 = EnvConfig()
        cfg2 = EnvConfig()
        assert len(cfg1.instance_id) == 8
        assert cfg1.instance_id != cfg2.instance_id

"""Tests for curriculum learning system."""

import pytest
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from curriculum import CurriculumStage, CurriculumScheduler, create_default_curriculum


class TestCurriculumStage:
    def test_exists_false_for_missing(self):
        stage = CurriculumStage("test", "/nonexistent/path.state")
        assert stage.exists() is False

    def test_exists_true_for_real_file(self, tmp_path):
        f = tmp_path / "test.state"
        f.write_bytes(b"data")
        stage = CurriculumStage("test", str(f))
        assert stage.exists() is True


class TestCurriculumScheduler:
    def _make_stages(self, tmp_path):
        f1 = tmp_path / "early.state"
        f1.write_bytes(b"1")
        f2 = tmp_path / "mid.state"
        f2.write_bytes(b"2")
        f3 = tmp_path / "late.state"
        f3.write_bytes(b"3")
        return [
            CurriculumStage("early", str(f1), weight=1.0, unlocked=True),
            CurriculumStage("mid", str(f2), weight=0.5, min_reward_to_unlock=50, unlocked=False),
            CurriculumStage("late", str(f3), weight=0.3, min_reward_to_unlock=200, unlocked=False),
        ]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            CurriculumScheduler([])

    def test_sample_only_unlocked(self, tmp_path):
        stages = self._make_stages(tmp_path)
        scheduler = CurriculumScheduler(stages)
        # Only "early" is unlocked, so sample must return it
        for _ in range(20):
            assert scheduler.sample() == stages[0].state_path

    def test_update_unlocks_stage(self, tmp_path):
        stages = self._make_stages(tmp_path)
        scheduler = CurriculumScheduler(stages)

        newly = scheduler.update(60)
        assert "mid" in newly
        assert stages[1].unlocked is True
        assert stages[2].unlocked is False  # needs 200

    def test_update_unlocks_multiple(self, tmp_path):
        stages = self._make_stages(tmp_path)
        scheduler = CurriculumScheduler(stages)

        newly = scheduler.update(300)
        assert "mid" in newly
        assert "late" in newly

    def test_sample_after_unlock(self, tmp_path):
        stages = self._make_stages(tmp_path)
        scheduler = CurriculumScheduler(stages)
        scheduler.update(300)

        # Should now sample from all 3
        seen_paths = set()
        for _ in range(200):
            seen_paths.add(scheduler.sample())
        assert len(seen_paths) >= 2  # at minimum early + one other

    def test_status(self, tmp_path):
        stages = self._make_stages(tmp_path)
        scheduler = CurriculumScheduler(stages)
        assert scheduler.status() == {"early": True, "mid": False, "late": False}

    def test_no_auto_advance(self, tmp_path):
        stages = self._make_stages(tmp_path)
        scheduler = CurriculumScheduler(stages, auto_advance=False)
        scheduler.update(1000)
        assert stages[1].unlocked is False


class TestDefaultCurriculum:
    def test_creates_without_error(self):
        scheduler = create_default_curriculum()
        assert len(scheduler.stages) == 3
        assert scheduler.stages[0].unlocked is True

"""Tests for milestone tracking."""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from milestones import Milestone, MilestoneTracker
from game_state import (
    GameState, PARTY_SIZE, BADGES, IN_BATTLE,
    MAP_NUMBER, OPPONENT_ACTIVE_HP, OPPONENT_ACTIVE_MAX_HP,
)


class MockMemory:
    def __init__(self):
        self._data = {}
    def __getitem__(self, addr):
        return self._data.get(addr, 0)
    def __setitem__(self, addr, value):
        self._data[addr] = value


class MockPyBoy:
    def __init__(self):
        self.memory = MockMemory()


class TestMilestone:
    def test_mark_first_time(self):
        m = Milestone("test", "A test")
        assert m.mark(100) is True
        assert m.achieved_at_step == 100

    def test_mark_second_time_noop(self):
        m = Milestone("test", "A test")
        m.mark(100)
        assert m.mark(200) is False
        assert m.achieved_at_step == 100

    def test_achieved_property(self):
        m = Milestone("test", "A test")
        assert m.achieved is False
        m.mark(5)
        assert m.achieved is True


class TestMilestoneTracker:
    @pytest.fixture
    def setup(self):
        pyboy = MockPyBoy()
        gs = GameState(pyboy)
        tracker = MilestoneTracker()
        return tracker, gs, pyboy

    def test_got_starter(self, setup):
        tracker, gs, pyboy = setup
        # Initially no party
        tracker.update(0, gs)
        # Get starter
        pyboy.memory[PARTY_SIZE] = 1
        newly = tracker.update(10, gs)
        assert "got_starter" in newly

    def test_caught_pokemon(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        tracker.update(0, gs)
        pyboy.memory[PARTY_SIZE] = 2
        newly = tracker.update(50, gs)
        assert "caught_pokemon" in newly

    def test_badge_milestone(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        tracker.update(0, gs)
        pyboy.memory[BADGES] = 0b00000001  # 1 badge
        newly = tracker.update(1000, gs)
        assert "badge_1" in newly

    def test_multiple_badges_at_once(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        tracker.update(0, gs)
        pyboy.memory[BADGES] = 0b00000111  # 3 badges
        newly = tracker.update(5000, gs)
        assert "badge_1" in newly
        assert "badge_2" in newly
        assert "badge_3" in newly

    def test_map_milestone(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        pyboy.memory[MAP_NUMBER] = 1  # Viridian
        newly = tracker.update(100, gs)
        assert "reached_viridian" in newly

    def test_no_duplicate_milestones(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        pyboy.memory[MAP_NUMBER] = 1
        tracker.update(100, gs)
        newly = tracker.update(200, gs)
        assert "reached_viridian" not in newly

    def test_reset_clears_milestones(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        tracker.update(10, gs)
        assert tracker.achievement_count() > 0
        tracker.reset()
        assert tracker.achievement_count() == 0

    def test_summary(self, setup):
        tracker, gs, pyboy = setup
        summary = tracker.summary()
        assert "got_starter" in summary
        assert summary["got_starter"] is None

    def test_achieved_milestones(self, setup):
        tracker, gs, pyboy = setup
        pyboy.memory[PARTY_SIZE] = 1
        tracker.update(10, gs)
        achieved = tracker.achieved_milestones()
        assert "got_starter" in achieved
        assert achieved["got_starter"] == 10
        assert "badge_1" not in achieved

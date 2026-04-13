"""Tests for the modular reward system."""

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rewards import (
    RewardContext,
    RewardSystem,
    event_reward,
    explore_reward,
    badge_reward,
    healing_reward,
    stuck_penalty,
    level_reward,
    create_default_reward_system,
)


class TestRewardFunctions:
    def test_event_reward_basic(self):
        ctx = RewardContext(event_flag_sum=15, base_event_flags=10, has_museum_ticket=False)
        assert event_reward(ctx) == 5.0

    def test_event_reward_excludes_museum_ticket(self):
        ctx = RewardContext(event_flag_sum=15, base_event_flags=10, has_museum_ticket=True)
        assert event_reward(ctx) == 4.0

    def test_event_reward_never_negative(self):
        ctx = RewardContext(event_flag_sum=5, base_event_flags=10)
        assert event_reward(ctx) == 0.0

    def test_explore_reward(self):
        ctx = RewardContext(seen_coords_count=42)
        assert explore_reward(ctx) == 42.0

    def test_badge_reward(self):
        ctx = RewardContext(badge_count=3)
        assert badge_reward(ctx) == 3.0

    def test_healing_reward(self):
        ctx = RewardContext(total_healing_reward=1.5)
        assert healing_reward(ctx) == 1.5

    def test_stuck_penalty_under_threshold(self):
        ctx = RewardContext(current_coord_visits=100)
        assert stuck_penalty(ctx) == 0.0

    def test_stuck_penalty_at_threshold(self):
        ctx = RewardContext(current_coord_visits=600)
        assert stuck_penalty(ctx) == -1.0

    def test_stuck_penalty_over_threshold(self):
        ctx = RewardContext(current_coord_visits=1000)
        assert stuck_penalty(ctx) == -1.0

    def test_level_reward_below_threshold(self):
        ctx = RewardContext(levels_sum=15, party_size=1)
        result = level_reward(ctx)
        assert result >= 0

    def test_level_reward_diminishing_returns(self):
        ctx_low = RewardContext(levels_sum=30, party_size=1)
        ctx_high = RewardContext(levels_sum=60, party_size=1)
        low = level_reward(ctx_low)
        high = level_reward(ctx_high)
        # High should be more but with diminishing returns
        assert high > low
        # The gap should be smaller than the raw difference
        assert (high - low) < (60 - 30)


class TestRewardSystem:
    def test_add_and_compute(self):
        rs = RewardSystem()
        rs.add("test", lambda ctx: 10.0, weight=2.0)
        ctx = RewardContext()
        result = rs.compute(ctx)
        assert result == {"test": 20.0}

    def test_multiple_components(self):
        rs = RewardSystem()
        rs.add("a", lambda ctx: 1.0, weight=1.0)
        rs.add("b", lambda ctx: 2.0, weight=3.0)
        result = rs.compute(RewardContext())
        assert result == {"a": 1.0, "b": 6.0}

    def test_names(self):
        rs = RewardSystem()
        rs.add("foo", lambda ctx: 0)
        rs.add("bar", lambda ctx: 0)
        assert rs.names == ["foo", "bar"]

    def test_default_system_has_expected_components(self):
        rs = create_default_reward_system()
        assert set(rs.names) == {"event", "heal", "badge", "explore", "stuck"}

    def test_default_system_scaling(self):
        rs = create_default_reward_system(reward_scale=2.0, explore_weight=0.5)
        ctx = RewardContext(
            seen_coords_count=100,
            badge_count=1,
            event_flag_sum=10,
            base_event_flags=5,
        )
        result = rs.compute(ctx)
        # badge: 2.0 * 10 * 1 = 20
        assert result["badge"] == 20.0
        # explore: 2.0 * 0.5 * 0.1 * 100 = 10
        assert result["explore"] == 10.0


class TestRewardContext:
    def test_defaults(self):
        ctx = RewardContext()
        assert ctx.seen_coords_count == 0
        assert ctx.hp_fraction == 1.0
        assert ctx.badge_count == 0
        assert ctx.died_count == 0

    def test_custom_values(self):
        ctx = RewardContext(badge_count=8, hp_fraction=0.5)
        assert ctx.badge_count == 8
        assert ctx.hp_fraction == 0.5

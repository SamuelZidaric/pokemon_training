"""Tests for enhanced reward functions (battle, PC softlock)."""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rewards import (
    RewardContext,
    battle_win_reward,
    opponent_damage_reward,
    pc_box_full_penalty,
    create_enhanced_reward_system,
)


class TestBattleWinReward:
    def test_zero_wins(self):
        ctx = RewardContext(battles_won=0)
        assert battle_win_reward(ctx) == 0.0

    def test_multiple_wins(self):
        ctx = RewardContext(battles_won=5)
        assert battle_win_reward(ctx) == 5.0


class TestOpponentDamageReward:
    def test_not_in_battle(self):
        ctx = RewardContext(in_battle=False)
        assert opponent_damage_reward(ctx) == 0.0

    def test_damage_dealt(self):
        ctx = RewardContext(
            in_battle=True,
            prev_opponent_hp_fraction=0.8,
            opponent_hp_fraction=0.5,
        )
        assert abs(opponent_damage_reward(ctx) - 0.3) < 1e-6

    def test_opponent_healed_no_reward(self):
        ctx = RewardContext(
            in_battle=True,
            prev_opponent_hp_fraction=0.5,
            opponent_hp_fraction=0.8,
        )
        assert opponent_damage_reward(ctx) == 0.0

    def test_no_change(self):
        ctx = RewardContext(
            in_battle=True,
            prev_opponent_hp_fraction=0.5,
            opponent_hp_fraction=0.5,
        )
        assert opponent_damage_reward(ctx) == 0.0


class TestPCBoxFullPenalty:
    def test_not_full(self):
        ctx = RewardContext(is_box_full=False)
        assert pc_box_full_penalty(ctx) == 0.0

    def test_full(self):
        ctx = RewardContext(is_box_full=True)
        assert pc_box_full_penalty(ctx) == -1.0


class TestEnhancedRewardSystem:
    def test_has_all_components(self):
        rs = create_enhanced_reward_system()
        names = set(rs.names)
        assert "battle_win" in names
        assert "opponent_dmg" in names
        assert "pc_full" in names
        assert "level" in names
        # Plus the base ones
        assert "event" in names
        assert "explore" in names
        assert "badge" in names

    def test_battle_win_weighted(self):
        rs = create_enhanced_reward_system(reward_scale=2.0)
        ctx = RewardContext(battles_won=3)
        result = rs.compute(ctx)
        # battle_win weight = reward_scale * 5 = 10, fn returns 3
        assert result["battle_win"] == 30.0

    def test_pc_full_penalty_weighted(self):
        rs = create_enhanced_reward_system(reward_scale=1.0)
        ctx = RewardContext(is_box_full=True)
        result = rs.compute(ctx)
        assert result["pc_full"] == -1.0

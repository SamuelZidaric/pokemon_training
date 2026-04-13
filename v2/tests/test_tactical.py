"""Tests for Phase 5 tactical features.

Covers:
- Type effectiveness engine (_TYPE_CHART, type_effectiveness)
- tactical_obs() vector shape, range, and content
- Tactical reward functions (hp_loss_penalty, super_effective_bonus,
  party_fainted_penalty, efficiency_reward)
- Tactical reward presets (battle, survival, speedrun)
- Battle curriculum
- PokemonNet tactical branch
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from game_state import (
    GameState,
    TACTICAL_OBS_SIZE,
    _TYPE_CHART,
    type_effectiveness,
    TYPE_ID_LIST,
    TYPE_ID_TO_INDEX,
    NUM_POKEMON_TYPES,
    IN_BATTLE,
    PARTY_SIZE,
    PARTY_HP,
    PARTY_MAX_HP,
    PARTY_LEVELS,
)
from rewards import (
    RewardContext,
    RewardSystem,
    hp_loss_penalty,
    super_effective_bonus,
    party_fainted_penalty,
    efficiency_reward,
    create_battle_focused_system,
    create_survival_system,
    create_speedrun_system,
)
from curriculum import create_battle_curriculum, create_gym_gauntlet_curriculum
from pokemon_model import PokemonNet


# ===========================================================================
# Type effectiveness
# ===========================================================================

class TestTypeEffectiveness:
    def test_neutral_matchup(self):
        # Normal vs Normal → 1.0
        assert type_effectiveness(0x00, 0x00, 0x00) == pytest.approx(1.0)

    def test_super_effective(self):
        # Fire (0x14) vs Grass (0x16) → 2.0
        assert type_effectiveness(0x14, 0x16, 0x16) == pytest.approx(2.0)

    def test_not_very_effective(self):
        # Fire vs Water → 0.5
        assert type_effectiveness(0x14, 0x15, 0x15) == pytest.approx(0.5)

    def test_immune(self):
        # Normal vs Ghost → 0.0
        assert type_effectiveness(0x00, 0x08, 0x08) == pytest.approx(0.0)
        # Electric vs Ground → 0.0
        assert type_effectiveness(0x17, 0x04, 0x04) == pytest.approx(0.0)

    def test_dual_type_double_super_effective(self):
        # Ground (0x04) vs Fire/Rock (0x14, 0x05) → 2.0 * 2.0 = 4.0
        assert type_effectiveness(0x04, 0x14, 0x05) == pytest.approx(4.0)

    def test_dual_type_cancels_out(self):
        # Fire (0x14) vs Grass/Water (0x16, 0x15) → 2.0 * 0.5 = 1.0
        assert type_effectiveness(0x14, 0x16, 0x15) == pytest.approx(1.0)

    def test_same_dual_types_no_double_count(self):
        # When def_type1 == def_type2, should only apply once
        eff_single = type_effectiveness(0x14, 0x16, 0x16)  # Fire vs Grass/Grass
        assert eff_single == pytest.approx(2.0)  # Not 4.0

    def test_immune_overrides_dual(self):
        # Normal (0x00) vs Ghost/Poison → 0.0 (immune to one type)
        assert type_effectiveness(0x00, 0x08, 0x03) == pytest.approx(0.0)

    def test_fighting_vs_ghost_immune(self):
        assert type_effectiveness(0x01, 0x08, 0x08) == pytest.approx(0.0)

    def test_chart_has_expected_entries(self):
        """Verify the type chart contains key Gen 1 matchups."""
        assert (0x14, 0x16) in _TYPE_CHART  # Fire → Grass
        assert (0x15, 0x14) in _TYPE_CHART  # Water → Fire
        assert (0x18, 0x01) in _TYPE_CHART  # Psychic → Fighting
        assert (0x08, 0x18) in _TYPE_CHART  # Ghost → Psychic (Gen 1 bug)

    def test_type_id_list_complete(self):
        assert len(TYPE_ID_LIST) == NUM_POKEMON_TYPES
        # All IDs in the chart should be known
        for atk, def_ in _TYPE_CHART:
            assert atk in TYPE_ID_TO_INDEX, f"Unknown atk type {atk:#x}"
            assert def_ in TYPE_ID_TO_INDEX, f"Unknown def type {def_:#x}"


# ===========================================================================
# Tactical observation vector
# ===========================================================================

def _make_game_state(mem_overrides: dict[int, int] | None = None) -> GameState:
    """Create a GameState with a mock PyBoy memory."""
    mem = [0] * 0x10000
    if mem_overrides:
        for addr, val in mem_overrides.items():
            mem[addr] = val

    pyboy = MagicMock()
    pyboy.memory.__getitem__ = lambda self, addr: mem[addr]
    return GameState(pyboy)


class TestTacticalObs:
    def test_shape(self):
        gs = _make_game_state()
        obs = gs.tactical_obs()
        assert obs.shape == (TACTICAL_OBS_SIZE,)
        assert obs.dtype == np.float32

    def test_size_constant(self):
        assert TACTICAL_OBS_SIZE == 22

    def test_out_of_battle_values(self):
        gs = _make_game_state()
        obs = gs.tactical_obs()
        assert obs[0] == 0.0  # not in battle
        assert obs[3] == 0.0  # opponent HP 0 when not in battle
        assert obs[4] == 0.0  # opponent level 0 when not in battle

    def test_in_battle_flag(self):
        gs = _make_game_state({IN_BATTLE: 1})
        obs = gs.tactical_obs()
        assert obs[0] == 1.0

    def test_values_in_range(self):
        """All tactical obs values should be in [-1, 1] range."""
        gs = _make_game_state({IN_BATTLE: 1, PARTY_SIZE: 3})
        obs = gs.tactical_obs()
        assert np.all(obs >= -1.0), f"Values below -1: {obs[obs < -1.0]}"
        assert np.all(obs <= 1.0), f"Values above 1: {obs[obs > 1.0]}"

    def test_party_fainted_count(self):
        # Give all 6 slots non-zero HP, then zero out slot 0
        overrides = {
            PARTY_SIZE: 3,
        }
        # Set all 6 HP slots to non-zero first (party_hp reads all 6 regardless of party_size)
        for i in range(6):
            overrides[PARTY_HP[i]] = 0
            overrides[PARTY_HP[i] + 1] = 25  # 25 HP
            overrides[PARTY_MAX_HP[i]] = 0
            overrides[PARTY_MAX_HP[i] + 1] = 50
        # Zero out slot 0 HP
        overrides[PARTY_HP[0] + 1] = 0
        gs = _make_game_state(overrides)
        assert gs.party_fainted_count == 1
        obs = gs.tactical_obs()
        assert obs[11] == pytest.approx(1.0 / 6.0)  # 1 fainted / 6

    def test_party_size_encoding(self):
        gs = _make_game_state({PARTY_SIZE: 4})
        obs = gs.tactical_obs()
        assert obs[12] == pytest.approx(4.0 / 6.0)


# ===========================================================================
# Type advantage properties
# ===========================================================================

class TestTypeAdvantageProperties:
    def test_type_advantage_signal_not_in_battle(self):
        gs = _make_game_state()
        # best_move_effectiveness returns 1.0 when not in battle
        assert gs.best_move_effectiveness == pytest.approx(1.0)
        assert gs.type_advantage_signal == pytest.approx(0.0)

    def test_party_fainted_count_all_alive(self):
        # All 6 slots have HP — none fainted
        overrides = {}
        for i in range(6):
            overrides[PARTY_HP[i]] = 0
            overrides[PARTY_HP[i] + 1] = 30
        gs = _make_game_state(overrides)
        assert gs.party_fainted_count == 0


# ===========================================================================
# Tactical reward functions
# ===========================================================================

class TestTacticalRewardFunctions:
    def test_hp_loss_penalty(self):
        ctx = RewardContext(hp_loss_this_step=0.3)
        assert hp_loss_penalty(ctx) == pytest.approx(-0.3)

    def test_hp_loss_penalty_zero(self):
        ctx = RewardContext(hp_loss_this_step=0.0)
        assert hp_loss_penalty(ctx) == pytest.approx(0.0)

    def test_super_effective_bonus_in_battle(self):
        ctx = RewardContext(in_battle=True, type_advantage=1.0)
        assert super_effective_bonus(ctx) == pytest.approx(1.0)

    def test_super_effective_bonus_resisted(self):
        ctx = RewardContext(in_battle=True, type_advantage=-1.0)
        assert super_effective_bonus(ctx) == pytest.approx(0.0)  # max(-1, 0) = 0

    def test_super_effective_bonus_not_in_battle(self):
        ctx = RewardContext(in_battle=False, type_advantage=1.0)
        assert super_effective_bonus(ctx) == pytest.approx(0.0)

    def test_party_fainted_penalty(self):
        ctx = RewardContext(party_fainted_count=2)
        assert party_fainted_penalty(ctx) == pytest.approx(-2.0)

    def test_party_fainted_penalty_zero(self):
        ctx = RewardContext(party_fainted_count=0)
        assert party_fainted_penalty(ctx) == pytest.approx(0.0)

    def test_efficiency_reward_dealing_damage(self):
        ctx = RewardContext(
            in_battle=True,
            prev_opponent_hp_fraction=1.0,
            opponent_hp_fraction=0.7,
            hp_fraction=0.8,
        )
        # damage_dealt = 0.3, * our hp 0.8 = 0.24
        assert efficiency_reward(ctx) == pytest.approx(0.24)

    def test_efficiency_reward_no_damage(self):
        ctx = RewardContext(
            in_battle=True,
            prev_opponent_hp_fraction=0.5,
            opponent_hp_fraction=0.5,
            hp_fraction=1.0,
        )
        assert efficiency_reward(ctx) == pytest.approx(0.0)

    def test_efficiency_reward_not_in_battle(self):
        ctx = RewardContext(in_battle=False)
        assert efficiency_reward(ctx) == pytest.approx(0.0)

    def test_efficiency_reward_opponent_healed(self):
        # Edge case: opponent HP goes up (healing)
        ctx = RewardContext(
            in_battle=True,
            prev_opponent_hp_fraction=0.5,
            opponent_hp_fraction=0.8,
            hp_fraction=1.0,
        )
        assert efficiency_reward(ctx) == pytest.approx(0.0)


# ===========================================================================
# Tactical reward presets
# ===========================================================================

class TestTacticalPresets:
    def test_battle_focused_has_expected_components(self):
        rs = create_battle_focused_system()
        names = {c.name for c in rs._components}
        assert "battle_win" in names
        assert "opponent_dmg" in names
        assert "super_eff" in names
        assert "efficiency" in names
        assert "hp_loss" in names
        assert "fainted" in names
        assert "badge" in names

    def test_battle_focused_low_explore_weight(self):
        rs = create_battle_focused_system()
        explore = next(c for c in rs._components if c.name == "explore")
        battle = next(c for c in rs._components if c.name == "battle_win")
        assert explore.weight < battle.weight * 0.01

    def test_survival_system_has_expected_components(self):
        rs = create_survival_system()
        names = {c.name for c in rs._components}
        assert "hp_loss" in names
        assert "fainted" in names
        assert "heal" in names

    def test_survival_system_hp_heavy(self):
        rs = create_survival_system()
        hp_loss = next(c for c in rs._components if c.name == "hp_loss")
        fainted = next(c for c in rs._components if c.name == "fainted")
        assert hp_loss.weight >= 8.0
        assert fainted.weight >= 10.0

    def test_speedrun_system_has_expected_components(self):
        rs = create_speedrun_system()
        names = {c.name for c in rs._components}
        assert "event" in names
        assert "badge" in names
        assert "efficiency" in names
        assert "stuck" in names

    def test_speedrun_badge_heavily_weighted(self):
        rs = create_speedrun_system()
        badge = next(c for c in rs._components if c.name == "badge")
        assert badge.weight >= 20.0

    def test_reward_scale_propagates(self):
        rs = create_battle_focused_system(reward_scale=2.0)
        battle = next(c for c in rs._components if c.name == "battle_win")
        # Default weight is 20 * scale, so 40 at scale=2.0
        assert battle.weight == pytest.approx(40.0)

    def test_all_presets_can_compute(self):
        """All tactical presets produce reward dicts from a default context."""
        ctx = RewardContext()
        for factory in [create_battle_focused_system, create_survival_system, create_speedrun_system]:
            rs = factory()
            result = rs.compute(ctx)
            assert isinstance(result, dict)
            total = sum(result.values())
            assert isinstance(total, float)


# ===========================================================================
# Battle curriculum
# ===========================================================================

class TestBattleCurriculum:
    def test_create_battle_curriculum(self, tmp_path):
        state_file = tmp_path / "pre_brock.state"
        state_file.write_bytes(b"fake state")
        sched = create_battle_curriculum(str(state_file))
        assert len(sched.stages) == 1
        assert sched.stages[0].name == "gym_battle"
        assert sched.sample() == str(state_file)

    def test_gym_gauntlet_stages(self):
        sched = create_gym_gauntlet_curriculum()
        assert len(sched.stages) == 8
        # First stage (Brock) should be unlocked
        assert sched.stages[0].unlocked is True
        assert sched.stages[0].name == "brock"
        # Others locked
        for stage in sched.stages[1:]:
            assert stage.unlocked is False

    def test_gym_gauntlet_progressive_unlock(self, tmp_path):
        sched = create_gym_gauntlet_curriculum()
        # Point Misty's stage at a real file so exists() passes
        misty_state = tmp_path / "pre_misty.state"
        misty_state.write_bytes(b"fake")
        sched.stages[1].state_path = str(misty_state)
        # Unlock Misty at reward >= 100
        newly = sched.update(100)
        assert "misty" in newly
        assert sched.stages[1].unlocked is True
        # Surge still locked (needs 200, and file doesn't exist)
        assert sched.stages[2].unlocked is False


# ===========================================================================
# PokemonNet with tactical branch
# ===========================================================================

class TestPokemonNetTactical:
    @pytest.fixture
    def obs_shapes_with_tactical(self):
        return {
            "screens": (72, 80, 3),
            "map": (48, 48, 1),
            "health": (1,),
            "level": (8,),
            "badges": (8,),
            "events": (1080,),
            "recent_actions": (3,),
            "tactical": (TACTICAL_OBS_SIZE,),
        }

    @pytest.fixture
    def obs_shapes_without_tactical(self):
        return {
            "screens": (72, 80, 3),
            "map": (48, 48, 1),
            "health": (1,),
            "level": (8,),
            "badges": (8,),
            "events": (1080,),
            "recent_actions": (3,),
        }

    def test_tactical_branch_exists(self, obs_shapes_with_tactical):
        net = PokemonNet(obs_shapes_with_tactical, num_actions=7)
        assert net._has_tactical is True
        assert hasattr(net, "tactical_fc")

    def test_no_tactical_branch_without_key(self, obs_shapes_without_tactical):
        net = PokemonNet(obs_shapes_without_tactical, num_actions=7)
        assert net._has_tactical is False

    def test_forward_with_tactical(self, obs_shapes_with_tactical):
        net = PokemonNet(obs_shapes_with_tactical, num_actions=7)
        B = 4
        obs = {
            "screens": torch.randint(0, 255, (B, 72, 80, 3), dtype=torch.float32),
            "map": torch.randint(0, 255, (B, 48, 48, 1), dtype=torch.float32),
            "health": torch.rand(B, 1),
            "level": torch.rand(B, 8),
            "badges": torch.rand(B, 8),
            "events": torch.rand(B, 1080),
            "recent_actions": torch.rand(B, 3),
            "tactical": torch.rand(B, TACTICAL_OBS_SIZE),
        }
        logits, value = net(obs)
        assert logits.shape == (B, 7)
        assert value.shape == (B, 1)

    def test_forward_without_tactical(self, obs_shapes_without_tactical):
        net = PokemonNet(obs_shapes_without_tactical, num_actions=7)
        B = 2
        obs = {
            "screens": torch.randint(0, 255, (B, 72, 80, 3), dtype=torch.float32),
            "map": torch.randint(0, 255, (B, 48, 48, 1), dtype=torch.float32),
            "health": torch.rand(B, 1),
            "level": torch.rand(B, 8),
            "badges": torch.rand(B, 8),
            "events": torch.rand(B, 1080),
            "recent_actions": torch.rand(B, 3),
        }
        logits, value = net(obs)
        assert logits.shape == (B, 7)
        assert value.shape == (B, 1)

    def test_tactical_branch_changes_output(self, obs_shapes_with_tactical, obs_shapes_without_tactical):
        """Model with tactical branch has different parameter count."""
        net_with = PokemonNet(obs_shapes_with_tactical, num_actions=7)
        net_without = PokemonNet(obs_shapes_without_tactical, num_actions=7)
        params_with = sum(p.numel() for p in net_with.parameters())
        params_without = sum(p.numel() for p in net_without.parameters())
        assert params_with > params_without

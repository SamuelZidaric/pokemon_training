"""Tests for PokemonNet — the custom CNN+MLP fusion model.

Tests run with plain PyTorch (no Ray required).
"""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from pokemon_model import PokemonNet, _conv_output_size


# ---------------------------------------------------------------------------
# Observation shape fixtures
# ---------------------------------------------------------------------------

# Matches the env's observation_space WITHOUT frame stacking
OBS_SHAPES_NO_STACK = {
    "screens": (72, 80, 3),       # H, W, frame_stacks
    "health": (1,),
    "level": (8,),
    "badges": (8,),
    "events": (1080,),            # (EVENT_FLAGS_END - START) * 8
    "map": (48, 48, 1),
    "recent_actions": (3,),
}

# Matches the env WITH DictFrameStack(n_stack=4)
OBS_SHAPES_STACKED = {
    "screens": (72, 80, 3, 4),    # H, W, C, N
    "health": (1, 4),
    "level": (8, 4),
    "badges": (8,),               # passthrough
    "events": (1080,),            # passthrough
    "map": (48, 48, 1),           # passthrough
    "recent_actions": (3,),       # passthrough
}

NUM_ACTIONS = 7
BATCH_SIZE = 4


def _make_batch(shapes: dict, batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """Create a fake batch of observations."""
    batch = {}
    for key, shape in shapes.items():
        if key == "screens":
            batch[key] = torch.randint(0, 256, (batch_size, *shape), dtype=torch.uint8)
        elif key == "map":
            batch[key] = torch.randint(0, 256, (batch_size, *shape), dtype=torch.uint8)
        elif key == "events":
            batch[key] = torch.randint(0, 2, (batch_size, *shape), dtype=torch.int8)
        elif key == "badges":
            batch[key] = torch.randint(0, 2, (batch_size, *shape), dtype=torch.int8)
        elif key == "recent_actions":
            batch[key] = torch.randint(0, NUM_ACTIONS, (batch_size, *shape))
        elif key == "health":
            batch[key] = torch.rand(batch_size, *shape)
        elif key == "level":
            batch[key] = torch.rand(batch_size, *shape) * 2 - 1
        else:
            batch[key] = torch.rand(batch_size, *shape)
    return batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConvOutputSize:
    def test_single_layer(self):
        h, w = _conv_output_size(72, 80, [5], [2])
        assert h == 34
        assert w == 38

    def test_multi_layer(self):
        h, w = _conv_output_size(72, 80, [5, 3, 3], [2, 2, 1])
        assert h > 0
        assert w > 0

    def test_map_size(self):
        h, w = _conv_output_size(48, 48, [5, 3], [2, 2])
        assert h > 0
        assert w > 0


class TestPokemonNetNoStack:
    """Test with the non-stacked observation shapes."""

    @pytest.fixture
    def model(self):
        return PokemonNet(OBS_SHAPES_NO_STACK, NUM_ACTIONS)

    def test_forward_output_shapes(self, model):
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, value = model(batch)
        assert logits.shape == (BATCH_SIZE, NUM_ACTIONS)
        assert value.shape == (BATCH_SIZE, 1)

    def test_logits_are_finite(self, model):
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, _ = model(batch)
        assert torch.isfinite(logits).all()

    def test_value_is_finite(self, model):
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        _, value = model(batch)
        assert torch.isfinite(value).all()

    def test_get_value_after_forward(self, model):
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        model(batch)
        v = model.get_value()
        assert v.shape == (BATCH_SIZE, 1)

    def test_get_value_before_forward_raises(self, model):
        with pytest.raises(AssertionError):
            model.get_value()

    def test_batch_size_1(self, model):
        batch = _make_batch(OBS_SHAPES_NO_STACK, batch_size=1)
        logits, value = model(batch)
        assert logits.shape == (1, NUM_ACTIONS)
        assert value.shape == (1, 1)

    def test_large_batch(self, model):
        batch = _make_batch(OBS_SHAPES_NO_STACK, batch_size=64)
        logits, value = model(batch)
        assert logits.shape == (64, NUM_ACTIONS)


class TestPokemonNetStacked:
    """Test with frame-stacked observation shapes."""

    @pytest.fixture
    def model(self):
        return PokemonNet(OBS_SHAPES_STACKED, NUM_ACTIONS)

    def test_forward_output_shapes(self, model):
        batch = _make_batch(OBS_SHAPES_STACKED)
        logits, value = model(batch)
        assert logits.shape == (BATCH_SIZE, NUM_ACTIONS)
        assert value.shape == (BATCH_SIZE, 1)

    def test_different_from_no_stack(self):
        model_ns = PokemonNet(OBS_SHAPES_NO_STACK, NUM_ACTIONS)
        model_s = PokemonNet(OBS_SHAPES_STACKED, NUM_ACTIONS)
        # Stacked model should have more CNN input channels
        ns_first_conv = list(model_ns.screen_cnn.children())[0]
        s_first_conv = list(model_s.screen_cnn.children())[0]
        assert s_first_conv.in_channels > ns_first_conv.in_channels

    def test_stacked_screen_channels(self):
        model = PokemonNet(OBS_SHAPES_STACKED, NUM_ACTIONS)
        first_conv = list(model.screen_cnn.children())[0]
        # (72, 80, 3, 4) → C*N = 3*4 = 12 input channels
        assert first_conv.in_channels == 12


class TestPokemonNetWithMacros:
    """Test with expanded action space from MacroActionWrapper."""

    def test_13_actions(self):
        model = PokemonNet(OBS_SHAPES_NO_STACK, num_actions=13)
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, _ = model(batch)
        assert logits.shape == (BATCH_SIZE, 13)


class TestPokemonNetCustomConfig:
    """Test that hyperparameters can be customised."""

    def test_wider_cnn(self):
        model = PokemonNet(
            OBS_SHAPES_NO_STACK, NUM_ACTIONS,
            cnn_channels=[64, 128, 128],
            cnn_kernels=[5, 3, 3],
            cnn_strides=[2, 2, 1],
        )
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, _ = model(batch)
        assert logits.shape == (BATCH_SIZE, NUM_ACTIONS)

    def test_deeper_fc(self):
        model = PokemonNet(
            OBS_SHAPES_NO_STACK, NUM_ACTIONS,
            fc_sizes=[512, 512, 256],
        )
        assert model.fusion_out_size == 256
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, _ = model(batch)
        assert logits.shape == (BATCH_SIZE, NUM_ACTIONS)

    def test_custom_map_cnn(self):
        model = PokemonNet(
            OBS_SHAPES_NO_STACK, NUM_ACTIONS,
            map_channels=[32, 64, 64],
            map_kernels=[3, 3, 3],
            map_strides=[2, 2, 1],
        )
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, _ = model(batch)
        assert logits.shape == (BATCH_SIZE, NUM_ACTIONS)


class TestGradientFlow:
    """Verify gradients flow through all branches."""

    def test_all_parameters_have_grad(self):
        model = PokemonNet(OBS_SHAPES_NO_STACK, NUM_ACTIONS)
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, value = model(batch)
        loss = logits.sum() + value.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_value_head_independent_grad(self):
        """Value head should receive gradients from value loss alone."""
        model = PokemonNet(OBS_SHAPES_NO_STACK, NUM_ACTIONS)
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        _, value = model(batch)
        value.sum().backward()
        assert model.value_head.weight.grad is not None

    def test_policy_head_independent_grad(self):
        model = PokemonNet(OBS_SHAPES_NO_STACK, NUM_ACTIONS)
        batch = _make_batch(OBS_SHAPES_NO_STACK)
        logits, _ = model(batch)
        logits.sum().backward()
        assert model.policy_head.weight.grad is not None


class TestRLlibModelFactory:
    """Test the RLlib wrapper factory (skipped if Ray not installed)."""

    def test_factory_returns_class(self):
        try:
            from pokemon_model import make_rllib_model_class
            cls = make_rllib_model_class()
            assert cls is not None
            assert cls.__name__ == "PokemonRLlibModel"
        except ImportError:
            pytest.skip("Ray not installed")

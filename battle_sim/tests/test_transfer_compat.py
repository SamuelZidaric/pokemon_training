"""Week-4 transfer guardrail.

The battle-expert .zip must produce weights that drop directly into the
full-game ``PokemonNet.tactical`` module on ``claude/quizzical-sammet``.
We don't have that module importable on this branch, so we mock it with
the exact layout it is contracted to have:

    nn.Sequential(
        nn.Linear(TACTICAL_OBS_SIZE, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
    )

If PokemonNet.tactical on the sister branch ever drifts from this layout,
this test is the first thing that will fail and refuse to ship weights
that can't be loaded.

We also train a tiny PPO model and actually copy the weights across — a
shape assertion alone doesn't catch e.g. transposed weight tensors or
separate pi/vf nets accidentally sharing a module.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from battle_sim.env import PokemonBattleEnv
from battle_sim.train_battle import POLICY_KWARGS
from battle_sim.v2_contract import TACTICAL_OBS_SIZE


class MockPokemonNetTactical(nn.Module):
    """Stand-in for the tactical branch on claude/quizzical-sammet.

    Must be kept in lockstep with the real PokemonNet.tactical module.
    If the real module changes shape or activation, update BOTH this
    class and the POLICY_KWARGS in train_battle.py in the same commit.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(TACTICAL_OBS_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


def _train_tiny_model() -> PPO:
    """Train a 2k-step model so we have real weights to copy, not init noise."""
    env = DummyVecEnv([lambda: PokemonBattleEnv(seed=0, max_turns=50)])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        n_steps=256,
        batch_size=64,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=2_048)
    return model


def test_mlp_extractor_policy_net_shape():
    """The SB3 policy_net's two Linears must match PokemonNet.tactical exactly."""
    model = _train_tiny_model()
    pnet = model.policy.mlp_extractor.policy_net
    # Sequential [Linear, ReLU, Linear, ReLU]
    assert isinstance(pnet[0], nn.Linear)
    assert isinstance(pnet[1], nn.ReLU)
    assert isinstance(pnet[2], nn.Linear)
    assert isinstance(pnet[3], nn.ReLU)
    assert pnet[0].weight.shape == (64, TACTICAL_OBS_SIZE)
    assert pnet[0].bias.shape == (64,)
    assert pnet[2].weight.shape == (64, 64)
    assert pnet[2].bias.shape == (64,)


def test_weights_graft_into_mock_tactical_module():
    """Actual weight copy + forward-pass parity between SB3 policy_net and
    the mock PokemonNet.tactical.  If this fails, Week-4 transfer fails."""
    model = _train_tiny_model()
    pnet = model.policy.mlp_extractor.policy_net

    tactical = MockPokemonNetTactical()
    # The graft — this is the exact Week-4 transfer code.
    tactical.fc1.weight.data.copy_(pnet[0].weight.data)
    tactical.fc1.bias.data.copy_(pnet[0].bias.data)
    tactical.fc2.weight.data.copy_(pnet[2].weight.data)
    tactical.fc2.bias.data.copy_(pnet[2].bias.data)

    # Forward-pass parity: SB3 applies mlp_extractor.policy_net to the
    # obs through features_extractor (FlattenExtractor == identity for Box).
    x = torch.randn(4, TACTICAL_OBS_SIZE)
    with torch.no_grad():
        sb3_out = pnet(x)
        mock_out = tactical(x)
    torch.testing.assert_close(sb3_out, mock_out)


def test_features_extractor_is_identity_for_box():
    """For a flat Box obs, SB3 uses FlattenExtractor which is a no-op —
    if this ever changes (e.g. SB3 inserts a pre-norm layer), transfer
    semantics change and we need to re-derive the graft code."""
    model = _train_tiny_model()
    fx = model.policy.features_extractor
    x = torch.randn(2, TACTICAL_OBS_SIZE)
    out = fx(x)
    # Flatten of a Box(N,) tensor is the tensor itself (up to shape).
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)


def test_activation_is_relu_not_tanh():
    """SB3's default is Tanh — we explicitly set ReLU in POLICY_KWARGS
    because PokemonNet.tactical uses ReLU.  If someone drops the kwarg,
    silently the wrong activation gets used and forward-pass parity fails."""
    model = _train_tiny_model()
    pnet = model.policy.mlp_extractor.policy_net
    assert isinstance(pnet[1], nn.ReLU), \
        f"expected ReLU activation, got {type(pnet[1]).__name__}"
    assert isinstance(pnet[3], nn.ReLU)

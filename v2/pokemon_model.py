"""Custom RLlib model for Pokemon Red's Dict observation space.

The default RLlib ComplexInputNetwork will either flatten everything
(losing spatial structure) or crash on the mixed-type Dict space.
This model explicitly routes each observation key through the right
network branch, then fuses them for the policy and value heads.

Architecture
------------

.. code-block:: text

    ┌──────────────────────────────────────────────────────┐
    │                   OBSERVATION DICT                   │
    └──┬────────┬─────────┬────────┬──────────┬───────────┘
       │        │         │        │          │
    screens   map    health/level  badges   events
       │        │     /recent_act   │          │
       ▼        ▼         │        │          │
    ┌──────┐ ┌──────┐     │        │          │
    │ CNN  │ │ CNN  │     │        │          │
    │ (3×  │ │ (2×  │     │        │          │
    │ conv)│ │ conv)│     │        │          │
    └──┬───┘ └──┬───┘     │        │          │
       │        │         │        │          │
       ▼        ▼         ▼        ▼          ▼
    ┌──────────────────────────────────────────────────────┐
    │                   CONCATENATE                        │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  FC (256)   │
                    │  ReLU       │
                    │  FC (256)   │
                    │  ReLU       │
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
         ┌────────────┐       ┌────────────┐
         │ Policy Head│       │ Value Head │
         │ FC → logits│       │ FC → V(s)  │
         └────────────┘       └────────────┘

Observation shapes (with n_stack=4, 7 actions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- screens:       (72, 80, 1, 4)  or (72, 80, 3) without stacking
- map:           (48, 48, 1)
- health:        (1, 4)          or (1,) without stacking
- level:         (8, 4)          or (8,) without stacking
- badges:        (8,)
- events:        (1080,)         passthrough, not stacked
- recent_actions:(3,)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CNN block builder
# ---------------------------------------------------------------------------

def _make_cnn(
    in_channels: int,
    channels: list[int],
    kernels: list[int],
    strides: list[int],
) -> tuple[nn.Sequential, int]:
    """Build a CNN stack and return (module, output_flat_size).

    We compute the output size analytically so we don't need a dummy
    forward pass.
    """
    layers: list[nn.Module] = []
    c_in = in_channels
    for c_out, k, s in zip(channels, kernels, strides):
        layers.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s))
        layers.append(nn.ReLU())
        c_in = c_out
    return nn.Sequential(*layers), c_in


def _conv_output_size(h: int, w: int, kernels: list[int], strides: list[int]) -> tuple[int, int]:
    """Compute spatial output size after a series of conv layers (no padding)."""
    for k, s in zip(kernels, strides):
        h = (h - k) // s + 1
        w = (w - k) // s + 1
    return h, w


# ---------------------------------------------------------------------------
# Custom PyTorch model — standalone (no RLlib dependency at import time)
# ---------------------------------------------------------------------------

class PokemonNet(nn.Module):
    """Two-branch CNN+MLP fusion network for Pokemon Red observations.

    This is a plain ``nn.Module`` that can be used standalone or wrapped
    by an RLlib ``TorchModelV2``.  Keeping it as a plain module makes it
    testable without Ray installed.

    Parameters
    ----------
    obs_space_shapes : dict[str, tuple[int, ...]]
        Shape of each observation key.  Pass
        ``{k: v.shape for k, v in obs_space.spaces.items()}``.
    num_actions : int
        Size of the discrete action space.
    cnn_channels : list[int]
        Channel counts for each screen-CNN layer.
    cnn_kernels : list[int]
        Kernel sizes for each screen-CNN layer.
    cnn_strides : list[int]
        Stride for each screen-CNN layer.
    map_channels : list[int]
        Channel counts for the minimap CNN.
    map_kernels : list[int]
        Kernel sizes for the minimap CNN.
    map_strides : list[int]
        Strides for the minimap CNN.
    fc_sizes : list[int]
        Hidden layer sizes for the fusion MLP.
    """

    def __init__(
        self,
        obs_space_shapes: dict[str, tuple[int, ...]],
        num_actions: int,
        cnn_channels: list[int] | None = None,
        cnn_kernels: list[int] | None = None,
        cnn_strides: list[int] | None = None,
        map_channels: list[int] | None = None,
        map_kernels: list[int] | None = None,
        map_strides: list[int] | None = None,
        fc_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.obs_keys = sorted(obs_space_shapes.keys())
        self.shapes = obs_space_shapes

        # Defaults tuned for Pokemon Red observation sizes
        cnn_channels = cnn_channels or [32, 64, 64]
        cnn_kernels = cnn_kernels or [5, 3, 3]
        cnn_strides = cnn_strides or [2, 2, 1]
        map_channels = map_channels or [16, 32]
        map_kernels = map_kernels or [5, 3]
        map_strides = map_strides or [2, 2]
        fc_sizes = fc_sizes or [256, 256]

        # -- Screen CNN ---------------------------------------------------
        screen_shape = obs_space_shapes["screens"]  # (72, 80, C) or (72, 80, C, N)
        if len(screen_shape) == 4:
            # Stacked: (H, W, C, N) → treat C*N as input channels
            screen_h, screen_w = screen_shape[0], screen_shape[1]
            screen_c = screen_shape[2] * screen_shape[3]
        else:
            screen_h, screen_w = screen_shape[0], screen_shape[1]
            screen_c = screen_shape[2]

        self.screen_cnn, last_c = _make_cnn(screen_c, cnn_channels, cnn_kernels, cnn_strides)
        out_h, out_w = _conv_output_size(screen_h, screen_w, cnn_kernels, cnn_strides)
        screen_flat = last_c * out_h * out_w

        # -- Map CNN ------------------------------------------------------
        map_shape = obs_space_shapes["map"]  # (48, 48, 1)
        map_h, map_w = map_shape[0], map_shape[1]
        map_c = map_shape[2] if len(map_shape) > 2 else 1

        self.map_cnn, map_last_c = _make_cnn(map_c, map_channels, map_kernels, map_strides)
        map_out_h, map_out_w = _conv_output_size(map_h, map_w, map_kernels, map_strides)
        map_flat = map_last_c * map_out_h * map_out_w

        # -- Tactical branch (dedicated battle logic) ---------------------
        self._has_tactical = "tactical" in obs_space_shapes
        tactical_out = 0
        if self._has_tactical:
            tactical_size = int(np.prod(obs_space_shapes["tactical"]))
            self.tactical_fc = nn.Sequential(
                nn.Linear(tactical_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            tactical_out = 64

        # -- Scalar branch ------------------------------------------------
        # Everything except screens, map, and tactical gets flattened
        scalar_size = 0
        for key in self.obs_keys:
            if key in ("screens", "map", "tactical"):
                continue
            scalar_size += int(np.prod(obs_space_shapes[key]))

        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 128),
            nn.ReLU(),
        )
        scalar_out = 128

        # -- Fusion -------------------------------------------------------
        fusion_in = screen_flat + map_flat + scalar_out + tactical_out

        fusion_layers: list[nn.Module] = []
        prev = fusion_in
        for size in fc_sizes:
            fusion_layers.append(nn.Linear(prev, size))
            fusion_layers.append(nn.ReLU())
            prev = size
        self.fusion = nn.Sequential(*fusion_layers)
        self.fusion_out_size = fc_sizes[-1]

        # -- Heads --------------------------------------------------------
        self.policy_head = nn.Linear(self.fusion_out_size, num_actions)
        self.value_head = nn.Linear(self.fusion_out_size, 1)

        # Cache for value function (RLlib calls value_function() separately)
        self._features: torch.Tensor | None = None

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        obs : dict[str, Tensor]
            Batch of observations.  Each value has shape ``(B, *obs_shape)``.

        Returns
        -------
        logits : Tensor  (B, num_actions)
        value  : Tensor  (B, 1)
        """
        B = obs["screens"].shape[0]

        # -- Screen CNN --
        # Input: (B, H, W, C[, N]) → need (B, C_in, H, W)
        screens = obs["screens"].float()
        if screens.dim() == 5:
            # (B, H, W, C, N) → (B, C*N, H, W)
            screens = screens.permute(0, 3, 4, 1, 2).reshape(
                B, -1, screens.shape[1], screens.shape[2]
            )
        else:
            # (B, H, W, C) → (B, C, H, W)
            screens = screens.permute(0, 3, 1, 2)
        screens = screens / 255.0  # normalize pixels
        screen_features = self.screen_cnn(screens).reshape(B, -1)

        # -- Map CNN --
        minimap = obs["map"].float()
        if minimap.dim() == 4:
            # (B, H, W, C) → (B, C, H, W)
            minimap = minimap.permute(0, 3, 1, 2)
        elif minimap.dim() == 3:
            # (B, H, W) → (B, 1, H, W)
            minimap = minimap.unsqueeze(1)
        minimap = minimap / 255.0
        map_features = self.map_cnn(minimap).reshape(B, -1)

        # -- Scalars --
        scalar_parts = []
        for key in self.obs_keys:
            if key in ("screens", "map", "tactical"):
                continue
            val = obs[key].float().reshape(B, -1)
            scalar_parts.append(val)
        scalars = torch.cat(scalar_parts, dim=1)
        scalar_features = self.scalar_fc(scalars)

        # -- Tactical branch --
        branches = [screen_features, map_features, scalar_features]
        if self._has_tactical and "tactical" in obs:
            tactical = obs["tactical"].float().reshape(B, -1)
            tactical_features = self.tactical_fc(tactical)
            branches.append(tactical_features)

        # -- Fusion --
        combined = torch.cat(branches, dim=1)
        self._features = self.fusion(combined)

        logits = self.policy_head(self._features)
        value = self.value_head(self._features)
        return logits, value

    def get_value(self) -> torch.Tensor:
        """Return the value estimate from the last forward pass."""
        assert self._features is not None, "Call forward() first"
        return self.value_head(self._features)


# ---------------------------------------------------------------------------
# RLlib wrapper — only imported when Ray is available
# ---------------------------------------------------------------------------

def make_rllib_model_class():
    """Build and return the RLlib TorchModelV2 subclass.

    Deferred import so the module works without Ray installed.
    """
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.annotations import override

    class PokemonRLlibModel(TorchModelV2, nn.Module):
        """RLlib-compatible wrapper around PokemonNet.

        Registered as ``"pokemon_cnn"`` so you can use::

            config.training(model={"custom_model": "pokemon_cnn"})
        """

        def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        ):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            custom = model_config.get("custom_model_config", {})

            # Extract shapes from the observation space
            obs_shapes = {}
            for key, space in obs_space.original_space.spaces.items():
                obs_shapes[key] = space.shape

            self.net = PokemonNet(
                obs_space_shapes=obs_shapes,
                num_actions=num_outputs,
                cnn_channels=custom.get("cnn_channels"),
                cnn_kernels=custom.get("cnn_kernels"),
                cnn_strides=custom.get("cnn_strides"),
                map_channels=custom.get("map_channels"),
                map_kernels=custom.get("map_kernels"),
                map_strides=custom.get("map_strides"),
                fc_sizes=custom.get("fc_sizes"),
            )

        @override(TorchModelV2)
        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            logits, _ = self.net(obs)
            return logits, state

        @override(TorchModelV2)
        def value_function(self):
            return self.net.get_value().squeeze(-1)

    return PokemonRLlibModel

"""Validated configuration for RedGymEnv."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EnvConfig:
    """Typed, validated configuration — replaces the raw dict.

    Construct from the legacy dict via ``EnvConfig.from_dict(old_config)``.
    """

    gb_path: str = "../PokemonRed.gb"
    init_state: str = "../init.state"
    session_path: Path = field(default_factory=lambda: Path("runs"))

    headless: bool = True
    save_final_state: bool = False
    print_rewards: bool = True
    save_video: bool = False
    fast_video: bool = True

    action_freq: int = 24
    max_steps: int = 2048 * 80

    explore_weight: float = 1.0
    reward_scale: float = 1.0

    instance_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # legacy / rarely used
    debug: bool = False
    early_stop: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.session_path, str):
            self.session_path = Path(self.session_path)
        if self.action_freq < 1:
            raise ValueError(f"action_freq must be >= 1, got {self.action_freq}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")

    @classmethod
    def from_dict(cls, d: dict) -> EnvConfig:
        """Create from the legacy config dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

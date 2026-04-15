"""Gymnasium-compatible battle environment for PPO training.

Observation space matches the full-game v2 Dict *exactly* — non-battle fields
are zero-padded — so a policy trained here can be loaded into the full-game
PokemonNet with no shape rework.

Action space is Discrete(9):
    0..3 — use move slot 0..3
    4..8 — switch to party slot 1..5 (no-op in v0.1, reserved for v0.2)

Rewards are lightweight and tactical — the goal of the specialist is to
shape the tactical branch's feature layer, not invent a new story.
    +1.0  on win
    -1.0  on loss
    +Δhp  shaping per turn (opp HP lost − our HP lost, in HP fraction)
    -0.01 step penalty (encourage decisive play)
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engine import BattleEngine, BattleResult, BattleState, random_opponent_policy
from .entities import Pokemon
from .obs import tactical_obs
from .rng import BattleRNG
from .v2_contract import (
    BADGES_SIZE,
    EVENTS_SIZE,
    LEVEL_ENC_SIZE,
    MAP_SHAPE,
    NUM_ACTIONS,
    RECENT_ACTIONS_SIZE,
    SCREEN_SHAPE,
    TACTICAL_OBS_SIZE,
)


# ---------------------------------------------------------------------------
# Battle initial-state sampling
# ---------------------------------------------------------------------------

def _default_teams(rng: np.random.Generator) -> tuple[Pokemon, Pokemon]:
    """Sample a rough early-game matchup.

    Pool intentionally small for v0.1 — enough for the sim to train against
    Brock-adjacent encounters but not so wide that specialist convergence
    stalls.
    """
    starters = [
        ("CHARMANDER", 10, ["SCRATCH", "GROWL", "EMBER"]),
        ("BULBASAUR",  10, ["TACKLE", "GROWL", "LEECH_SEED"]),
        ("SQUIRTLE",   10, ["TACKLE", "TAIL_WHIP", "BUBBLE"]),
    ]
    wild_pool = [
        ("PIDGEY",   3, ["TACKLE", "SAND_ATTACK"]),
        ("RATTATA",  4, ["TACKLE", "TAIL_WHIP"]),
        ("SPEAROW",  5, ["PECK", "GROWL"]),
        ("GEODUDE",  10, ["TACKLE", "DEFENSE_CURL"]),
        ("ONIX",     12, ["TACKLE", "SCREECH", "BIND"]),
        ("CATERPIE", 3, ["TACKLE", "STRING_SHOT"]),
        ("WEEDLE",   3, ["POISON_STING", "STRING_SHOT"]),
    ]
    p = starters[rng.integers(0, len(starters))]
    o = wild_pool[rng.integers(0, len(wild_pool))]
    player = Pokemon.build(p[0], p[1], p[2])
    opponent = Pokemon.build(o[0], o[1], o[2])
    return player, opponent


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class PokemonBattleEnv(gym.Env):
    """Single-battle Gymnasium env.  Episode = one complete battle."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        sample_teams=None,
        opponent_policy=None,
        max_turns: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._sample_teams = sample_teams or _default_teams
        self._opponent_policy = opponent_policy or random_opponent_policy
        self._max_turns = max_turns

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "screens": spaces.Box(low=0, high=255, shape=SCREEN_SHAPE, dtype=np.uint8),
            "health":  spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "level":   spaces.Box(low=-1, high=1, shape=(LEVEL_ENC_SIZE,), dtype=np.float32),
            "badges":  spaces.MultiBinary(BADGES_SIZE),
            "events":  spaces.MultiBinary(EVENTS_SIZE),
            "map":     spaces.Box(low=0, high=255, shape=MAP_SHAPE, dtype=np.uint8),
            "recent_actions": spaces.MultiDiscrete([NUM_ACTIONS] * RECENT_ACTIONS_SIZE),
            "tactical": spaces.Box(
                low=-1, high=1, shape=(TACTICAL_OBS_SIZE,), dtype=np.float32
            ),
        })

        self._rng_np = np.random.default_rng(seed)
        self._battle_rng = BattleRNG(seed or 0)
        self.state: BattleState | None = None
        self.engine: BattleEngine | None = None

    # -- gym API -----------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng_np = np.random.default_rng(seed)
            self._battle_rng = BattleRNG(seed)

        player, opponent = self._sample_teams(self._rng_np)
        self.state = BattleState(player=player, opponent=opponent, battle_type=1)
        self.engine = BattleEngine(
            self.state, self._battle_rng, opponent_policy=self._opponent_policy,
        )
        return self._obs(), {"turn": 0}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self.engine is not None and self.state is not None
        s = self.state

        prev_opp_hp = s.opponent.hp / s.opponent.max_hp if s.opponent.max_hp else 0.0
        prev_our_hp = s.player.hp / s.player.max_hp if s.player.max_hp else 0.0

        result = self.engine.step(int(action))

        new_opp_hp = s.opponent.hp / s.opponent.max_hp if s.opponent.max_hp else 0.0
        new_our_hp = s.player.hp / s.player.max_hp if s.player.max_hp else 0.0

        dealt = max(0.0, prev_opp_hp - new_opp_hp)
        taken = max(0.0, prev_our_hp - new_our_hp)

        shaping = dealt - taken
        step_pen = -0.01

        if result == BattleResult.PLAYER_WIN:
            terminal_rew = 1.0
            terminated = True
        elif result == BattleResult.OPPONENT_WIN:
            terminal_rew = -1.0
            terminated = True
        else:
            terminal_rew = 0.0
            terminated = False

        truncated = (not terminated) and s.turn >= self._max_turns
        reward = float(shaping + step_pen + terminal_rew)

        info = {
            "turn": s.turn,
            "result": int(result),
            "events": list(s.last_events),
        }
        return self._obs(), reward, terminated, truncated, info

    # -- observation composition -----------------------------------------

    def _obs(self) -> dict[str, np.ndarray]:
        assert self.state is not None
        return {
            "screens": np.zeros(SCREEN_SHAPE, dtype=np.uint8),
            "health":  np.array(
                [self.state.player.hp / max(self.state.player.max_hp, 1)],
                dtype=np.float32,
            ),
            "level":   np.zeros(LEVEL_ENC_SIZE, dtype=np.float32),
            "badges":  np.zeros(BADGES_SIZE, dtype=np.int8),
            "events":  np.zeros(EVENTS_SIZE, dtype=np.int8),
            "map":     np.zeros(MAP_SHAPE, dtype=np.uint8),
            "recent_actions": np.zeros(RECENT_ACTIONS_SIZE, dtype=np.int64),
            "tactical": tactical_obs(self.state),
        }

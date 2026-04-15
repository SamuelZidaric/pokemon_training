"""Gymnasium-compatible battle environment for PPO training.

v0.4 (6v6 + full sensors): emits a flat Box(92,) containing the full tactical
obs spec — see TRANSFER_CONTRACT.md.  The env wraps a multi-mon sim with
real switching and rejection-by-penalty for invalid atomic actions.

Action space is Discrete(9):
    0..3 — use move slot 0..3 on active mon
    4..8 — switch to bench display position 0..4 (the k-th non-active party
           member in original-team order).  Invalid switches (out-of-range,
           fainted, already-active) are rejected with a -0.05 penalty and
           still cost the turn; the opponent takes its normal action.

Rewards (per TRANSFER_CONTRACT.md §4):
    +1.0  on win (all opp fainted)
    -1.0  on loss (all player fainted)
    +Δhp shaping per turn — whole-party HP fractions on both sides, so
          fainting a full-health opp mon and having an ally die for it
          nets roughly zero shaping.  Independent of which mon is active.
    -0.01 step penalty
    -0.05 iff the atomic action was rejected by the engine
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engine import BattleEngine, BattleResult, BattleState, random_opponent_policy
from .entities import Pokemon
from .obs import tactical_obs
from .rng import BattleRNG
from .v2_contract import (
    INVALID_ACTION_PENALTY,
    MAX_PARTY_SIZE,
    TACTICAL_OBS_SIZE,
)


# ---------------------------------------------------------------------------
# Multi-mon team sampling
# ---------------------------------------------------------------------------

_STARTERS = [
    ("CHARMANDER", ["SCRATCH", "GROWL", "EMBER", "LEER"]),
    ("BULBASAUR",  ["TACKLE", "GROWL", "LEECH_SEED", "VINE_WHIP"]),
    ("SQUIRTLE",   ["TACKLE", "TAIL_WHIP", "BUBBLE", "WATER_GUN"]),
]

# Extra party pool — mons the player would plausibly have caught on Route 1-3
# with level-appropriate learnsets.  Each entry is a fallback team-member
# candidate for slots 1..5.
_PARTY_POOL = [
    ("PIDGEY",    ["TACKLE", "SAND_ATTACK", "GUST"]),
    ("RATTATA",   ["TACKLE", "TAIL_WHIP", "QUICK_ATTACK"]),
    ("SPEAROW",   ["PECK", "GROWL", "LEER"]),
    ("NIDORAN_M", ["TACKLE", "LEER", "POISON_STING"]),
    ("NIDORAN_F", ["TACKLE", "GROWL", "SCRATCH"]),
    ("MANKEY",    ["SCRATCH", "LEER", "KARATE_CHOP"]),
    ("CATERPIE",  ["TACKLE", "STRING_SHOT"]),
    ("WEEDLE",    ["POISON_STING", "STRING_SHOT"]),
    ("ZUBAT",     ["LEECH_LIFE", "SUPERSONIC"]),
    ("ODDISH",    ["ABSORB", "POISONPOWDER", "SLEEP_POWDER"]),
    ("BELLSPROUT",["VINE_WHIP", "GROWTH", "SLEEP_POWDER"]),
]

# Same opponent pool as v0.3 — used for each opp party slot independently.
_WILD_POOL = [
    ("PIDGEY",    ["TACKLE", "SAND_ATTACK", "GUST"]),
    ("RATTATA",   ["TACKLE", "TAIL_WHIP", "QUICK_ATTACK"]),
    ("SPEAROW",   ["PECK", "GROWL", "LEER"]),
    ("GEODUDE",   ["TACKLE", "DEFENSE_CURL"]),
    ("ONIX",      ["TACKLE", "SCREECH", "BIND"]),
    ("CATERPIE",  ["TACKLE", "STRING_SHOT"]),
    ("WEEDLE",    ["POISON_STING", "STRING_SHOT"]),
    ("ODDISH",    ["ABSORB", "POISONPOWDER", "SLEEP_POWDER"]),
    ("BELLSPROUT",["VINE_WHIP", "GROWTH", "SLEEP_POWDER"]),
    ("ZUBAT",     ["LEECH_LIFE", "SUPERSONIC"]),
    ("EKANS",     ["WRAP", "POISON_STING", "LEER"]),
    ("SANDSHREW", ["SCRATCH", "DEFENSE_CURL", "SAND_ATTACK"]),
    ("MANKEY",    ["SCRATCH", "LEER", "KARATE_CHOP"]),
    ("NIDORAN_M", ["TACKLE", "LEER", "POISON_STING"]),
    ("NIDORAN_F", ["TACKLE", "GROWL", "SCRATCH"]),
    ("PIKACHU",   ["THUNDERSHOCK", "GROWL", "THUNDER_WAVE"]),
    ("PARAS",     ["SCRATCH", "STUN_SPORE"]),
]


def _sample_mon(pool: list[tuple[str, list[str]]], level: int,
                rng: np.random.Generator) -> Pokemon:
    species, moves = pool[rng.integers(0, len(pool))]
    return Pokemon.build(species, level, moves)


def _default_teams(rng: np.random.Generator) -> tuple[list[Pokemon], list[Pokemon]]:
    """Sample a 6v6 matchup.  Lead mon for the player is always a starter;
    the rest of the party is sampled from the broader route-1-3 pool.

    Team sizes vary: player 3-6 mons, opp 1-4 mons.  Wider variance than
    v0.3 so the policy sees both early-game 1v1 and mid-game 4v4 scenarios.
    """
    p_level = int(rng.integers(8, 19))
    # Player lead: starter.
    lead_species, lead_moves = _STARTERS[rng.integers(0, len(_STARTERS))]
    player_team = [Pokemon.build(lead_species, p_level, lead_moves)]

    # Player bench: sample without duplicating the starter species.
    p_size = int(rng.integers(3, MAX_PARTY_SIZE + 1))  # 3..6
    pool_filtered = [e for e in _PARTY_POOL if e[0] != lead_species]
    while len(player_team) < p_size:
        entry = pool_filtered[rng.integers(0, len(pool_filtered))]
        # Level variance within the party: each bench mon ±2 of lead.
        lv = max(3, min(25, p_level + int(rng.integers(-2, 3))))
        player_team.append(Pokemon.build(entry[0], lv, entry[1]))

    # Opp team: 1..4 mons, each sampled from wild pool.  Opp level tracks
    # player level with the same ±3 variance v0.3 used.
    o_size = int(rng.integers(1, 5))
    opp_team = []
    for _ in range(o_size):
        lv = max(3, min(25, p_level + int(rng.integers(-3, 4))))
        opp_team.append(_sample_mon(_WILD_POOL, lv, rng))

    return player_team, opp_team


# ---------------------------------------------------------------------------
# Party-wide HP fraction for reward shaping
# ---------------------------------------------------------------------------

def _party_hp_frac(party: list[Pokemon]) -> float:
    """Sum of (hp / max_hp) across all non-empty party slots, normalized by
    party size.  Fainted = 0 contribution.  Stays in [0, 1]."""
    if not party:
        return 0.0
    total = 0.0
    for m in party:
        total += (m.hp / m.max_hp) if m.max_hp else 0.0
    return total / len(party)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class PokemonBattleEnv(gym.Env):
    """6v6 Gymnasium env.  Episode = one complete battle (all of one side KO'd)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        sample_teams=None,
        opponent_policy=None,
        max_turns: int = 200,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._sample_teams = sample_teams or _default_teams
        self._opponent_policy = opponent_policy or random_opponent_policy
        self._max_turns = max_turns

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(TACTICAL_OBS_SIZE,), dtype=np.float32,
        )

        self._rng_np = np.random.default_rng(seed)
        self._battle_rng = BattleRNG(seed or 0)
        self.state: BattleState | None = None
        self.engine: BattleEngine | None = None

    # -- gym API -----------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng_np = np.random.default_rng(seed)
            self._battle_rng = BattleRNG(seed)

        player_team, opp_team = self._sample_teams(self._rng_np)
        self.state = BattleState(
            player_party=player_team, opp_party=opp_team, battle_type=1,
        )
        self.engine = BattleEngine(
            self.state, self._battle_rng, opponent_policy=self._opponent_policy,
        )
        return self._obs(), {
            "turn": 0,
            "player_team_size": len(player_team),
            "opp_team_size": len(opp_team),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.engine is not None and self.state is not None
        s = self.state

        prev_opp_hp = _party_hp_frac(s.opp_party)
        prev_our_hp = _party_hp_frac(s.player_party)

        result = self.engine.step(int(action))

        new_opp_hp = _party_hp_frac(s.opp_party)
        new_our_hp = _party_hp_frac(s.player_party)

        dealt = max(0.0, prev_opp_hp - new_opp_hp)
        taken = max(0.0, prev_our_hp - new_our_hp)

        shaping = dealt - taken
        step_pen = -0.01
        invalid_pen = INVALID_ACTION_PENALTY if s.last_action_was_invalid else 0.0

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
        reward = float(shaping + step_pen + terminal_rew + invalid_pen)

        info = {
            "turn": s.turn,
            "result": int(result),
            "events": list(s.last_events),
            "invalid_action": s.last_action_was_invalid,
            "active_slot": s.player_active,
        }
        return self._obs(), reward, terminated, truncated, info

    # -- observation composition -----------------------------------------

    def _obs(self) -> np.ndarray:
        assert self.state is not None
        return tactical_obs(self.state)

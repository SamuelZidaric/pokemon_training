"""Transfer contract: constants duplicated from v2/game_state.py.

Source of truth lives on branch ``claude/quizzical-sammet`` at
``v2/game_state.py``.  Because this sim lives on a sibling branch that can't
import across, we duplicate the values here and cover drift with
``tests/test_contract_drift.py`` which is run before every training session.

If any value in this file changes, the tactical-branch embedding layout
breaks and transferring weights to the full-game agent is unsafe.
"""
from __future__ import annotations


# --- Type index map (see v2/game_state.py:TYPE_ID_LIST) ---
TYPE_ID_LIST: list[int] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08,
    0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
]
TYPE_ID_TO_INDEX: dict[int, int] = {tid: i for i, tid in enumerate(TYPE_ID_LIST)}
NUM_POKEMON_TYPES: int = 15  # divisor is NUM_POKEMON_TYPES - 1 = 14


# --- Tactical observation vector ---
# v0.3: expanded 22 → 28 to give the policy direct access to major status
# and attack/defense stages on both actors.  This BREAKS the v0.1/v0.2
# transfer contract — v2/game_state.py on claude/quizzical-sammet must
# append the same 6 dims in the same order before any weight transfer.
TACTICAL_OBS_SIZE: int = 28

# Status ordinal encoding (shared with v2 — must match):
STATUS_ORDINAL: dict[str, int] = {
    "OK": 0, "PAR": 1, "SLP": 2, "BRN": 3, "PSN": 4, "FRZ": 5,
}
# Divisor for normalizing ordinal into [0, 1]
STATUS_ORDINAL_MAX: int = 5


# --- Full observation Dict space shapes (for zero-pad stubs in env.py) ---
# These mirror v2/red_gym_env_v2.py so PokemonBattleEnv can hand a complete
# Dict to any code expecting the full-game observation.
SCREEN_SHAPE = (72, 80, 4)          # frame_stacks=4
MAP_SHAPE = (48, 48, 1)             # coords_pad=12 → 12*4
BADGES_SIZE = 8
RECENT_ACTIONS_SIZE = 4
LEVEL_ENC_SIZE = 8                  # enc_freqs
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
EVENTS_SIZE = (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8   # 2232
NUM_ACTIONS = 7                     # DOWN, LEFT, RIGHT, UP, A, B, START

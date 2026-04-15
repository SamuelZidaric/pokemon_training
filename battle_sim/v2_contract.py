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
# v0.4: expanded 36 → 92.  Full sensor suite (remaining stat stages) + 6v6
# switching support (5 bench slots + battle-global meta).  See
# TRANSFER_CONTRACT.md for the full block layout.
#
# Block A (0..35)   : unchanged v0.3 active-mon tactical.
# Block B (36..43)  : 8 remaining stat stages — spe/spc/acc/eva × (self, opp).
# Block C (44..88)  : 5 bench slots × 9 dims each (hp, level, status onehot,
#                     offensive eff rollup, defensive eff rollup vs active opp).
# Block D (89..91)  : active_slot_index, self_alive_count, opp_remaining.
#
# Status is one-hot (5 dims per actor slot: PAR/SLP/BRN/PSN/FRZ; all-zero = OK).
# Ordinal was rejected in v0.3 as a DL anti-pattern.  Type matchups on bench
# are encoded as effectiveness ROLLUPS not type-index /14 — same reasoning
# (categorical-as-ordinal is noise) and keeps bench compact at 9 dims/slot.
TACTICAL_OBS_SIZE: int = 92

# Block boundaries (for asserts and layout sanity-checks).
TACTICAL_BLOCK_A_END: int = 36   # v0.3 active-mon tactical
TACTICAL_BLOCK_B_END: int = 44   # + 8 extra stages (spe/spc/acc/eva × 2 actors)
TACTICAL_BLOCK_C_END: int = 89   # + 5 bench slots × 9 dims
TACTICAL_BLOCK_D_END: int = 92   # + 3 battle-global meta dims

# Party / bench sizing — Gen 1 max party is 6.  Bench is party minus active.
MAX_PARTY_SIZE: int = 6
MAX_BENCH_SIZE: int = 5          # party - 1 active
BENCH_SLOT_DIMS: int = 9         # hp, level, status×5, off_eff, def_eff

# Status one-hot layout — index within the 5-slot per-actor sub-vector.
# "OK" is encoded as all-zeros, so it has no slot.
STATUS_ONEHOT_INDEX: dict[str, int] = {
    "PAR": 0, "SLP": 1, "BRN": 2, "PSN": 3, "FRZ": 4,
}
STATUS_ONEHOT_SIZE: int = 5

# Invalid-action penalty applied when the engine rejects an atomic action.
# See TRANSFER_CONTRACT.md §3.  PyBoy-side must match this exactly.
INVALID_ACTION_PENALTY: float = -0.05


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

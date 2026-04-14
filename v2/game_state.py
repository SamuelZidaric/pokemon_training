"""Centralized Pokemon Red memory address reading and game state access.

Replaces scattered magic hex addresses with a typed, readable API.
All addresses sourced from:
  - https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
  - https://github.com/pret/pokered/blob/master/constants/event_constants.asm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyboy import PyBoy


# ---------------------------------------------------------------------------
# Address constants
# ---------------------------------------------------------------------------

# Player position
PLAYER_X = 0xD362
PLAYER_Y = 0xD361
MAP_NUMBER = 0xD35E

# Battle state
IN_BATTLE = 0xD057

# Party
PARTY_SIZE = 0xD163
PARTY_SPECIES = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]

# Party Pokemon levels (one per slot, max 6)
PARTY_LEVELS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# Party Pokemon current HP (high byte; low byte is addr+1)
PARTY_HP = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]

# Party Pokemon max HP (high byte; low byte is addr+1)
PARTY_MAX_HP = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

# Badges
BADGES = 0xD356

# Opponent party levels
OPPONENT_LEVELS = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]

# Opponent active Pokemon
OPPONENT_ACTIVE_HP = 0xCFE6  # high byte; low byte is addr+1
OPPONENT_ACTIVE_MAX_HP = 0xCFF4  # high byte; low byte is addr+1
OPPONENT_ACTIVE_SPECIES = 0xCFE5
OPPONENT_ACTIVE_LEVEL = 0xCFF3
OPPONENT_ACTIVE_TYPE1 = 0xCFEA
OPPONENT_ACTIVE_TYPE2 = 0xCFEB

# Party Pokemon types (type1 and type2 for each slot)
PARTY_TYPE1 = [0xD170, 0xD19C, 0xD1C8, 0xD1F4, 0xD220, 0xD24C]
PARTY_TYPE2 = [0xD171, 0xD19D, 0xD1C9, 0xD1F5, 0xD221, 0xD24D]

# Party Pokemon moves (4 moves per slot, 6 slots)
PARTY_MOVES = [
    [0xD173, 0xD174, 0xD175, 0xD176],  # slot 0
    [0xD19F, 0xD1A0, 0xD1A1, 0xD1A2],  # slot 1
    [0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE],  # slot 2
    [0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA],  # slot 3
    [0xD223, 0xD224, 0xD225, 0xD226],  # slot 4
    [0xD24F, 0xD250, 0xD251, 0xD252],  # slot 5
]

# Party Pokemon move PP (4 PP values per slot, 6 slots)
PARTY_MOVE_PP = [
    [0xD188, 0xD189, 0xD18A, 0xD18B],  # slot 0
    [0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7],  # slot 1
    [0xD1E0, 0xD1E1, 0xD1E2, 0xD1E3],  # slot 2
    [0xD20C, 0xD20D, 0xD20E, 0xD20F],  # slot 3
    [0xD238, 0xD239, 0xD23A, 0xD23B],  # slot 4
    [0xD264, 0xD265, 0xD266, 0xD267],  # slot 5
]

# Battle menu state
BATTLE_MENU_CURSOR = 0xCC2D

# PC / box state
CURRENT_BOX_NUMBER = 0xD5A0
NUM_POKEMON_IN_BOX = 0xDA80
PC_BOX_FULL_FLAG = 0xD5A0  # bits indicate which boxes are full

# Menu state (helps detect PC interaction)
MENU_ITEM_ID = 0xCC26
TEXT_BOX_ID = 0xD125

# Pokemon type chart (Gen 1 type IDs)
POKEMON_TYPES = {
    0x00: "Normal",    0x01: "Fighting",  0x02: "Flying",
    0x03: "Poison",    0x04: "Ground",    0x05: "Rock",
    0x07: "Bug",       0x08: "Ghost",     0x14: "Fire",
    0x15: "Water",     0x16: "Grass",     0x17: "Electric",
    0x18: "Psychic",   0x19: "Ice",       0x1A: "Dragon",
}

# Number of distinct type IDs used in Gen 1 (for one-hot encoding)
NUM_POKEMON_TYPES = 15  # 0x00-0x1A, with gaps

# Type IDs ordered for encoding (index → type_id)
TYPE_ID_LIST = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08,
                0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A]
TYPE_ID_TO_INDEX = {tid: i for i, tid in enumerate(TYPE_ID_LIST)}

# Gen 1 type effectiveness chart: (atk_type, def_type) → multiplier
# Only super-effective (2.0) and not-very-effective (0.5) / immune (0.0)
# are listed.  Anything missing defaults to 1.0 (neutral).
_TYPE_CHART: dict[tuple[int, int], float] = {
    # Fire
    (0x14, 0x16): 2.0, (0x14, 0x19): 2.0, (0x14, 0x07): 2.0,  # → Grass, Ice, Bug
    (0x14, 0x14): 0.5, (0x14, 0x15): 0.5, (0x14, 0x05): 0.5, (0x14, 0x1A): 0.5,
    # Water
    (0x15, 0x14): 2.0, (0x15, 0x04): 2.0, (0x15, 0x05): 2.0,  # → Fire, Ground, Rock
    (0x15, 0x15): 0.5, (0x15, 0x16): 0.5, (0x15, 0x1A): 0.5,
    # Grass
    (0x16, 0x15): 2.0, (0x16, 0x04): 2.0, (0x16, 0x05): 2.0,  # → Water, Ground, Rock
    (0x16, 0x14): 0.5, (0x16, 0x16): 0.5, (0x16, 0x03): 0.5,
    (0x16, 0x02): 0.5, (0x16, 0x07): 0.5, (0x16, 0x1A): 0.5,
    # Electric
    (0x17, 0x15): 2.0, (0x17, 0x02): 2.0,  # → Water, Flying
    (0x17, 0x16): 0.5, (0x17, 0x17): 0.5, (0x17, 0x1A): 0.5,
    (0x17, 0x04): 0.0,  # → Ground: immune
    # Ice
    (0x19, 0x16): 2.0, (0x19, 0x04): 2.0, (0x19, 0x02): 2.0, (0x19, 0x1A): 2.0,
    (0x19, 0x14): 0.5, (0x19, 0x15): 0.5, (0x19, 0x19): 0.5,
    # Fighting
    (0x01, 0x00): 2.0, (0x01, 0x05): 2.0, (0x01, 0x19): 2.0,  # → Normal, Rock, Ice
    (0x01, 0x03): 0.5, (0x01, 0x02): 0.5, (0x01, 0x18): 0.5, (0x01, 0x07): 0.5,
    (0x01, 0x08): 0.0,  # → Ghost: immune
    # Psychic
    (0x18, 0x01): 2.0, (0x18, 0x03): 2.0,  # → Fighting, Poison
    (0x18, 0x18): 0.5,
    # Ground
    (0x04, 0x14): 2.0, (0x04, 0x17): 2.0, (0x04, 0x03): 2.0, (0x04, 0x05): 2.0,
    (0x04, 0x16): 0.5, (0x04, 0x07): 0.5,
    (0x04, 0x02): 0.0,  # → Flying: immune
    # Flying
    (0x02, 0x16): 2.0, (0x02, 0x01): 2.0, (0x02, 0x07): 2.0,
    (0x02, 0x05): 0.5, (0x02, 0x17): 0.5,
    # Poison
    (0x03, 0x16): 2.0,  # → Grass
    (0x03, 0x03): 0.5, (0x03, 0x04): 0.5, (0x03, 0x05): 0.5, (0x03, 0x08): 0.5,
    # Rock
    (0x05, 0x14): 2.0, (0x05, 0x19): 2.0, (0x05, 0x02): 2.0, (0x05, 0x07): 2.0,
    (0x05, 0x01): 0.5, (0x05, 0x04): 0.5,
    # Bug
    (0x07, 0x16): 2.0, (0x07, 0x03): 2.0, (0x07, 0x18): 2.0,
    (0x07, 0x14): 0.5, (0x07, 0x01): 0.5, (0x07, 0x02): 0.5, (0x07, 0x08): 0.5,
    # Ghost
    (0x08, 0x08): 2.0, (0x08, 0x18): 2.0,  # Gen 1: Ghost SE vs Psychic (bug in game)
    (0x08, 0x00): 0.0,  # → Normal: immune
    # Dragon
    (0x1A, 0x1A): 2.0,
    # Normal
    (0x00, 0x05): 0.5,
    (0x00, 0x08): 0.0,  # → Ghost: immune
}


def type_effectiveness(atk_type: int, def_type1: int, def_type2: int) -> float:
    """Compute type effectiveness multiplier for an attack type vs a defender.

    Returns the combined multiplier (e.g. 4.0 for double SE, 0.25 for
    double resist, 0.0 for immune).
    """
    mult1 = _TYPE_CHART.get((atk_type, def_type1), 1.0)
    if def_type1 == def_type2:
        return mult1
    mult2 = _TYPE_CHART.get((atk_type, def_type2), 1.0)
    return mult1 * mult2

# Event flags
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E  # expanded for SS Anne (old: 0xD7F6)
MUSEUM_TICKET_ADDR = 0xD754
MUSEUM_TICKET_BIT = 0

# Oak's Parcel quest flags (sourced from pokered/constants/event_constants.asm,
# verified against live tensorboard all_flags output from tactical_enhanced runs)
# EVENT_GOT_OAKS_PARCEL: flag 0x3E → byte 0xD74E bit 6 — obtained parcel at Viridian Mart
# EVENT_GOT_POKEDEX: flag 0x2D → byte 0xD74B bit 5 — Oak gave Pokedex (post-delivery)
# EVENT_GOT_POKEBALLS_FROM_OAK: flag 0x2C → byte 0xD74B bit 4 — Oak gave starter pokeballs
# EVENT_OAK_APPEARED_IN_PALLET: byte 0xD74B bit 7 — Oak blocks Route 1; fires pre-delivery
# EVENT_PALLET_AFTER_GETTING_POKEBALLS: byte 0xD74B bit 6 — post-delivery return to Pallet
# EVENT_FOLLOWED_OAK_INTO_LAB_2: byte 0xD74B bit 0 — initial cutscene
# Delivery itself fires Pokedex+Pokeballs simultaneously, so we use pokedex as the
# delivery proxy (no separate flag is reliably exposed at the RL step granularity).
OAKS_PARCEL_ADDR = 0xD74E
OAKS_PARCEL_BIT = 6
DELIVERED_PARCEL_ADDR = 0xD74B  # uses POKEDEX flag as delivery proxy
DELIVERED_PARCEL_BIT = 5
POKEDEX_ADDR = 0xD74B
POKEDEX_BIT = 5
OAKS_POKEBALLS_ADDR = 0xD74B
OAKS_POKEBALLS_BIT = 4


@dataclass(frozen=True)
class Position:
    x: int
    y: int
    map_id: int

    def __str__(self) -> str:
        return f"x:{self.x} y:{self.y} m:{self.map_id}"


class GameState:
    """High-level, read-only view of the Pokemon Red game state.

    Wraps a PyBoy instance and exposes named properties instead of
    raw hex addresses.  Every method is a pure read -- no side effects.
    """

    def __init__(self, pyboy: PyBoy) -> None:
        self._mem = pyboy.memory

    # -- helpers ----------------------------------------------------------

    def read(self, addr: int) -> int:
        """Read a single byte from memory."""
        return self._mem[addr]

    def read_word(self, addr: int) -> int:
        """Read a big-endian 16-bit word (e.g. HP values)."""
        return 256 * self._mem[addr] + self._mem[addr + 1]

    def read_bit(self, addr: int, bit: int) -> bool:
        return bool((self._mem[addr] >> bit) & 1)

    @staticmethod
    def bit_count(value: int) -> int:
        return bin(value).count("1")

    # -- position ---------------------------------------------------------

    @property
    def position(self) -> Position:
        return Position(
            x=self.read(PLAYER_X),
            y=self.read(PLAYER_Y),
            map_id=self.read(MAP_NUMBER),
        )

    @property
    def map_id(self) -> int:
        return self.read(MAP_NUMBER)

    # -- battle -----------------------------------------------------------

    @property
    def in_battle(self) -> bool:
        return self.read(IN_BATTLE) != 0

    # -- party ------------------------------------------------------------

    @property
    def party_size(self) -> int:
        return self.read(PARTY_SIZE)

    @property
    def party_species(self) -> list[int]:
        return [self.read(addr) for addr in PARTY_SPECIES]

    @property
    def party_levels(self) -> list[int]:
        return [self.read(addr) for addr in PARTY_LEVELS]

    @property
    def levels_sum(self) -> int:
        return sum(self.party_levels)

    @property
    def party_hp(self) -> list[int]:
        return [self.read_word(addr) for addr in PARTY_HP]

    @property
    def party_max_hp(self) -> list[int]:
        return [self.read_word(addr) for addr in PARTY_MAX_HP]

    @property
    def hp_fraction(self) -> float:
        total_hp = sum(self.party_hp)
        total_max = sum(self.party_max_hp)
        if total_max == 0:
            return 1.0
        return total_hp / total_max

    # -- badges -----------------------------------------------------------

    @property
    def badges_raw(self) -> int:
        return self.read(BADGES)

    @property
    def badge_count(self) -> int:
        return self.bit_count(self.badges_raw)

    @property
    def badges_array(self) -> np.ndarray:
        return np.array(
            [int(b) for b in f"{self.badges_raw:08b}"], dtype=np.int8
        )

    # -- opponent ---------------------------------------------------------

    @property
    def max_opponent_level(self) -> int:
        return max(self.read(addr) for addr in OPPONENT_LEVELS)

    # -- events -----------------------------------------------------------

    @property
    def event_bits(self) -> list[int]:
        return [
            int(bit)
            for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            for bit in f"{self.read(addr):08b}"
        ]

    @property
    def event_flag_sum(self) -> int:
        return sum(
            self.bit_count(self.read(addr))
            for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        )

    @property
    def has_museum_ticket(self) -> bool:
        return self.read_bit(MUSEUM_TICKET_ADDR, MUSEUM_TICKET_BIT)

    # -- Oak's Parcel quest chain ----------------------------------------

    @property
    def has_oaks_parcel(self) -> bool:
        """Got the parcel from Viridian Mart."""
        return self.read_bit(OAKS_PARCEL_ADDR, OAKS_PARCEL_BIT)

    @property
    def delivered_oaks_parcel(self) -> bool:
        """Delivered the parcel back to Oak."""
        return self.read_bit(DELIVERED_PARCEL_ADDR, DELIVERED_PARCEL_BIT)

    @property
    def has_pokedex(self) -> bool:
        """Oak gave the Pokedex (unlocks catching)."""
        return self.read_bit(POKEDEX_ADDR, POKEDEX_BIT)

    @property
    def has_oaks_pokeballs(self) -> bool:
        """Got 5 Pokeballs from Oak after delivering parcel."""
        return self.read_bit(OAKS_POKEBALLS_ADDR, OAKS_POKEBALLS_BIT)

    # -- battle details ---------------------------------------------------

    @property
    def battle_type(self) -> int:
        """0 = not in battle, 1 = wild, 2 = trainer."""
        return self.read(IN_BATTLE)

    @property
    def opponent_species(self) -> int:
        return self.read(OPPONENT_ACTIVE_SPECIES)

    @property
    def opponent_level(self) -> int:
        return self.read(OPPONENT_ACTIVE_LEVEL)

    @property
    def opponent_hp_fraction(self) -> float:
        hp = self.read_word(OPPONENT_ACTIVE_HP)
        max_hp = self.read_word(OPPONENT_ACTIVE_MAX_HP)
        if max_hp == 0:
            return 1.0
        return hp / max_hp

    @property
    def opponent_types(self) -> tuple[int, int]:
        return (self.read(OPPONENT_ACTIVE_TYPE1), self.read(OPPONENT_ACTIVE_TYPE2))

    @property
    def opponent_type_names(self) -> tuple[str, str]:
        t1, t2 = self.opponent_types
        return (
            POKEMON_TYPES.get(t1, "Unknown"),
            POKEMON_TYPES.get(t2, "Unknown"),
        )

    # -- party types & moves ----------------------------------------------

    @property
    def party_types(self) -> list[tuple[int, int]]:
        return [
            (self.read(t1), self.read(t2))
            for t1, t2 in zip(PARTY_TYPE1, PARTY_TYPE2)
        ]

    @property
    def lead_pokemon_types(self) -> tuple[int, int]:
        return (self.read(PARTY_TYPE1[0]), self.read(PARTY_TYPE2[0]))

    @property
    def party_moves(self) -> list[list[int]]:
        return [
            [self.read(addr) for addr in slot]
            for slot in PARTY_MOVES
        ]

    @property
    def party_move_pp(self) -> list[list[int]]:
        return [
            [self.read(addr) for addr in slot]
            for slot in PARTY_MOVE_PP
        ]

    @property
    def lead_moves(self) -> list[int]:
        return [self.read(addr) for addr in PARTY_MOVES[0]]

    @property
    def lead_pp(self) -> list[int]:
        return [self.read(addr) for addr in PARTY_MOVE_PP[0]]

    # -- PC / box state ---------------------------------------------------

    @property
    def current_box_number(self) -> int:
        return self.read(CURRENT_BOX_NUMBER) & 0x7F

    @property
    def pokemon_in_current_box(self) -> int:
        return self.read(NUM_POKEMON_IN_BOX)

    @property
    def is_box_full(self) -> bool:
        return self.pokemon_in_current_box >= 20

    # -- menu state -------------------------------------------------------

    @property
    def text_box_id(self) -> int:
        return self.read(TEXT_BOX_ID)

    # -- tactical: type advantage -----------------------------------------

    @property
    def best_move_effectiveness(self) -> float:
        """Best type effectiveness multiplier among lead Pokemon's moves.

        Returns the highest effectiveness of any move with PP > 0
        against the current opponent.  Returns 1.0 if not in battle.

        Note: We don't have a move→type lookup table in memory, so this
        uses the lead Pokemon's *own types* as a proxy for its STAB moves.
        For a full implementation, a move-type table would be needed.
        """
        if not self.in_battle:
            return 1.0
        def_t1, def_t2 = self.opponent_types
        lead_t1, lead_t2 = self.lead_pokemon_types

        eff1 = type_effectiveness(lead_t1, def_t1, def_t2)
        eff2 = type_effectiveness(lead_t2, def_t1, def_t2)
        return max(eff1, eff2)

    @property
    def type_advantage_signal(self) -> float:
        """Normalised type advantage: +1 super effective, 0 neutral, -1 resisted."""
        eff = self.best_move_effectiveness
        if eff >= 2.0:
            return 1.0
        elif eff <= 0.5:
            return -1.0
        elif eff == 0.0:
            return -1.0
        return 0.0

    @property
    def party_fainted_count(self) -> int:
        """Number of party Pokemon with 0 HP."""
        return sum(1 for hp in self.party_hp if hp == 0)

    # -- tactical observation encoding ------------------------------------

    def tactical_obs(self) -> np.ndarray:
        """Fixed-size tactical observation vector for the battle branch.

        Layout (22 floats):
          [0]      in_battle (0 or 1)
          [1]      battle_type (0/1/2 normalised)
          [2]      our HP fraction
          [3]      opponent HP fraction
          [4]      opponent level (normalised /100)
          [5]      type advantage signal (-1/0/+1)
          [6]      best move effectiveness (raw, capped at 4)
          [7..10]  lead move PP (normalised /40)
          [11]     party fainted count (normalised /6)
          [12]     party size (normalised /6)
          [13..14] lead types as indices (/15)
          [15..16] opponent types as indices (/15)
          [17..20] lead moves (normalised /165)
          [21]     num pokemon in box (/20)
        """
        in_b = float(self.in_battle)
        b_type = self.battle_type / 2.0

        our_hp = self.hp_fraction
        opp_hp = self.opponent_hp_fraction if self.in_battle else 0.0
        opp_lvl = self.opponent_level / 100.0 if self.in_battle else 0.0

        ta = self.type_advantage_signal
        bme = min(self.best_move_effectiveness, 4.0) / 4.0

        pp = [min(p, 40) / 40.0 for p in self.lead_pp]

        fainted = self.party_fainted_count / 6.0
        p_size = self.party_size / 6.0

        lt1, lt2 = self.lead_pokemon_types
        lt1_n = TYPE_ID_TO_INDEX.get(lt1, 0) / max(NUM_POKEMON_TYPES - 1, 1)
        lt2_n = TYPE_ID_TO_INDEX.get(lt2, 0) / max(NUM_POKEMON_TYPES - 1, 1)

        ot1, ot2 = self.opponent_types if self.in_battle else (0, 0)
        ot1_n = TYPE_ID_TO_INDEX.get(ot1, 0) / max(NUM_POKEMON_TYPES - 1, 1)
        ot2_n = TYPE_ID_TO_INDEX.get(ot2, 0) / max(NUM_POKEMON_TYPES - 1, 1)

        moves = [min(m, 165) / 165.0 for m in self.lead_moves]

        box = min(self.pokemon_in_current_box, 20) / 20.0

        return np.array([
            in_b, b_type, our_hp, opp_hp, opp_lvl,
            ta, bme,
            *pp,
            fainted, p_size,
            lt1_n, lt2_n, ot1_n, ot2_n,
            *moves,
            box,
        ], dtype=np.float32)


TACTICAL_OBS_SIZE: int = 22

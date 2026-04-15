"""Pokemon and Move entities for the battle simulator.

Design choice: dataclasses with mutable stat_stages, HP, and PP.  Pokemon
are constructed from species JSON but live in the battle state.

Stat-stage semantics (Gen 1):
    stage  -6 -5 -4 -3 -2 -1  0 +1 +2 +3 +4 +5 +6
    mult  25/100 28/100 33/100 40/100 50/100 66/100 1 1.5 2.0 2.5 3.0 3.5 4.0
The ratios are enforced via a lookup table; we store the stage integer and
apply the multiplier in ``effective_attack`` / ``effective_defense`` so the
crit formula can bypass it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .data_loader import load_moves, load_moves_by_id, load_pokemon


# Gen 1 stat-stage multipliers expressed as (num, den) to preserve the
# "floor after each multiplication" contract.
STAGE_MULT: dict[int, tuple[int, int]] = {
    -6: (25, 100), -5: (28, 100), -4: (33, 100), -3: (40, 100),
    -2: (50, 100), -1: (66, 100),
     0: (1, 1),
     1: (15, 10),  2: (20, 10),  3: (25, 10),
     4: (30, 10),  5: (35, 10),  6: (40, 10),
}


@dataclass
class Move:
    id: int
    name: str
    type_id: int
    power: int
    accuracy: int         # byte 0..255
    pp_max: int
    pp: int
    effect: str

    @classmethod
    def from_name(cls, name: str) -> "Move":
        d = load_moves()[name]
        return cls(
            id=d["id"], name=name, type_id=d["type_id"], power=d["power"],
            accuracy=d["accuracy"], pp_max=d["pp"], pp=d["pp"], effect=d["effect"],
        )

    @classmethod
    def from_id(cls, move_id: int) -> "Move":
        d = load_moves_by_id()[move_id]
        return cls(
            id=d["id"], name=d["name"], type_id=d["type_id"], power=d["power"],
            accuracy=d["accuracy"], pp_max=d["pp"], pp=d["pp"], effect=d["effect"],
        )


@dataclass
class Pokemon:
    species: str
    level: int
    type1_id: int
    type2_id: int
    base_hp: int
    base_atk: int
    base_def: int
    base_spd: int
    base_spc: int
    max_hp: int
    hp: int
    moves: list[Move]
    # stat stages
    atk_stage: int = 0
    def_stage: int = 0
    spd_stage: int = 0
    spc_stage: int = 0
    acc_stage: int = 0
    eva_stage: int = 0
    # status (v0.2 — stubbed for v0.1)
    status: str = "OK"

    @classmethod
    def build(cls, species: str, level: int, move_names: list[str]) -> "Pokemon":
        """Create a Pokemon with Gen 1 stat formulas (no IVs / EVs).

        Gen 1 stat formula at zero IVs/EVs:
            HP   = floor(((base + 0) * 2) * level / 100) + level + 10
            STAT = floor(((base + 0) * 2) * level / 100) + 5
        We ignore DVs/stat-exp for the sim — matches the tactical branch's
        level-only observation.
        """
        data = load_pokemon()[species.upper()]
        hp = ((data["hp"] * 2) * level) // 100 + level + 10
        atk = ((data["atk"] * 2) * level) // 100 + 5
        df  = ((data["def"] * 2) * level) // 100 + 5
        spd = ((data["spd"] * 2) * level) // 100 + 5
        spc = ((data["spc"] * 2) * level) // 100 + 5
        return cls(
            species=species.upper(), level=level,
            type1_id=data["type1_id"], type2_id=data["type2_id"],
            base_hp=data["hp"], base_atk=data["atk"], base_def=data["def"],
            base_spd=data["spd"], base_spc=data["spc"],
            max_hp=hp, hp=hp,
            moves=[Move.from_name(n) for n in move_names],
            # Runtime effective stats start equal to base-derived values;
            # the canonical values live on the instance as atk/def/etc.
        )

    # --- derived getters --------------------------------------------------

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    @property
    def atk(self) -> int:
        return ((self.base_atk * 2) * self.level) // 100 + 5

    @property
    def df(self) -> int:
        return ((self.base_def * 2) * self.level) // 100 + 5

    @property
    def spd(self) -> int:
        return ((self.base_spd * 2) * self.level) // 100 + 5

    @property
    def spc(self) -> int:
        return ((self.base_spc * 2) * self.level) // 100 + 5

    def effective_atk(self, physical: bool) -> int:
        base = self.atk if physical else self.spc
        stage = self.atk_stage if physical else self.spc_stage
        num, den = STAGE_MULT[stage]
        return (base * num) // den

    def effective_def(self, physical: bool) -> int:
        base = self.df if physical else self.spc
        stage = self.def_stage if physical else self.spc_stage
        num, den = STAGE_MULT[stage]
        return (base * num) // den

"""One-off parser: pret/pokered ASM → battle_sim/data/*.json.

Usage (from repo root):

    git clone --depth 1 https://github.com/pret/pokered.git pokered_src
    python battle_sim/scripts/parse_pokered.py

Outputs (written to battle_sim/data/):
    type_ids.json      — {TYPE_NAME: byte_id}
    move_ids.json      — {MOVE_NAME: int 1..165}
    moves.json         — {MOVE_NAME: {id, power, type, accuracy, pp, effect}}
    pokemon.json       — {SPECIES_NAME: {dex, hp, atk, def, spd, spc, type1, type2, catch_rate, base_exp}}
    type_chart.json    — [[atk_id, def_id, mult_str], ...]   mult_str ∈ {"SUPER", "NVE", "NO_EFFECT"}
    learnsets.json     — {SPECIES_NAME: {evolutions: [...], learnset: [[level, move], ...]}}
    POKERED_VERSION.txt — commit SHA of pokered_src HEAD at parse time
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
POKERED = REPO_ROOT / "pokered_src"
OUT = REPO_ROOT / "battle_sim" / "data"


# ---------------------------------------------------------------------------
# Type constants
# ---------------------------------------------------------------------------

def parse_type_ids() -> dict[str, int]:
    """Parse constants/type_constants.asm → {NAME: byte_id}.

    Handles `const NAME` (increments), `const_next N` (jump to N), and the
    NORMAL..GHOST physical block + UNUSED gap + FIRE..DRAGON special block.
    """
    text = (POKERED / "constants/type_constants.asm").read_text()
    out: dict[str, int] = {}
    val = 0
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("const_def"):
            val = 0
        elif m := re.match(r"const_next\s+(\d+)", line):
            val = int(m.group(1))
        elif m := re.match(r"const\s+(\w+)", line):
            out[m.group(1)] = val
            val += 1
        # skip DEF lines — they're derived
    # normalise the pokered name for psychic
    if "PSYCHIC_TYPE" in out:
        out["PSYCHIC"] = out["PSYCHIC_TYPE"]
    return out


# ---------------------------------------------------------------------------
# Move constants + move table
# ---------------------------------------------------------------------------

def parse_move_ids() -> dict[str, int]:
    """Parse constants/move_constants.asm → {NAME: int}."""
    text = (POKERED / "constants/move_constants.asm").read_text()
    out: dict[str, int] = {}
    val = 0
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if line.startswith("const_def"):
            val = 0
        elif m := re.match(r"const\s+(\w+)", line):
            out[m.group(1)] = val
            val += 1
    return out


def parse_moves(type_ids: dict[str, int]) -> dict[str, dict]:
    """Parse data/moves/moves.asm `move` macros → {NAME: {...}}."""
    text = (POKERED / "data/moves/moves.asm").read_text()
    # Macro: move NAME, EFFECT, POWER, TYPE, ACCURACY percent, PP
    pat = re.compile(
        r"^\s*move\s+(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)",
        re.MULTILINE,
    )
    out: dict[str, dict] = {}
    move_id = 0
    for m in pat.finditer(text):
        move_id += 1
        name, effect, power, type_name, acc_pct, pp = m.groups()
        # percent → byte: X percent = floor(X * 255 / 100). Gen 1 quirk.
        # Empirically: 100 percent = 255, 85 percent = 216, 70 percent = 178.
        # Using the RGBDS definition: percent = 1.0 * 255/100 = 2.55, so
        # "100 percent" = floor(100 * 65536 / 100) / 256 = 255 (roughly).
        # We store the byte value directly to avoid confusion downstream.
        acc_byte = (int(acc_pct) * 256 // 100) - 1 if int(acc_pct) == 100 \
                   else int(acc_pct) * 256 // 100
        # The above is an approximation of RGBDS's fixed-point math.
        # For exact Gen 1 behaviour we want: 100% → 0xFF (255).
        if int(acc_pct) == 100:
            acc_byte = 255
        else:
            acc_byte = int(round(int(acc_pct) * 255 / 100))
        out[name] = {
            "id": move_id,
            "effect": effect,
            "power": int(power),
            "type": type_name,
            "type_id": type_ids[type_name],
            "accuracy": acc_byte,
            "accuracy_pct": int(acc_pct),
            "pp": int(pp),
        }
    assert move_id == 165, f"expected 165 moves, got {move_id}"
    return out


# ---------------------------------------------------------------------------
# Base stats
# ---------------------------------------------------------------------------

def parse_base_stats(type_ids: dict[str, int]) -> dict[str, dict]:
    """Walk data/pokemon/base_stats/*.asm → {SPECIES: {...}}."""
    out: dict[str, dict] = {}
    base_dir = POKERED / "data/pokemon/base_stats"
    for f in sorted(base_dir.glob("*.asm")):
        text = f.read_text()
        # dex id
        dex_m = re.search(r"db\s+DEX_(\w+)", text)
        # stats: db HP, ATK, DEF, SPD, SPC
        stats_m = re.search(
            r"db\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\n\s*;\s*hp",
            text,
        )
        if stats_m is None:
            stats_m = re.search(
                r"db\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
                text,
            )
        # types: db TYPE1, TYPE2 ; type
        type_m = re.search(r"db\s+(\w+)\s*,\s*(\w+)\s*;\s*type", text)
        # catch rate
        catch_m = re.search(r"db\s+(\d+)\s*;\s*catch rate", text)
        # base exp
        exp_m = re.search(r"db\s+(\d+)\s*;\s*base exp", text)

        if not (dex_m and stats_m and type_m and catch_m and exp_m):
            continue

        name = dex_m.group(1)
        hp, atk, df, spd, spc = map(int, stats_m.groups())
        t1, t2 = type_m.group(1), type_m.group(2)
        out[name] = {
            "dex": None,  # filled via Pokedex order below
            "hp": hp, "atk": atk, "def": df, "spd": spd, "spc": spc,
            "type1": t1, "type2": t2,
            "type1_id": type_ids[t1], "type2_id": type_ids[t2],
            "catch_rate": int(catch_m.group(1)),
            "base_exp": int(exp_m.group(1)),
        }
    return out


# ---------------------------------------------------------------------------
# Type chart
# ---------------------------------------------------------------------------

MULT_MAP = {"SUPER_EFFECTIVE": "SUPER",
            "NOT_VERY_EFFECTIVE": "NVE",
            "NO_EFFECT": "NO_EFFECT"}

def parse_type_chart(type_ids: dict[str, int]) -> list[list]:
    """Parse data/types/type_matchups.asm → list of [atk_id, def_id, mult]."""
    text = (POKERED / "data/types/type_matchups.asm").read_text()
    pat = re.compile(
        r"^\s*db\s+(\w+)\s*,\s*(\w+)\s*,\s*(SUPER_EFFECTIVE|NOT_VERY_EFFECTIVE|NO_EFFECT)",
        re.MULTILINE,
    )
    rows = []
    for m in pat.finditer(text):
        atk, dfn, mult = m.groups()
        rows.append([type_ids[atk], type_ids[dfn], MULT_MAP[mult]])
    assert rows, "type_matchups.asm produced 0 rows"
    return rows


# ---------------------------------------------------------------------------
# Learnsets + evolutions
# ---------------------------------------------------------------------------

def parse_learnsets() -> dict[str, dict]:
    """Parse data/pokemon/evos_moves.asm.

    Format per species:
        NameEvosMoves:
            [; Evolutions]
            db EVOLVE_LEVEL, level, SPECIES
            db EVOLVE_ITEM,  ITEM,  1, SPECIES
            db EVOLVE_TRADE, 1,     SPECIES
            db 0
            [; Learnset]
            db LEVEL, MOVE
            ...
            db 0
    """
    text = (POKERED / "data/pokemon/evos_moves.asm").read_text()
    # Split on labels ending in EvosMoves:
    blocks = re.split(r"^(\w+)EvosMoves:\s*$", text, flags=re.MULTILINE)
    # blocks = ["<prelude>", name1, body1, name2, body2, ...]
    out: dict[str, dict] = {}
    for i in range(1, len(blocks), 2):
        name = blocks[i].upper()
        body = blocks[i + 1]
        evos, learnset = [], []
        # Take lines up to the next label or section break
        seen_zero = 0
        for raw in body.splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("db 0") or line == "db 0":
                seen_zero += 1
                if seen_zero >= 2:
                    break
                continue
            if line.startswith("db "):
                parts = [p.strip() for p in line[3:].split(",")]
                if seen_zero == 0:
                    # evolution
                    evos.append(parts)
                else:
                    # learnset: [level, move]
                    if len(parts) >= 2:
                        try:
                            lvl = int(parts[0])
                        except ValueError:
                            continue
                        learnset.append([lvl, parts[1]])
        out[name] = {"evolutions": evos, "learnset": learnset}
    return out


# ---------------------------------------------------------------------------
# Pokedex order (dex numbers)
# ---------------------------------------------------------------------------

def parse_pokedex_order(base_stats: dict[str, dict]) -> None:
    """Fill dex numbers using constants/pokedex_constants.asm if present."""
    f = POKERED / "constants/pokedex_constants.asm"
    if not f.exists():
        return
    text = f.read_text()
    val = 0
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if line.startswith("const_def"):
            val = 0
        elif m := re.match(r"const\s+DEX_(\w+)", line):
            val += 1
            name = m.group(1)
            if name in base_stats:
                base_stats[name]["dex"] = val


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    if not POKERED.exists():
        print(f"error: {POKERED} not found. Clone pret/pokered first:",
              file=sys.stderr)
        print("  git clone --depth 1 https://github.com/pret/pokered.git pokered_src",
              file=sys.stderr)
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)

    type_ids = parse_type_ids()
    move_ids = parse_move_ids()
    moves = parse_moves(type_ids)
    base_stats = parse_base_stats(type_ids)
    parse_pokedex_order(base_stats)
    chart = parse_type_chart(type_ids)
    learn = parse_learnsets()

    # Sanity checks
    assert len(moves) == 165, f"moves: expected 165 got {len(moves)}"
    assert len(base_stats) == 151, f"base_stats: expected 151 got {len(base_stats)}"
    # Ghost→Psychic must be NO_EFFECT per cartridge data
    psy = type_ids["PSYCHIC_TYPE"]
    ghost = type_ids["GHOST"]
    ghost_psy = [r for r in chart if r[0] == ghost and r[1] == psy]
    assert ghost_psy == [[ghost, psy, "NO_EFFECT"]], \
        f"expected Ghost→Psychic NO_EFFECT, got {ghost_psy}"

    out_map = {
        "type_ids.json": type_ids,
        "move_ids.json": move_ids,
        "moves.json": moves,
        "pokemon.json": base_stats,
        "type_chart.json": chart,
        "learnsets.json": learn,
    }
    for name, payload in out_map.items():
        (OUT / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {OUT / name} ({len(payload)} entries)")

    # Pokered pin
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=POKERED
        ).decode().strip()
        (OUT / "POKERED_VERSION.txt").write_text(sha + "\n")
        print(f"pokered pinned at {sha}")
    except Exception as e:
        print(f"warning: could not read pokered SHA: {e}")


if __name__ == "__main__":
    main()

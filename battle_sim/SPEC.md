# Battle Simulator — v0.1 Spec

Phase A of the specialist-agent architecture. Scripted Gen 1 battle simulator
in pure Python designed for **weight-transfer-compatible** training: the
observation and action interfaces mirror the existing `tactical` branch of
`PokemonNet`, so a policy trained on this sim can be loaded into the full-game
agent without shape or index rework.

All references to `v2/` in this document point at the source-of-truth branch
`claude/quizzical-sammet` (worktree: `.claude/worktrees/quizzical-sammet/v2/`).
Do not read this session's `v2/` — it is stale and lacks `game_state.py`.

---

## Part 1 — Transfer Contract Audit

### 1.1 Tactical observation vector

Defined in [game_state.py:492](.claude/worktrees/quizzical-sammet/v2/game_state.py) as
`GameState.tactical_obs()`, consumed as `obs["tactical"]` by `PokemonNet`
(see [pokemon_model.py:182](.claude/worktrees/quizzical-sammet/v2/pokemon_model.py)).

- **Shape:** `(22,)`, `dtype=np.float32`
- **Constant name:** `TACTICAL_OBS_SIZE = 22`
- **Gym space:** `spaces.Box(low=-1, high=1, shape=(22,), dtype=np.float32)` at
  [red_gym_env_v2.py:131](.claude/worktrees/quizzical-sammet/v2/red_gym_env_v2.py)
- **Network branch:** `tactical_fc = Linear(22, 64) → ReLU → Linear(64, 64) → ReLU`,
  concatenated with screen-CNN / map-CNN / scalar features before the fusion MLP.
  This is the block whose weights the sim-trained policy will own.

Field table (must be reproduced byte-for-byte in the sim):

| Idx | Field | Scaling | Source RAM (PyBoy) | Sim mirror requirement |
|----:|-------|---------|--------------------|------------------------|
| 0 | `in_battle` | `float(bool)`, 0 or 1 | `0xD057 != 0` | Always 1.0 inside sim. Keep the slot for transfer. |
| 1 | `battle_type` | `raw / 2.0` → {0, 0.5, 1.0} | `0xD057` (0/1/2) | 0.5 = wild, 1.0 = trainer. Emit whichever is being trained. |
| 2 | our HP fraction | `sum(party_hp)/sum(party_max_hp)` (clamp 1.0 if max=0) | party HP/maxHP tables | Recompute the same way every step. |
| 3 | opponent HP fraction | `opp_hp/opp_max_hp`, 0.0 if not in battle | `0xCFE6`/`0xCFF4` (word each) | Active opponent only. |
| 4 | opponent level | `lvl/100`, 0 if not in battle | `0xCFF3` | Same divisor. |
| 5 | type-advantage signal | `{-1, 0, +1}` from `type_advantage_signal` | derived | Use **identical lookup** (see 1.2). Critically: current code uses lead's *own types* as a move-type proxy — sim MUST do the same in v0.1 even though it has real move types available (else the network sees a different distribution). Document as Known Divergence #1 and revisit in v0.2. |
| 6 | best-move effectiveness | `min(eff, 4.0) / 4.0` | derived | Same proxy as idx 5. |
| 7–10 | lead PP (4 slots) | `min(pp, 40) / 40.0`, per slot | `PARTY_MOVE_PP[0][0..3]` | Same cap, same divisor. Zero-filled for empty move slots (value 0). |
| 11 | party fainted count | `sum(hp==0) / 6.0` | derived | Same. |
| 12 | party size | `party_size / 6.0` | `0xD163` | Same. |
| 13 | lead type 1 index | `TYPE_ID_TO_INDEX[t1] / 14` | `0xD170` | **Must use the same `TYPE_ID_LIST` ordering** (see 1.3). |
| 14 | lead type 2 index | `TYPE_ID_TO_INDEX[t2] / 14` | `0xD171` | Same. |
| 15 | opponent type 1 | `/14`, 0 if not in battle | `0xCFEA` | Same. |
| 16 | opponent type 2 | `/14`, 0 if not in battle | `0xCFEB` | Same. |
| 17–20 | lead moves (4 slots) | `min(move_id, 165) / 165.0` | `PARTY_MOVES[0][0..3]` | Raw Gen 1 move ID 1..165. Zero-fill empty slots. |
| 21 | box count | `min(n, 20) / 20.0` | `0xDA80` | Battle sim has no PC; emit **0.0** (zero-pad). Flagged as a full-game-only field. |

**Zero-pad fields for sim:** index 21 (no PC in battle sim). All others are
legitimately populated from sim state.

**Ranges:** every value is in `[0, 1]` except idx 5 which is in `{-1, 0, +1}`
(the Box `low=-1` exists to accommodate exactly this field). Do not clip.

### 1.2 Type chart (must match exactly)

Source of truth: `_TYPE_CHART` dict at
[game_state.py:115](.claude/worktrees/quizzical-sammet/v2/game_state.py).
Multipliers are `{0.0, 0.5, 1.0, 2.0}`. `type_effectiveness()` multiplies the
two defender-type lookups — double-effective = 4.0, double-resist = 0.25,
either-immune → 0.0.

**Known incorrect entries to resolve before freezing the contract** (see Part 5):

- `(0x08, 0x18): 2.0` — labelled "Gen 1: Ghost SE vs Psychic (bug in game)"
  but the in-game result of that bug is that Ghost moves do **0 damage** to
  Psychic, not 2×. Current env teaches the agent a counterfactual.
- Missing `(0x03, 0x07): 2.0` — Gen 1 Poison is SE vs Bug (removed in Gen 2).
  Current env defaults to 1.0.
- Possibly more omissions — a full diff against
  `pret/pokered/data/types/type_matchups.asm` is a v0.1 prerequisite and
  the output of `parse_pokered.py` will be authoritative.

Whatever we decide for the env chart, the sim chart must be bit-identical.

### 1.3 Type ID index map

`TYPE_ID_LIST` at [game_state.py:108](.claude/worktrees/quizzical-sammet/v2/game_state.py):

```
[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x08,
 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A]
```

15 entries (Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost, Fire,
Water, Grass, Electric, Psychic, Ice, Dragon). The gaps (0x06, 0x09–0x13) are
reserved/Gen-2 and never appear in Gen 1 RAM. `NUM_POKEMON_TYPES = 15`, divisor
for normalization is `14`. Sim **must import and reuse this list directly** (or
duplicate it verbatim) — any reordering breaks the tactical branch's learned
embeddings.

### 1.4 Move ID space

`PARTY_MOVES` reads raw bytes `1..165` (Gen 1 move count). Normalized `/165.0`.
Move ID `0` = empty slot. No one-hot — the network sees the raw scalar. The
sim must:

- Emit the **same Gen 1 move IDs** (1=POUND, 2=KARATE_CHOP, ..., 165=STRUGGLE).
- Use the authoritative mapping from `pret/pokered/data/moves/moves.asm`
  macro order (the `move` macro invocation index is the ID — see Part 4).

### 1.5 Action space

Env action space is `Discrete(len(self.valid_actions)) = Discrete(7)`:
`[DOWN, LEFT, RIGHT, UP, A, B, START]` — raw D-pad + buttons. The full-game
policy head outputs logits over these 7 actions.

**Design decision for the sim (needs user approval):** Two options.

1. **Button-parity action space (recommended for transfer).** Sim advances a
   scripted battle UI state machine; the agent must output the same 7-action
   sequence (UP/DOWN to navigate the fight menu, A to confirm, etc.). Preserves
   policy-head shape exactly. Cost: ~3–6 wasted steps per decision on menu
   navigation; damages the 50k battles/sec goal; needs a menu state model.

2. **Semantic action space `Discrete(9)` = `[move0, move1, move2, move3, switch1..switch5, run]`.**
   10× simpler sim, hits throughput targets, but the policy head needs a
   translation layer when loaded into the full-game agent — an adapter module
   that maps semantic actions into the 7-button sequence. The tactical *branch
   features* still transfer cleanly (they're pre-fusion); only the final policy
   logits need remapping.

Recommendation: **option 2** with a thin `BattleActionAdapter` loaded in Week
4. The transfer guarantee in this project is about the *tactical feature
branch*, not the final policy head. Option 1's throughput hit kills the
fast-iteration premise.

### 1.6 Sim-side non-goals (confirmed zero-pad or absent)

- `screens`, `map`, `health`, `level`, `badges`, `events`, `recent_actions` —
  the battle sim does **not** produce these. The sim's `PokemonBattleEnv`
  (Week 3) will emit them as fixed zero arrays matching their shapes so the
  full-game `PokemonNet` can accept sim observations unchanged. Shapes from
  [red_gym_env_v2.py:111](.claude/worktrees/quizzical-sammet/v2/red_gym_env_v2.py):
  - `screens`: `(72, 80, 4)` uint8 zeros
  - `health`: `(1,)` float (can be real)
  - `level`: `(8,)` float (Fourier encoded — emit zeros; see note)
  - `badges`: `MultiBinary(8)` zeros
  - `events`: `MultiBinary((0xD87E - 0xD747) * 8) = MultiBinary(2232)` zeros
  - `map`: `(48, 48, 1)` uint8 zeros
  - `recent_actions`: `MultiDiscrete([7]*4)` zeros
- Note on `level`: it's the Fourier encoding of `0.02 * levels_sum`, not raw.
  If the sim emits zeros, the agent loses one scalar — acceptable for a battle
  specialist, and better than feeding noise.

---

## Part 2 — Module Layout (confirmed)

Proposed layout in the original task stands, with one addition (`obs.py`) for
the transfer-contract encoder so the field table above is owned by code, not
scattered across engine call sites.

```
battle_sim/
    SPEC.md                      ← this document
    data/
        pokemon.json             ← parsed base stats + types
        moves.json               ← parsed move attributes
        type_chart.json          ← parsed type matchups
        learnsets.json           ← parsed level-up moves + evolutions
    scripts/
        parse_pokered.py         ← one-off ASM → JSON parser
    engine.py                    ← turn-order state machine, faint checks, battle loop
    entities.py                  ← Pokemon + Move dataclasses, stat stages
    damage.py                    ← pure damage/hit/crit functions
    rng.py                       ← seeded RNG wrapper (stream of bytes)
    obs.py                       ← builds the 22-float tactical vector + zero-padded
                                    full Dict obs; single source of truth for
                                    encoding, imports TYPE_ID_LIST from v2/
    env.py                       ← (Week 3) Gymnasium env wrapping engine + obs
    adapter.py                   ← (Week 4) semantic-action ↔ 7-button translator
    tests/
        test_v01.py              ← differential harness (PyBoy diff)
        test_damage.py           ← damage-formula unit tests with Bulbapedia vectors
        test_obs_parity.py       ← asserts sim obs == env obs on matched states
```

**Key architectural points:**

- `obs.py` should `from v2.game_state import TYPE_ID_LIST, TYPE_ID_TO_INDEX,
  TACTICAL_OBS_SIZE` so drift is impossible. If cross-package imports are
  painful, duplicate the list with a unit test asserting equality against the
  v2 version.
- Engine is decoupled from observation — `obs.py` is the only module that
  knows the vector layout. Lets us change the sim internals without touching
  the transfer contract.
- `damage.py` is pure functions over `(attacker, defender, move, rng_bytes)`
  so damage can be unit-tested against Bulbapedia vectors independently of
  the battle loop.

**Deferred from v0.1:** legendaries, trades, 6v6, mid-battle evolution, HM
moves, full status condition set (v0.2 delivers par/slp/brn/psn/frz).

---

## Part 3 — Gen 1 Damage Formula (canonical)

Pseudocode below is the contract for `damage.py`. Every `floor()` is mandatory.

```python
def damage(attacker, defender, move, rng, is_crit):
    # Crit in Gen 1: uses unmodified base Atk/Def (ignore stat stages entirely)
    if is_crit:
        atk = attacker.base_atk if move.is_physical else attacker.base_spc
        dfn = defender.base_def if move.is_physical else defender.base_spc
        level = 2 * attacker.level  # crit doubles the level term, not a ×2 postmult
        crit_mult = 1                # no additional multiplier; the ×2 is folded via level
    else:
        atk = attacker.effective_atk(move)   # applies stat stages
        dfn = defender.effective_def(move)
        level = attacker.level
        crit_mult = 1

    base = ((2 * level * crit_mult // 5 + 2) * move.power * atk // dfn) // 50 + 2

    # Sequential multipliers, each truncated
    if move.type in (attacker.type1, attacker.type2):
        base = base * 3 // 2                     # STAB = 1.5, via integer math

    type_mult_num, type_mult_den = type_matchup_ratio(move.type,
                                                       defender.type1,
                                                       defender.type2)
    base = base * type_mult_num // type_mult_den  # 0, 1/2, 1, 2, 4 combos

    if base == 0:
        return 0                                  # immunity short-circuit

    # Damage roll: byte in [217, 255] inclusive (217/255 ≈ 0.85)
    roll = rng.next_damage_roll()                 # returns int in [217, 255]
    base = base * roll // 255

    return max(1, base)                           # Gen 1: non-zero hits floor at 1
```

**Critical-hit probability** (Gen 1): `threshold = base_speed // 2` (doubled for
high-crit moves like Slash, Karate Chop, Crabhammer, Razor Leaf). Roll a byte;
crit if `roll < threshold` (cap at 255). Record this; the differential harness
will force the RNG to reproduce PyBoy's crit exactly.

**Hit check:** `accuracy_byte = move.accuracy` (already 0..255). `rng.next() < accuracy_byte` ⇒ hit. Gen 1 quirk: even `accuracy=255` misses ~1/256 of the
time because the comparison is `<`, not `<=`. **Replicate this.**

---

## Part 4 — `pret/pokered` Data to Parse

Prerequisite: `git clone https://github.com/pret/pokered` into the repo root
(gitignored). Tag/commit pinned in `battle_sim/data/POKERED_VERSION.txt` by
the parser for reproducibility.

| Output JSON | Source file | Macro / section | Notes |
|-------------|-------------|-----------------|-------|
| `pokemon.json` | `data/pokemon/base_stats/*.asm` (151 files, one per species) | Each file begins with a `db DEX_NUM` line, then `db HP, ATK, DEF, SPD, SPC` (base stats), `db TYPE1, TYPE2`, `db CATCH_RATE`, `db BASE_EXP_YIELD`. Types are `_CONSTANT` names (e.g. `FIRE`) resolved from `constants/type_constants.asm`. | Parser walks the directory; filename is the species key (`bulbasaur.asm` → `BULBASAUR`). |
| `moves.json` | `data/moves/moves.asm` | One `move NAME, EFFECT, POWER, TYPE, ACCURACY, PP` macro per line. **The line index is the move ID** (starting at 1). | Also grab `EFFECT_*` constants from `data/moves/effects_pointers.asm` for v0.2 status-effect dispatch. |
| `type_chart.json` | `data/types/type_matchups.asm` | Sequence of `db ATK, DEF, MULT` rows. Multiplier is the raw byte (20 = 2×, 05 = 0.5×, 00 = 0×). Terminated by `db $ff`. | Must also consult `engine/battle/core.asm` for the Ghost-vs-Psychic code path — the *data* says 2×, the *engine* bug makes it 0×. Decide which the sim emulates (see Part 5). |
| `learnsets.json` | `data/pokemon/evos_moves.asm` | Per species: a block of `db EVOLVE_LEVEL/STONE/TRADE, ...` evolution entries terminated by `db 0`, then `db LEVEL, MOVE_NAME` level-up entries terminated by `db 0`. | v0.1 can emit the data but not use evolution; v0.2 uses level-up moves for trainer AI team-building. |
| *(side output)* `type_ids.json` | `constants/type_constants.asm` | `const NORMAL` / `const FIGHTING` / … ordered definitions. Produces `NAME → ID` map. | Cross-check against `TYPE_ID_LIST` — any mismatch is a transfer-breaker. |
| *(side output)* `move_ids.json` | `constants/move_constants.asm` | `const POUND` / `const KARATE_CHOP` / … | Gives us the canonical name↔ID map. |

### `parse_pokered.py` skeleton

```python
"""One-off parser: pret/pokered ASM → battle_sim/data/*.json.

Run once after cloning pokered; commits the JSON. Re-run if pokered is bumped.
Output format prioritizes O(1) lookup at sim startup.
"""
from __future__ import annotations
import json, re
from pathlib import Path

POKERED_ROOT = Path("pokered")          # adjust
DATA_OUT    = Path("battle_sim/data")

def parse_type_constants(): ...         # build {"NORMAL": 0x00, ...}
def parse_move_constants(): ...         # build {"POUND": 1, ...}
def parse_base_stats(type_ids, species_ids) -> dict:
    # Walk data/pokemon/base_stats/*.asm; parse db lines; return
    # {"BULBASAUR": {"dex":1, "hp":45, "atk":49, "def":49, "spd":45,
    #                "spc":65, "type1":"GRASS", "type2":"POISON",
    #                "catch_rate":45, "base_exp":64}, ...}
    ...
def parse_moves(type_ids) -> dict: ...  # {"TACKLE": {"id":33, "power":35, "type":"NORMAL", "acc":255, "pp":35, "effect":"NO_ADDITIONAL_EFFECT"}, ...}
def parse_type_chart(type_ids) -> list: # [(atk_id, def_id, mult_byte), ...]
    # Mult byte: 0x14→2.0, 0x05→0.5, 0x00→0.0
    ...
def parse_evos_moves(species_ids, move_ids) -> dict: ...

def main():
    type_ids    = parse_type_constants()
    move_ids    = parse_move_constants()
    species_ids = {f.stem.upper(): i+1 for i, f in enumerate(sorted(
        (POKERED_ROOT / "data/pokemon/base_stats").glob("*.asm")))}

    out = {
        "type_ids.json":  type_ids,
        "move_ids.json":  move_ids,
        "pokemon.json":   parse_base_stats(type_ids, species_ids),
        "moves.json":     parse_moves(type_ids),
        "type_chart.json": parse_type_chart(type_ids),
        "learnsets.json": parse_evos_moves(species_ids, move_ids),
    }
    for name, payload in out.items():
        (DATA_OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Sanity check: assert TYPE_ID_LIST ordering matches pokered constants
    ...

if __name__ == "__main__":
    main()
```

Line-range targeting per file isn't reliable — pokered is hand-maintained —
but each file has stable section markers (`; evolutions`, `; learnset`, etc.)
that regexes can pin to.

### Gotchas

- **Accuracy is byte 0..255, not percent.** Raw `0xFF` = 100% (nominally).
  Keep in byte domain to reproduce the 1/256-miss quirk.
- **Effects are indirected via pointer tables.** For v0.1 we only care about
  `power > 0` direct-damage moves; status/one-hit-KO/multi-hit moves are v0.2.
- **Some "moves" have `power=0`** but still deal damage via fixed formulas
  (Dragon Rage, Night Shade, Seismic Toss, Sonic Boom, Psywave). Flag these
  in `moves.json` with `fixed_damage: {...}` so the engine can dispatch.
- **Struggle** is move ID 165 with special no-PP semantics — v0.2 at earliest.
- **Counter** reads the opponent's last-used move category (physical vs
  special). Out of scope for v0.1.

---

## Part 5 — Gen 1 Quirks: Audit of Current PyBoy Env

Every item below is a place where `v2/` currently models Gen 1 differently
from how the cartridge actually behaves. These cause **transfer drift**: the
sim either matches the env (and thus also mis-models reality) or matches
reality (and produces observations that contradict the env the agent trains
against). Each needs an explicit decision before v0.1 freezes.

### 5.1 Ghost → Psychic effectiveness — **⚠ Critical**

Location: [game_state.py:157](.claude/worktrees/quizzical-sammet/v2/game_state.py),
`(0x08, 0x18): 2.0`.

- **Data (type_matchups.asm):** Ghost is 2× vs Psychic.
- **Actual in-game behavior:** Ghost moves do 0× damage to Psychic. This is
  the famous Gen 1 bug — the engine's type lookup ends early when it hits the
  Psychic→Ghost immunity row first, treating Ghost→Psychic as "no effect".
- **Current env:** follows the data (2×), mis-teaches the agent.

**Recommendation:** sim replicates the **actual bug** (0×). Then patch
`game_state.py` env-side to match. Impact is small (Lick is the only damaging
Ghost move; Night Shade is typeless). Guardrail in the task says "replicate
Gen 1 bugs, don't fix them" — this is the poster child.

### 5.2 Poison → Bug / Bug → Poison — **Missing from env**

- **Gen 1:** Poison is 2× vs Bug; Bug is 2× vs Poison.
- **Current env:** `(0x03, 0x07)` absent → defaults to 1×. (Bug→Poison is
  present at [game_state.py:154](.claude/worktrees/quizzical-sammet/v2/game_state.py)
  line with `(0x07, 0x03): 2.0`, correct.)

**Recommendation:** add to env chart; sim uses the authoritative pokered chart.

### 5.3 Full type-chart diff — **TODO before v0.1 freeze**

The in-code chart was hand-written from memory. We should diff the parser
output (`type_chart.json`) against `_TYPE_CHART` and reconcile every
discrepancy. This is a one-shot audit, not recurring work.

### 5.4 `best_move_effectiveness` uses lead's own types as move-type proxy

Location: [game_state.py:463](.claude/worktrees/quizzical-sammet/v2/game_state.py),
comment says "no move→type lookup in memory, so uses own types as STAB proxy".

- **Reality:** a Charmander (Fire/—) knowing Scratch (Normal), Growl (Normal),
  Ember (Fire) has real move types {Normal, Fire}, not {Fire} from the species.
- **Impact:** the tactical branch's "type advantage" signal is noisier than it
  could be.
- **Sim implication:** v0.1 sim **must reproduce the same proxy** to keep the
  transfer contract clean. Once `moves.json` exists, v0.2 should:
  (a) upgrade both the env and the sim simultaneously to use true move types,
  (b) add ~4 dims to `tactical_obs` for per-move effectiveness, and
  (c) re-train from scratch (tactical branch weights don't transfer across
  observation changes).

### 5.5 Crit formula

Current env has no crit model (tactical obs doesn't expose crit likelihood,
and damage dealt is inferred from HP deltas). Sim must implement the
`base_speed/2` Gen 1 crit formula so that damage distributions match PyBoy
turn-by-turn in the differential harness. Not an env bug — just a sim
requirement.

### 5.6 1/256 miss on "100% accurate" moves

Gen 1 `accuracy=255` moves miss ~0.39% of the time due to the `<` comparison
bug. Env doesn't model this explicitly (it reads RAM outcomes). Sim must
model it; otherwise differential tests will drift over long horizons.

### 5.7 Stat stages on crits

Gen 1 crits ignore **both** attacker buffs and defender buffs — and
controversially, the "self-buff and then crit" strategy is actively *worse*
than not buffing. Must implement exactly (see pseudocode in Part 3).
Current env has no stat-stage observation, so this is purely an
internal-sim correctness issue, not a transfer issue.

### 5.8 `battle_type` values

`IN_BATTLE` (`0xD057`) actually holds richer info: 0 = none, 1 = wild,
2 = trainer battle, some higher values for "lost battle" / safari zone.
Env's `battle_type` divides by 2.0, implicitly assuming values ∈ {0, 1, 2}.
Sim should emit exactly 0.0 / 0.5 / 1.0 and never produce higher.

---

## Part 6 — Week 1 Success Metrics (unchanged from task)

1. **50,000 battles/sec** single-core (1v1, no status, deterministic RNG).
   Measured via `pytest tests/test_v01.py::test_throughput`.
2. **100% parity** on 100 seeded differential tests vs PyBoy, turn-by-turn,
   across HP / PP / status flags / stat stages.
3. **Unit coverage** of the damage formula against published Bulbapedia
   vectors (≥ 20 manually verified cases covering: same-type, STAB, crit,
   2× SE, 4× SE, 0.5× resist, 0.25× resist, 0× immune, min damage roll,
   max damage roll).

Non-metrics (explicitly don't chase in Week 1): training integration,
reward signal design, any RL loop at all. Sim correctness first.

---

## Part 7 — Open Questions for User Approval

1. **Action space:** confirm option 2 (semantic `Discrete(9)` + Week-4
   button adapter) over option 1 (native 7-button UI state machine)?
2. **Ghost vs Psychic:** confirm sim implements the 0× bug and env is
   patched to match? (Alternative: keep env at 2× and sim at 2×,
   accepting both diverge from the cartridge.)
3. **Differential harness state seeding:** user or Claude to produce the 100
   mock initial states? Suggest a structured spec: 20 type-matchup coverage,
   30 crit/damage-roll coverage, 30 miss/hit boundary, 20 faint-timing
   coverage. User curates Pokémon/level/move choices, Claude writes the
   harness runner.
4. **Pokered pin:** fix to a specific commit (recommend current `master`
   head at first clone) or always follow `master`? Recommend pin.
5. **Where does `battle_sim/` live?** This session scaffolded it in the
   jovial-carson worktree. Proposed actual home: inside `v2/` on
   `claude/quizzical-sammet` (or a new `claude/battle-sim` sibling branch)
   so it can eventually `from v2.game_state import …`. Current placement is
   provisional for spec-review only.

---

## Appendix A — File References

All source-of-truth paths on `claude/quizzical-sammet`:

- `v2/game_state.py` — RAM reader, type chart, `tactical_obs()`
- `v2/pokemon_model.py` — `PokemonNet` with tactical branch
- `v2/red_gym_env_v2.py` — observation space, step loop
- `v2/rewards.py` — reward system (battle-relevant components: `catch_reward`,
  `opponent_damage_reward`, `battle_win_reward` — sim will emit analogous
  signals in Week 3)
- `v2/DEVLOG.md` — full training-run history

---

_Spec drafted 2026-04-15. Review, then build._

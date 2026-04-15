# Pokemon Red RL — Development & Action Log

A chronological log of training runs, reward shaping iterations, bug fixes, and
architectural decisions on the `claude/quizzical-sammet` branch.

---

## Phase 5 — Tactical Features (pre-run-1)

**Goal:** Add battle-aware observations and tactical rewards so the agent could
reason about type matchups, HP, and PP.

**Delivered:**
- `game_state.py` — Centralized typed memory reader (player, party, battle,
  opponent, badges, move types/PP).
- `rewards.py` — Composable `RewardSystem` with `RewardContext` snapshot and
  weighted `RewardComponent` registry. Five presets: default, enhanced, battle,
  survival, speedrun.
- `curriculum.py` — `CurriculumScheduler` and `CurriculumStage` for rotating save
  states (gym states, dungeons, etc.).
- `PokemonNet` — Multi-input policy network with dedicated tactical branch.
- `train_tactical.py` — Entry point tying curriculum + rewards + obs together.
- 47 new unit tests (type effectiveness, tactical obs, rewards, PokemonNet,
  curriculum) — **181 tests passing total**.

**PR:** opened on `claude/quizzical-sammet`.

---

## Training Run 1 — 2026-04-14

**Command:**
```
python train_tactical.py --mode enhanced --num-cpu 4 --ep-length 32768 \
    --iterations 268 --explore-weight 0.25
```

**Setup:** `init.state` (bedroom start), 4 workers, enhanced reward mode
(original weights: event=4, heal=10, badge=10, explore=0.025, stuck=0.05).

**Duration:** ~14 hours, ~20M steps.

**Result — PLATEAU AT VIRIDIAN:**
- ✅ 100% reach Viridian City (map 1)
- ❌ 0% get Pokédex, 0% catch, 0% evolve, 0% badge
- Agent settled into a Route 1 wild-battle grinding loop
- Policy collapsed (low entropy), exploration stalled

**Diagnosed root causes:**
1. `explore_weight=0.25` too low — new tiles barely rewarded
2. Battle rewards unbounded — each Rattata KO was a reliable +reward
3. No specific story milestone rewards (parcel, pokedex, deliver)
4. Backtracking to Pallet felt like negative exploration (old tiles)
5. No signal for catching or evolving Pokémon
6. Episodes too short — agent can't both grind AND do Oak quest
7. Entropy coefficient default was too low

---

## Reward Shaping Iteration 1 — 2026-04-15 (for run 2)

**Intent:** Fix all six root causes from run 1 with additive reward shaping,
not curriculum.

### Changes

**`game_state.py`** — Added Oak's Parcel quest flag addresses:
```python
OAKS_PARCEL_ADDR = 0xD74E, BIT = 1       # ⚠ WRONG — fixed later
DELIVERED_PARCEL_ADDR = 0xD74E, BIT = 0  # ⚠ WRONG — fixed later
POKEDEX_ADDR = 0xD74B, BIT = 5           # ✓
OAKS_POKEBALLS_ADDR = 0xD74B, BIT = 6    # ⚠ WRONG — fixed later
```
Properties: `has_oaks_parcel`, `delivered_oaks_parcel`, `has_pokedex`,
`has_oaks_pokeballs`.

**`rewards.py`** — New `RewardContext` fields:
- Story flags (has_oaks_parcel, delivered_oaks_parcel, has_pokedex, has_oaks_pokeballs)
- Catch/evolution (species_changed_count, new_species_caught)
- Level-gating (lead_level, opponent_level)

Seven new reward functions:
- `oaks_parcel_reward`, `delivered_parcel_reward`, `pokedex_reward`, `pokeballs_reward`
- `catch_reward`, `evolution_reward`, `party_growth_reward`
- **Modified** `opponent_damage_reward` with level gate: skip wild Pokémon
  more than 2 levels below the lead

Rewrote `create_enhanced_reward_system` with story-dominant weights:
| Component      | Weight |
|----------------|--------|
| pokedex        | 120    |
| delivered      | 100    |
| parcel         | 60     |
| badge          | 50     |
| pokeballs      | 30     |
| evolve         | 25     |
| catch          | 15     |
| event          | 4      |
| heal           | 10     |
| explore        | 0.1    |
| level          | 1      |
| party_growth   | 5      |
| battle_win     | 5      |
| opponent_dmg   | 2      |
| stuck          | 0.05   |
| pc_full        | 1      |

**`red_gym_env_v2.py`** — Added:
- `_track_battles` gates by level: trainers always count, wild only if
  `opp_level >= lead_level - 2`
- `_track_species` — detects catches vs evolutions via species-at-slot changes
- `battles_won_gated` and `trainer_battles_won` counters
- `starting_species`, `seen_species`, `new_species_caught`, `species_changed_count`
  reset each episode
- All new context fields wired into `_compute_rewards`

**`train_tactical.py`** — Entropy bump for enhanced mode:
```python
if args.mode == "battle":    ent_coef = 0.02
elif args.mode == "enhanced": ent_coef = 0.05   # was 0.01
else:                         ent_coef = 0.01
```
Added `--ep-length` help text recommending 65536+.

**Commit:** `aa9cee5` — "Add Oak's Parcel story milestones, catch/evolve
rewards, level-gated battles"

---

## Training Run 2 — started 2026-04-15

**Command:**
```
python train_tactical.py --mode enhanced \
    --gb-path "C:\Users\samuel\Documents\GitHub\pokemon_training\PokemonRed.gb" \
    --init-state "../init.state" \
    --num-cpu 4 --ep-length 65536 --iterations 268 --explore-weight 3.0
```

**Partial observations (at iteration ~315, ~20M steps, ~7.5h in):**
| Metric                    | Value |
|---------------------------|-------|
| `frac_got_starter`        | 1.00  |
| `frac_reached_viridian`   | 1.00  |
| `frac_first_battle_won`   | 0.25-0.75 (inconsistent) |
| `frac_caught_pokemon`     | 0     |
| `frac_first_evolution`    | 0     |
| `frac_reached_pewter`     | 0     |
| `frac_badge_1`            | 0     |
| `pcount`                  | 1     |
| `max_map_progress`        | 3 (Viridian) |
| `deaths` / episode        | 2-3   |
| `entropy_loss`            | -1.85 to -1.92 (healthy) |

**Trajectory flags observed:**
- Early steps (2k-32k): `{}` — confirms `init.state` IS bedroom start
- Mid-episode (~131k+): `Got Pokedex`, `Got Pokeballs From Oak`,
  `Oak Appeared In Pallet`, `Pallet After Getting Pokeballs 2`,
  `Followed Oak Into Lab 2` — agent completes the quest chain
- Unnamed flag `0xD74E-6: 03E` present — this is `GOT_OAKS_PARCEL` (flag 0x3E)

**Diagnosis:** Agent learned the full Oak quest, but post-Pokéballs it has no
signal pushing it north. It returns to Viridian and stays there. Starter dies
2-3× per episode, resetting progress.

---

## Bug Fix — Event Flag Bit Positions — 2026-04-15

**Discovery:** While reading the live trajectory log, noticed:
- `0xD74B-4: Got Pokeballs From Oak` — but my code had `OAKS_POKEBALLS_BIT = 6`
- `0xD74B-6: Pallet After Getting Pokeballs 2` — that's the bit I was reading
- `0xD74E-6: 03E` (unnamed, raw) — that IS `GOT_OAKS_PARCEL` (flag 0x3E =
  byte offset 7, bit 6), but my code had `OAKS_PARCEL_BIT = 1`

**Consequence:** In run 2, parcel and pokeballs rewards **never fired** even
though the agent earned both flags. The 60 + 30 reward weight was silently
unused. Only the pokedex reward (bit 5, correctly set) fired during delivery.

**Fix** (`game_state.py`):
```python
OAKS_PARCEL_ADDR = 0xD74E, BIT = 6       # was BIT = 1
DELIVERED_PARCEL_ADDR = 0xD74B, BIT = 5  # use pokedex as proxy (fires together)
POKEDEX_ADDR = 0xD74B, BIT = 5           # unchanged ✓
OAKS_POKEBALLS_ADDR = 0xD74B, BIT = 4    # was BIT = 6
```

**Commit:** `03ef934` — "Fix Oak's Parcel and Pokeballs event flag bit positions"

---

## Reward Shaping Iteration 2 — Map Progress Reward — 2026-04-15

**Motivation:** Run 2 shows `max_map_progress` stuck at 3 (Viridian) with 0%
reaching Pewter. The existing reward stack has no direct signal for linear
story progression — only `coord_count` (flat exploration) and event flags
(sparse).

**Solution:** Add `map_progress_reward` that returns the linear progression
index along the existing `essential_map_locations` path:

| Index | Map ID | Location          |
|-------|--------|-------------------|
| 0     | 40     | Oak's lab         |
| 1     | 0      | Pallet Town       |
| 2     | 12     | Route 1           |
| 3     | 1      | Viridian City     |
| 4     | 13     | Route 2           |
| 5     | 51     | Viridian Forest   |
| 6     | 2      | Pewter City       |
| 7     | 54     | Pewter Gym        |
| 8     | 14     | Route 3           |
| 9-11  | 59-61  | Mt. Moon          |
| 12    | 15     | Route 4           |
| 13    | 3      | Cerulean City     |
| 14    | 65     | Cerulean Gym      |

**Changes:**
- `RewardContext.max_map_progress: int = 0`
- `map_progress_reward(ctx)` returns `float(max(ctx.max_map_progress, 0))`
- Added to enhanced preset with **weight 20**
- Env wires `max_map_progress=self.max_map_progress` into context

Each new story map adds a persistent +20 to episode return, creating a strong
directional gradient northward.

**Commit:** `c868352` — "Add map_progress_reward to push past Viridian wall"

---

## Architectural Notes & Clarifications

### Worktree vs main repo

- **Main repo:** `C:\Users\samuel\Documents\GitHub\pokemon_training` — probably
  on `main` branch. Has the ROM file but NOT the reward-shaping code.
- **Worktree:** `C:\Users\samuel\Desktop\PokemonRedExperiments\.claude\worktrees\quizzical-sammet`
  — checkout of `claude/quizzical-sammet` branch, where all the new code lives.
- **Always run training from the worktree's `v2/` dir** until the PR merges.
- After merge, can delete worktree and run from main repo normally.

### `init.state` content

- Confirmed: `init.state` is the **bedroom start** (no quest flags set).
- Early-episode trajectory logs showing `{}` prove this.
- Agent does the Oak quest from scratch each episode.

### `delivered_oaks_parcel` reward

- The actual "parcel delivered" bit in RAM is ambiguous; the visible event
  flags at the moment of delivery are Pokedex + Pokeballs + Pallet-after.
- We point `DELIVERED_PARCEL_ADDR` at the pokedex bit so it fires simultaneously
  with `pokedex_reward`. Net effect: delivery triggers **pokedex (120) +
  delivered (100) = 220** reward bolus. This is intentional.

### Curriculum (deferred)

- User decision: do reward shaping first, curriculum later if still plateauing.
- `has_pokedex.state` and `has_pokedex_nballs.state` are NOT distinct curriculum
  stages — they represent the same in-game moment, differing only by Pokéball
  inventory.
- Save states needed for a real curriculum (user must create manually):
  `viridian_pre_parcel`, `post_brock`, `cerulean`, `pre_mtmoon`, etc.

### Dependency setup (encountered during run 1 setup)

Required pip installs on fresh machine:
- `scikit-image` (for `skimage`)
- `pyboy`
- `mediapy`
- `tensorboard`

Quirks:
- PyBoy parameter to silence sound is `sound_emulated=False`, not `sound=False`.
- Tensorboard heatmap crashes with single worker (rearrange `r=2` fails on
  shape `(1, 484, 476)`). Fixed with `n_envs` check and single-env fallback
  in tensorboard callback.

---

## Recommended Next Run Command

```bash
cd v2
python train_tactical.py --mode enhanced \
    --gb-path "../PokemonRed.gb" \
    --init-state "../init.state" \
    --num-cpu 4 --ep-length 65536 --iterations 268 --explore-weight 3.0
```

**What to watch in TensorBoard:**
- `env_stats_max/max_map_progress` — should climb past 3 (Viridian) toward 5
  (Viridian Forest) and 6 (Pewter)
- `milestones/frac_reached_pewter` — target: > 0 by iteration 100
- `milestones/frac_caught_pokemon` — target: > 0
- `milestones/frac_first_evolution` — target: > 0 by iteration 150
- `milestones/frac_badge_1` — target: > 0 by end of run
- `trajectory/all_flags` — should see `0xD74E-6: 03E` (parcel) and more named
  flags as the quest progresses
- `entropy_loss` — stay below -1.5 (healthy); above = policy collapse

---

## Future Work (if run 3 still plateaus)

1. **Pokéball use reward** — explicit signal for throwing a ball, even if the
   catch fails. Currently `catch_reward` only fires on success.
2. **Item-use tracking** — detect menu → ball selection → throw action
   sequence.
3. **Curriculum** — user creates save states at story checkpoints
   (viridian_pre_parcel, post_brock, etc.) and rotates them via
   `CurriculumScheduler`.
4. **Shaped distance reward** — bonus for decreasing Manhattan distance to the
   next story map while holding a relevant item (parcel carrier bonus).
5. **Trainer-only battle rewards** — drop wild battle rewards entirely; only
   reward trainer battles (which are tied to story progress).
6. **Death penalty** — currently no penalty for starter fainting. Could add
   `-10` per faint to discourage risky play.
7. **Level ceiling** — if avg party level > 15 and no new badge, zero out
   `level_reward` to force progression-seeking.

---

## Commit Log (`claude/quizzical-sammet` branch)

| Hash       | Description |
|------------|-------------|
| `aa9cee5`  | Add Oak's Parcel story milestones, catch/evolve rewards, level-gated battles |
| `03ef934`  | Fix Oak's Parcel and Pokeballs event flag bit positions |
| `c868352`  | Add map_progress_reward to push past Viridian wall |

---

_Last updated: 2026-04-15_

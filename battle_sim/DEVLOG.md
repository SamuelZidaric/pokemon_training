# Battle Simulator — Development Log

Phase A of the specialist-agent architecture. This log tracks design
decisions, training runs, and v-version bumps for the scripted battle
simulator on branch `claude/jovial-carson`. Parent-project history lives
in `v2/DEVLOG.md` on `claude/quizzical-sammet`.

---

## v0.1 — scaffold + scripted engine — 2026-04-15

**Goal:** prove the sim-train-then-transfer pattern end-to-end with a
minimum-viable engine (damage only, no status, no switching). Validate
the 22-dim transfer contract with a real trained policy.

**Delivered:** PR #2 on `claude/jovial-carson`.

- `parse_pokered.py` — ASM → JSON for types (17), moves (165), base stats (151),
  type chart (82 rows), learnsets (190). Pokered pinned at `9441f1a`.
- `engine.py` — 1v1 turn-order state machine.
- `damage.py` — Gen 1 integer-truncated damage formula with STAB, type mult,
  crit (doubles level + bypasses stat stages), damage roll [217..255],
  1/256 miss on 100%-accurate moves.
- `obs.py` — 22-float `tactical_obs` mirroring `v2/game_state.py` exactly.
- `env.py` — Gymnasium env with full-shape obs Dict, non-battle keys
  zero-padded.
- `train_battle.py` — SB3 PPO entry point.
- 27 unit tests passing.

### Training Run 1 — v0.1 — 2026-04-15

**Command:**
```powershell
python -m battle_sim.train_battle --num-cpu 4 --steps 1000000
```

**Duration:** ~23 min (700–3000 it/s depending on phase).

**Headline results (1000-episode deterministic evaluation):**

| Metric | vs random | vs greedy |
|--------|----------:|----------:|
| Win rate | **88.8%** | **80.2%** |
| Mean reward | +1.48 | +1.18 |
| Median battle length | 2 turns | 2 turns |

**Per-matchup breakdown (first-move choice was 100%-deterministic per species):**

| Player | Chosen move | Reason it's right |
|--------|------------|-------------------|
| Charmander (Fire) | **EMBER** (slot 2) | STAB Fire > neutral Normal Scratch |
| Squirtle (Water) | **BUBBLE** (slot 2) | STAB Water |
| Bulbasaur (Grass) | **TACKLE** (slot 0) | No damaging Grass move at L10; Leech Seed stubbed |

**Action usage:** slots 0 + 2 only. Slots 1 (status) and 3 (empty/stub) never selected. Never switched (actions 4–8).

**TensorBoard convergence:**

| Metric | Start | End | Read |
|--------|------:|----:|------|
| `train/entropy_loss` | -2.19 | -0.23 | Near-deterministic — appropriate given only 2 useful moves |
| `train/explained_variance` | -0.01 | **+0.96** | Value function nearly perfect |
| `train/value_loss` | 0.92 | 0.07 | Critic converged |
| `train/approx_kl` | 0.017 | 0.0008 | Updates petered out |
| `train/policy_gradient_loss` | -0.018 | ~0 | Policy frozen |

### Diagnosis: the 20% loss rate is structural, not a training failure

1. **Bulbasaur vs Geodude/Onix: 0% win rate** — physically unwinnable at L10.
   Bulbasaur's Vine Whip (Grass, SE vs Rock) unlocks at L13; at L10 its only
   damaging move is Tackle (Normal, 0.5× resisted by Rock). The type chart
   forbids a win here — no policy can fix it without higher level or swapped
   moveset.
2. **Charmander vs Geodude: 85%** — Ember is 0.5× resisted (Fire vs Rock),
   sometimes Char doesn't out-damage Geodude's crit-Tackle bursts.
3. **Scattered losses to Spearow Peck crits on Bulbasaur** — irreducible
   variance from Gen 1 crit formula.

### What v0.1 proves

- Transfer contract holds: 22-dim `tactical_obs` matches v2 byte-for-byte,
  drift-check tests pass.
- Pipeline end-to-end: pokered → JSON → engine → env → PPO → saved zip.
- Policy extracts **all** learnable signal in v0.1: "pick STAB damaging move,
  avoid status moves." Full convergence at 1M steps.

### What v0.1 does not exercise

- Status conditions — engine ignores `effect` field on moves entirely.
- Stat-stage changes — `atk_stage`/`def_stage` fields exist on `Pokemon` but
  no move writes to them.
- Switching — no-op in 1v1.
- Level variance — fixed matchups (starters L10 vs wild L3–12) train a
  level-agnostic policy by accident.
- Trainer AI — opponent is random-move or greedy-highest-power.

### Known divergences from v2 (flagged for future reconciliation)

- **Ghost → Psychic:** sim = 0× (correct per pokered data), v2 = 2× (bug).
  Will matter once Gastly/Haunter show up in the opponent pool.
- **Poison ↔ Bug:** sim = 2× both ways (correct), v2 missing from chart.
- **`best_move_effectiveness` proxy:** both sim and v2 use attacker's own
  types as move-type stand-in. Deliberate parity; fixing requires expanding
  `tactical_obs` and retraining v2 too.

### Open follow-ups (shipped as v0.2)

- Status effect dispatch (par/slp/brn/psn/frz).
- Stat-stage moves writing to `atk_stage`/`def_stage`.
- Secondary effects (Ember's 10% burn, Body Slam's 30% para, flinch moves).
- `Monitor` wrapper in `train_battle.py` so `rollout/ep_rew_mean` logs
  to TensorBoard (missing in v0.1 — had to evaluate the saved zip post-hoc
  to get the win-rate signal).
- Expand tactical obs with status + stage dims (**breaks transfer contract**
  — needs a simultaneous update of `v2/game_state.py`).
- Level variance on reset + wider opponent pool (include trainer teams).

---

## v0.2 — status effects + stat stages + wider opponent pool — 2026-04-15

**Goal:** give the policy something to learn beyond "pick highest-power move".
v0.1 converged at 1M steps because the state-action space collapsed to
`species → one damaging move`.  v0.2 keeps the 22-dim transfer contract
intact but adds the mechanics the v0.1 policy was blind to, and widens the
opponent pool + level distribution so the optimal choice varies per matchup.

**Engine / mechanics delivered:**

- `effects.py` — full Gen 1 effect dispatcher covering:
  - Primary status: SLEEP, PARALYZE, POISON, TOXIC, CONFUSION, burn/freeze.
  - Secondary status on hit: 10% variants (Ember burn, Thunder-type para)
    and 30% variants (Body-Slam para, Fire-Blast burn).  Uses the same
    `rng.next_byte() < 26` / `< 77` thresholds as pokered.
  - Stat-stage moves: ATTACK/DEFENSE/SPEED/SPECIAL/ACCURACY/EVASION up/down
    at both single-stage and double-stage magnitudes; applied to attacker
    (user-buff moves) or defender (debuff moves).
  - Turn-start gating: flinch (single-turn), freeze (locked — no self-thaw
    in v0.2), sleep (counts down, wake-up turn is also skipped per Gen 1),
    paralysis (25% full-paralysis roll), confusion (50% skip in v0.2; full
    self-hit calc present but gated behind turn-skip).
  - Type immunity: Poison-type immune to PSN, Fire-type immune to BRN,
    Ice-type immune to FRZ.  Paralysis applies to Electric-types (Gen 1
    quirk).
  - End-of-turn residual: 1/16 max-HP tick for BRN / PSN, floor 1.
  - Stat multipliers: burn halves physical Atk (applies even through crit's
    stat-stage bypass — it's a status mult, not a stage), paralysis quarters
    Speed (applied in turn-order decision).
- `engine.py` — wired in:
  - `check_action_allowed` before move selection (PP not consumed on a
    skipped turn).
  - `apply_move_effect` after damage resolves, with `move_hit` / `did_damage`
    flags so secondary effects only roll on landed damaging moves.
  - `end_of_turn_residuals` for both actors after the turn.
  - `status_spd_multiplier` folded into the first/second decision.
- `damage.py` — consumes `status_atk_multiplier` for physical moves.
- `entities.py` — added `sleep_turns`, `confusion_turns`, `flinched` fields
  on `Pokemon`.

**Env / training delivered:**

- `_default_teams` in `env.py`:
  - Player level sampled uniformly in [8, 18] (was fixed L10).
  - Opponent level sampled as `clip(p_level + U[-3, 3], 3, 25)`.
  - Player starters now carry 4 moves each (added LEER / WATER_GUN / VINE_WHIP)
    so the policy has meaningful stat-move vs damage-move trade-offs.
  - Opponent pool widened from 7 to 17 species — adds ODDISH/BELLSPROUT
    (SLEEP_POWDER), ZUBAT (SUPERSONIC/LEECH_LIFE), EKANS (POISON_STING),
    SANDSHREW, MANKEY (KARATE_CHOP — high-crit), NIDORAN_M/F, PIKACHU
    (THUNDER_WAVE), PARAS (STUN_SPORE).
- `train_battle.py` now wraps each sub-env in SB3 `Monitor`, so
  `rollout/ep_rew_mean` + `rollout/ep_len_mean` log to TensorBoard directly
  (v0.1 workaround — post-hoc eval — no longer needed).
- Tests: +19 in `test_effects.py` (type immunity, sleep countdown, burn Atk
  halving, par speed quartering, end-to-end Thunder-Wave / Growl / residual
  integration).  46 tests total passing.

**Obs contract:** unchanged.  Status + stage dims are deliberately deferred
to v0.3 — expanding tactical obs requires a lockstep update of
`v2/game_state.py` on `claude/quizzical-sammet` (the transfer contract is
shared).  v0.2 policy learns status indirectly via HP deltas and episode
outcomes — crude signal but matches the v0.1 baseline methodology.

### Training Run 2 — v0.2 — 2026-04-15

**Command:**
```powershell
python -m battle_sim.train_battle --num-cpu 4 --steps 1000000 --run-name battle_v0_2
```

**Duration:** 24 min (~700 fps, steady across the run).

**Headline results (1000-episode deterministic evaluation):**

| Metric | vs random | vs greedy | Δ vs v0.1 |
|--------|----------:|----------:|----------:|
| Win rate | **84.2%** | **71.2%** | −4.6pp / −9.0pp |
| Mean reward | +1.18 | +0.74 | −0.30 / −0.44 |
| Median battle length | 4 turns | 3 turns | +2 / +1 |

The regression vs v0.1's 89% / 80% is **expected and healthy** — it comes
from the harder opponent pool (17 species vs 7) and level variance, not
from the status machinery being broken.  Battles are longer because status
moves now meaningfully cost turns on both sides.

**Per-opponent win rate (vs greedy, 1000 games sampled uniformly):**

| Opponent | Win% | Opponent | Win% |
|----------|-----:|----------|-----:|
| EKANS | 98% | NIDORAN_F | 61% |
| CATERPIE | 96% | ZUBAT | 56% |
| SANDSHREW | 95% | PARAS | 50% |
| WEEDLE | 95% | MANKEY | 44% |
| ONIX | 90% | SPEAROW | 36% |
| GEODUDE | 84% | **PIKACHU** | **29%** |
| NIDORAN_M | 83% | | |
| RATTATA | 71% | | |
| BELLSPROUT | 70% | | |
| ODDISH | 67% | | |
| PIDGEY | 65% | | |

The bottom five are all **speed+status** threats: Pikachu (Thunder Wave →
25% skip / turn), Mankey (base 70 spd + high-crit Karate Chop), Spearow
(base 70 spd + Peck), Paras (Stun Spore), Zubat (Supersonic confusion).
The v0.2 obs has no status channel, so the policy can't directly observe
"I'm paralyzed, I should pick a different slot" — this is the main thing
v0.3's obs expansion should fix.

**Per-player starter:**

| Starter | Win vs greedy |
|---------|--------------:|
| CHARMANDER | 81% |
| SQUIRTLE   | 69% |
| BULBASAUR  | 64% |

Charmander pulls ahead: its new 4-move set (SCRATCH / GROWL / EMBER / LEER)
gives it a real answer (Ember STAB) to most wilds, while Bulbasaur's best
damaging move remains Grass-typed in a pool with several Rock/Flying
resists.

**Action usage distribution (per-step, vs greedy):**

| Action | Fraction |
|--------|---------:|
| slot 0 (SCRATCH / TACKLE)           | 37.4% |
| slot 1 (GROWL / TAIL_WHIP)          | **0.0%** |
| slot 2 (EMBER / LEECH_SEED / BUBBLE) | 30.9% |
| slot 3 (LEER / VINE_WHIP / WATER_GUN)| 31.7% |
| 4..8 (switch — no-op)               | 0.0% |

**Slot 1 — the single-stat debuff move — is never selected.**  The policy
correctly identifies that a one-turn Atk−1 / Def−1 in a 3-5-turn fight is
strictly worse than the available damaging slot.  This is the first
training artifact that goes beyond v0.1's "pick STAB" — the policy is now
making a *comparative-value* choice between three damaging options rather
than one.

**TensorBoard convergence:**

| Metric | Start | End | Read |
|--------|------:|----:|------|
| `rollout/ep_rew_mean` | ~0.1 | ~1.1 | Healthy climb; noisier than v0.1 due to status RNG |
| `rollout/ep_len_mean` | ~8 | ~4.5 | Policy decisively ends fights |
| `train/entropy_loss` | -2.19 | -0.28 | Near-deterministic; slightly more exploration than v0.1 (-0.23) because pool is harder |
| `train/explained_variance` | -0.01 | **+0.55** | Value fn carries more residual variance — status RNG + level variance is ~harder to predict than v0.1's deterministic pool |
| `train/approx_kl` | 0.017 | 0.002 | Small-but-nonzero updates still happening at 1M |
| `train/value_loss` | 0.92 | 0.50 | Critic plateau — v0.3 obs expansion should drop this further |

### What v0.2 proves

- Effect dispatcher, status machinery, and stat-stage system all land
  cleanly into the engine — engine tests + 19 new effect tests all pass.
- Policy can learn comparative choice among damaging moves (slot 0 vs 2
  vs 3 chosen per-species) without an explicit move-type feature.
- Status moves in the opponent arsenal materially change matchup
  difficulty in a way the policy can't fully answer yet — signal we're
  now training on something harder than v0.1.

### What v0.2 does not exercise

- Obs-side status awareness — policy doesn't see "I'm paralyzed" directly;
  must infer from turn-skip log.  **v0.3's primary target.**
- Switching — still a no-op (action space dim 4..8 unused, and opponent
  pool is still 1-mon).
- Confusion self-damage path — present in code but gated behind a 50/50
  skip in v0.2; full self-hit is a v0.3 cleanup.
- Freeze thaw — no Fire-move-hits-frozen logic; freeze is permanent.
- Multi-hit moves / fixed-damage / Substitute / Mirror Move — deferred.

### Open follow-ups (shipped as v0.3)

- Expand tactical obs (+6 dims: self/opp status 5-way one-hot folded into
  a 3-float sub-vector + stat-stage summary).  **Breaks the 22-dim
  transfer contract — requires simultaneous v2/game_state.py update on
  `claude/quizzical-sammet`.**
- Multi-mon opponent parties + switching logic.
- Confusion self-hit damage path.
- Freeze / thaw on Fire move hit.
- Trainer-team sampler (gym leaders).

### Known divergences still present

- Ghost → Psychic: sim = 0×, v2 = 2× (bug on `claude/quizzical-sammet`).
- Poison ↔ Bug: sim = 2× both ways, v2 missing from chart.
- `best_move_effectiveness` proxy still uses attacker's own types.

---

_v0.2 committed on `claude/jovial-carson`; training-run 2 table filled
in above._

---

## v0.3 — obs expansion + confusion self-hit + freeze thaw — 2026-04-15

**Goal:** address the v0.2 weakness analysis by giving the policy direct
access to status and stat-stage state.  The Pikachu/Spearow/Mankey 29-44%
matchups in v0.2 losses were all speed+status threats; without `self.status`
in tactical obs the policy could only infer paralysis from turn-skip lag
in the log.

**⚠ Transfer-contract break.**  `TACTICAL_OBS_SIZE` grows from **22 → 36**.
This is incompatible with v0.1 / v0.2 saved checkpoints.  Before any weight
transfer into the full-game agent happens, `v2/game_state.py` on
`claude/quizzical-sammet` must append the same 14 dims in the same order.
The drift-check test on this branch now asserts 36 — when v2 is updated,
both sides should pass simultaneously, at which point it's safe to
transfer.

**Obs layout — appended indices:**

| Idx | Meaning | Encoding | Range |
|----:|---------|----------|------:|
| 22..26 | self.status (PAR/SLP/BRN/PSN/FRZ) | 5-dim one-hot; OK = all zeros | {0, 1} |
| 27..31 | opp.status  (PAR/SLP/BRN/PSN/FRZ) | 5-dim one-hot; OK = all zeros | {0, 1} |
| 32 | self.atk_stage | stage / 6 | [-1, 1] |
| 33 | self.def_stage | stage / 6 | [-1, 1] |
| 34 | opp.atk_stage  | stage / 6 | [-1, 1] |
| 35 | opp.def_stage  | stage / 6 | [-1, 1] |

**Status encoding — one-hot, not ordinal.**  An earlier draft used an
ordinal-and-normalize scheme (OK=0..FRZ=5 → `/5`) to keep the addition
small.  That was rejected pre-training: ordinal imposes a spurious
magnitude ordering (treating PAR as "between" OK and SLP, or BRN as
"half of FRZ") that a small MLP has to waste samples un-learning.  The
whole point of this obs expansion is sample efficiency against
paralysis/sleep threats — using a DL anti-pattern there would be
self-defeating.  5-dim one-hot with all-zeros for OK gives the policy
a sparse, orthogonal signal; "OK is the absence of signal" is cheap to
encode and common in Pokemon obs schemes.

Speed/Special stages are deliberately **not** added: none of the opponent
pool's current moves alter them beyond STRING_SHOT (SPEED_DOWN1) on the
lone Caterpie/Weedle matchups.  Added only if v0.4 expands the pool with
Agility/Amnesia users.

Speed/Special stages are deliberately **not** added: none of the opponent
pool's current moves alter them beyond STRING_SHOT (SPEED_DOWN1) on the
lone Caterpie/Weedle matchups.  Added only if v0.4 expands the pool with
Agility/Amnesia users.

**Engine / mechanics added:**

- Confusion self-hit upgraded from v0.2's 50/50 skip to the proper Gen 1
  fixed-formula typeless 40-power physical calc:
  `((2*L/5 + 2) * 40 * Atk / Def) / 50 + 2`.  No crit, no STAB, no roll —
  matches pokered's confusion branch.
- Fire-type damaging hits (`move.type_id == 0x14 && move.power > 0`) thaw
  a frozen defender as part of the hit.  Previously freeze was permanent
  in v0.2.

**Tests:** +5 in `test_effects.py` (confusion self-hit damage, confusion
wear-off, Fire-thaw positive case, non-Fire-doesn't-thaw negative case) +2
in `test_obs_contract.py` (v0.3 one-hot status+stage layout, OK-is-all-zeros),
plus the 22 → 36 assertion update.  **52 tests total passing.**

### v0.3 speedup — thin-obs + MlpPolicy — 2026-04-15

Mid-v0.3 refactor, before the main 1M-step run.  The v0.1/v0.2 env emitted
the full-game Dict obs with zero-padded `screens` (72×80×4), `map` (48×48),
`events` (2232-bit MultiBinary), etc.  The transfer-contract rationale was
that a policy trained here could be loaded into the full-game env with no
shape rework — but that has never been needed; weight transfer is
layer-by-layer, not env-by-env.

**Old per-step cost budget (v0.2, 700 fps per worker):**

| Cost | ~µs | Why |
|---|---:|---|
| Engine + effect dispatch | 22 | Pure Gen 1 math |
| Dict obs build (mostly zeros) | 50-80 | Allocation + copies for 15 KB of padding |
| Pickle + IPC across SubprocVecEnv boundary | ~200 | 15 KB payload per step |
| MultiInputPolicy forward (NatureCNN over zero screens!) | 700-900 | Wasted compute on padded tensors |
| **Total** | ~1,500 | **~670 fps/worker** |

The NatureCNN and the IPC pickle are both pure overhead — neither contributes
signal the policy can use.

**Changes:**

- `env.py`: `observation_space = Box(-1, 1, (36,), float32)`.  `_obs()` returns
  `tactical_obs(state)` directly — no Dict wrapper.
- `train_battle.py`: `policy="MlpPolicy"`, explicit
  `policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64]), activation_fn=nn.ReLU)`.
  Both fields *must* be locked — SB3's default activation is Tanh (wrong),
  default depth is version-dependent (silent transfer failure risk).
- `train_battle.py`: `--vec {subproc,dummy}` flag.  DummyVecEnv runs all
  `num_cpu` envs in one process, no IPC.
- New test file `tests/test_transfer_compat.py` with 4 tests:
  1. `policy_net` is `Linear(36→64)→ReLU→Linear(64→64)→ReLU` — shape gate.
  2. Weight-graft from SB3 policy_net into a mock `PokemonNet.tactical`
     reproduces the forward pass bit-for-bit — semantic gate.
  3. `features_extractor` is identity (FlattenExtractor on a Box) — so no
     hidden pre-norm layer sneaks in.
  4. Activation is ReLU not Tanh — guard against someone dropping
     POLICY_KWARGS.

### Benchmark — SubprocVecEnv vs DummyVecEnv (100k steps)

| Config | Wall | Aggregate fps | vs v0.2 |
|--------|---:|---:|---:|
| SubprocVecEnv, 4 workers | 14 s | 6,854 | **2.4×** |
| **DummyVecEnv, 4 envs** (chosen) | **12 s** | **8,163** | **2.9×** |
| DummyVecEnv, 1 env | 29 s | 3,400 | 1.2× |

DummyVecEnv (single-process, 4 vectorized envs) wins because:
- Zero pickle cost per step.
- SB3 batches policy forward across all n_envs in a single call (one
  tensor of shape `[4, 36]` vs four serial calls of `[36]` each).
- 4× parallelism from SubprocVecEnv gets ~swallowed by GIL since the
  engine is pure Python and step latency is ~20 µs.

**Projected 1M-step wall: ~2 min (down from 24 min).** Headline numbers
for the main run follow.

### Training Run 3 — v0.3 — PENDING

_Run the training when ready:_

```bash
python -m battle_sim.train_battle --num-cpu 4 --vec dummy --steps 1000000 --run-name battle_v0_3
```

Fill this section in after the run completes — same format as Run 2:
headline table with vs-random / vs-greedy / Δ vs v0.2, per-opponent
breakdown (especially watching Pikachu/Spearow/Mankey), per-starter,
action-slot usage, and TensorBoard convergence.

**Hypothesis:** Pikachu 29% → 55%+, Spearow 36% → 55%+, Mankey 44% → 55%+.
Overall vs-greedy 71% → 78%+.  If these numbers don't move, the signal
isn't the bottleneck — likely candidates are (a) 1M steps isn't enough
for the expanded state space, (b) opponent pool structurally favors
speed+status even with perfect observability, or (c) reward shaping
doesn't incentivize defensive play (e.g. burn-stalling).  In that order
of likelihood — bump to 2M steps first.

### Deferred to v0.4

- Multi-mon opponent parties + switching (actions 4..8 finally meaningful).
- Trainer AI (Brock / Misty scripted teams).
- Speed / Special stage dims if the pool adds Agility/Amnesia users.
- Contract reconciliation PR on `claude/quizzical-sammet` to unlock
  weight transfer.

---

_v0.3 code ready on `claude/jovial-carson`; awaiting Training Run 3._

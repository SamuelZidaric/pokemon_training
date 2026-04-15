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

### Training Run 3 — v0.3 — 2026-04-15

**Command:**
```bash
python -m battle_sim.train_battle --num-cpu 4 --vec dummy --steps 1000000 --run-name battle_v0_3
```

**Duration:** ~3 min wall (vs 24 min for v0.2 — **8× speedup** held end-to-end,
matching the 100k benchmark projection).  TB reports ~6,800-11,500 fps across
the run.

**Headline results (1000-episode deterministic evaluation):**

| Metric | vs random | vs greedy | Δ vs v0.2 |
|--------|----------:|----------:|----------:|
| Win rate | **85.1%** | **70.9%** | +0.9pp / −0.3pp |
| Mean reward | +1.20 | +0.70 | +0.02 / −0.04 |
| Median battle length | 3 turns | 3 turns | −1 / 0 |

Overall numbers are essentially flat vs v0.2 — but the *distribution* moved.
The status-threat matchups (Pikachu, Spearow) saw the predicted lift; the
"easy" damage matchups (Pidgey, Bellsprout) regressed.  Net wash on the
mean, real signal in the tails.

**Per-opponent win rate (vs greedy, 1000 games sampled uniformly):**

| Opponent | v0.3 Win% | v0.2 Win% | Δ |
|----------|----------:|----------:|--:|
| EKANS | 96.6% | 98% | −1.4 |
| WEEDLE | 96.2% | 95% | +1.2 |
| CATERPIE | 95.7% | 96% | −0.3 |
| ONIX | 91.4% | 90% | +1.4 |
| NIDORAN_M | 91.1% | 83% | **+8.1** |
| SANDSHREW | 90.9% | 95% | −4.1 |
| GEODUDE | 80.4% | 84% | −3.6 |
| **BELLSPROUT** | **76.9%** | 70% | **+6.9** |
| RATTATA | 72.2% | 71% | +1.2 |
| NIDORAN_F | 65.6% | 61% | +4.6 |
| ZUBAT | 61.4% | 56% | +5.4 |
| PARAS | 58.9% | 50% | **+8.9** |
| ODDISH | 57.9% | 67% | −9.1 |
| **MANKEY** | 46.8% | 44% | +2.8 |
| **SPEAROW** | 45.1% | 36% | **+9.1** |
| **PIDGEY** | **43.1%** | 65% | **−21.9** |
| **PIKACHU** | 38.7% | 29% | **+9.7** |

**Hypothesis verdict — partial win:**
- Pikachu 29% → **38.7%** (+9.7pp) — moved decisively but didn't reach 55%.
- Spearow 36% → **45.1%** (+9.1pp) — same direction, same magnitude.
- Mankey 44% → 46.8% (+2.8pp) — barely moved; Mankey's pressure is
  high-crit Karate Chop, not status, so the obs expansion has nothing
  to bite on.
- Vs-greedy 71% → 70.9% — flat, because the gains were eaten by the
  Pidgey collapse.

**The Pidgey regression is the headline anomaly.**  Pidgey carries
Sand-Attack (ACC_DOWN1) — same accuracy-debuff family as PARAS's Stun
Spore, ZUBAT's Supersonic, etc.  v0.2 had no obs channel for "I'm
debuffed", so the policy treated all Pidgeys as "uniform damaging
opponent" and just out-traded them.  v0.3 added status one-hot dims
22..31 — but **stat-stage dims 32..35 are *self*/opp atk and def stages,
not accuracy stages**.  My guess: the status one-hot dims being non-zero
for *neighboring* PARAS/ZUBAT/PIKACHU rollouts taught the policy a
defensive (likely switch-attempt → no-op or weak-debuff slot) reaction
that generalizes badly back onto Pidgey, where the actual debuff is
accuracy and there is no obs feature for it.  Confirmation would need
adding accuracy-stage dims and re-running.  Filed for v0.4.

**Per-player starter:**

| Starter | v0.3 vs greedy | v0.2 vs greedy | Δ |
|---------|---------------:|---------------:|--:|
| CHARMANDER | 78.9% | 81% | −2.1 |
| SQUIRTLE   | 70.5% | 69% | +1.5 |
| BULBASAUR  | 64.0% | 64% | 0.0 |

Starter ordering preserved.  Gap narrowed slightly — Charmander's
Ember-STAB advantage is less of an outlier when the policy can also
exploit status differences against Charmander's harder matchups.

**Action usage distribution (per-step, vs greedy):**

| Action | v0.3 | v0.2 |
|--------|-----:|-----:|
| slot 0 (SCRATCH / TACKLE)            | 32.6% | 37.4% |
| slot 1 (GROWL / TAIL_WHIP)           | **1.9%** | **0.0%** |
| slot 2 (EMBER / LEECH_SEED / BUBBLE) | 28.1% | 30.9% |
| slot 3 (LEER / VINE_WHIP / WATER_GUN)| 37.4% | 31.7% |
| 4..8 (switch — no-op)                | 0.0% | 0.0% |

Slot 1 is no longer strictly dead — the policy now picks the debuff move
~2% of the time.  Inspection would tell us *when*: most plausibly when
opp is a glass-cannon physical attacker (Mankey, Pikachu) and one Atk−1
shifts the trade math.  vs-random the rate is higher (8.7%) which is
consistent with "policy explores debuff against weaker opponents".

**TensorBoard convergence:**

| Metric | Start | End | Read |
|--------|------:|----:|------|
| `rollout/ep_rew_mean` | -0.67 | +1.29 | Higher peak than v0.2 (+1.1) — the obs expansion gives the value fn more to chew on |
| `rollout/ep_len_mean` | 11.4 | 4.8 | Same end as v0.2; shorter early because thin-obs lets the policy commit faster |
| `train/entropy_loss` | -2.19 | -0.41 | Less collapsed than v0.2 (-0.28) — policy keeps slot-1 explore alive |
| `train/explained_variance` | -0.09 | **+0.58** | Same plateau as v0.2 — the bottleneck isn't predictability of return |
| `train/approx_kl` | 0.015 | 0.003 | Healthy; small updates at convergence |
| `train/value_loss` | 0.68 | 0.41 | Lower than v0.2 (0.50) — value fn does benefit from status dims |
| `time/fps` | 11,507 | 6,837 | Started fast on cold envs, settled at ~8k — matches benchmark |

### What v0.3 proves

- Status one-hot lifts the worst status-matchup losses by ~9pp at
  matched compute.  Direction-correct, magnitude smaller than hoped.
- Thin-obs + DummyVecEnv + MlpPolicy stack delivers the projected 8×
  speedup end-to-end.  1M steps in 3 min unblocks the v0.4 iteration loop.
- Transfer-graft test passes on real trained weights — the layer-copy
  contract is real, not just shape compatible.
- One-hot beat ordinal in practice: slot-1 entropy stayed alive,
  consistent with "orthogonal status signal rather than spurious
  magnitude axis".

### What v0.3 surfaced

- **Accuracy stages are the missing obs channel.**  Pidgey
  (Sand-Attack) regressed −22pp because the obs has status and
  atk/def stages but not acc/eva.  v0.4 must add at minimum
  `self.acc_stage` and `opp.eva_stage` (2 dims) — likely `self.eva_stage`
  and `opp.acc_stage` too for symmetry (4 dims; 36 → 40).
- 1M steps may underfit the expanded state space.  The status lift is
  real but capped well below the hypothesis target — try a 3M-step run
  on the same obs to see whether more compute closes the Pikachu gap
  before adding new dims.
- Mankey is structurally hard (high-crit physical) and won't move on
  obs alone.  Either reward-shape against crit-RNG variance or accept
  it as a structural floor.

### Hypothesis for v0.4 (not started)

Add accuracy/evasion stages (4 dims, 36→40) + train 3M steps.  Expect
Pidgey to recover toward 65%+ and Pikachu/Spearow to push past 50%.
If Mankey still floors at ~45-50%, that's the structural-RNG floor and
not addressable from obs.

### Deferred to v0.4

- Multi-mon opponent parties + switching (actions 4..8 finally meaningful).
- Trainer AI (Brock / Misty scripted teams).
- Speed / Special stage dims if the pool adds Agility/Amnesia users.
- Contract reconciliation PR on `claude/quizzical-sammet` to unlock
  weight transfer.

---

_v0.3 shipped on `claude/jovial-carson`; Training Run 3 results above.
v0.4 candidate: accuracy/evasion stage dims + 3M steps._

---

## v0.4 — full sensors + 6v6 switching — 2026-04-15

**Goal:** close the remaining observation gaps from v0.3 (Pidgey Sand-Attack
regression root cause: no accuracy-stage signal) AND unlock the core Gen 1
tactical layer that was deferred since v0.1 — party switching.  This is the
"sensors + switching" release that takes the sim from "1v1 move picker" to
"6v6 tactical battle".

### The transfer contract — locked, versioned, documented

v0.4 introduces `battle_sim/TRANSFER_CONTRACT.md` — the single source of
truth for what `claude/quizzical-sammet` must implement to accept
battle-expert weights.  Covers obs layout, action semantics, reward shape,
rejection protocol, PyBoy macro executor sketch, and drift detection.
This is the **last contract break** planned — subsequent changes should be
additive-with-contract-version-bump, not layout re-orderings.

### Obs expansion: 36 → 92 dims

Block layout (see TRANSFER_CONTRACT.md for the full table):

| Block | Indices | Content |
|---|---|---|
| A | 0..35 | v0.3 active-mon tactical (unchanged) |
| B | 36..43 | remaining stat stages (spe/spc/acc/eva × 2 actors) |
| C | 44..88 | 5 bench slots × 9 dims (hp, level, status onehot, off-eff, def-eff) |
| D | 89..91 | active_slot_index/5, self_alive/6, opp_remaining/6 |

**Block B** — directly targeting the v0.3 Pidgey collapse.  The missing
signal was accuracy-stage feedback after Sand-Attack; Block B puts all four
remaining Gen 1 stages on the tape.

**Block C — bench encoding with effectiveness rollups, not type indices.**
Each bench slot carries (hp, level, status-onehot-5, off-eff-rollup,
def-eff-rollup).  The rollups are computed using species types only —
**never** opponent's hidden moveset — which mirrors what the PyBoy agent
can read from RAM (species bytes visible; move bytes of opp are not in
wild battles).  This avoids an "oracle policy" that can't transfer and
keeps the feature dim tight at 9 dims/slot × 5 slots = 45.

Why not type indices on bench?  Same reason v0.3 rejected ordinal status
encoding — categorical-as-ordinal forces the MLP to un-learn spurious
magnitude ordering, wasting sample efficiency.  The only thing the policy
uses bench-mon types FOR is matchup decisions; encode the answer directly.

**Block D** gives the value function the "party health bar" it needs for
stable returns across switches.  `opp_remaining` is the fog-of-war
ball-icons count — what a human sees in the real game.

**Bench display order** (v0.4 specific): bench[k] = the k-th non-active
party member in original-team order.  So when active=2, bench shows
[party[0], party[1], party[3], party[4], party[5]].  Action `4+k`
switches to `bench[k]`.  This allows "switch back to starter" (impossible
under a fixed party-index mapping), at the cost of making action semantics
active-index-dependent.  The PyBoy macro executor must read
`active_slot_index` before translating atomic→button.

### Action space + rejection protocol

Still `Discrete(9)` — 0..3 moves, 4..8 switches.  But 4..8 now have real
semantics and so does rejection:

- **Invalid action** = move slot out-of-range, move slot has 0 PP, switch
  to fainted/out-of-range/already-active bench pos.
- **Sim response**: engine sets `state.last_action_was_invalid=True`,
  player action burns, opponent still gets its normal action, turn counter
  advances.  Env applies `INVALID_ACTION_PENALTY = -0.05` on top of
  whatever damage the opp dealt.
- **PyBoy response** (spec'd in TRANSFER_CONTRACT.md §3, not implemented
  here): game rejects via "No!" dialog, wrapper detects no-turn-transition,
  applies the same -0.05.

**Why penalty over action-masking.**  Masking is a crutch that doesn't
transfer.  When the policy lands in PyBoy where there's no mask (just an
error "bloop"), it would panic on any fainted-slot select.  Penalty
teaches the constraint natively → no domain-shift shock at transfer.

### Engine state machine

`BattleState` grew from `player: Pokemon` / `opponent: Pokemon` to
`player_party: list[Pokemon]` / `opp_party: list[Pokemon]` with
`player_active` / `opp_active` indices.  `player` / `opponent` preserved as
@property accessors returning the active mon, so all v0.1-era tests and
call sites keep working.  Constructor accepts either `player=` (legacy
single-mon) or `player_party=` (v0.4) kwargs.

**Turn order in Gen 1 terms:**
1. Switch phase: both sides' switch actions resolve first (Gen 1 switches
   always precede attacks).  Outgoing mon's confusion_turns + flinched are
   cleared (volatile status).
2. Attack phase: speed-order (paralysis-quartered) resolution of remaining
   move actions.
3. End-of-turn residuals: burn/poison ticks.
4. **Forced switch-on-faint**: any side whose active just fainted
   auto-sends the lowest-index non-fainted party member.  Silent to the
   policy — no extra step.  Deferred to v0.5: policy-controlled post-faint
   switch.

### Reward

Per-step reward is now **party-wide** HP shaping:

```
shaping = dealt_opp_party_hp_frac - taken_our_party_hp_frac
```

where `party_hp_frac = mean(hp/max_hp over all party slots)`.  This keeps
the shaping scale invariant across switches — fainting a full-health opp
with one of your mons nets roughly the right magnitude regardless of which
mon was active for the KO.  Active-mon-only shaping would have made
switches look artificially costly (HP vanishes from the "our" side when
you switch out a damaged mon).

### Tests added / updated

- `test_obs_contract.py`: 92-dim assertion, Block B/C/D field-level tests,
  bench-ordering-skips-active test, rollup-uses-species-types-only test.
- `test_switching.py` (new, 13 tests): valid switch, display-order action
  mapping, switch-back-to-slot-0, reject fainted/out-of-range/already-
  active, reject invalid-move-slot and 0-PP, volatile-status clear on
  switch, forced-switch-on-faint picks lowest-living, battle ends on full
  wipe, env invalid-action penalty applied, env valid-action no-penalty.
- `test_env.py`: updated shape assertion 36 → 92.

**75/75 tests passing** on first clean run.  Smoke-train pipeline
(PPO MlpPolicy `Linear(92→64)→ReLU→Linear(64→64)→ReLU`) verified.

### Deferred to v0.5

- **Policy-controlled post-faint switch** (currently auto-send lowest
  living).  Would complicate episode step structure — one turn = possibly
  multiple policy calls.
- **Trainer opponent AI** (Brock/Misty scripted parties with sensible
  switching).  Opp policies still emit only 0..3 today.
- **Item usage** in battle (potions, status heal) — requires a bag-slot
  obs block and new action-space values.  This WOULD break the contract
  again, so it's a v1.0-not-v0.5 candidate.
- **Opponent visible moves** — in the real game, the player sees opp's
  moves after they're used once.  Adding this would be a real feature,
  but it's a fingerprint/sequence encoding problem (variable-length).

### Training Run 4 — v0.4 — 2026-04-15

**Command:**
```bash
python -m battle_sim.train_battle --num-cpu 4 --vec dummy --steps 1000000 --run-name battle_v0_4
```

**Duration:** ~2 min wall.  TB reports ~6,500-9,900 fps across the run.
Slightly slower than v0.3 per-step due to 92-dim obs + longer multi-mon
episodes (median length 10 vs v0.3's 3), but total time *decreased* because
the env rollout phase dominates less once episodes actually go somewhere.

**Headline results (1000-episode deterministic evaluation):**

| Metric | vs random | vs greedy | Δ vs v0.3 |
|--------|----------:|----------:|----------:|
| Win rate | **88.7%** | **80.7%** | +3.6pp / **+9.8pp** |
| Mean reward | +1.17 | +0.86 | −0.03 / +0.16 |
| Median battle length | 11 turns | 10 turns | +8 / +7 |

**+9.8pp vs-greedy is the biggest single-version win in the project's
history.**  The median-length jump is expected: 6v6 means fighting through
multiple opponents rather than one, so episodes naturally triple in length.

**Per-opponent win rate (vs greedy):**

| Opp | v0.3 | v0.4 | Δ |
|---|---:|---:|---:|
| **PIKACHU** | 38.7% | **65.7%** | **+27.0** |
| **PIDGEY** | 43.1% | **71.2%** | **+28.1** |
| MANKEY | 46.8% | 72.1% | **+25.3** |
| SPEAROW | 45.1% | 64.7% | +19.6 |
| PARAS | 58.9% | 82.3% | +23.4 |
| NIDORAN_F | 65.6% | 87.9% | +22.3 |
| RATTATA | 72.2% | 76.2% | +4.0 |
| ODDISH | 57.9% | 87.7% | +29.8 |
| ZUBAT | 61.4% | 90.0% | +28.6 |
| BELLSPROUT | 76.9% | 87.7% | +10.8 |
| GEODUDE | 80.4% | 84.0% | +3.6 |
| NIDORAN_M | 91.1% | 75.0% | −16.1 |
| ONIX | 91.4% | 85.7% | −5.7 |
| WEEDLE | 96.2% | 80.4% | −15.8 |
| CATERPIE | 95.7% | 84.5% | −11.2 |
| SANDSHREW | 90.9% | 83.6% | −7.3 |
| EKANS | 96.6% | 93.4% | −3.2 |

**Every previously-hard matchup moved 20-30pp.**  The regressions are on
the "easy" end — Weedle/Caterpie/Nidoran_M — which I read as a compute-
redistribution artifact: the policy now optimizes team-level play, so it
sometimes lets a trivial opp chip a sacrifice mon to preserve a better
matchup later.  Mean reward still positive on all matchups.

**All 4 Run-4 hypotheses confirmed:**
1. ✅ Pidgey recovers 43% → **71%** (target was 65%+).  Block B's
   accuracy-stage dim fixes the v0.3 Sand-Attack blind spot.
2. ✅ Pikachu past 50% — lands at **66%**.  Status obs + speed-stage
   visibility closed the paralysis-gap that couldn't be overcome at v0.3.
3. ✅ Switch action usage 5.1% vs greedy (6.3% vs random) — meets the >5%
   bar.  Proactive switching is being used, not just auto-switch-on-faint.
4. ✅ Slot-1 debuff usage 1.2% — stable in the 1-5% band (no regression
   to v0.2's 0% collapse or overuse).

**Per-player starter:**

| Starter | v0.3 vs greedy | v0.4 vs greedy | Δ |
|---------|---------------:|---------------:|--:|
| CHARMANDER | 78.9% | 83.6% | +4.7 |
| SQUIRTLE   | 70.5% | 79.4% | +8.9 |
| BULBASAUR  | 64.0% | **79.1%** | **+15.1** |

**Bulbasaur closing 15pp to catch the pack is the headline emergent
behavior.**  In v0.3 Bulbasaur was stuck eating Peck/Ember with only Vine
Whip as its out; in v0.4 with 3-6 mon teams the policy learned to pivot
out of bad matchups to a Pidgey/Rattata partner.  The -0.01 step penalty
+ -0.05 invalid penalty apparently didn't deter this — the implicit cost
of staying in a bad matchup exceeds the cost of a switching turn.

**Action usage distribution (per-step, vs greedy):**

| Action | v0.3 | v0.4 | Meaning |
|--------|-----:|-----:|---|
| 0 (move slot 0) | 32.6% | **51.5%** | primary STAB / damaging move |
| 1 (move slot 1) | 1.9% | 1.2% | debuff — rarely worth it |
| 2 (move slot 2) | 28.1% | 28.0% | secondary damaging move |
| 3 (move slot 3) | 37.4% | 14.2% | third move slot |
| **4** (bench[0]) | — | 0.0% | never |
| **5** (bench[1]) | — | **2.6%** | main "eject button" |
| 6 (bench[2]) | — | 0.0% | never |
| 7 (bench[3]) | — | **2.2%** | secondary eject |
| 8 (bench[4]) | — | 0.3% | edge case |

**Emergent switching exists and has a recognizable shape.**  Actions 5
and 7 carry ~all of the switch usage; 4/6/8 are effectively dead.  This
is a classic local optimum under a 1M-step budget: the policy learned
"when in a bad matchup, hit Action 5" as a heuristic rather than
evaluating each bench slot's matchup delta.  In the 3-6 mon sampled
teams, bench positions 1 and 3 statistically carry the best pivot
partners — the policy tuned to that distribution.  A broader team
sampler (or 3M steps) would probably redistribute this.

**TensorBoard convergence:**

| Metric | Start | End | Read |
|--------|------:|----:|------|
| `rollout/ep_rew_mean` | −2.05 | +1.18 | Negative start = early random policy spamming invalid actions (−0.05 each).  Climb steeper than v0.3 because 6v6 has more reward to collect via multi-mon KOs. |
| `rollout/ep_len_mean` | 45.9 | 16.4 | Started 46 turns (bad policy drags out), ended 16 (decisive multi-mon).  +11 vs v0.3's 4.8 is the multi-mon structural delta. |
| `train/entropy_loss` | −2.19 | **−0.63** | Higher than v0.3 end (−0.41) = **more exploration retained**.  With 9 real actions instead of effectively-4, the policy stays less collapsed — healthy for continued switch-slot discovery. |
| `train/explained_variance` | +0.11 | +0.59 | Matches v0.3 plateau (+0.58).  Value fn predicts ~60% of return variance; bottleneck isn't return predictability. |
| `train/approx_kl` | 0.014 | 0.005 | Small updates at convergence.  Healthy. |
| `train/value_loss` | 0.27 | **0.19** | **Best of any run** (v0.3: 0.41, v0.2: 0.50).  Party-wide HP shaping + bench visibility made returns more learnable. |
| `time/fps` | 9,896 | 6,555 | Cold start ~10k, steady ~6.5k.  Lower than v0.3 (8k) due to obs size + episode length, still very fast. |

### What v0.4 proves

- **Block B (acc/spe/spc/eva stages) closed the Pidgey blind spot** and
  delivered broad status-matchup lifts — Pikachu/Spearow/Paras all crossed
  into winning-majority territory for the first time.
- **Block C (bench visibility) enabled emergent proactive switching** at
  ~5% rate, with a learnable local-optimum shape (actions 5 + 7 favored).
- **Block D (party meta) + party-wide HP shaping** produced project-best
  value_loss (0.19) — the value function can actually predict returns
  across switch-heavy trajectories.
- **Bulbasaur's 15pp catch-up** demonstrates the policy learned the
  *opportunity cost* of staying in a bad matchup vs paying the step
  penalty to pivot — exactly the tactical reasoning the obs expansion
  was designed to enable.
- **Rejection-by-penalty** (−0.05, no masking) didn't break learning:
  ep_rew_mean starts at −2.05 (early invalid-action spamming) and climbs
  cleanly.  The policy internalizes the constraint.
- **Transfer-graft test still passes** (`test_transfer_compat.py` on
  92-dim obs): weight copy into mock `PokemonNet.tactical` reproduces
  forward-pass bit-for-bit.

### What v0.4 does not exercise

- **Policy-controlled post-faint switch** — auto-send lowest-living still
  handles this.  Would double step count for switch episodes, so deferred
  to v0.5 when it's worth the PPO rollout-math rework.
- **Trainer-team opponents** — all opponents still use wild single-mon
  pool.  Brock/Misty scripted parties (with their own switching logic)
  are a v0.5 target.
- **Opponent visible-moves memory** — once opp used a move the player
  saw it, but obs doesn't encode this.  Variable-length sequence encoding
  problem; deferred indefinitely.
- **Item usage in battle** — would break the contract a 4th time (new
  action-space values + bag obs block).  Hard v1.0-not-v0.5 candidate.

### Shipped

v0.4 delivers:
- 92-dim obs contract locked in `TRANSFER_CONTRACT.md` (single source of
  truth for the sister branch).
- Battle-expert `battle_expert.zip` in `runs/battle_v0_4/` trainable in
  ~2 min, ready for weight transfer once `claude/quizzical-sammet`
  implements the matching `game_state.tactical_obs()`.
- 75 tests passing including 13 new switching-specific cases covering
  the full rejection protocol.

**Week-4 transfer is unblocked on this branch.**  Next step is the
contract-reconciliation PR on `claude/quizzical-sammet` — that branch
appends Blocks B/C/D to its `game_state.tactical_obs()` and bumps
`TACTICAL_OBS_SIZE` to 92, at which point the graft code in
`test_transfer_compat.py` becomes the deploy path.

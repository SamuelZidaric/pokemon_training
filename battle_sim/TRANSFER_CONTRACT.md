# Transfer Contract — `claude/jovial-carson` (sim) ↔ `claude/quizzical-sammet` (full-game)

This document is the single source of truth for what the **full-game PokemonNet
agent on `claude/quizzical-sammet` must implement** to accept battle-expert
weights trained in this simulator.  When the two branches disagree, this
document wins and the other branch conforms.  Weight transfer is only
authorized when both branches pass the contract-drift guard.

**Version:** v0.4  (obs `Box(92,)` · atomic `Discrete(9)` · rejection-by-penalty)

---

## 1. What transfers, what doesn't

The only weights copied across are the **tactical feature trunk**:

```
mlp_extractor.policy_net   ──copy──►   PokemonNet.tactical
    Linear(92, 64) → ReLU → Linear(64, 64) → ReLU
```

Not transferred:
- The SB3 `pi` action head — full-game retrains its own, potentially over a
  different action distribution.
- The SB3 `vf` value head — the full-game value function depends on full-game
  reward, not battle reward.
- Feature extractors for non-tactical obs keys (screen CNN, event MLP, etc.).

This is what `battle_sim/tests/test_transfer_compat.py` verifies with a mock
of `PokemonNet.tactical` — shape, ReLU-not-Tanh, FlattenExtractor identity,
and bit-exact forward-pass parity after weight graft.

---

## 2. Obs contract — `Box(-1, 1, (92,), float32)`

`v2/game_state.py::tactical_obs()` on the sister branch must emit a 92-dim
float32 vector with index-for-index matching semantics to
`battle_sim/obs.py::tactical_obs()`.  Block A is **unchanged from v0.3** to
preserve the compatibility path for v0.3 weights — only Blocks B/C/D are new.

### Block A — active-mon tactical (indices 0..35, unchanged)
See `battle_sim/obs.py` and `v0.3` DEVLOG entry.  22 v0.1/v0.2 dims + 14
v0.3 dims (status one-hot × 2, atk/def stages × 2).

### Block B — remaining stat stages (indices 36..43)
Each is `stage / 6` clamped to `[-1, 1]`.  These are the **four Gen 1 stages
v0.3 didn't encode**: speed, special, accuracy, evasion.

| Idx | Field | Source |
|---:|---|---|
| 36 | `self.spe_stage / 6` | active player mon |
| 37 | `self.spc_stage / 6` | active player mon |
| 38 | `self.acc_stage / 6` | active player mon |
| 39 | `self.eva_stage / 6` | active player mon |
| 40 | `opp.spe_stage / 6` | active opponent mon |
| 41 | `opp.spc_stage / 6` | active opponent mon |
| 42 | `opp.acc_stage / 6` | active opponent mon |
| 43 | `opp.eva_stage / 6` | active opponent mon |

### Block C — bench party (indices 44..88, 5 slots × 9 dims)
Bench slots are the **5 non-active party members in original-team order** (active is
skipped from the enumeration).  If the full team has fewer than 6 members,
trailing bench slots are zero-filled.  This means bench[k] is a specific party
member that can shift position when the active mon changes — but it always
refers to the same party member the switch action on that index selects.

Switch action index coupling: atomic action `4+k` maps to **bench display
position `k`**, which is the `k`-th non-active party member in original-team
order.  So:
- If active = party[0], bench display = [1, 2, 3, 4, 5]; action 4 → party[1].
- If active = party[2], bench display = [0, 1, 3, 4, 5]; action 4 → party[0].

This allows "switch back to starter" (impossible under a fixed party-index
mapping) at the cost of a position-dependent action semantic.  The PyBoy
macro executor must read `active_slot_index` before translating the button
sequence.

Per bench slot `k ∈ {0..4}`, offset `44 + 9*k`:

| Offset | Field | Encoding | Range |
|---:|---|---|---|
| +0 | `hp_frac` | `hp / max_hp` | [0, 1]; 0 = fainted |
| +1 | `level / 100` | — | [0, 1] |
| +2..+6 | status one-hot | PAR/SLP/BRN/PSN/FRZ; OK = all zeros | {0, 1} |
| +7 | best-own-move-eff vs current opp (rollup) | `max over own types of type_eff(self_type, opp.type1, opp.type2)`, capped /4 | [0, 1] |
| +8 | opp's best-move-eff vs this bench mon (rollup) | `max over opp's types of type_eff(opp_type, self_type1, self_type2)`, capped /4 | [0, 1] |

**Contractual constraint on +7 and +8 — visible-info only.**  The effectiveness
rollup must be computed from **species types only**, never from the opponent's
hidden moveset.  Using `attacker_types` (not `attacker_moves`) keeps the
calculation honest against PyBoy RAM, which cannot read opp's moves in wild
battles and cannot read opp's bench moves in trainer battles.  This is why
it's called a "rollup" not an "analyzer" — it's the same information a human
gets from looking at sprites.

Bench slots beyond the team size are zero-filled.  A fainted party member at
a given position emits its real hp (0) / status / level / types for the
rollup — do not zero-fill on faint, because the policy learning "that slot
is dead, don't pick it" needs a stable signal for position + faint.

### Block D — battle-global meta (indices 89..91)

| Idx | Field | Encoding | Range |
|---:|---|---|---|
| 89 | `active_slot_index / 5` | which party pos 0..5 is on field | [0, 1] |
| 90 | `self_alive_count / 6` | party members with hp > 0 | [0, 1] |
| 91 | `opp_remaining / 6` | fog-of-war opp party size (the ball-icons signal) | [0, 1] |

`opp_remaining` is the **number of opp party members the player can see
remaining** — i.e. opp_alive_count in the sim, since the sim grants full
visibility.  On PyBoy this reads the opp-party-ball-icon state, not the
real RAM party struct.

---

## 3. Action contract — `Discrete(9)`, atomic

| Atomic | Meaning |
|---|---|
| 0..3 | use active-mon move at slot k |
| 4..8 | switch active mon out, send in bench display position (k - 4), i.e. bench[0..4]; see Block C for display-order semantics |

The sim **never** exposes button-level actions.  Both environments operate
at the atomic layer.  PyBoy env wraps a `BattleMacroExecutor` that translates
atomic → button sequence (see §5).

### Rejection protocol (critical)

An atomic action is **invalid** if any of:
- moves 0..3: the move slot is out of range, or the move's PP is 0.
- switches 4..8: the target party slot is beyond team size, is currently
  active, or is fainted.

On an invalid action, **both environments must**:
1. Not advance the battle state (player does nothing this turn).
2. Still let the opponent take its turn (the "wasted turn" cost).
3. Apply an **additional `-0.05` reward penalty** on top of whatever the
   opponent does to the player.
4. Not terminate the episode (unless the opponent's move results in a faint).
5. Advance the turn counter by 1.

This is enforced in the sim via `BattleState.last_action_was_invalid` and a
reward hook in `PokemonBattleEnv.step`.  On PyBoy, the game itself rejects
the input ("No!" dialog / ball bounce), the wrapper detects no-turn-transition,
and applies the same -0.05.

**Do not action-mask invalid actions in either environment.**  The policy
must learn the constraint natively so it generalizes without fine-tune cliff.

### Forced-switch-on-faint

When the active mon faints and at least one non-active, non-fainted party
member remains, both environments must auto-send the **lowest-index living
party member** as a silent engine-level transition.  This is not an atomic
action and is not exposed to the policy.  If no living member remains, the
battle resolves as a loss.

Rationale: keeping one atomic action = one policy step.  Exposing a
post-faint switch choice to the policy would double the step count for
switch episodes, which breaks PPO's rollout arithmetic and complicates
reward attribution.  Deferred to v0.5.

---

## 4. Reward contract

Per step:

```
reward = terminal_rew + shaping + step_pen + invalid_pen
```

| Component | Value |
|---|---|
| `terminal_rew` | +1.0 on win, -1.0 on loss, 0 otherwise |
| `shaping` | `dealt_opp_hp_frac - taken_our_hp_frac` (each in [0, 1]) |
| `step_pen` | -0.01 |
| `invalid_pen` | -0.05 iff `state.last_action_was_invalid` else 0 |

HP fractions are **whole-party** on the sim side in v0.4 — both `dealt` and
`taken` sum across all opp/self party members respectively, normalized by
party size.  PyBoy must mirror this: do not use active-mon-only deltas or
the reward scale changes across switches.

---

## 5. PyBoy-side macro executor (informational, not this branch)

This section is not binding on this branch — it's a handoff sketch of what
`claude/quizzical-sammet` needs to implement.  The canonical implementation
lives there.

Atomic action → button sequence, starting and ending at **battle main menu**
(canonical state; verify via RAM handshake, never by frame-count):

```
0..3  (move k)    : A → {cursor_to_FIGHT} A → {D-pad k times} A → wait_for_main_menu()
4..8  (switch)    : A → {cursor_to_POKEMON} A → {D-pad (k-3) times} A → A(SWITCH) → wait_for_main_menu()
```

`wait_for_main_menu()` requirements:
- Block until RAM indicates the battle main menu is idle and HP bar drain is complete.
- Do **not** sample the tactical obs until this returns — stale HP / in-flight
  status animation are the two most common transfer-killing pitfalls.
- If the game rejected the action (faint, no PP, whited-out dialog), detect
  via no-turn-counter advance after a bounded frame budget, back cursor out
  to main menu, return with `rejected=True` so the env can apply `-0.05`.

Menu-drift hazards to handle:
- Pokémon-just-learned-a-move prompt after a level-up battle.
- Low-HP warning beep (purely cosmetic; consumes no frames but can desync naive sequences).
- Catch tutorial overlay (Viridian Forest first-encounter).
- Trainer dialog boxes between opponent mons in a party battle.

---

## 6. Drift detection

Both branches must carry a copy of these constants and validate them.
Sim-side: `battle_sim/v2_contract.py` + `battle_sim/tests/test_obs_contract.py`.
Full-game-side: `v2/game_state.py` + a mirror test that imports from both and
asserts shape equality.

When this file is updated, increment the version header at the top AND bump
`TACTICAL_OBS_SIZE`.  Any PR that changes `TACTICAL_OBS_SIZE` on one side
without the other is a hard block on merging weights.

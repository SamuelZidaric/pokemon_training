# Battle Simulator — Phase A

Scripted Gen 1 Pokemon battle simulator for training a battle-expert policy
that transfers into the full-game agent's tactical branch.

See [SPEC.md](SPEC.md) for the transfer-contract audit and design rationale.

## Setup

```bash
# 1. Clone pret/pokered once (gitignored)
git clone --depth 1 https://github.com/pret/pokered.git pokered_src

# 2. Regenerate data JSON (commit the outputs)
python battle_sim/scripts/parse_pokered.py

# 3. Run the test suite
python -m pytest battle_sim/tests/ -v
```

## Quick train (smoke)

```bash
python -m battle_sim.train_battle --num-cpu 1 --steps 10000 --run-name smoke
```

Full training run:

```bash
python -m battle_sim.train_battle --num-cpu 4 --steps 1000000
```

## Module map

| File | Purpose |
|------|---------|
| `data/` | Parsed pokered JSON (types, moves, base stats, chart, learnsets) |
| `scripts/parse_pokered.py` | ASM → JSON parser (run once after cloning pokered) |
| `v2_contract.py` | Transfer-contract constants duplicated from `v2/game_state.py` |
| `rng.py` | Seeded byte-stream RNG |
| `data_loader.py` | JSON load + `type_effectiveness` helper |
| `entities.py` | `Pokemon` / `Move` dataclasses, stat formulas, stage multipliers |
| `damage.py` | Pure Gen 1 damage / hit / crit functions |
| `engine.py` | 1v1 `BattleEngine` state machine |
| `obs.py` | 22-float `tactical_obs` — the transfer contract |
| `env.py` | Gymnasium `PokemonBattleEnv` with zero-padded full-game obs Dict |
| `train_battle.py` | SB3 PPO entry point |
| `tests/` | Unit + smoke tests (27 passing) |

## v0.1 scope / non-goals

**In:** 1v1, damage formula, type chart, STAB, crit (with stat-stage bypass),
hit check (incl. 1/256 miss on 100% moves), Ghost→Psychic = 0× bug, RNG
determinism, Gymnasium env with full-shape obs Dict.

**Deferred (v0.2):** status conditions (par/slp/brn/psn/frz), switching,
multi-hit moves, fixed-damage moves (Dragon Rage, Seismic Toss), Substitute,
Struggle semantics, trainer AI.

**Deferred (Week 4):** weight transfer into full-game PokemonNet. Requires
merging with `claude/quizzical-sammet` first (or cherry-picking `pokemon_model.py`).

## Decisions made for v0.1 (see SPEC.md Part 7)

1. **Action space:** semantic `Discrete(9)` (`[move0..3, switch1..5]`). Week-4
   button adapter will translate to the full-game 7-button space.
2. **Ghost→Psychic:** sim follows pokered data (0×, NO_EFFECT). `v2` chart
   currently reads 2× — **separate commit on `claude/quizzical-sammet`
   needed** to patch `_TYPE_CHART` or the sim and env will disagree.
3. **Differential harness vs PyBoy:** deferred. Requires cross-branch
   tooling and a ROM path; structural scaffold is in place but Week 1's
   50k/s + 100-case parity metric isn't verified yet. Current tests cover
   formula correctness and internal determinism.
4. **Pokered pin:** commit `9441f1aafe96c1bb8160b94936a29207ad6a12c3`
   (written to `data/POKERED_VERSION.txt` at parse time).

## Known divergences from the eventual full-game env

Flagged so they don't silently break weight transfer:

- **`best_move_effectiveness` uses attacker's own types, not real move types.**
  Parity with `v2/game_state.py:463`. Upgrading both together is a v0.2 task
  — plan to expand `tactical_obs` by ~4 dims and retrain from scratch.
- **Ghost→Psychic currently 2× in `v2/game_state.py`** but 0× here. Reconcile
  before any weight transfer happens.
- **Poison ↔ Bug absent from v2 type chart.** Gen 1 had both as 2×; sim
  follows pokered, v2 needs a fix.

## Transfer contract (short form)

Observation fed to the policy's tactical branch:

- `Box(low=-1, high=1, shape=(22,), dtype=float32)`
- Index layout: see `SPEC.md` Part 1.1 table
- Type index map: `v2_contract.TYPE_ID_LIST` (15 entries, Gen 1 type IDs with gaps)
- Move ID domain: raw Gen 1 1..165, normalized /165

`tests/test_obs_contract.py` asserts all three stay in sync with the v2
authoritative values.

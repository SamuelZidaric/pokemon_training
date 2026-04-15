"""Deterministic eval harness for trained battle expert.

Mirrors the v0.2 eval methodology: 1000 episodes vs random opponent +
1000 episodes vs greedy (highest_power) opponent.  Reports overall
win-rate, mean reward, median episode length, per-opponent breakdown,
per-starter breakdown, and player action-slot usage.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict

import numpy as np
from stable_baselines3 import PPO

from battle_sim.engine import (
    highest_power_opponent_policy,
    random_opponent_policy,
)
from battle_sim.env import PokemonBattleEnv


def run_eval(model_path: str, n_episodes: int, opponent: str, seed_base: int = 10_000):
    if opponent == "random":
        opp_policy = random_opponent_policy
    elif opponent == "greedy":
        opp_policy = highest_power_opponent_policy
    else:
        raise ValueError(opponent)

    model = PPO.load(model_path, device="cpu")

    wins = 0
    rewards = []
    lengths = []
    per_opp = defaultdict(lambda: [0, 0])  # species -> [wins, total]
    per_starter = defaultdict(lambda: [0, 0])
    action_counter = Counter()

    for ep in range(n_episodes):
        env = PokemonBattleEnv(
            opponent_policy=opp_policy, max_turns=200, seed=seed_base + ep,
        )
        obs, _ = env.reset(seed=seed_base + ep)
        starter = env.state.player.species
        opp_species = env.state.opponent.species

        ep_rew = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counter[action] += 1
            obs, r, term, trunc, _ = env.step(action)
            ep_rew += r
            steps += 1
            done = term or trunc

        won = env.state.result == 1
        wins += int(won)
        rewards.append(ep_rew)
        lengths.append(steps)
        per_opp[opp_species][1] += 1
        per_opp[opp_species][0] += int(won)
        per_starter[starter][1] += 1
        per_starter[starter][0] += int(won)

    return {
        "n": n_episodes,
        "win_rate": wins / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "median_length": int(statistics.median(lengths)),
        "per_opp": {k: (v[0] / v[1], v[1]) for k, v in per_opp.items()},
        "per_starter": {k: (v[0] / v[1], v[1]) for k, v in per_starter.items()},
        "actions": dict(action_counter),
    }


def fmt(label, r):
    print(f"\n=== {label} ({r['n']} eps) ===")
    print(f"  win_rate     : {r['win_rate']*100:.1f}%")
    print(f"  mean_reward  : {r['mean_reward']:+.2f}")
    print(f"  median_length: {r['median_length']} turns")
    print("  per-opponent:")
    for sp, (wr, n) in sorted(r["per_opp"].items(), key=lambda x: -x[1][0]):
        print(f"    {sp:12s} {wr*100:5.1f}%  (n={n})")
    print("  per-starter:")
    for sp, (wr, n) in sorted(r["per_starter"].items(), key=lambda x: -x[1][0]):
        print(f"    {sp:12s} {wr*100:5.1f}%  (n={n})")
    total = sum(r["actions"].values())
    print("  action usage:")
    for a in range(9):
        c = r["actions"].get(a, 0)
        print(f"    slot {a}: {c/total*100:5.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="battle_sim/runs/battle_v0_3/battle_expert.zip")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()

    rnd = run_eval(args.model, args.n, "random", seed_base=10_000)
    grd = run_eval(args.model, args.n, "greedy", seed_base=20_000)
    fmt("vs RANDOM", rnd)
    fmt("vs GREEDY", grd)


if __name__ == "__main__":
    main()

"""PPO training entry point for the battle specialist.

End-to-end: spins up N parallel PokemonBattleEnvs and trains a small MLP
policy on the tactical observation.  The non-tactical keys are zero-padded
so the resulting policy's feature extractor is compatible with the full
PokemonNet tactical branch.

Usage:

    python -m battle_sim.train_battle --num-cpu 4 --steps 1_000_000

After training, the saved ``.zip`` is loadable into the full-game PokemonNet
by copying the policy's tactical-branch weights into the corresponding
module.  See ``transfer_to_full_game.py`` (Week 4, TODO).

Design note — we use a simple ``MultiInputPolicy`` here, not PokemonNet.
PokemonNet lives on ``claude/quizzical-sammet`` and imports aren't available
on this branch; Week 4 will build the transfer tool once both branches
merge or we cherry-pick ``pokemon_model.py``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from battle_sim.env import PokemonBattleEnv


def _make_env(seed: int, monitor_dir: str | None = None):
    def _thunk():
        env = PokemonBattleEnv(seed=seed, max_turns=200)
        # Monitor wraps the env so ep_rew_mean / ep_len_mean log to TB.
        # Each worker gets its own monitor.csv; SB3 aggregates across workers.
        return Monitor(env, filename=(f"{monitor_dir}/monitor_{seed}" if monitor_dir else None))
    return _thunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpu", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=str, default="battle_sim/runs")
    parser.add_argument("--run-name", type=str, default="battle_v0_1")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    mon_dir = str(run_dir / "monitor")
    Path(mon_dir).mkdir(exist_ok=True)

    # Vec env
    if args.num_cpu == 1:
        venv = DummyVecEnv([_make_env(0, mon_dir)])
    else:
        venv = SubprocVecEnv([_make_env(i, mon_dir) for i in range(args.num_cpu)])

    model = PPO(
        policy="MultiInputPolicy",
        env=venv,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        tensorboard_log=str(run_dir / "tb"),
        verbose=1,
        device=args.device,
    )
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(str(run_dir / "battle_expert.zip"))
    print(f"saved {run_dir / 'battle_expert.zip'}")


if __name__ == "__main__":
    main()

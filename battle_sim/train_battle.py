"""PPO training entry point for the battle specialist.

v0.3 thin-obs: the env emits a flat Box(36,) of tactical floats and we
train with ``MlpPolicy`` whose ``mlp_extractor.policy_net`` is shaped
exactly like PokemonNet's tactical branch — Linear(36→64)→ReLU→Linear(64→64)→ReLU.
At transfer time those two Linears copy straight into the full-game
network.  See ``tests/test_transfer_compat.py`` for the shape/weight guard.

Usage:

    python -m battle_sim.train_battle --num-cpu 4 --steps 1_000_000
    # Or single-process (often faster post thin-obs):
    python -m battle_sim.train_battle --num-cpu 1 --steps 1_000_000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from battle_sim.env import PokemonBattleEnv


# Network architecture — MUST match PokemonNet.tactical on
# claude/quizzical-sammet for weight transfer to succeed.  Do NOT rely on
# SB3 defaults here: depths / activations drift between versions and
# silently break transfer.
POLICY_KWARGS = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    activation_fn=torch.nn.ReLU,
)


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
    parser.add_argument("--run-name", type=str, default="battle_v0_3")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--vec", type=str, default="subproc", choices=["subproc", "dummy"],
        help="VecEnv backend.  'dummy' is single-process — often faster for "
             "this thin-obs env since IPC cost dominates engine cost.",
    )
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    mon_dir = str(run_dir / "monitor")
    Path(mon_dir).mkdir(exist_ok=True)

    # Vec env — DummyVecEnv is often faster post thin-obs because the
    # engine runs in tens of microseconds and SubprocVecEnv's pickle cost
    # dominates.  Pass --vec dummy to force single-process, --vec subproc
    # to keep the multi-process path.
    thunks = [_make_env(i, mon_dir) for i in range(args.num_cpu)]
    if args.vec == "dummy" or args.num_cpu == 1:
        venv = DummyVecEnv(thunks)
    else:
        venv = SubprocVecEnv(thunks)

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        policy_kwargs=POLICY_KWARGS,
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

"""Tactical training script — Hyperbolic Time Chamber for Pokemon Red.

This script ties together the full tactical training pipeline:

1. **Curriculum** — focused save states (gym battles, dungeons)
2. **Reward system** — battle-focused, survival, or speedrun presets
3. **Observations** — tactical battle data (types, HP, PP, effectiveness)
4. **Neural network** — dedicated battle branch in PokemonNet

Usage::

    # Battle training against Brock (create pre_brock.state first):
    python train_tactical.py --mode battle --gym-state ../pre_brock.state

    # Survival training through Mt. Moon:
    python train_tactical.py --mode survival --init-state ../pre_mtmoon.state

    # Speedrun mode (maximise progress per step):
    python train_tactical.py --mode speedrun

    # Full pipeline (standard training but with tactical obs + enhanced rewards):
    python train_tactical.py --mode enhanced

    # With frame stacking and action macros:
    python train_tactical.py --mode battle --gym-state ../pre_brock.state \\
        --frame-stack 4 --action-macros --num-cpu 96
"""

from __future__ import annotations

import argparse
import sys
from os.path import exists
from pathlib import Path

from red_gym_env_v2 import RedGymEnv
from config import EnvConfig
from rewards import (
    create_default_reward_system,
    create_enhanced_reward_system,
    create_battle_focused_system,
    create_survival_system,
    create_speedrun_system,
)
from curriculum import CurriculumScheduler, CurriculumStage, create_battle_curriculum

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


# ---------------------------------------------------------------------------
# Reward system factory
# ---------------------------------------------------------------------------

REWARD_MODES = {
    "default": create_default_reward_system,
    "enhanced": create_enhanced_reward_system,
    "battle": create_battle_focused_system,
    "survival": create_survival_system,
    "speedrun": create_speedrun_system,
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(rank: int, env_conf: dict, reward_mode: str, seed: int = 0):
    def _init():
        env = RedGymEnv(env_conf)

        # Swap in the tactical reward system
        if reward_mode in REWARD_MODES:
            factory = REWARD_MODES[reward_mode]
            if reward_mode in ("battle", "survival", "speedrun"):
                env.reward_system = factory(
                    reward_scale=env_conf.get("reward_scale", 1.0),
                )
            else:
                env.reward_system = factory(
                    reward_scale=env_conf.get("reward_scale", 1.0),
                    explore_weight=env_conf.get("explore_weight", 1.0),
                )

        # Optional wrappers
        if env_conf.get("use_action_macros"):
            from action_macros import MacroActionWrapper
            env = MacroActionWrapper(env)

        if env_conf.get("use_frame_stack"):
            from frame_stack import DictFrameStack
            env = DictFrameStack(
                env,
                n_stack=env_conf.get("frame_stack_n", 4),
                stack_keys=["screens", "health", "level", "tactical"],
                passthrough_keys=["events", "map"],
            )

        # Optional streaming
        if env_conf.get("use_stream"):
            from stream_agent_wrapper import StreamWrapper
            env = StreamWrapper(env, stream_metadata={
                "user": f"tactical-{reward_mode}",
                "env_id": rank,
                "color": "#FF4444" if reward_mode == "battle" else "#4499FF",
                "extra": f"mode:{reward_mode}",
            })

        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tactical Pokemon Red trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", choices=list(REWARD_MODES.keys()), default="enhanced",
        help="Reward preset to use",
    )

    # Environment
    parser.add_argument("--gb-path", default="../PokemonRed.gb")
    parser.add_argument("--init-state", default="../init.state",
                        help="Default save state (used for non-battle modes)")
    parser.add_argument("--gym-state", default=None,
                        help="Save state for battle mode (right before a gym)")
    parser.add_argument("--reward-scale", type=float, default=0.5)
    parser.add_argument("--explore-weight", type=float, default=0.25)

    # Wrappers
    parser.add_argument("--frame-stack", type=int, default=0)
    parser.add_argument("--action-macros", action="store_true")
    parser.add_argument("--stream", action="store_true")

    # Training
    parser.add_argument("--num-cpu", type=int, default=64)
    parser.add_argument("--ep-length", type=int, default=2048 * 80)
    parser.add_argument("--session-path", default="runs/tactical")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--iterations", type=int, default=10000)

    args = parser.parse_args()

    # Resolve save state
    if args.mode == "battle":
        if args.gym_state is None:
            print("ERROR: --gym-state is required for battle mode")
            print("  Create a save state right before a gym leader battle,")
            print("  then pass it with: --gym-state ../pre_brock.state")
            sys.exit(1)
        init_state = args.gym_state
    else:
        init_state = args.init_state

    sess_path = Path(args.session_path) / args.mode
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "init_state": init_state,
        "max_steps": args.ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": args.gb_path,
        "reward_scale": args.reward_scale,
        "explore_weight": args.explore_weight,
        # Wrapper flags
        "use_action_macros": args.action_macros,
        "use_frame_stack": args.frame_stack > 0,
        "frame_stack_n": args.frame_stack,
        "use_stream": args.stream,
    }

    print(f"=== Tactical Training: {args.mode.upper()} mode ===")
    print(f"Save state: {init_state}")
    print(f"Workers: {args.num_cpu}")
    print(f"Episode length: {args.ep_length}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Action macros: {args.action_macros}")
    print(f"Reward scale: {args.reward_scale}")
    print()

    env = SubprocVecEnv([
        make_env(i, env_config, args.mode) for i in range(args.num_cpu)
    ])

    checkpoint_callback = CheckpointCallback(
        save_freq=args.ep_length // 2,
        save_path=str(sess_path),
        name_prefix=f"tactical_{args.mode}",
    )
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    train_steps_batch = args.ep_length // args.num_cpu

    if args.checkpoint and exists(args.checkpoint + ".zip"):
        print(f"Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = args.num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = args.num_cpu
        model.rollout_buffer.reset()
    else:
        # Battle mode: faster learning with higher entropy for exploration
        ent_coef = 0.02 if args.mode == "battle" else 0.01
        lr = 3e-4 if args.mode == "battle" else 2.5e-4

        model = PPO(
            "MultiInputPolicy", env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=512,
            n_epochs=1 if args.mode != "battle" else 3,
            gamma=0.997 if args.mode != "battle" else 0.99,
            ent_coef=ent_coef,
            learning_rate=lr,
            tensorboard_log=str(sess_path),
        )

    print(model.policy)
    print()

    model.learn(
        total_timesteps=args.ep_length * args.num_cpu * args.iterations,
        callback=CallbackList(callbacks),
        tb_log_name=f"tactical_{args.mode}",
    )


if __name__ == "__main__":
    main()

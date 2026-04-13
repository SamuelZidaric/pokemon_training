"""Ray RLlib training script for Pokemon Red.

Replaces SubprocVecEnv-based training with Ray's distributed architecture.
Scales from a single machine to a cluster without code changes.

Usage::

    # Single machine, 64 workers:
    python ray_trainer.py --num-workers 64

    # Resume from checkpoint:
    python ray_trainer.py --num-workers 64 --checkpoint runs/ray/checkpoint_000100

    # With frame stacking and action macros:
    python ray_trainer.py --num-workers 64 --frame-stack 4 --action-macros

    # Cluster mode (after `ray start --head` on driver):
    python ray_trainer.py --num-workers 256 --address auto

Architecture
------------
Ray RLlib manages the full lifecycle:

1. **EnvRunner workers** — each worker runs one PyBoy emulator in its
   own process.  No GIL contention, no pickle overhead for observations.
2. **Learner** — batches rollouts from all workers and runs GPU-accelerated
   PPO updates.  Supports multi-GPU out of the box.
3. **Checkpointing** — automatic periodic saves, resumable.
4. **TensorBoard** — RLlib logs to TB natively; our custom metrics
   (milestones, heatmaps) are injected via callbacks.

vs SubprocVecEnv
~~~~~~~~~~~~~~~~
- SubprocVecEnv serialises *every observation* through multiprocessing
  pipes each step.  For 64 envs × (72×80×3 + event_bits + ...) this is
  a significant bottleneck.
- Ray workers own their env and only send *rollout batches* to the
  learner, dramatically reducing IPC.
- Async sampling: workers collect experience while the learner trains
  on the previous batch (AsyncSamplesOptimizer).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def env_creator(env_config: dict[str, Any]):
    """Factory function for Ray to create RedGymEnv instances.

    Ray calls this once per worker, passing the ``env_config`` dict
    from the algorithm config.  We layer on optional wrappers.
    """
    # Imports inside the function so workers don't need the full
    # module graph at import time (Ray serialises the function ref).
    from red_gym_env_v2 import RedGymEnv
    from config import EnvConfig

    # Separate wrapper flags from env config
    use_frame_stack = env_config.pop("use_frame_stack", False)
    frame_stack_n = env_config.pop("frame_stack_n", 4)
    use_macros = env_config.pop("use_action_macros", False)
    use_stream = env_config.pop("use_stream", False)
    stream_metadata = env_config.pop("stream_metadata", None)

    # Build base env
    cfg = EnvConfig.from_dict(env_config)
    env = RedGymEnv(cfg)

    # Optional wrappers
    if use_macros:
        from action_macros import MacroActionWrapper
        env = MacroActionWrapper(env)

    if use_frame_stack:
        from frame_stack import DictFrameStack
        env = DictFrameStack(
            env,
            n_stack=frame_stack_n,
            stack_keys=["screens", "health", "level"],
            passthrough_keys=["events", "map"],
        )

    if use_stream:
        from stream_agent_wrapper import StreamWrapper
        env = StreamWrapper(env, stream_metadata=stream_metadata or {})

    return env


# ---------------------------------------------------------------------------
# Custom metrics callback
# ---------------------------------------------------------------------------

def _make_callbacks_class():
    """Build the RLlib callbacks class.  Deferred to avoid import error
    if ray is not installed."""
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    class PokemonCallbacks(DefaultCallbacks):
        """Injects game-specific metrics into RLlib's TensorBoard logs."""

        def on_episode_end(self, *, episode, env_runner, base_env, **kwargs):
            # Pull metrics from the underlying RedGymEnv
            env = base_env.get_sub_environments()[0]

            # Unwrap to find the RedGymEnv
            unwrapped = env
            while hasattr(unwrapped, "env"):
                if hasattr(unwrapped, "milestone_tracker"):
                    break
                unwrapped = unwrapped.env

            if hasattr(unwrapped, "milestone_tracker"):
                ms = unwrapped.milestone_tracker.achieved_milestones()
                episode.custom_metrics["milestones_achieved"] = len(ms)
                for name, step in ms.items():
                    episode.custom_metrics[f"milestone_step/{name}"] = step

            if hasattr(unwrapped, "seen_coords"):
                episode.custom_metrics["unique_coords"] = len(unwrapped.seen_coords)

            if hasattr(unwrapped, "battles_won"):
                episode.custom_metrics["battles_won"] = unwrapped.battles_won

            if hasattr(unwrapped, "died_count"):
                episode.custom_metrics["deaths"] = unwrapped.died_count

            if hasattr(unwrapped, "game"):
                episode.custom_metrics["badge_count"] = unwrapped.game.badge_count
                episode.custom_metrics["party_size"] = unwrapped.game.party_size

    return PokemonCallbacks


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> dict:
    """Build the RLlib algorithm config dict from CLI args."""
    from ray.rllib.algorithms.ppo import PPOConfig

    ep_length = 2048 * 80

    env_config = {
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "init_state": args.init_state,
        "max_steps": ep_length,
        "print_rewards": False,  # too noisy with many workers
        "save_video": False,
        "fast_video": True,
        "session_path": args.session_path,
        "gb_path": args.gb_path,
        "reward_scale": args.reward_scale,
        "explore_weight": args.explore_weight,
        # Wrapper flags (popped by env_creator)
        "use_frame_stack": args.frame_stack > 0,
        "frame_stack_n": args.frame_stack,
        "use_action_macros": args.action_macros,
        "use_stream": args.stream,
    }

    train_batch = ep_length  # total frames per training iteration

    # Register custom model
    from ray.rllib.models import ModelCatalog
    from pokemon_model import make_rllib_model_class
    ModelCatalog.register_custom_model("pokemon_cnn", make_rllib_model_class())

    config = (
        PPOConfig()
        .environment(
            env="pokemon_red",
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length=ep_length // args.num_workers,
        )
        .training(
            train_batch_size=train_batch,
            sgd_minibatch_size=512,
            num_sgd_iter=1,
            gamma=0.997,
            entropy_coeff=0.01,
            lr=2.5e-4,
            clip_param=0.2,
            vf_clip_param=10.0,
            model={
                "custom_model": "pokemon_cnn",
                "custom_model_config": {
                    # Defaults are tuned for Pokemon Red, but can be
                    # overridden here for experimentation:
                    # "cnn_channels": [32, 64, 64],
                    # "cnn_kernels": [5, 3, 3],
                    # "cnn_strides": [2, 2, 1],
                    # "fc_sizes": [256, 256],
                },
            },
        )
        .framework("torch")
        .callbacks(_make_callbacks_class())
        .checkpointing(
            checkpoint_frequency=args.checkpoint_freq,
        )
    )

    if args.num_gpus > 0:
        config = config.learners(num_gpus_per_learner=args.num_gpus)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train Pokemon Red agent with Ray RLlib",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument("--gb-path", default="../PokemonRed.gb")
    parser.add_argument("--init-state", default="../init.state")
    parser.add_argument("--session-path", default="runs/ray")
    parser.add_argument("--reward-scale", type=float, default=0.5)
    parser.add_argument("--explore-weight", type=float, default=0.25)

    # Wrappers
    parser.add_argument(
        "--frame-stack", type=int, default=0,
        help="Number of frames to stack (0 = disabled)",
    )
    parser.add_argument("--action-macros", action="store_true")
    parser.add_argument("--stream", action="store_true")

    # Ray / RLlib
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--address", default=None, help="Ray cluster address")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=10000)

    args = parser.parse_args()

    # -- Ray init ---------------------------------------------------------
    import ray
    from ray import tune

    ray.init(address=args.address)

    # Register our custom environment
    tune.register_env("pokemon_red", env_creator)

    # -- Build and run ----------------------------------------------------
    config = build_config(args)

    Path(args.session_path).mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        from ray.rllib.algorithms.ppo import PPO
        algo = PPO(config=config)
        algo.restore(args.checkpoint)
        print(f"Resumed from: {args.checkpoint}")
    else:
        from ray.rllib.algorithms.ppo import PPO
        algo = PPO(config=config)

    print(f"Training with {args.num_workers} workers for {args.iterations} iterations")
    print(f"Frame stack: {args.frame_stack}, Action macros: {args.action_macros}")
    print(algo.get_policy().model)

    best_reward = float("-inf")
    for i in range(1, args.iterations + 1):
        result = algo.train()

        mean_reward = result.get("env_runners", {}).get("episode_reward_mean", 0)
        ep_len = result.get("env_runners", {}).get("episode_len_mean", 0)

        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint = algo.save(args.session_path)
            print(f"  [NEW BEST] saved: {checkpoint}")

        if i % 10 == 0 or i == 1:
            custom = result.get("env_runners", {}).get("custom_metrics", {})
            badges = custom.get("badge_count_mean", 0)
            coords = custom.get("unique_coords_mean", 0)
            battles = custom.get("battles_won_mean", 0)
            milestones = custom.get("milestones_achieved_mean", 0)

            print(
                f"Iter {i:5d} | "
                f"reward: {mean_reward:8.1f} | "
                f"ep_len: {ep_len:6.0f} | "
                f"badges: {badges:.1f} | "
                f"coords: {coords:.0f} | "
                f"battles: {battles:.0f} | "
                f"milestones: {milestones:.1f}"
            )

    algo.save(args.session_path)
    print(f"Training complete.  Final checkpoint saved to {args.session_path}")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

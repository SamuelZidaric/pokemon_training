from __future__ import annotations

import os
import json
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce

from global_map import GLOBAL_MAP_SHAPE


def merge_dicts(dicts: list[dict]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    sum_dict: dict[str, float] = {}
    count_dict: dict[str, int] = {}
    distrib_dict: dict[str, list] = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict


class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir: str | os.PathLike, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.log_dir, "histogram")
            )

    def _on_step(self) -> bool:
        if not self.training_env.env_method("check_if_done", indices=[0])[0]:
            return True

        all_infos = self.training_env.get_attr("agent_stats")
        all_final_infos = [stats[-1] for stats in all_infos]
        mean_infos, distributions = merge_dicts(all_final_infos)

        # -- standard metrics --
        for key, val in mean_infos.items():
            self.logger.record(f"env_stats/{key}", val)

        for key, distrib in distributions.items():
            self.writer.add_histogram(
                f"env_stats_distribs/{key}", distrib, self.n_calls
            )
            self.logger.record(f"env_stats_max/{key}", max(distrib))

        # -- exploration maps --
        explore_map = np.array(self.training_env.get_attr("explore_map"))
        map_sum = reduce(explore_map, "f h w -> h w", "max")
        self.logger.record(
            "trajectory/explore_sum",
            Image(map_sum, "HW"),
            exclude=("stdout", "log", "json", "csv"),
        )

        n_envs = explore_map.shape[0]
        if n_envs >= 2:
            # Arrange into a grid with 2 rows
            n_rows = min(2, n_envs)
            # Pad to even number if needed
            if n_envs % n_rows != 0:
                pad_n = n_rows - (n_envs % n_rows)
                pad = np.zeros((pad_n, *explore_map.shape[1:]), dtype=explore_map.dtype)
                explore_map_padded = np.concatenate([explore_map, pad], axis=0)
            else:
                explore_map_padded = explore_map
            map_row = rearrange(explore_map_padded, "(r f) h w -> (r h) (f w)", r=n_rows)
        else:
            map_row = explore_map[0]
        self.logger.record(
            "trajectory/explore_map",
            Image(map_row, "HW"),
            exclude=("stdout", "log", "json", "csv"),
        )

        # -- coordinate heatmap --
        self._log_coord_heatmap(all_infos)

        # -- event flags --
        list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
        merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
        self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

        # -- milestones --
        self._log_milestones()

        return True

    def _log_coord_heatmap(self, all_infos: list[list[dict]]) -> None:
        """Build a heatmap of visited coordinates across all envs."""
        heatmap = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)

        try:
            from global_map import local_to_global

            for env_stats in all_infos:
                for stat in env_stats:
                    x, y, m = stat.get("x", 0), stat.get("y", 0), stat.get("map", 0)
                    try:
                        gy, gx = local_to_global(y, x, m)
                        if 0 <= gy < heatmap.shape[0] and 0 <= gx < heatmap.shape[1]:
                            heatmap[gy, gx] += 1
                    except (KeyError, IndexError):
                        continue

            if heatmap.max() > 0:
                heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

            self.logger.record(
                "trajectory/coord_heatmap",
                Image(heatmap, "HW"),
                exclude=("stdout", "log", "json", "csv"),
            )

            # also log as histogram to writer for finer-grained analysis
            self.writer.add_histogram(
                "env_stats_distribs/coord_density",
                heatmap[heatmap > 0],
                self.n_calls,
            )
        except Exception:
            pass  # don't crash training if heatmap fails

    def _log_milestones(self) -> None:
        """Log milestone data if the env has a milestone tracker."""
        try:
            milestone_summaries = self.training_env.get_attr("milestone_summary")
        except AttributeError:
            return

        # Aggregate across envs: for each milestone, record the
        # fraction of envs that achieved it and the mean step
        all_names: set[str] = set()
        for summary in milestone_summaries:
            all_names.update(summary.keys())

        for name in sorted(all_names):
            steps = [
                s[name] for s in milestone_summaries
                if s.get(name) is not None
            ]
            achieved_frac = len(steps) / len(milestone_summaries)
            self.logger.record(f"milestones/frac_{name}", achieved_frac)
            if steps:
                self.logger.record(f"milestones/mean_step_{name}", np.mean(steps))
                self.logger.record(f"milestones/min_step_{name}", min(steps))

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()

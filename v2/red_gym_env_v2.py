from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE
from game_state import (
    GameState,
    EVENT_FLAGS_START,
    EVENT_FLAGS_END,
    MUSEUM_TICKET_ADDR,
    MUSEUM_TICKET_BIT,
    TACTICAL_OBS_SIZE,
)
from config import EnvConfig
from rewards import (
    RewardContext,
    RewardSystem,
    create_default_reward_system,
)
from milestones import MilestoneTracker


class RedGymEnv(Env):
    """Pokemon Red gymnasium environment (v2).

    Accepts either the legacy ``dict`` config or the new ``EnvConfig``
    dataclass — so existing training scripts keep working unchanged.
    """

    def __init__(self, config: dict[str, Any] | EnvConfig | None = None) -> None:
        # ---- normalise config ------------------------------------------
        if isinstance(config, dict):
            self.cfg = EnvConfig.from_dict(config)
        elif isinstance(config, EnvConfig):
            self.cfg = config
        else:
            self.cfg = EnvConfig()

        self.s_path = self.cfg.session_path
        self.save_final_state = self.cfg.save_final_state
        self.print_rewards = self.cfg.print_rewards
        self.headless = self.cfg.headless
        self.init_state = self.cfg.init_state
        self.act_freq = self.cfg.action_freq
        self.max_steps = self.cfg.max_steps
        self.save_video = self.cfg.save_video
        self.fast_video = self.cfg.fast_video
        self.explore_weight = self.cfg.explore_weight
        self.reward_scale = self.cfg.reward_scale
        self.instance_id = self.cfg.instance_id

        self.frame_stacks = 3
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer: Optional[media.VideoWriter] = None
        self.model_frame_writer: Optional[media.VideoWriter] = None
        self.map_frame_writer: Optional[media.VideoWriter] = None
        self.reset_count = 0
        self.all_runs: list[dict] = []

        self.essential_map_locations: dict[int, int] = {
            v: i
            for i, v in enumerate(
                [40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65]
            )
        }

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        # event names from pokered source
        with open(Path(__file__).parent / "events.json") as f:
            self.event_names: dict[str, str] = json.load(f)

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12
        self.enc_freqs = 8

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(
                    low=0, high=255, shape=self.output_shape, dtype=np.uint8
                ),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary(
                    (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8
                ),
                "map": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                    dtype=np.uint8,
                ),
                "recent_actions": spaces.MultiDiscrete(
                    [len(self.valid_actions)] * self.frame_stacks
                ),
                "tactical": spaces.Box(
                    low=-1, high=1,
                    shape=(TACTICAL_OBS_SIZE,), dtype=np.float32
                ),
            }
        )

        head = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(self.cfg.gb_path, window=head, sound_emulated=False)
        if not self.headless:
            self.pyboy.set_emulation_speed(6)

        # ---- game state reader & reward system -------------------------
        self.game = GameState(self.pyboy)
        self.reward_system = create_default_reward_system(
            reward_scale=self.reward_scale,
            explore_weight=self.explore_weight,
        )
        self.milestone_tracker = MilestoneTracker()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        self.seed = seed
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats: list[dict] = []
        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1.0
        self.total_healing_rew = 0.0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.base_event_flags = self.game.event_flag_sum
        self.current_event_flags_set: dict[str, str] = {}
        self.max_map_progress = 0

        # battle tracking
        self.battles_won = 0
        self.battles_won_gated = 0   # only counts battles vs opponents near our level
        self.trainer_battles_won = 0
        self.prev_in_battle = False
        self.prev_opponent_hp = 1.0
        self.prev_battle_type = 0
        self.prev_opponent_level = 0

        # catching / evolution tracking
        self.starting_species = self.game.party_species
        self.seen_species: set[int] = set(s for s in self.starting_species if s != 0)
        self.new_species_caught = 0
        self.species_changed_count = 0

        # milestones
        self.milestone_tracker.reset()

        self.progress_reward = self._compute_rewards()
        self.total_reward = sum(self.progress_reward.values())
        self.reset_count += 1
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_heal_reward()

        self.party_size = self.game.party_size

        self._track_battles()
        self._track_species()
        new_reward = self.update_reward()
        self.last_health = self.game.hp_fraction
        self.update_map_progress()

        # milestone tracking
        self.milestone_tracker.update(self.step_count, self.game)

        step_limit_reached = self.check_if_done()
        obs = self._get_obs()

        if self.step_count % 100 == 0:
            self._update_event_flag_names()

        self.step_count += 1
        return obs, new_reward, False, step_limit_reached, {}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def render(self, reduce_res: bool = True) -> np.ndarray:
        game_pixels = self.pyboy.screen.ndarray[:, :, 0:1]
        if reduce_res:
            game_pixels = downscale_local_mean(game_pixels, (2, 2, 1)).astype(
                np.uint8
            )
        return game_pixels

    def _get_obs(self) -> dict[str, np.ndarray]:
        screen = self.render()
        self.update_recent_screens(screen)

        level_sum = 0.02 * self.game.levels_sum

        return {
            "screens": self.recent_screens,
            "health": np.array([self.game.hp_fraction]),
            "level": self.fourier_encode(level_sum),
            "badges": self.game.badges_array,
            "events": np.array(self.game.event_bits, dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
            "tactical": self.game.tactical_obs(),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def run_action_on_emulator(self, action: int) -> None:
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()

    # ------------------------------------------------------------------
    # Coordinates & exploration
    # ------------------------------------------------------------------

    def init_map_mem(self) -> None:
        self.seen_coords: dict[str, int] = {}

    def get_game_coords(self) -> tuple[int, int, int]:
        pos = self.game.position
        return pos.x, pos.y, pos.map_id

    def update_seen_coords(self) -> None:
        if not self.game.in_battle:
            x, y, m = self.get_game_coords()
            key = f"x:{x} y:{y} m:{m}"
            self.seen_coords[key] = self.seen_coords.get(key, 0) + 1

    def get_current_coord_count_reward(self) -> int:
        x, y, m = self.get_game_coords()
        key = f"x:{x} y:{y} m:{m}"
        return 0 if self.seen_coords.get(key, 0) < 600 else 1

    def get_global_coords(self) -> tuple[int, int]:
        x, y, m = self.get_game_coords()
        return local_to_global(y, x, m)

    def update_explore_map(self) -> None:
        r, c = self.get_global_coords()
        if r < self.explore_map.shape[0] and c < self.explore_map.shape[1]:
            self.explore_map[r, c] = 255
        else:
            print(f"coord out of bounds! global: ({r},{c}) game: {self.get_game_coords()}")

    def get_explore_map(self) -> np.ndarray:
        r, c = self.get_global_coords()
        if r >= self.explore_map.shape[0] or c >= self.explore_map.shape[1]:
            out = np.zeros(
                (self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8
            )
        else:
            out = self.explore_map[
                r - self.coords_pad : r + self.coords_pad,
                c - self.coords_pad : c + self.coords_pad,
            ]
        return repeat(out, "h w -> (h h2) (w w2)", h2=2, w2=2)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def update_reward(self) -> float:
        self.progress_reward = self._compute_rewards()
        new_total = sum(self.progress_reward.values())
        step_reward = new_total - self.total_reward
        self.total_reward = new_total
        return step_reward

    def _compute_rewards(self) -> dict[str, float]:
        """Build a RewardContext and run the reward system."""
        ctx = RewardContext(
            seen_coords_count=len(self.seen_coords),
            current_coord_visits=self._current_coord_visits(),
            event_flag_sum=self.game.event_flag_sum,
            base_event_flags=self.base_event_flags,
            has_museum_ticket=self.game.has_museum_ticket,
            hp_fraction=self.game.hp_fraction,
            last_hp_fraction=self.last_health,
            party_size=self.game.party_size,
            last_party_size=self.party_size,
            levels_sum=self.game.levels_sum,
            badge_count=self.game.badge_count,
            total_healing_reward=self.total_healing_rew,
            died_count=self.died_count,
            max_opponent_level=self.max_opponent_level,
            in_battle=self.game.in_battle,
            opponent_hp_fraction=self.game.opponent_hp_fraction if self.game.in_battle else 1.0,
            prev_opponent_hp_fraction=self.prev_opponent_hp,
            battles_won=self.battles_won_gated,
            is_box_full=self.game.is_box_full,
            pokemon_in_box=self.game.pokemon_in_current_box,
            type_advantage=self.game.type_advantage_signal,
            hp_loss_this_step=max(self.last_health - self.game.hp_fraction, 0.0),
            party_fainted_count=self.game.party_fainted_count,
            # Oak's Parcel quest chain
            has_oaks_parcel=self.game.has_oaks_parcel,
            delivered_oaks_parcel=self.game.delivered_oaks_parcel,
            has_pokedex=self.game.has_pokedex,
            has_oaks_pokeballs=self.game.has_oaks_pokeballs,
            # Catching / evolution
            species_changed_count=self.species_changed_count,
            new_species_caught=self.new_species_caught,
            # Level-gating: use gated counter instead of raw battles_won
            lead_level=self.game.party_levels[0] if self.game.party_size > 0 else 1,
            opponent_level=self.game.opponent_level if self.game.in_battle else 0,
            # Linear map progression index (0=Oak's lab … 14=Cerulean gym)
            max_map_progress=self.max_map_progress,
        )
        scores = self.reward_system.compute(ctx)
        # preserve max-ever semantics for event reward
        scores["event"] = max(scores.get("event", 0), self.max_event_rew * self.reward_scale * 4)
        self.max_event_rew = scores["event"] / (self.reward_scale * 4) if self.reward_scale else 0
        return scores

    def _current_coord_visits(self) -> int:
        x, y, m = self.get_game_coords()
        return self.seen_coords.get(f"x:{x} y:{y} m:{m}", 0)

    def get_game_state_reward(self, print_stats: bool = False) -> dict[str, float]:
        """Legacy interface — delegates to the reward system."""
        return self._compute_rewards()

    def _track_battles(self) -> None:
        """Track battle wins for reward computation.

        Gated counter: only rewards wins vs opponents within ~2 levels of
        the lead Pokemon, or any trainer battle. This kills the Route 1
        grinding loop (mashing weak Rattatas gives no reward).
        """
        in_battle = self.game.in_battle
        if self.prev_in_battle and not in_battle and self.prev_opponent_hp <= 0:
            self.battles_won += 1
            # Evaluate level-gated reward using previous battle state
            lead_level = self.game.party_levels[0] if self.game.party_size > 0 else 1
            if self.prev_battle_type == 2:
                # Trainer battle — always rewarded (tied to story)
                self.battles_won_gated += 1
                self.trainer_battles_won += 1
            elif self.prev_opponent_level >= max(lead_level - 2, 1):
                # Wild battle against a non-pushover — rewarded
                self.battles_won_gated += 1
            # else: wild vs weak opponent → no reward (grinding filter)
        # Record battle context while still in battle, for post-battle evaluation
        if in_battle:
            self.prev_battle_type = self.game.battle_type
            self.prev_opponent_level = self.game.opponent_level
        self.prev_in_battle = in_battle
        self.prev_opponent_hp = (
            self.game.opponent_hp_fraction if in_battle else 1.0
        )

    def _track_species(self) -> None:
        """Track new species caught and evolution events this episode."""
        current = self.game.party_species
        for i, sp in enumerate(current):
            if sp == 0:
                continue
            if sp not in self.seen_species:
                # Either a catch (new party slot) or an evolution (known slot, new species)
                prev_at_slot = self.starting_species[i] if i < len(self.starting_species) else 0
                if prev_at_slot != 0 and prev_at_slot != sp:
                    # Slot previously had a different species → evolution
                    self.species_changed_count += 1
                else:
                    self.new_species_caught += 1
                self.seen_species.add(sp)
        self.starting_species = current

    @property
    def milestone_summary(self) -> dict[str, int | None]:
        """Expose milestone data for TensorBoard callback."""
        return self.milestone_tracker.summary()

    def update_heal_reward(self) -> None:
        cur_health = self.game.hp_fraction
        if cur_health > self.last_health and self.game.party_size == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def group_rewards(self) -> tuple[float, float, float]:
        prog = self.progress_reward
        return (
            prog.get("level", 0) * 100 / self.reward_scale,
            self.game.hp_fraction * 2000,
            prog.get("explore", 0) * 150 / (self.explore_weight * self.reward_scale),
        )

    # ------------------------------------------------------------------
    # Game progress helpers
    # ------------------------------------------------------------------

    def get_badges(self) -> int:
        return self.game.badge_count

    def get_levels_sum(self) -> int:
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [max(lv - min_poke_level, 0) for lv in self.game.party_levels]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def read_party(self) -> list[int]:
        return self.game.party_species

    def read_hp_fraction(self) -> float:
        return self.game.hp_fraction

    def update_max_op_level(self) -> int:
        opp_base_level = 5
        opponent_level = self.game.max_opponent_level - opp_base_level
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_map_progress(self) -> None:
        map_idx = self.game.map_id
        self.max_map_progress = max(
            self.max_map_progress, self.get_map_progress(map_idx)
        )

    def get_map_progress(self, map_idx: int) -> int:
        return self.essential_map_locations.get(map_idx, -1)

    # ------------------------------------------------------------------
    # Low-level memory access (kept for backward compat)
    # ------------------------------------------------------------------

    def read_m(self, addr: int) -> int:
        return self.game.read(addr)

    def read_bit(self, addr: int, bit: int) -> bool:
        return self.game.read_bit(addr, bit)

    def read_event_bits(self) -> list[int]:
        return self.game.event_bits

    def read_hp(self, start: int) -> int:
        return self.game.read_word(start)

    @staticmethod
    def bit_count(bits: int) -> int:
        return bin(bits).count("1")

    def fourier_encode(self, val: float) -> np.ndarray:
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def check_if_done(self) -> bool:
        return self.step_count >= self.max_steps - 1

    def update_recent_screens(self, cur_screen: np.ndarray) -> None:
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def update_recent_actions(self, action: int) -> None:
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def _update_event_flag_names(self) -> None:
        for address in range(EVENT_FLAGS_START, EVENT_FLAGS_END):
            val = self.game.read(address)
            for idx, bit in enumerate(f"{val:08b}"):
                if bit == "1":
                    key = f"0x{address:X}-{idx}"
                    if key in self.event_names:
                        self.current_event_flags_set[key] = self.event_names[key]

    def append_agent_stats(self, action: int) -> None:
        x, y, m = self.get_game_coords()
        levels = self.game.party_levels
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x,
                "y": y,
                "map": m,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.game.party_size,
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.game.party_species,
                "hp": self.game.hp_fraction,
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.game.badge_count,
                "event": self.progress_reward.get("event", 0),
                "healr": self.total_healing_rew,
                "battles_won": self.battles_won,
                "milestones": self.milestone_tracker.achievement_count(),
            }
        )

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_video(self) -> None:
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / "rollouts"
        base_dir.mkdir(exist_ok=True)

        full_name = f"full_reset_{self.reset_count}_id{self.instance_id}.mp4"
        model_name = f"model_reset_{self.reset_count}_id{self.instance_id}.mp4"
        map_name = f"map_reset_{self.reset_count}_id{self.instance_id}.mp4"

        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self) -> None:
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])
        self.map_frame_writer.add_image(self.get_explore_map())

    def save_and_print_info(self, done: bool, obs: dict) -> None:
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / f"curframe_{self.instance_id}.jpeg",
                self.render(reduce_res=False)[:, :, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / "final_states"
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg",
                    obs["map"][:, :, 0],
                )
                plt.imsave(
                    fs_path / f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg",
                    self.explore_map,
                )
                plt.imsave(
                    fs_path / f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg",
                    self.render(reduce_res=False)[:, :, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

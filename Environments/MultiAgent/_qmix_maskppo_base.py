import os
import random
from dataclasses import dataclass

import numpy as np

from absl import flags

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _install_pysc2_shuffle_compat():
    try:
        random.shuffle([], lambda: 0.5)
        return
    except TypeError:
        pass

    def _compat_shuffle(seq, rand=None):
        if rand is None:
            return random._inst.shuffle(seq)

        for idx in range(len(seq) - 1, 0, -1):
            swap_idx = int(rand() * (idx + 1))
            seq[idx], seq[swap_idx] = seq[swap_idx], seq[idx]

    random.shuffle = _compat_shuffle


_install_pysc2_shuffle_compat()

from pysc2.env import sc2_env
from pysc2.lib import actions, features
from pysc2.maps import lib


N_FRIEND = 5

FRIEND_STRIDE = 4
ENEMY_STRIDE = 4

OBS_FRIEND_STRIDE = 2
OBS_ENEMY_STRIDE = 2

FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([""])


RAW = actions.RAW_FUNCTIONS
BEACON_TYPE_ID = 317
BEACON_RADIUS = 5.0

STEP_MUL = 8
FIVE_MIN_LOOPS = 5 * 60 * 16
DEFAULT_EPISODE_LIMIT = FIVE_MIN_LOOPS // STEP_MUL
STEP_PIX = 2
SCR_RES = 64
DEFAULT_ATTACK_RANGE = 6.0

MINIMAP_PATHABLE_INDEX = int(features.MINIMAP_FEATURES.pathable.index)
MINIMAP_PLAYER_RELATIVE_INDEX = int(features.MINIMAP_FEATURES.player_relative.index)
MINI_CH = 2

MOVE_DIRS = [
    (0, -STEP_PIX),
    (0, STEP_PIX),
    (-STEP_PIX, 0),
    (STEP_PIX, 0),
    (STEP_PIX, -STEP_PIX),
    (-STEP_PIX, -STEP_PIX),
    (STEP_PIX, STEP_PIX),
    (-STEP_PIX, STEP_PIX),
]

N_MOVE_ACTIONS = len(MOVE_DIRS)

KILL_BONUS = 1.0
HP_SCALE = 0.05

NAV_WIN_BONUS = 25.0
COMBAT_WIN_BONUS = 10.0
COMBAT_LOSS_PENALTY = -10.0
NAV_TIMEOUT_PENALTY = -15.0
TIE_BONUS = 0.0


@dataclass(frozen=True)
class TwoBridgeMapConfig:
    alias: str
    registry_name: str
    filename: str
    directory: str
    n_enemies: int
    players: int = 2


def register_two_bridge_map(config: TwoBridgeMapConfig):
    map_cls = type(
        config.registry_name,
        (lib.Map,),
        {
            "name": config.registry_name,
            "directory": config.directory,
            "filename": config.filename,
            "players": config.players,
        },
    )
    lib.get_maps().pop(config.registry_name, None)
    lib.get_maps()[config.registry_name] = map_cls()


class TwoBridgeQMixMaskPPOEnvBase:
    """
    QMIX environment aligned with the NS MaskPPO setup.

    Each agent controls one friendly unit with the same per-unit action
    semantics used by the NS single-agent envs. Agent observations are the
    same compact actor vector broadcast to each unit, while the shared minimap
    is exposed separately for centralized visual encoding.
    """

    def __init__(
        self,
        *,
        map_config: TwoBridgeMapConfig,
        map_name: str,
        seed=None,
        episode_limit=None,
        visualize=False,
        realtime=False,
        replay_dir="",
        save_replay_episodes=0,
        attack_range: float = DEFAULT_ATTACK_RANGE,
    ):
        if map_name != map_config.alias:
            raise ValueError(
                f"Unsupported map_name={map_name!r}. Only {map_config.alias!r} is implemented."
            )

        self.map_config = map_config
        self.map_name = map_name
        self.n_agents = N_FRIEND
        self.n_enemies = int(map_config.n_enemies)
        self.n_actions = 1 + N_MOVE_ACTIONS + self.n_enemies
        self.attack_action_offset = 1 + N_MOVE_ACTIONS
        self.attack_range = float(attack_range)
        self.episode_limit = (
            DEFAULT_EPISODE_LIMIT if episode_limit is None else int(episode_limit)
        )
        if self.episode_limit < 1:
            raise ValueError("episode_limit must be >= 1")
        if self.attack_range <= 0.0:
            raise ValueError("attack_range must be > 0")

        self.vec_enemy = self.n_agents * FRIEND_STRIDE
        self.vec_bxy = self.vec_enemy + self.n_enemies * ENEMY_STRIDE
        self.vec_dist = self.vec_bxy + 2
        self.vec_time = self.vec_dist + 1
        self.vec_ecount = self.vec_time + 1
        self.vec_size = self.vec_ecount + 1

        self.obs_friend = 0
        self.obs_enemy = self.obs_friend + self.n_agents * OBS_FRIEND_STRIDE
        self.obs_time = self.obs_enemy + self.n_enemies * OBS_ENEMY_STRIDE
        self.obs_ecount = self.obs_time + 1
        self.obs_size = self.obs_ecount + 1

        register_two_bridge_map(map_config)

        self._seed = None
        self._env = sc2_env.SC2Env(
            map_name=self.map_config.registry_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy),
            ],
            step_mul=STEP_MUL,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=SCR_RES,
                raw_crop_to_playable_area=True,
                feature_dimensions=features.Dimensions(screen=SCR_RES, minimap=SCR_RES),
            ),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes,
        )

        self._my_tags = np.zeros(self.n_agents, np.int64)
        self._friend_alive = np.zeros(self.n_agents, dtype=bool)
        self._enemy_tags = np.zeros(self.n_enemies, np.int64)
        self._enemy_alive = np.zeros(self.n_enemies, dtype=bool)
        self._friend_enemy_attackable = np.zeros(
            (self.n_agents, self.n_enemies), dtype=bool
        )
        self._fx = np.zeros(self.n_agents, np.float32)
        self._fy = np.zeros(self.n_agents, np.float32)
        self._raw_x_max = float(SCR_RES - 1)
        self._raw_y_max = float(SCR_RES - 1)

        self._step_ctr = 0
        self._prev_beacon_dists = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp = np.zeros(self.n_enemies, np.float32)
        self._prev_friend_hp = np.zeros(self.n_agents, np.float32)

        self._state = np.zeros(self.vec_size, np.float32)
        self._obs = np.zeros((self.n_agents, self.obs_size), np.float32)
        self._avail_actions = np.zeros((self.n_agents, self.n_actions), np.int8)
        self._minimap = np.zeros((MINI_CH, SCR_RES, SCR_RES), np.uint8)
        self._last_reward_components = {
            "nav_r": 0.0,
            "combat_r": 0.0,
            "term_r": 0.0,
            "friend_hp": 0.0,
            "enemy_hp": 0.0,
            "nav_dist": 0.0,
            "combat_dist": 0.0,
        }

        if seed is not None:
            self.seed(seed)
        self._refresh_raw_bounds()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self._step_ctr = 0
        self._prev_beacon_dists = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp.fill(0.0)
        self._prev_friend_hp.fill(0.0)
        self._my_tags.fill(0)
        self._friend_alive.fill(False)
        self._enemy_tags.fill(0)
        self._enemy_alive.fill(False)
        self._friend_enemy_attackable.fill(False)
        self._fx.fill(0.0)
        self._fy.fill(0.0)
        self._last_reward_components = {
            "nav_r": 0.0,
            "combat_r": 0.0,
            "term_r": 0.0,
            "friend_hp": 0.0,
            "enemy_hp": 0.0,
            "nav_dist": 0.0,
            "combat_dist": 0.0,
        }

        ts = self._env.reset()[0]
        self._refresh_raw_bounds()
        self._update_observations(ts)
        return self.get_obs(), {}

    def step(self, actions_):
        actions_arr = np.asarray(actions_, dtype=np.int64).reshape(-1)
        if actions_arr.shape[0] != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} actions, received {actions_arr.shape[0]}."
            )

        cmds = self._translate_actions(actions_arr)
        ts = self._env.step(cmds)[0]
        self._step_ctr += 1
        self._update_observations(ts)

        if ts.last():
            result = "victory" if ts.reward > 0 else "defeat" if ts.reward < 0 else "tie"
            info = self._build_result_info(result)
            return self.get_obs(), float(ts.reward), True, False, info

        result = None
        friend_alive = (self._state[2 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE] > 0).sum()
        no_friend = friend_alive == 0
        no_enemy = self._state[self.vec_ecount] == 0
        beacon_win = self._state[self.vec_dist] < BEACON_RADIUS

        if beacon_win:
            result = "nav_win"
        elif no_enemy and no_friend:
            result = "tie"
        elif no_enemy:
            result = "combat_win"
        elif no_friend:
            result = "combat_loss"
        elif self._step_ctr >= self.episode_limit:
            result = "timeout_loss"

        terminated = result is not None
        reward = self._shape_reward(self._state, terminated, result)
        info = self._build_result_info(result)
        info["rew"] = dict(self._last_reward_components)
        return self.get_obs(), reward, terminated, False, info

    def get_obs(self):
        return self._obs.copy()

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id].copy()

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        return self._state.copy()

    def get_state_size(self):
        return self.vec_size

    def get_minimap(self):
        return self._minimap.copy()

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.get_obs_size(),
            "state_shape": self.get_state_size(),
            "minimap_shape": self._minimap.shape,
            "episode_limit": self.episode_limit,
        }

    def get_avail_actions(self):
        return self._avail_actions.copy()

    def get_avail_agent_actions(self, agent_id):
        return self._avail_actions[agent_id].copy()

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self):
        return None

    def seed(self, seed=None):
        if seed is None:
            return self._seed
        self._seed = int(seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        return self._seed

    def save_replay(self):
        if self._env is not None and hasattr(self._env, "save_replay"):
            self._env.save_replay("two_bridge_qmix")

    def get_stats(self):
        return {}

    def get_reward_components(self):
        return dict(self._last_reward_components)

    def _build_result_info(self, result):
        if result is None:
            return {"result": None, "episode_limit": False}

        won = result in {"nav_win", "combat_win", "victory"}
        return {
            "result": result,
            "battle_won": int(won),
            "nav_win": int(result == "nav_win"),
            "combat_win": int(result in {"combat_win", "victory"}),
            "combat_loss": int(result in {"combat_loss", "defeat"}),
            "tie": int(result == "tie"),
            "episode_limit": result == "timeout_loss",
        }

    def _refresh_raw_bounds(self):
        game_info = self._env.game_info[0].start_raw.map_size
        self._raw_x_max = max(0.0, float(game_info.x) - 1.0)
        self._raw_y_max = max(0.0, float(game_info.y) - 1.0)

    def _populate_slot_tags(self, units, slot_tags):
        assigned_tags = {int(tag) for tag in slot_tags if int(tag) != 0}
        next_free = 0
        for unit in units:
            tag = int(unit.tag)
            if tag in assigned_tags:
                continue
            while next_free < len(slot_tags) and int(slot_tags[next_free]) != 0:
                next_free += 1
            if next_free >= len(slot_tags):
                break
            slot_tags[next_free] = tag
            assigned_tags.add(tag)
            next_free += 1

    def _translate_actions(self, actions_arr):
        cmds = []

        for agent_id, action_id in enumerate(actions_arr):
            if not self._friend_alive[agent_id] or self._my_tags[agent_id] == 0:
                continue

            tag = int(self._my_tags[agent_id])
            x = float(self._fx[agent_id])
            y = float(self._fy[agent_id])

            if action_id == 0:
                continue

            if 1 <= action_id <= N_MOVE_ACTIONS:
                dx, dy = MOVE_DIRS[action_id - 1]
                target = (
                    float(np.clip(x + dx, 0.0, self._raw_x_max)),
                    float(np.clip(y + dy, 0.0, self._raw_y_max)),
                )
                cmds.append(RAW.Move_pt("now", [tag], target))
                continue

            enemy_slot = int(action_id) - self.attack_action_offset
            if (
                0 <= enemy_slot < self.n_enemies
                and self._enemy_alive[enemy_slot]
                and self._friend_enemy_attackable[agent_id, enemy_slot]
            ):
                cmds.append(RAW.Attack_unit("now", [tag], int(self._enemy_tags[enemy_slot])))

        if not cmds:
            cmds.append(RAW.no_op())
        return cmds

    def _update_observations(self, ts):
        observation = ts.observation
        raw_units = observation.raw_units
        feature_minimap = np.asarray(observation.feature_minimap, np.uint8)
        self._minimap = feature_minimap[
            [MINIMAP_PATHABLE_INDEX, MINIMAP_PLAYER_RELATIVE_INDEX]
        ]

        friends = sorted((u for u in raw_units if u.owner == 1), key=lambda unit: unit.tag)
        enemies = sorted((u for u in raw_units if u.owner == 2), key=lambda unit: unit.tag)
        beacon = next((u for u in raw_units if u.unit_type == BEACON_TYPE_ID), None)

        self._populate_slot_tags(friends[: self.n_agents], self._my_tags)
        self._populate_slot_tags(enemies[: self.n_enemies], self._enemy_tags)

        self._friend_alive.fill(False)
        self._enemy_alive.fill(False)
        self._friend_enemy_attackable.fill(False)
        self._fx.fill(0.0)
        self._fy.fill(0.0)

        friend_by_tag = {int(unit.tag): unit for unit in friends}
        enemy_by_tag = {int(unit.tag): unit for unit in enemies}

        state = np.zeros(self.vec_size, np.float32)

        for agent_id, tag in enumerate(self._my_tags):
            if int(tag) == 0:
                continue
            unit = friend_by_tag.get(int(tag))
            if unit is None:
                continue
            alive = unit.health > 0
            self._friend_alive[agent_id] = alive
            self._fx[agent_id] = unit.x
            self._fy[agent_id] = unit.y
            start = agent_id * FRIEND_STRIDE
            state[start : start + FRIEND_STRIDE] = (
                unit.x,
                unit.y,
                unit.health,
                float(alive),
            )

        for enemy_id, tag in enumerate(self._enemy_tags):
            if int(tag) == 0:
                continue
            unit = enemy_by_tag.get(int(tag))
            if unit is None:
                continue
            alive = unit.health > 0
            self._enemy_alive[enemy_id] = alive
            start = self.vec_enemy + enemy_id * ENEMY_STRIDE
            state[start : start + ENEMY_STRIDE] = (
                unit.x,
                unit.y,
                unit.health,
                float(alive),
            )

        beacon_x, beacon_y = (beacon.x, beacon.y) if beacon is not None else (-1.0, -1.0)
        state[self.vec_bxy : self.vec_bxy + 2] = (beacon_x, beacon_y)

        friend_x = state[0 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_y = state[1 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_hp = state[2 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_alive = friend_hp > 0

        if friend_alive.any() and beacon_x >= 0 and beacon_y >= 0:
            state[self.vec_dist] = float(
                np.hypot(
                    friend_x[friend_alive] - beacon_x,
                    friend_y[friend_alive] - beacon_y,
                ).min()
            )
        else:
            state[self.vec_dist] = 128.0

        enemy_x = state[
            self.vec_enemy : self.vec_enemy + self.n_enemies * ENEMY_STRIDE : ENEMY_STRIDE
        ]
        enemy_y = state[
            self.vec_enemy + 1 : self.vec_enemy + self.n_enemies * ENEMY_STRIDE : ENEMY_STRIDE
        ]
        for agent_id in range(self.n_agents):
            if not self._friend_alive[agent_id]:
                continue
            for enemy_id in range(self.n_enemies):
                if not self._enemy_alive[enemy_id]:
                    continue
                dist = np.hypot(
                    self._fx[agent_id] - enemy_x[enemy_id],
                    self._fy[agent_id] - enemy_y[enemy_id],
                )
                self._friend_enemy_attackable[agent_id, enemy_id] = dist <= self.attack_range

        state[self.vec_time] = observation.game_loop[0] / 16.0
        state[self.vec_ecount] = float(self._enemy_alive.sum())

        actor_vec = np.zeros(self.obs_size, np.float32)
        for agent_id in range(self.n_agents):
            src = agent_id * FRIEND_STRIDE
            dst = self.obs_friend + agent_id * OBS_FRIEND_STRIDE
            actor_vec[dst : dst + OBS_FRIEND_STRIDE] = (
                state[src + 2],
                state[src + 3],
            )

        for enemy_id in range(self.n_enemies):
            src = self.vec_enemy + enemy_id * ENEMY_STRIDE
            dst = self.obs_enemy + enemy_id * OBS_ENEMY_STRIDE
            actor_vec[dst : dst + OBS_ENEMY_STRIDE] = (
                state[src + 2],
                state[src + 3],
            )

        actor_vec[self.obs_time] = state[self.vec_time]
        actor_vec[self.obs_ecount] = state[self.vec_ecount]

        avail_actions = np.zeros((self.n_agents, self.n_actions), np.int8)
        avail_actions[:, 0] = 1
        for agent_id in range(self.n_agents):
            if not self._friend_alive[agent_id]:
                continue
            avail_actions[agent_id, 1 : 1 + N_MOVE_ACTIONS] = 1
            avail_actions[
                agent_id,
                self.attack_action_offset : self.attack_action_offset + self.n_enemies,
            ] = self._friend_enemy_attackable[agent_id].astype(np.int8)

        self._state = state
        self._obs = np.repeat(actor_vec[None, :], self.n_agents, axis=0)
        self._avail_actions = avail_actions

    def _shape_reward(self, vec, done, result):
        friend_x = vec[0 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_y = vec[1 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_hp = vec[2 : self.n_agents * FRIEND_STRIDE : FRIEND_STRIDE]

        enemy_x = vec[
            self.vec_enemy : self.vec_enemy + self.n_enemies * ENEMY_STRIDE : ENEMY_STRIDE
        ]
        enemy_y = vec[
            self.vec_enemy + 1 : self.vec_enemy + self.n_enemies * ENEMY_STRIDE : ENEMY_STRIDE
        ]
        enemy_hp = vec[
            self.vec_enemy + 2 : self.vec_enemy + self.n_enemies * ENEMY_STRIDE : ENEMY_STRIDE
        ]

        friend_alive = friend_hp > 0
        enemy_alive = enemy_hp > 0
        beacon_x, beacon_y = vec[self.vec_bxy : self.vec_bxy + 2]

        if self._prev_enemy_hp.sum() == 0 and self._prev_friend_hp.sum() == 0:
            self._prev_beacon_dists = (
                np.hypot(friend_x - beacon_x, friend_y - beacon_y)
                if beacon_x >= 0 and beacon_y >= 0
                else None
            )
            if enemy_alive.any():
                centroid_x = enemy_x[enemy_alive].mean()
                centroid_y = enemy_y[enemy_alive].mean()
                self._prev_centroid_dists = np.hypot(friend_x - centroid_x, friend_y - centroid_y)
            else:
                self._prev_centroid_dists = None

            self._prev_enemy_hp[:] = enemy_hp
            self._prev_friend_hp[:] = friend_hp
            self._last_reward_components = {
                "nav_r": 0.0,
                "combat_r": 0.0,
                "term_r": 0.0,
                "friend_hp": float(friend_hp.sum()),
                "enemy_hp": float(enemy_hp.sum()),
                "nav_dist": 0.0,
                "combat_dist": 0.0,
            }
            return 0.0

        nav_r = 0.0
        if beacon_x >= 0 and beacon_y >= 0:
            beacon_dists = np.hypot(friend_x - beacon_x, friend_y - beacon_y)
            if self._prev_beacon_dists is not None:
                nav_delta = self._prev_beacon_dists - beacon_dists
                valid_nav = nav_delta[friend_alive]
                nav_r = float(valid_nav.mean()) if valid_nav.size > 0 else 0.0
            self._prev_beacon_dists = beacon_dists
        else:
            self._prev_beacon_dists = None

        combat_dist_r = 0.0
        if enemy_alive.any():
            centroid_x = enemy_x[enemy_alive].mean()
            centroid_y = enemy_y[enemy_alive].mean()
            centroid_dists = np.hypot(friend_x - centroid_x, friend_y - centroid_y)
            if self._prev_centroid_dists is not None:
                centroid_delta = self._prev_centroid_dists - centroid_dists
                valid_centroid = centroid_delta[friend_alive]
                combat_dist_r = float(valid_centroid.mean()) if valid_centroid.size > 0 else 0.0
            self._prev_centroid_dists = centroid_dists
        else:
            self._prev_centroid_dists = None

        enemy_hp_delta = float((self._prev_enemy_hp - enemy_hp).clip(min=0.0).sum())
        friend_hp_delta = float((self._prev_friend_hp - friend_hp).clip(min=0.0).sum())
        enemy_kills = float(((self._prev_enemy_hp > 0) & (enemy_hp <= 0)).sum())

        combat_r = (
            combat_dist_r
            + HP_SCALE * enemy_hp_delta
            - HP_SCALE * friend_hp_delta
            + KILL_BONUS * enemy_kills
        )

        term_r = 0.0
        if done:
            if result == "nav_win":
                term_r = NAV_WIN_BONUS
            elif result in {"combat_win", "victory"}:
                term_r = COMBAT_WIN_BONUS
            elif result in {"combat_loss", "defeat"}:
                term_r = COMBAT_LOSS_PENALTY
            elif result == "timeout_loss":
                term_r = NAV_TIMEOUT_PENALTY
            elif result == "tie":
                term_r = TIE_BONUS

        self._prev_enemy_hp[:] = enemy_hp
        self._prev_friend_hp[:] = friend_hp
        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": float(friend_hp.sum()),
            "enemy_hp": float(enemy_hp.sum()),
            "nav_dist": float(nav_r),
            "combat_dist": float(combat_dist_r),
        }
        return float(nav_r + combat_r + term_r)

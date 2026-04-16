import os
import random

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
N_ENEMY = 3

FRIEND_STRIDE = 5
ENEMY_STRIDE = 5

VEC_FRIEND = 0
VEC_ENEMY = VEC_FRIEND + N_FRIEND * FRIEND_STRIDE
VEC_BXY = VEC_ENEMY + N_ENEMY * ENEMY_STRIDE
VEC_DIST = VEC_BXY + 2
VEC_TIME = VEC_DIST + 1
VEC_ECOUNT = VEC_TIME + 1
VEC_SIZE = VEC_ECOUNT + 1

OBS_SIZE = (
    FRIEND_STRIDE
    + (N_FRIEND - 1) * FRIEND_STRIDE
    + N_ENEMY * ENEMY_STRIDE
    + 2
    + 3
)


class TwoBridgeMap_V1_Navigate(lib.Map):
    name = "TwoBridgeMap_V1_Navigate"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
    filename = "TwoBridgeMap_V1_Navigate.SC2Map"
    players = 2


lib.get_maps().pop("TwoBridgeMap_V1_Navigate", None)
lib.get_maps()["TwoBridgeMap_V1_Navigate"] = TwoBridgeMap_V1_Navigate()


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
MINI_CH = len(features.MINIMAP_FEATURES)

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
N_ACTIONS = 1 + N_MOVE_ACTIONS + N_ENEMY
ATTACK_ACTION_OFFSET = 1 + N_MOVE_ACTIONS

KILL_BONUS = 1.0
HP_SCALE = 0.05

NAV_WIN_BONUS = 25.0
COMBAT_WIN_BONUS = 10.0
COMBAT_LOSS_PENALTY = -10.0
NAV_TIMEOUT_PENALTY = -15.0
TIE_BONUS = 0.0


class TwoBridgeQMixEnv:
    """
    Multi-agent raw-action environment for TwoBridgeMap_V1_Navigate.

    Per-agent discrete actions:
    0            -> no-op
    1..8         -> move using the same direction ordering as MaskPPO
    9..N -> attack enemy slot 0..(N_ENEMY - 1)
    """

    def __init__(
        self,
        map_name="V1_Navigate",
        seed=None,
        episode_limit=None,
        visualize=False,
        realtime=False,
        replay_dir="",
        save_replay_episodes=0,
    ):
        if map_name != "V1_Navigate":
            raise ValueError(f"Unsupported map_name={map_name!r}. Only 'V1_Navigate' is implemented.")

        self.map_name = map_name
        self.n_agents = N_FRIEND
        self.n_enemies = N_ENEMY
        self.n_actions = N_ACTIONS
        self.episode_limit = (
            DEFAULT_EPISODE_LIMIT if episode_limit is None else int(episode_limit)
        )
        if self.episode_limit < 1:
            raise ValueError("episode_limit must be >= 1")

        self._seed = None
        self._env = sc2_env.SC2Env(
            map_name="TwoBridgeMap_V1_Navigate",
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
                # PySC2 couples feature_screen and feature_minimap when feature
                # layers are enabled. We request them here only to access the
                # minimap; the screen tensor is never surfaced by this env.
                feature_dimensions=features.Dimensions(screen=SCR_RES, minimap=SCR_RES),
            ),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes,
        )

        self._my_tags = np.zeros(N_FRIEND, np.int64)
        self._enemy_tags = np.zeros(N_ENEMY, np.int64)
        self._enemy_alive = np.zeros(N_ENEMY, dtype=bool)
        self._fx = np.zeros(N_FRIEND, np.float32)
        self._fy = np.zeros(N_FRIEND, np.float32)
        self._raw_x_max = float(SCR_RES - 1)
        self._raw_y_max = float(SCR_RES - 1)

        self._step_ctr = 0
        self._prev_beacon_dists = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp = np.zeros(N_ENEMY, np.float32)
        self._prev_friend_hp = np.zeros(N_FRIEND, np.float32)

        self._state = np.zeros(VEC_SIZE, np.float32)
        self._obs = np.zeros((self.n_agents, OBS_SIZE), np.float32)
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
        friend_alive = (self._state[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE] > 0).sum()
        no_friend = friend_alive == 0
        no_enemy = self._state[VEC_ECOUNT] == 0
        beacon_win = self._state[VEC_DIST] < BEACON_RADIUS

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
        return OBS_SIZE

    def get_state(self):
        return self._state.copy()

    def get_state_size(self):
        return VEC_SIZE

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

    def _translate_actions(self, actions_arr):
        cmds = []
        friend_hp = self._state[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]

        for agent_id, action_id in enumerate(actions_arr):
            if friend_hp[agent_id] <= 0 or self._my_tags[agent_id] == 0:
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

            enemy_slot = int(action_id) - ATTACK_ACTION_OFFSET
            if 0 <= enemy_slot < N_ENEMY and self._enemy_alive[enemy_slot]:
                cmds.append(RAW.Attack_unit("now", [tag], int(self._enemy_tags[enemy_slot])))

        if not cmds:
            cmds.append(RAW.no_op())
        return cmds

    def _update_observations(self, ts):
        observation = ts.observation
        raw_units = observation.raw_units
        self._minimap = np.asarray(observation.feature_minimap, np.uint8)

        friends = sorted((u for u in raw_units if u.owner == 1), key=lambda unit: unit.tag)
        enemies = sorted((u for u in raw_units if u.owner == 2), key=lambda unit: unit.tag)
        beacon = next((u for u in raw_units if u.unit_type == BEACON_TYPE_ID), None)

        self._my_tags.fill(0)
        self._enemy_tags.fill(0)
        self._enemy_alive.fill(False)
        self._fx.fill(0.0)
        self._fy.fill(0.0)

        for idx, unit in enumerate(friends[:N_FRIEND]):
            self._my_tags[idx] = unit.tag
            self._fx[idx] = unit.x
            self._fy[idx] = unit.y

        for idx, unit in enumerate(enemies[:N_ENEMY]):
            self._enemy_tags[idx] = unit.tag
            self._enemy_alive[idx] = unit.health > 0

        state = np.zeros(VEC_SIZE, np.float32)

        for idx, unit in enumerate(friends[:N_FRIEND]):
            start = idx * FRIEND_STRIDE
            state[start : start + FRIEND_STRIDE] = (
                unit.x,
                unit.y,
                unit.health,
                unit.weapon_cooldown,
                1.0,
            )

        enemy_base = VEC_ENEMY
        for idx, unit in enumerate(enemies[:N_ENEMY]):
            start = enemy_base + idx * ENEMY_STRIDE
            state[start : start + ENEMY_STRIDE] = (
                unit.x,
                unit.y,
                unit.health,
                unit.weapon_cooldown,
                float(unit.health > 0),
            )

        beacon_x, beacon_y = (beacon.x, beacon.y) if beacon is not None else (-1.0, -1.0)
        state[VEC_BXY : VEC_BXY + 2] = (beacon_x, beacon_y)

        friend_x = state[0 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_y = state[1 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_hp = state[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_alive = friend_hp > 0

        if friend_alive.any() and beacon is not None and beacon_x >= 0 and beacon_y >= 0:
            state[VEC_DIST] = float(
                np.hypot(friend_x[friend_alive] - beacon_x, friend_y[friend_alive] - beacon_y).min()
            )
        else:
            state[VEC_DIST] = 128.0

        state[VEC_TIME] = observation.game_loop[0] / 16.0
        state[VEC_ECOUNT] = float(self._enemy_alive.sum())

        self._state = state
        self._obs = self._build_all_agent_obs()
        self._avail_actions = self._build_avail_actions()

    def _build_all_agent_obs(self):
        obs = np.zeros((self.n_agents, OBS_SIZE), np.float32)
        beacon_x, beacon_y = self._state[VEC_BXY : VEC_BXY + 2]

        for agent_id in range(self.n_agents):
            cursor = 0
            own_start = agent_id * FRIEND_STRIDE
            own = self._state[own_start : own_start + FRIEND_STRIDE]
            own_x, own_y, _, _, own_alive = own

            obs[agent_id, cursor : cursor + FRIEND_STRIDE] = own
            cursor += FRIEND_STRIDE

            for other_id in range(self.n_agents):
                if other_id == agent_id:
                    continue
                other_start = other_id * FRIEND_STRIDE
                other = self._state[other_start : other_start + FRIEND_STRIDE]
                rel_x = other[0] - own_x if own_alive > 0 else 0.0
                rel_y = other[1] - own_y if own_alive > 0 else 0.0
                obs[agent_id, cursor : cursor + FRIEND_STRIDE] = (
                    rel_x,
                    rel_y,
                    other[2],
                    other[3],
                    other[4],
                )
                cursor += FRIEND_STRIDE

            for enemy_id in range(N_ENEMY):
                enemy_start = VEC_ENEMY + enemy_id * ENEMY_STRIDE
                enemy = self._state[enemy_start : enemy_start + ENEMY_STRIDE]
                rel_x = enemy[0] - own_x if own_alive > 0 else 0.0
                rel_y = enemy[1] - own_y if own_alive > 0 else 0.0
                obs[agent_id, cursor : cursor + ENEMY_STRIDE] = (
                    rel_x,
                    rel_y,
                    enemy[2],
                    enemy[3],
                    enemy[4],
                )
                cursor += ENEMY_STRIDE

            if own_alive > 0 and beacon_x >= 0 and beacon_y >= 0:
                obs[agent_id, cursor : cursor + 2] = (beacon_x - own_x, beacon_y - own_y)
            cursor += 2

            obs[agent_id, cursor] = self._state[VEC_DIST]
            obs[agent_id, cursor + 1] = self._state[VEC_TIME]
            obs[agent_id, cursor + 2] = self._state[VEC_ECOUNT]

        return obs

    def _build_avail_actions(self):
        avail_actions = np.zeros((self.n_agents, self.n_actions), np.int8)
        enemy_alive = self._enemy_alive.astype(np.int8)

        for agent_id in range(self.n_agents):
            friend_start = agent_id * FRIEND_STRIDE
            friend_alive = self._state[friend_start + 2] > 0
            avail_actions[agent_id, 0] = 1
            if not friend_alive:
                continue

            avail_actions[agent_id, 1 : 1 + N_MOVE_ACTIONS] = 1
            avail_actions[
                agent_id, ATTACK_ACTION_OFFSET : ATTACK_ACTION_OFFSET + N_ENEMY
            ] = enemy_alive

        return avail_actions

    def _shape_reward(self, vec, done, result):
        friend_x = vec[0 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_y = vec[1 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        friend_hp = vec[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]

        enemy_x = vec[VEC_ENEMY : VEC_ENEMY + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]
        enemy_y = vec[VEC_ENEMY + 1 : VEC_ENEMY + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]
        enemy_hp = vec[VEC_ENEMY + 2 : VEC_ENEMY + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]

        friend_alive = friend_hp > 0
        enemy_alive = enemy_hp > 0
        beacon_x, beacon_y = vec[VEC_BXY : VEC_BXY + 2]

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
                deltas = self._prev_beacon_dists - beacon_dists
                alive_deltas = deltas[friend_alive]
                nav_r = float(alive_deltas.mean()) if alive_deltas.size > 0 else 0.0
            self._prev_beacon_dists = beacon_dists
        else:
            self._prev_beacon_dists = None

        combat_r = 0.0
        centroid_x = np.nan
        centroid_y = np.nan
        if enemy_alive.any():
            centroid_x = enemy_x[enemy_alive].mean()
            centroid_y = enemy_y[enemy_alive].mean()
            centroid_dists = np.hypot(friend_x - centroid_x, friend_y - centroid_y)
            if self._prev_centroid_dists is not None:
                deltas = self._prev_centroid_dists - centroid_dists
                alive_deltas = deltas[friend_alive]
                combat_r += float(alive_deltas.mean()) if alive_deltas.size > 0 else 0.0
            self._prev_centroid_dists = centroid_dists
        else:
            self._prev_centroid_dists = None

        combat_r += HP_SCALE * float(self._prev_enemy_hp.sum() - enemy_hp.sum())
        combat_r -= HP_SCALE * float(self._prev_friend_hp.sum() - friend_hp.sum())

        kills = (~enemy_alive & (self._prev_enemy_hp > 0)).sum()
        losses = (~friend_alive & (self._prev_friend_hp > 0)).sum()
        combat_r += KILL_BONUS * float(kills)
        combat_r -= KILL_BONUS * float(losses)

        self._prev_enemy_hp[:] = enemy_hp
        self._prev_friend_hp[:] = friend_hp

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
            else:
                term_r = 0.0
        else:
            term_r = 0.0

        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": float(friend_hp.sum()),
            "enemy_hp": float(enemy_hp.sum()),
            "nav_dist": (
                float(np.mean(np.hypot(friend_x - beacon_x, friend_y - beacon_y)))
                if friend_alive.any() and beacon_x >= 0 and beacon_y >= 0
                else 0.0
            ),
            "combat_dist": (
                float(np.mean(np.hypot(friend_x - centroid_x, friend_y - centroid_y)))
                if friend_alive.any() and enemy_alive.any()
                else 0.0
            ),
        }

        return float(nav_r + combat_r + term_r)


TwoBridgeEnv = TwoBridgeQMixEnv



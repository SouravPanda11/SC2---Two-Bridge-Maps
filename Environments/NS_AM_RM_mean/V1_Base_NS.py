import json
from pathlib import Path

import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ────────────────────── configuration ──────────────────────────
N_FRIEND = 5
N_ENEMY  = 3

# index helpers for the compact vector  ─────────────────────────
FRIEND_STRIDE = 4
ENEMY_STRIDE  = 4

VEC_FRIEND  = 0
VEC_ENEMY   = VEC_FRIEND  + N_FRIEND * FRIEND_STRIDE
VEC_BXY     = VEC_ENEMY   + N_ENEMY  * ENEMY_STRIDE      # 2 × float32
VEC_DIST    = VEC_BXY     + 2                 # 1 × float32
VEC_TIME    = VEC_DIST    + 1                 # 1 × float32
VEC_ECOUNT  = VEC_TIME    + 1                 # 1 × float32
VEC_SIZE    = VEC_ECOUNT  + 1

OBS_FRIEND_STRIDE = 2
OBS_ENEMY_STRIDE  = 2
OBS_FRIEND  = 0
OBS_ENEMY   = OBS_FRIEND + N_FRIEND * OBS_FRIEND_STRIDE
OBS_TIME    = OBS_ENEMY + N_ENEMY * OBS_ENEMY_STRIDE
OBS_ECOUNT  = OBS_TIME + 1
OBS_VEC_SIZE = OBS_ECOUNT + 1

# ───────────────────── Map registration ───────────────────────
class TwoBridgeMap_V1_Base(lib.Map):
    name      = "TwoBridgeMap_V1_Base"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
    filename  = "TwoBridgeMap_V1_Base.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V1_Base", None)
lib.get_maps()["TwoBridgeMap_V1_Base"] = TwoBridgeMap_V1_Base()

# ───────────────────────── constants ───────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW              = actions.RAW_FUNCTIONS
MARINE_HP        = 45
BEACON_TYPE_ID   = 317
BEACON_RADIUS    = 5.0

STEP_MUL         = 8
FIVE_MIN_LOOPS   = 5 * 60 * 16
MAX_STEPS        = FIVE_MIN_LOOPS // STEP_MUL
STEP_PIX         = 2
ATTACK_RANGE     = 6.0

SCR_RES          = 64
MINIMAP_PATHABLE_INDEX = int(features.MINIMAP_FEATURES.pathable.index)
MINIMAP_PLAYER_RELATIVE_INDEX = int(features.MINIMAP_FEATURES.player_relative.index)
MINI_CH          = 2

MOVE_DIRS = [
    ( 0, -STEP_PIX), ( 0,  STEP_PIX), (-STEP_PIX, 0), ( STEP_PIX, 0),
    ( STEP_PIX,-STEP_PIX), (-STEP_PIX,-STEP_PIX),
    ( STEP_PIX, STEP_PIX), (-STEP_PIX, STEP_PIX)
]
ATTACK_ACTION_OFFSET = 1 + len(MOVE_DIRS)
N_UNIT_ACTIONS       = ATTACK_ACTION_OFFSET + N_ENEMY

# ───────────────────── reward constants ────────────────────────
KILL_BONUS   = 1.0
HP_SCALE     = 0.05

NAV_WIN_BONUS       = 25.0
COMBAT_WIN_BONUS    = 10.0
COMBAT_LOSS_PENALTY = -10.0
NAV_TIMEOUT_PENALTY = -15.0
TIE_BONUS           = 0.0

# ─────────────────────── environment ───────────────────────────
class TwoBridgeEnv(gym.Env):
    """
    5 v 3 Two-Bridge – navigation & combat.
    Action space = per-friendly discrete actions.
    """
    metadata = {}

    action_space = spaces.MultiDiscrete([N_UNIT_ACTIONS] * N_FRIEND)

    observation_space = spaces.Dict({
        "minimap":     spaces.Box(0, 4, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
        "vector":      spaces.Box(0.0, np.inf, (OBS_VEC_SIZE,), np.float32),
        # Per-friendly action mask:
        # 0 noop | 1..8 move | ATTACK_ACTION_OFFSET.. attack enemy slots
        "action_mask": spaces.MultiBinary((N_FRIEND, N_UNIT_ACTIONS)),
    })

    def __init__(self,
                 screen_res: int = 64,
                 visualize: bool = False,
                 realtime: bool = False,
                 replay_dir: str = None,
                 save_replay_episodes: int = 0,
                 action_log_path: str = None):
        super().__init__()

        self._env = sc2_env.SC2Env(
            map_name="TwoBridgeMap_V1_Base",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot  (sc2_env.Race.terran,
                                   sc2_env.Difficulty.easy)],
            step_mul=STEP_MUL,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=SCR_RES,
                raw_crop_to_playable_area=True,
                # PySC2 couples feature_screen and feature_minimap in the interface.
                # We keep the feature layer enabled for minimap but only expose minimap in obs.
                feature_dimensions=features.Dimensions(
                    screen=SCR_RES, minimap=SCR_RES)),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes)

        # caches
        self._my_tags     = np.zeros(N_FRIEND, np.int64)
        self._friend_alive = np.zeros(N_FRIEND, bool)
        self._enemy_tags  = np.zeros(N_ENEMY,  np.int64)
        self._enemy_alive = np.zeros(N_ENEMY,  bool)
        self._friend_enemy_attackable = np.zeros((N_FRIEND, N_ENEMY), bool)
        self._fx = np.zeros(N_FRIEND, np.float32)
        self._fy = np.zeros(N_FRIEND, np.float32)
        self._raw_x_max = float(SCR_RES - 1)
        self._raw_y_max = float(SCR_RES - 1)

        self._step_ctr            = 0
        self._episode_ctr         = -1
        self._episode_step_ctr    = 0
        self._prev_beacon_dists   = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp       = np.zeros(N_ENEMY,  np.float32)
        self._prev_friend_hp      = np.zeros(N_FRIEND, np.float32)

        self._last_act = {"actions": np.zeros(N_FRIEND, np.int64)}
        self._last_action_debug = {}

        # instrumentation caches
        self._last_reward_components = {
            "nav_r": 0.0, "combat_r": 0.0, "term_r": 0.0,
            "friend_hp": 0.0, "enemy_hp": 0.0,
            "nav_dist": 0.0, "combat_dist": 0.0
        }
        self._last_unit_metrics = {"friend": {}, "enemy": {}}
        self._last_internal_vec = np.zeros(VEC_SIZE, np.float32)
        self._action_log_fp = None
        if action_log_path:
            log_path = Path(action_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._action_log_fp = log_path.open("w", encoding="utf-8")
        self._refresh_raw_bounds()

    def close(self):
        if self._action_log_fp is not None:
            self._action_log_fp.close()
            self._action_log_fp = None
        self._env.close()

    def _refresh_raw_bounds(self):
        gi = self._env.game_info[0].start_raw.map_size
        self._raw_x_max = max(0.0, float(gi.x) - 1.0)
        self._raw_y_max = max(0.0, float(gi.y) - 1.0)

    def _write_action_record(self, record):
        if self._action_log_fp is None:
            return
        self._action_log_fp.write(json.dumps(record) + "\n")
        self._action_log_fp.flush()

    def _serialize_action_mask(self, action_mask):
        if isinstance(action_mask, dict):
            return {
                key: np.asarray(value, dtype=np.int8).tolist()
                for key, value in action_mask.items()
            }
        return np.asarray(action_mask, dtype=np.int8).tolist()

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

    def _friend_units_from_vec(self, vec):
        units = []
        for i in range(N_FRIEND):
            base = i * FRIEND_STRIDE
            tag = int(self._my_tags[i])
            hp = float(vec[base + 2])
            if tag != 0:
                units.append({
                    "slot": i,
                    "tag": tag,
                    "x": float(vec[base]),
                    "y": float(vec[base + 1]),
                    "hp": hp,
                })
        return units

    def _enemy_units_from_vec(self, vec):
        units = []
        base = VEC_ENEMY
        for i in range(N_ENEMY):
            idx = base + i * ENEMY_STRIDE
            tag = int(self._enemy_tags[i])
            hp = float(vec[idx + 2])
            if tag != 0:
                units.append({
                    "slot": i,
                    "tag": tag,
                    "x": float(vec[idx]),
                    "y": float(vec[idx + 1]),
                    "hp": hp,
                    "alive": bool(self._enemy_alive[i]),
                })
        return units

    def _summarize_step_debug(self, obs, ts, reward, done, result):
        internal_vec = self._last_internal_vec
        debug = dict(self._last_action_debug)
        debug.update({
            "episode": int(self._episode_ctr),
            "episode_step": int(self._episode_step_ctr),
            "game_loop": int(ts.observation.game_loop[0]),
            "reward": float(reward),
            "done": bool(done),
            "result": result,
            "friendly_units_after": self._friend_units_from_vec(internal_vec),
            "enemy_units_after": self._enemy_units_from_vec(internal_vec),
            "beacon_after": {
                "x": float(internal_vec[VEC_BXY]),
                "y": float(internal_vec[VEC_BXY + 1]),
            },
            "action_mask_after": self._serialize_action_mask(obs["action_mask"]),
        })
        self._write_action_record({"event": "step", **debug})
        return debug

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0
        self._episode_ctr += 1
        self._episode_step_ctr = 0

        self._prev_beacon_dists   = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp[:]    = 0.0
        self._prev_friend_hp[:]   = 0.0

        self._last_act = {"actions": np.zeros(N_FRIEND, np.int64)}
        self._last_action_debug = {}
        self._last_unit_metrics = {"friend": {}, "enemy": {}}
        self._last_reward_components = {
            "nav_r": 0.0, "combat_r": 0.0, "term_r": 0.0,
            "friend_hp": 0.0, "enemy_hp": 0.0,
            "nav_dist": 0.0, "combat_dist": 0.0
        }
        self._last_internal_vec.fill(0.0)
        self._my_tags[:] = 0
        self._friend_alive[:] = False
        self._enemy_tags[:] = 0
        self._enemy_alive[:] = False
        self._friend_enemy_attackable.fill(False)
        self._fx[:] = 0.0
        self._fy[:] = 0.0

        ts = self._env.reset()[0]
        self._refresh_raw_bounds()
        obs = self._build_obs(ts)
        internal_vec = self._last_internal_vec
        self._write_action_record({
            "event": "reset",
            "episode": int(self._episode_ctr),
            "game_loop": int(ts.observation.game_loop[0]),
            "raw_bounds": {
                "x_min": 0.0,
                "x_max": float(self._raw_x_max),
                "y_min": 0.0,
                "y_max": float(self._raw_y_max),
            },
            "friendly_units": self._friend_units_from_vec(internal_vec),
            "enemy_units": self._enemy_units_from_vec(internal_vec),
            "beacon": {
                "x": float(internal_vec[VEC_BXY]),
                "y": float(internal_vec[VEC_BXY + 1]),
            },
            "action_mask": self._serialize_action_mask(obs["action_mask"]),
        })
        return obs, {}

    def step(self, action):
        cmds = self._translate_actions(action)
        ts   = self._env.step(cmds)[0]
        obs  = self._build_obs(ts)
        self._episode_step_ctr += 1

        # built-in victory / defeat
        if ts.last():
            res = "victory" if ts.reward > 0 else "defeat" if ts.reward < 0 else "tie"
            info = {"result": res}
            info["action_debug"] = self._summarize_step_debug(
                obs, ts, float(ts.reward), True, info["result"]
            )
            return obs, float(ts.reward), True, False, info

        # custom termination
        internal_vec = self._last_internal_vec
        friend_alive = (internal_vec[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE] > 0).sum()
        no_friend    = friend_alive == 0
        no_enemy     = internal_vec[VEC_ECOUNT] == 0
        beacon_win   = internal_vec[VEC_DIST] < BEACON_RADIUS

        info = {"result": None}
        if beacon_win:               info["result"] = "nav_win"
        elif no_enemy and no_friend: info["result"] = "tie"
        elif no_enemy:               info["result"] = "combat_win"
        elif no_friend:              info["result"] = "combat_loss"

        if self._step_ctr >= MAX_STEPS and info["result"] is None:
            info["result"] = "timeout_loss"

        done   = info["result"] is not None
        reward = self._shape_reward(internal_vec, done, info["result"])
        info["rew"] = self.get_reward_components()
        info["action_debug"] = self._summarize_step_debug(
            obs, ts, reward, done, info["result"]
        )
        return obs, reward, done, False, info

    def _translate_actions(self, act):
        actions_arr = np.asarray(act, dtype=np.int64).reshape(-1)
        if actions_arr.shape[0] != N_FRIEND:
            raise ValueError(f"Expected {N_FRIEND} per-unit actions, received {actions_arr.shape[0]}.")

        self._last_act = {"actions": actions_arr.copy()}
        self._last_action_debug = {
            "requested_action": {"actions": actions_arr.astype(int).tolist()},
            "raw_bounds": {
                "x_min": 0.0,
                "x_max": float(self._raw_x_max),
                "y_min": 0.0,
                "y_max": float(self._raw_y_max),
            },
            "per_unit_actions": [],
        }

        cmds = []
        for friend_idx, action_id in enumerate(actions_arr):
            tag = int(self._my_tags[friend_idx])
            alive = bool(self._friend_alive[friend_idx])
            unit_debug = {
                "slot": int(friend_idx),
                "tag": tag,
                "alive": alive,
                "action_id": int(action_id),
                "x": float(self._fx[friend_idx]),
                "y": float(self._fy[friend_idx]),
            }

            if not alive or tag == 0:
                unit_debug["translated_action"] = {"command": "no_op", "reason": "unit_not_alive"}
                self._last_action_debug["per_unit_actions"].append(unit_debug)
                continue

            if action_id == 0:
                unit_debug["translated_action"] = {"command": "no_op", "reason": "noop_requested"}
                self._last_action_debug["per_unit_actions"].append(unit_debug)
                continue

            if 1 <= action_id < ATTACK_ACTION_OFFSET:
                dx, dy = MOVE_DIRS[int(action_id) - 1]
                tx = float(np.clip(self._fx[friend_idx] + dx, 0.0, self._raw_x_max))
                ty = float(np.clip(self._fy[friend_idx] + dy, 0.0, self._raw_y_max))
                cmds.append(RAW.Move_pt("now", [tag], (tx, ty)))
                unit_debug["translated_action"] = {
                    "command": "Move_pt",
                    "delta": {"x": int(dx), "y": int(dy)},
                    "target_after_clip": {"x": tx, "y": ty},
                }
                self._last_action_debug["per_unit_actions"].append(unit_debug)
                continue

            enemy_idx = int(action_id) - ATTACK_ACTION_OFFSET
            if (0 <= enemy_idx < N_ENEMY and self._enemy_alive[enemy_idx]
                    and self._friend_enemy_attackable[friend_idx, enemy_idx]):
                cmds.append(RAW.Attack_unit("now", [tag], int(self._enemy_tags[enemy_idx])))
                unit_debug["translated_action"] = {
                    "command": "Attack_unit",
                    "target_enemy_slot": int(enemy_idx),
                    "target_enemy_tag": int(self._enemy_tags[enemy_idx]),
                }
            else:
                reason = "invalid_or_masked_action"
                if not (0 <= enemy_idx < N_ENEMY):
                    reason = "attack_with_invalid_enemy_slot"
                elif not self._enemy_alive[enemy_idx]:
                    reason = "attack_target_not_alive"
                else:
                    reason = "attack_target_out_of_range"
                unit_debug["translated_action"] = {"command": "no_op", "reason": reason}
            self._last_action_debug["per_unit_actions"].append(unit_debug)

        if not cmds:
            self._last_action_debug["translated_action"] = {
                "command": "no_op",
                "reason": "no_valid_unit_commands",
            }
            return [RAW.no_op()]

        self._last_action_debug["translated_action"] = {
            "command_count": len(cmds),
            "commands": [
                entry["translated_action"]["command"]
                for entry in self._last_action_debug["per_unit_actions"]
            ],
        }
        return cmds

    def _build_obs(self, ts):
        ob   = ts.observation
        ru   = ob.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        ene  = sorted([u for u in ru if u.owner == 2], key=lambda u: u.tag)
        bea  = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)

        bx, by = (bea.x, bea.y) if bea is not None else (-1.0, -1.0)

        self._populate_slot_tags(fri[:N_FRIEND], self._my_tags)
        self._populate_slot_tags(ene[:N_ENEMY], self._enemy_tags)

        self._friend_alive[:] = False
        self._enemy_alive[:] = False
        self._friend_enemy_attackable.fill(False)
        self._fx[:] = self._fy[:] = 0.0

        friend_by_tag = {int(u.tag): u for u in fri}
        enemy_by_tag = {int(u.tag): u for u in ene}

        internal_vec = np.zeros(VEC_SIZE, np.float32)

        # friends
        for i, tag in enumerate(self._my_tags):
            if int(tag) == 0:
                continue
            unit = friend_by_tag.get(int(tag))
            if unit is None:
                continue
            alive = unit.health > 0
            self._friend_alive[i] = alive
            self._fx[i], self._fy[i] = unit.x, unit.y
            internal_vec[i * FRIEND_STRIDE:(i + 1) * FRIEND_STRIDE] = (
                unit.x, unit.y, unit.health, float(alive)
            )

        # enemies
        base = VEC_ENEMY
        for i, tag in enumerate(self._enemy_tags):
            if int(tag) == 0:
                continue
            unit = enemy_by_tag.get(int(tag))
            if unit is None:
                continue
            alive = unit.health > 0
            self._enemy_alive[i] = alive
            internal_vec[base + i * ENEMY_STRIDE:base + (i + 1) * ENEMY_STRIDE] = (
                unit.x, unit.y, unit.health, float(alive)
            )

        enemy_x = internal_vec[base : base + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]
        enemy_y = internal_vec[base + 1 : base + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]
        for friend_idx in range(N_FRIEND):
            if not self._friend_alive[friend_idx]:
                continue
            for enemy_idx in range(N_ENEMY):
                if not self._enemy_alive[enemy_idx]:
                    continue
                dist = np.hypot(self._fx[friend_idx] - enemy_x[enemy_idx],
                                self._fy[friend_idx] - enemy_y[enemy_idx])
                self._friend_enemy_attackable[friend_idx, enemy_idx] = dist <= ATTACK_RANGE

        # beacon / misc
        internal_vec[VEC_BXY:VEC_BXY+2] = (bx, by)

        if fri and bea is not None:
            fx  = internal_vec[0 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
            fy  = internal_vec[1 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
            fhp = internal_vec[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
            f_alive = fhp > 0
            if f_alive.any() and (bx >= 0) and (by >= 0):
                dists = np.hypot(fx - bx, fy - by)
                internal_vec[VEC_DIST] = float(dists[f_alive].min())
            else:
                internal_vec[VEC_DIST] = 128.0
        else:
            internal_vec[VEC_DIST] = 128.0

        internal_vec[VEC_TIME]   = ob.game_loop[0] / 16.0
        internal_vec[VEC_ECOUNT] = float(self._enemy_alive.sum())

        action_mask = np.zeros((N_FRIEND, N_UNIT_ACTIONS), np.int8)
        action_mask[:, 0] = 1
        for friend_idx in range(N_FRIEND):
            if not self._friend_alive[friend_idx]:
                continue
            action_mask[friend_idx, 1:ATTACK_ACTION_OFFSET] = 1
            action_mask[
                friend_idx,
                ATTACK_ACTION_OFFSET:ATTACK_ACTION_OFFSET + N_ENEMY
            ] = self._friend_enemy_attackable[friend_idx].astype(np.int8)

        self._last_internal_vec = internal_vec
        actor_vec = np.zeros(OBS_VEC_SIZE, np.float32)
        for i in range(N_FRIEND):
            src = i * FRIEND_STRIDE
            dst = OBS_FRIEND + i * OBS_FRIEND_STRIDE
            actor_vec[dst:dst + OBS_FRIEND_STRIDE] = (
                internal_vec[src + 2],
                internal_vec[src + 3],
            )
        for i in range(N_ENEMY):
            src = VEC_ENEMY + i * ENEMY_STRIDE
            dst = OBS_ENEMY + i * OBS_ENEMY_STRIDE
            actor_vec[dst:dst + OBS_ENEMY_STRIDE] = (
                internal_vec[src + 2],
                internal_vec[src + 3],
            )
        actor_vec[OBS_TIME] = internal_vec[VEC_TIME]
        actor_vec[OBS_ECOUNT] = internal_vec[VEC_ECOUNT]

        minimap = np.asarray(ob.feature_minimap, np.uint8)[
            [MINIMAP_PATHABLE_INDEX, MINIMAP_PLAYER_RELATIVE_INDEX]
        ]
        self._step_ctr += 1
        return {
            "minimap":     minimap,
            "vector":      actor_vec,
            "action_mask": action_mask,
        }

    def _shape_reward(self, vec, done, res):
        """
        Team reward = navigation Δ-distance + combat (centroid Δ-distance + HP + kill/loss) + terminal.
        Also emits per-unit diagnostics keyed by tags.
        """
        # unpack
        fx  = vec[0 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        fy  = vec[1 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
        fhp = vec[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]

        ex  = vec[VEC_ENEMY     : VEC_ENEMY + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]
        ey  = vec[VEC_ENEMY + 1 : VEC_ENEMY + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]
        ehp = vec[VEC_ENEMY + 2 : VEC_ENEMY + N_ENEMY * ENEMY_STRIDE : ENEMY_STRIDE]

        f_alive = fhp > 0
        e_alive = ehp > 0
        bx, by  = vec[VEC_BXY : VEC_BXY+2]

        # first-frame guard
        if self._prev_enemy_hp.sum() == 0 and self._prev_friend_hp.sum() == 0:
            self._prev_beacon_dists = np.hypot(fx - bx, fy - by) if (bx >= 0 and by >= 0) else None
            if e_alive.any():
                cx, cy = ex[e_alive].mean(), ey[e_alive].mean()
                self._prev_centroid_dists = np.hypot(fx - cx, fy - cy)
            else:
                self._prev_centroid_dists = None

            self._prev_enemy_hp[:]  = ehp
            self._prev_friend_hp[:] = fhp

            self._last_reward_components = {
                "nav_r": 0.0, "combat_r": 0.0, "term_r": 0.0,
                "friend_hp": float(fhp.sum()), "enemy_hp": float(ehp.sum()),
                "nav_dist": 0.0, "combat_dist": 0.0
            }
            self._last_unit_metrics = {"friend": {}, "enemy": {}}
            return 0.0

        # NAV shaping (+per-unit)
        nav_per = np.zeros(N_FRIEND, np.float32)
        nav_r   = 0.0
        if (bx >= 0) and (by >= 0):
            beacon_dists = np.hypot(fx - bx, fy - by)
            if self._prev_beacon_dists is not None:
                nav_per = self._prev_beacon_dists - beacon_dists
                sel = nav_per[f_alive]
                nav_r = float(sel.mean()) if sel.size > 0 else 0.0
            self._prev_beacon_dists = beacon_dists
        else:
            self._prev_beacon_dists = None

        # COMBAT shaping (+per-unit centroid)
        combat_per = np.zeros(N_FRIEND, np.float32)
        combat_r   = 0.0
        cx = cy = np.nan
        if e_alive.any():
            cx, cy = ex[e_alive].mean(), ey[e_alive].mean()
            centroid_dists = np.hypot(fx - cx, fy - cy)
            if self._prev_centroid_dists is not None:
                combat_per = self._prev_centroid_dists - centroid_dists
                sel = combat_per[f_alive]
                combat_r += float(sel.mean()) if sel.size > 0 else 0.0
            self._prev_centroid_dists = centroid_dists
        else:
            self._prev_centroid_dists = None

        # HP shaping
        combat_r +=  HP_SCALE * float(self._prev_enemy_hp.sum()  - ehp.sum())
        combat_r += -HP_SCALE * float(self._prev_friend_hp.sum() - fhp.sum())

        # kill / loss bonuses
        kills  = (~e_alive & (self._prev_enemy_hp > 0)).sum()
        losses = (~f_alive & (self._prev_friend_hp > 0)).sum()
        combat_r +=  KILL_BONUS * float(kills)
        combat_r += -KILL_BONUS * float(losses)

        # update HP caches
        self._prev_enemy_hp[:]  = ehp
        self._prev_friend_hp[:] = fhp

        # terminal
        if done:
            if   res == "nav_win":      term_r = NAV_WIN_BONUS
            elif res == "combat_win":   term_r = COMBAT_WIN_BONUS
            elif res == "combat_loss":  term_r = COMBAT_LOSS_PENALTY
            elif res == "timeout_loss": term_r = NAV_TIMEOUT_PENALTY
            elif res == "tie":          term_r = TIE_BONUS
            elif res == "victory":      term_r = COMBAT_WIN_BONUS
            elif res == "defeat":       term_r = COMBAT_LOSS_PENALTY
            else:                       term_r = 0.0
        else:
            term_r = 0.0

        # log components
        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": float(fhp.sum()),
            "enemy_hp": float(ehp.sum()),
            "nav_dist": float(np.mean(np.hypot(fx - bx, fy - by))) if f_alive.any() and (bx >= 0) and (by >= 0) else 0.0,
            "combat_dist": float(np.mean(np.hypot(fx - cx, fy - cy))) if (e_alive.any() and f_alive.any()) else 0.0,
        }

        # per-unit diagnostics keyed by tags
        friend_dict = {}
        for i in range(N_FRIEND):
            tag = int(self._my_tags[i])
            if tag != 0:
                friend_dict[tag] = {
                    "nav_r":    float(nav_per[i]),
                    "combat_r": float(combat_per[i]),
                    "hp":       float(fhp[i]),
                }

        enemy_dict = {}
        for j in range(N_ENEMY):
            etag = int(self._enemy_tags[j])
            if etag != 0:
                enemy_dict[etag] = {"hp": float(ehp[j])}

        self._last_unit_metrics = {"friend": friend_dict, "enemy": enemy_dict}

        return float(nav_r + combat_r + term_r)

    # ───────────────────────── instrumentation getters ─────────────────────────
    def get_reward_components(self):
        return getattr(self, "_last_reward_components", {})

    def get_friendly_tags(self):
        return self._my_tags.copy()

    def get_unit_metrics(self):
        return getattr(self, "_last_unit_metrics", {"friend": {}, "enemy": {}})

    def get_last_action_debug(self):
        return getattr(self, "_last_action_debug", {})

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
N_ENEMY  = 5

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

# ───────────────────── Map registration ───────────────────────
class TwoBridgeMap_V2_Combat(lib.Map):
    name      = "TwoBridgeMap_V2_Combat"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
    filename  = "TwoBridgeMap_V2_Combat.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V2_Combat", None)
lib.get_maps()["TwoBridgeMap_V2_Combat"] = TwoBridgeMap_V2_Combat()

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

SCR_RES          = 64
MINI_CH          = len(features.MINIMAP_FEATURES)   # 7

MOVE_DIRS = [
    ( 0, -STEP_PIX), ( 0,  STEP_PIX), (-STEP_PIX, 0), ( STEP_PIX, 0),
    ( STEP_PIX,-STEP_PIX), (-STEP_PIX,-STEP_PIX),
    ( STEP_PIX, STEP_PIX), (-STEP_PIX, STEP_PIX)
]

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
    5 v 5 Two-Bridge – navigation & combat.
    Action space = {verb, who-mask, direction, enemy_idx}
    """
    metadata = {}

    action_space = spaces.Dict({
        "verb":      spaces.Discrete(3),              # 0 noop | 1 move | 2 attack
        "who":       spaces.MultiBinary(N_FRIEND),    # which friendlies receive the order
        "direction": spaces.Discrete(9),              # 0 unused | 1-8 compass
        "enemy_idx": spaces.Discrete(N_ENEMY + 1)     # 0 none | 1..N_ENEMY slots
    })

    observation_space = spaces.Dict({
        "minimap":     spaces.Box(0, 255, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
        "vector":      spaces.Box(0.0, np.inf, (VEC_SIZE,), np.float32),
        # Branch masks used by the NS trainers. Direction is left unmasked
        # because move targets are clipped to the playable bounds.
        "action_mask": spaces.Dict({
            "verb":      spaces.MultiBinary(3),
            "who":       spaces.MultiBinary(N_FRIEND),
            "enemy_idx": spaces.MultiBinary(N_ENEMY + 1),
        }),
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
            map_name="TwoBridgeMap_V2_Combat",
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
        self._enemy_tags  = np.zeros(N_ENEMY,  np.int64)
        self._enemy_alive = np.zeros(N_ENEMY,  bool)
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

        self._last_act = {"verb": 0, "who_bits": np.zeros(N_FRIEND, bool), "enemy_idx": -1}
        self._last_action_debug = {}

        # instrumentation caches
        self._last_reward_components = {
            "nav_r": 0.0, "combat_r": 0.0, "term_r": 0.0,
            "friend_hp": 0.0, "enemy_hp": 0.0,
            "nav_dist": 0.0, "combat_dist": 0.0
        }
        self._last_unit_metrics = {"friend": {}, "enemy": {}}
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
        debug = dict(self._last_action_debug)
        debug.update({
            "episode": int(self._episode_ctr),
            "episode_step": int(self._episode_step_ctr),
            "game_loop": int(ts.observation.game_loop[0]),
            "reward": float(reward),
            "done": bool(done),
            "result": result,
            "friendly_units_after": self._friend_units_from_vec(obs["vector"]),
            "enemy_units_after": self._enemy_units_from_vec(obs["vector"]),
            "beacon_after": {
                "x": float(obs["vector"][VEC_BXY]),
                "y": float(obs["vector"][VEC_BXY + 1]),
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

        self._last_act = {"verb": 0, "who_bits": np.zeros(N_FRIEND, bool), "enemy_idx": -1}
        self._last_action_debug = {}
        self._last_unit_metrics = {"friend": {}, "enemy": {}}
        self._last_reward_components = {
            "nav_r": 0.0, "combat_r": 0.0, "term_r": 0.0,
            "friend_hp": 0.0, "enemy_hp": 0.0,
            "nav_dist": 0.0, "combat_dist": 0.0
        }

        ts = self._env.reset()[0]
        self._refresh_raw_bounds()
        obs = self._build_obs(ts)
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
            "friendly_units": self._friend_units_from_vec(obs["vector"]),
            "enemy_units": self._enemy_units_from_vec(obs["vector"]),
            "beacon": {
                "x": float(obs["vector"][VEC_BXY]),
                "y": float(obs["vector"][VEC_BXY + 1]),
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
        friend_alive = (obs["vector"][2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE] > 0).sum()
        no_friend    = friend_alive == 0
        no_enemy     = obs["vector"][VEC_ECOUNT] == 0
        beacon_win   = obs["vector"][VEC_DIST] < BEACON_RADIUS

        info = {"result": None}
        if beacon_win:               info["result"] = "nav_win"
        elif no_enemy and no_friend: info["result"] = "tie"
        elif no_enemy:               info["result"] = "combat_win"
        elif no_friend:              info["result"] = "combat_loss"

        if self._step_ctr >= MAX_STEPS and info["result"] is None:
            info["result"] = "timeout_loss"

        done   = info["result"] is not None
        reward = self._shape_reward(obs["vector"], done, info["result"])
        info["rew"] = self.get_reward_components()
        info["action_debug"] = self._summarize_step_debug(
            obs, ts, reward, done, info["result"]
        )
        return obs, reward, done, False, info

    def _translate_actions(self, act):
        verb      = int(act["verb"])
        who_bits  = act["who"].astype(bool)
        dir_id    = int(act["direction"])
        enemy_idx = int(act["enemy_idx"]) - 1   # shift => 0..N_ENEMY-1

        tags = [int(t) for t, b in zip(self._my_tags, who_bits) if b]
        self._last_act = {"verb": verb, "who_bits": who_bits.copy(), "enemy_idx": enemy_idx}
        self._last_action_debug = {
            "requested_action": {
                "verb": verb,
                "who_bits": who_bits.astype(int).tolist(),
                "direction": dir_id,
                "enemy_idx_input": int(act["enemy_idx"]),
                "enemy_slot": enemy_idx,
            },
            "selected_tags": tags,
            "selected_units_before": [
                {
                    "slot": i,
                    "tag": int(self._my_tags[i]),
                    "x": float(self._fx[i]),
                    "y": float(self._fy[i]),
                }
                for i, selected in enumerate(who_bits)
                if selected and int(self._my_tags[i]) != 0
            ],
            "raw_bounds": {
                "x_min": 0.0,
                "x_max": float(self._raw_x_max),
                "y_min": 0.0,
                "y_max": float(self._raw_y_max),
            },
        }

        # MOVE
        if verb == 1 and tags and 1 <= dir_id <= 8:
            dx, dy = MOVE_DIRS[dir_id-1]
            cx = np.mean(self._fx[who_bits]) + dx
            cy = np.mean(self._fy[who_bits]) + dy
            pt = (float(np.clip(cx, 0.0, self._raw_x_max)),
                  float(np.clip(cy, 0.0, self._raw_y_max)))
            self._last_action_debug["translated_action"] = {
                "command": "Move_pt",
                "delta": {"x": int(dx), "y": int(dy)},
                "target_before_clip": {"x": float(cx), "y": float(cy)},
                "target_after_clip": {"x": pt[0], "y": pt[1]},
            }
            return [RAW.Move_pt("now", tags, pt)]

        # ATTACK
        if (verb == 2 and tags and
                0 <= enemy_idx < N_ENEMY and self._enemy_alive[enemy_idx]):
            self._last_action_debug["translated_action"] = {
                "command": "Attack_unit",
                "target_enemy_slot": enemy_idx,
                "target_enemy_tag": int(self._enemy_tags[enemy_idx]),
            }
            return [RAW.Attack_unit("now", tags, int(self._enemy_tags[enemy_idx]))]

        reason = "invalid_or_masked_action"
        if verb == 0:
            reason = "noop_requested"
        elif verb == 1:
            if not tags:
                reason = "move_without_selected_units"
            elif not (1 <= dir_id <= 8):
                reason = "move_with_invalid_direction"
        elif verb == 2:
            if not tags:
                reason = "attack_without_selected_units"
            elif not (0 <= enemy_idx < N_ENEMY):
                reason = "attack_with_invalid_enemy_slot"
            elif not self._enemy_alive[enemy_idx]:
                reason = "attack_target_not_alive"
        self._last_action_debug["translated_action"] = {
            "command": "no_op",
            "reason": reason,
        }
        return [RAW.no_op()]

    def _build_obs(self, ts):
        ob   = ts.observation
        ru   = ob.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        ene  = sorted([u for u in ru if u.owner == 2], key=lambda u: u.tag)
        bea  = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)

        bx, by = (bea.x, bea.y) if bea is not None else (-1.0, -1.0)

        self._my_tags[:]     = 0
        self._enemy_tags[:]  = 0
        self._enemy_alive[:] = False
        self._fx[:] = self._fy[:] = 0.0

        for i, u in enumerate(fri[:N_FRIEND]):
            self._my_tags[i] = u.tag
            self._fx[i], self._fy[i] = u.x, u.y

        for i, u in enumerate(ene[:N_ENEMY]):
            self._enemy_tags[i]  = u.tag
            self._enemy_alive[i] = u.health > 0

        vec = np.zeros(VEC_SIZE, np.float32)

        # friends
        for i, u in enumerate(fri[:N_FRIEND]):
            vec[i * FRIEND_STRIDE:(i + 1) * FRIEND_STRIDE] = (u.x, u.y, u.health, 1.0)

        # enemies
        base = VEC_ENEMY
        for i, u in enumerate(ene[:N_ENEMY]):
            vec[base + i * ENEMY_STRIDE:base + (i + 1) * ENEMY_STRIDE] = (
                u.x, u.y, u.health, float(u.health > 0)
            )

        # beacon / misc
        vec[VEC_BXY:VEC_BXY+2] = (bx, by)

        if fri and bea is not None:
            fx  = vec[0 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
            fy  = vec[1 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
            fhp = vec[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE]
            f_alive = fhp > 0
            if f_alive.any() and (bx >= 0) and (by >= 0):
                dists = np.hypot(fx - bx, fy - by)
                vec[VEC_DIST] = float(dists[f_alive].min())
            else:
                vec[VEC_DIST] = 128.0
        else:
            vec[VEC_DIST] = 128.0

        vec[VEC_TIME]   = ob.game_loop[0] / 16.0
        vec[VEC_ECOUNT] = float(self._enemy_alive.sum())

        who_mask = (vec[2 : N_FRIEND * FRIEND_STRIDE : FRIEND_STRIDE] > 0).astype(np.int8)
        any_friend_alive = int(who_mask.any())

        verb_mask = np.zeros(3, np.int8)
        verb_mask[0] = 1
        verb_mask[1] = any_friend_alive
        verb_mask[2] = int(vec[VEC_ECOUNT] > 0)

        enemy_mask = np.zeros(N_ENEMY + 1, np.int8)
        enemy_mask[0] = 1
        if verb_mask[2]:
            enemy_mask[1 : 1 + N_ENEMY] = self._enemy_alive.astype(np.int8)

        action_mask = {
            "verb": verb_mask,
            "who": who_mask,
            "enemy_idx": enemy_mask,
        }

        self._step_ctr += 1
        return {
            "minimap":     np.asarray(ob.feature_minimap, np.uint8),
            "vector":      vec,
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

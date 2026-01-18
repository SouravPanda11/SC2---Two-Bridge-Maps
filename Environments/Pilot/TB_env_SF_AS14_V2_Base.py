# 5 v 5 Two-Bridge – 14-choice action space + spatial features (SF)
# Eval-compatible: emits info["result"] in {nav_win, combat_win, combat_loss, timeout_loss, tie}
# and info["rew"] = get_reward_components() with nav/combat/term decomposition.

import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env   import sc2_env
from pysc2.lib   import actions, features
from pysc2.maps  import lib
from absl        import flags

# ───────────────────── Map registration ──────────────────────────────
class TwoBridgeMap_V2_Base(lib.Map):
    name      = "TwoBridgeMap_V2_Base"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
    filename  = "TwoBridgeMap_V2_Base.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V2_Base", None)
lib.get_maps()["TwoBridgeMap_V2_Base"] = TwoBridgeMap_V2_Base()

# ────────────────────────── constants ────────────────────────────────
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
SCR_CH           = len(features.SCREEN_FEATURES)    # 17
MINI_CH          = len(features.MINIMAP_FEATURES)   # 7

# Reward constants (match NSF behavior)
TERM_WIN_BONUS     = 10.0
TERM_LOSS_PENALTY  = -10.0
TERM_TIE_BONUS     = 0.0

# movement vectors for 8 directions
MOVE_DIRS = [
    ( 0, -STEP_PIX), ( 0,  STEP_PIX), (-STEP_PIX, 0), ( STEP_PIX, 0),
    ( STEP_PIX,-STEP_PIX), (-STEP_PIX,-STEP_PIX),
    ( STEP_PIX, STEP_PIX), (-STEP_PIX, STEP_PIX)
]

# ──────────────────────── environment ────────────────────────────────
class TwoBridgeEnv(gym.Env):
    metadata = {}

    def __init__(self,
                 screen_res: int = 64,
                 visualize: bool = False,
                 realtime: bool = False,
                 replay_dir: str = None,
                 save_replay_episodes: int = 0):
        super().__init__()

        self._env = sc2_env.SC2Env(
            map_name="TwoBridgeMap_V2_Base",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot  (sc2_env.Race.terran,
                                   sc2_env.Difficulty.easy)],
            step_mul=STEP_MUL,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=SCR_RES,
                feature_dimensions=features.Dimensions(screen=SCR_RES, minimap=SCR_RES)),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes
        )

        # 14 choices per marine  → flat 5-length vector
        self.action_space = spaces.MultiDiscrete([14] * 5)

        self.observation_space = spaces.Dict({
            "screen":  spaces.Box(0, 255, (SCR_CH,  SCR_RES, SCR_RES), np.uint8),
            "minimap": spaces.Box(0, 255, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
            "vector":  spaces.Box(0.0, np.inf, (55,), np.float32),
        })

        # caches for decoding & reward
        self._my_tags     = np.zeros(5, np.int64)
        self._enemy_tags  = np.zeros(5, np.int64)
        self._enemy_alive = np.zeros(5, bool)
        self._fx = self._fy = np.zeros(5, np.float32)

        self._step_ctr          = 0
        self._prev_beacon_dist  = None
        self._prev_enemy_alive  = np.zeros(5, bool)
        self._prev_friend_alive = np.zeros(5, bool)

        # instrumentation caches (same schema as NSF)
        self._last_reward_components = {
            "nav_r": 0.0,
            "combat_r": 0.0,
            "term_r": 0.0,
            "friend_hp": 0.0,
            "enemy_hp": 0.0,
            "nav_dist": 0.0,
            "enemy_alive": 0.0,
        }
        self._last_unit_metrics = {"friend": {}, "enemy": {}}

    # ───────────── gym API ─────────────────────────────────────────────
    def close(self):
        self._env.close()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0

        self._prev_beacon_dist = None
        self._prev_enemy_alive[:]  = False
        self._prev_friend_alive[:] = False

        self._last_reward_components = {
            "nav_r": 0.0,
            "combat_r": 0.0,
            "term_r": 0.0,
            "friend_hp": 0.0,
            "enemy_hp": 0.0,
            "nav_dist": 0.0,
            "enemy_alive": 0.0,
        }
        self._last_unit_metrics = {"friend": {}, "enemy": {}}

        ts = self._env.reset()[0]
        return self._build_obs(ts), {}

    def step(self, act_vec):
        cmds = self._translate_actions(np.asarray(act_vec, np.int64))
        ts   = self._env.step(cmds)[0]
        obs  = self._build_obs(ts)

        # Built-in termination (Galaxy triggers, surrender, etc.)
        # Map into our canonical labels
        if ts.last():
            pr = getattr(ts.observation, "player_result", None)
            if pr and len(pr) > 0:
                r0 = pr[0].result
                if r0 == 1:
                    res = "combat_win"
                elif r0 == 2:
                    res = "combat_loss"
                else:
                    res = "tie"
            else:
                res = "tie"

            reward = self._shape_reward(obs["vector"], done=True, res=res)
            info = {"result": res, "rew": self.get_reward_components()}
            return obs, float(reward), True, False, info

        # Custom termination logic
        vec = obs["vector"]
        friend_alive_ct = int((vec[2:25:5] > 0).sum())
        enemy_alive_ct  = int(vec[54])

        no_friend  = (friend_alive_ct == 0)
        no_enemy   = (enemy_alive_ct == 0)
        beacon_win = (vec[52] < BEACON_RADIUS)

        res = None
        if beacon_win:
            res = "nav_win"
        elif no_enemy and no_friend:
            res = "tie"
        elif no_enemy:
            res = "combat_win"
        elif no_friend:
            res = "combat_loss"

        if self._step_ctr >= MAX_STEPS and res is None:
            res = "timeout_loss"

        done = (res is not None)
        reward = self._shape_reward(vec, done=done, res=res)
        info = {"result": res, "rew": self.get_reward_components()}
        return obs, float(reward), done, False, info

    # ───────────── helper functions ───────────────────────────────────
    def _translate_actions(self, a_vec):
        cmds = []
        for slot, a in enumerate(a_vec):
            tag = int(self._my_tags[slot])
            if tag == 0 or a == 0:
                continue

            if 1 <= a <= 8:  # move
                dx, dy = MOVE_DIRS[a - 1]
                x = float(np.clip(self._fx[slot] + dx, 0, SCR_RES - 1))
                y = float(np.clip(self._fy[slot] + dy, 0, SCR_RES - 1))
                cmds.append(RAW.Move_pt("now", tag, (x, y)))
            else:            # attack idx
                ei = a - 9
                if 0 <= ei < 5 and self._enemy_alive[ei]:
                    cmds.append(RAW.Attack_unit("now", tag, int(self._enemy_tags[ei])))

        return cmds or [RAW.no_op()]

    def _build_obs(self, ts):
        ob   = ts.observation
        ru   = ob.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        ene  = sorted([u for u in ru if u.owner == 2], key=lambda u: u.tag)
        bea  = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)
        bx, by = (bea.x, bea.y) if bea is not None else (-1., -1.)

        self._my_tags[:]     = 0
        self._enemy_tags[:]  = 0
        self._enemy_alive[:] = False
        self._fx[:] = self._fy[:] = 0.0

        for i, u in enumerate(fri[:5]):
            self._my_tags[i] = u.tag
            self._fx[i], self._fy[i] = u.x, u.y
        for i, u in enumerate(ene[:5]):
            self._enemy_tags[i]  = u.tag
            self._enemy_alive[i] = (u.health > 0)

        vec = np.zeros(55, np.float32)
        for i, u in enumerate(fri[:5]):
            vec[i*5:(i+1)*5] = (u.x, u.y, u.health, u.weapon_cooldown, 1.0)
        for i, u in enumerate(ene[:5]):
            vec[25+i*5:25+(i+1)*5] = (u.x, u.y, u.health, u.weapon_cooldown, float(u.health > 0))

        vec[50:52] = (bx, by)
        if len(fri) > 0 and bea is not None:
            vec[52] = np.hypot(fri[0].x - bx, fri[0].y - by)
        else:
            vec[52] = 128.0

        vec[53] = ob.game_loop[0] / 16.0
        vec[54] = float(self._enemy_alive.sum())

        self._step_ctr += 1
        return {
            "screen":  np.asarray(ob.feature_screen,  np.uint8),
            "minimap": np.asarray(ob.feature_minimap, np.uint8),
            "vector":  vec
        }

    # ───── reward shaping + instrumentation ─────────────────────────────
    def _shape_reward(self, vec, done: bool, res: str):
        """
        - combat_r: kill/loss delta (alive-count deltas)
        - nav_r: beacon distance delta
        - term_r: terminal bonus/penalty
        """
        enemy_alive_ct  = float(vec[54])
        friend_alive_ct = float((vec[2:25:5] > 0).sum())

        prev_enemy_ct  = float(self._prev_enemy_alive.sum())
        prev_friend_ct = float(self._prev_friend_alive.sum())

        # (a) kill/loss delta
        combat_r = (prev_enemy_ct - enemy_alive_ct) - (prev_friend_ct - friend_alive_ct)

        # (b) beacon distance delta
        d_now = float(vec[52])
        nav_r = 0.0
        if self._prev_beacon_dist is not None:
            nav_r = float(self._prev_beacon_dist - d_now)
        self._prev_beacon_dist = d_now

        # (c) terminal shaping
        term_r = 0.0
        if done:
            if res in ("nav_win", "combat_win"):
                term_r = TERM_WIN_BONUS
            elif res in ("combat_loss", "timeout_loss"):
                term_r = TERM_LOSS_PENALTY
            elif res == "tie":
                term_r = TERM_TIE_BONUS

        # Update caches AFTER computing deltas
        self._prev_enemy_alive  = self._enemy_alive.copy()
        self._prev_friend_alive = (vec[2:25:5] > 0).astype(bool)

        # HP sums (friend HP at indices 2,7,12,17,22; enemy HP at 27,32,37,42,47)
        friend_hp_sum = float(vec[2:25:5].sum())
        enemy_hp_sum  = float(vec[27:50:5].sum())

        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": friend_hp_sum,
            "enemy_hp": enemy_hp_sum,
            "nav_dist": float(d_now),
            "enemy_alive": float(enemy_alive_ct),
        }

        # Per-unit metrics keyed by tags (hp only)
        friend_dict = {}
        for i in range(5):
            tag = int(self._my_tags[i])
            if tag != 0:
                friend_dict[tag] = {"hp": float(vec[i*5 + 2])}

        enemy_dict = {}
        for j in range(5):
            etag = int(self._enemy_tags[j])
            if etag != 0:
                enemy_dict[etag] = {"hp": float(vec[25 + j*5 + 2])}

        self._last_unit_metrics = {"friend": friend_dict, "enemy": enemy_dict}

        return float(combat_r + nav_r + term_r)

    # ───────────────────────── instrumentation getters ─────────────────────────
    def get_reward_components(self):
        return getattr(self, "_last_reward_components", {})

    def get_friendly_tags(self):
        return self._my_tags.copy()

    def get_unit_metrics(self):
        return getattr(self, "_last_unit_metrics", {"friend": {}, "enemy": {}})

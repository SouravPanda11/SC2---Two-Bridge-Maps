import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ────────────────────── configuration ──────────────────────────
N_FRIEND = 5

# index helpers for the compact vector  ─────────────────────────
# friend features: [x, y, hp, weapon_cd, alive] × N_FRIEND
VEC_FRIEND   = 0
VEC_BXY      = VEC_FRIEND + N_FRIEND * 5   # 2 × float32 (beacon x,y)
VEC_DIST     = VEC_BXY    + 2             # 1 × float32 (min dist any friend→beacon)
VEC_TIME     = VEC_DIST   + 1             # 1 × float32 (game time in sec)
VEC_FCOUNT   = VEC_TIME   + 1             # 1 × float32 (#alive friendlies)
VEC_SIZE     = VEC_FCOUNT + 1

# ───────────────────── Map registration ───────────────────────
class TwoBridgeMap_V2_Base_Beacon(lib.Map):
    name      = "TwoBridgeMap_V2_Base_Beacon"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_V2_Base_Beacon.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V2_Base_Beacon", None)
lib.get_maps()["TwoBridgeMap_V2_Base_Beacon"] = TwoBridgeMap_V2_Base_Beacon()

# ───────────────────────── constants ───────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW            = actions.RAW_FUNCTIONS
BEACON_TYPE_ID = 317
BEACON_RADIUS  = 2.0

STEP_MUL       = 8
FIVE_MIN_LOOPS = 5*60*16
MAX_STEPS      = FIVE_MIN_LOOPS // STEP_MUL
STEP_PIX       = 2

SCR_RES        = 64
SCR_CH         = len(features.SCREEN_FEATURES)    # 17
MINI_CH        = len(features.MINIMAP_FEATURES)   # 7

MOVE_DIRS = [
    ( 0, -STEP_PIX), ( 0,  STEP_PIX), (-STEP_PIX, 0), ( STEP_PIX, 0),
    ( STEP_PIX,-STEP_PIX), (-STEP_PIX,-STEP_PIX),
    ( STEP_PIX, STEP_PIX), (-STEP_PIX, STEP_PIX)
]

# ───────────────────── navigation reward constants ─────────────
NAV_WIN_BONUS       = 25.0
NAV_TIMEOUT_PENALTY = -15.0

# ─────────────────────── environment ───────────────────────────
class TwoBridgeEnv(gym.Env):
    """
    Two-Bridge V2_Base_Beacon (NAVIGATION ONLY).
    5 friendly marines must navigate to the beacon.

    Action space = {verb, who-mask, direction}
      - verb: 0 noop | 1 move
      - who:  5-bit mask over marines
      - direction: 0 unused | 1-8 compass
    """
    metadata = {}

    # -------------- Gym spaces ---------------------------------
    action_space = spaces.Dict({
        "verb":      spaces.Discrete(2),            # 0 noop | 1 move
        "who":       spaces.MultiBinary(N_FRIEND),  # 5 marines
        "direction": spaces.Discrete(9),            # 0 unused | 1-8 compass
    })

    observation_space = spaces.Dict({
        "screen":  spaces.Box(0, 255, (SCR_CH,  SCR_RES, SCR_RES), np.uint8),
        "minimap": spaces.Box(0, 255, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
        "vector":  spaces.Box(0.0, np.inf, (VEC_SIZE,), np.float32),

        "action_mask": spaces.Dict({
            "verb":      spaces.MultiBinary(2),
            "who":       spaces.MultiBinary(N_FRIEND),
            "direction": spaces.MultiBinary(9),
        })
    })

    # -------------- ctor / close -------------------------------
    def __init__(self,
                 screen_res: int = 64,
                 visualize: bool = False,
                 realtime: bool = False,
                 replay_dir: str = None,
                 save_replay_episodes: int = 0):
        super().__init__()

        self._env = sc2_env.SC2Env(
            map_name="TwoBridgeMap_V2_Base_Beacon",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot  (sc2_env.Race.terran,
                                   sc2_env.Difficulty.easy)],
            step_mul=STEP_MUL,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=SCR_RES,
                feature_dimensions=features.Dimensions(
                    screen=SCR_RES, minimap=SCR_RES)),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes)

        # caches ────────────────────────────────────────────────
        self._my_tags   = np.zeros(N_FRIEND, np.int64)
        self._fx = self._fy = np.zeros(N_FRIEND, np.float32)

        self._step_ctr          = 0
        self._prev_beacon_dists = None          # shape (N_FRIEND,)
        self._last_reward_components = {}

    def close(self):
        self._env.close()

    # -------------- Gym API ------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0
        self._prev_beacon_dists = None

        ts = self._env.reset()[0]
        return self._build_obs(ts), {}

    def step(self, action):
        cmds = self._translate_actions(action)
        ts   = self._env.step(cmds)[0]
        obs  = self._build_obs(ts)

        vec        = obs["vector"]
        beacon_win = vec[VEC_DIST] < BEACON_RADIUS
        timeout    = self._step_ctr >= MAX_STEPS

        done = bool(ts.last() or beacon_win or timeout)

        if done:
            if beacon_win:
                res = "nav_win"
            elif timeout:
                res = "timeout_loss"
            else:
                # SC2 ended by itself (e.g., scenario script) → treat as timeout
                res = "timeout_loss"
        else:
            res = None

        reward = self._shape_reward(vec, done, res)
        info   = {
            "result": res,
            "rew": self.get_reward_components()
        }
        return obs, reward, done, False, info

    # -------------- translate actions --------------------------
    def _translate_actions(self, act):
        verb   = int(act["verb"])
        who    = act["who"].astype(bool)
        dir_id = int(act["direction"])

        tags = [int(t) for t, b in zip(self._my_tags, who) if b]

        # MOVE
        if verb == 1 and tags and 1 <= dir_id <= 8:
            dx, dy = MOVE_DIRS[dir_id - 1]
            cx = np.mean(self._fx[who]) + dx
            cy = np.mean(self._fy[who]) + dy
            pt = (float(np.clip(cx, 0, SCR_RES - 1)),
                  float(np.clip(cy, 0, SCR_RES - 1)))
            return [RAW.Move_pt("now", tags, pt)]

        # anything else → no-op
        return [RAW.no_op()]

    # -------------- build observation + FULL MASKS --------------
    def _build_obs(self, ts):
        ob   = ts.observation
        ru   = ob.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        bea  = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)

        # beacon coords
        if bea is not None:
            bx, by = bea.x, bea.y
        else:
            bx, by = -1., -1.

        self._my_tags[:] = 0
        self._fx[:] = self._fy[:] = 0

        for i, u in enumerate(fri[:N_FRIEND]):
            self._my_tags[i] = u.tag
            self._fx[i], self._fy[i] = u.x, u.y

        vec = np.zeros(VEC_SIZE, np.float32)

        # friend features
        for i, u in enumerate(fri[:N_FRIEND]):
            vec[i*5:(i+1)*5] = (
                u.x,
                u.y,
                u.health,
                u.weapon_cooldown,
                1.0,   # alive flag (no combat, but keeps layout consistent)
            )

        # beacon / misc
        vec[VEC_BXY : VEC_BXY+2] = (bx, by)

        if fri and bea is not None:
            fx = vec[0 : N_FRIEND*5 : 5]
            fy = vec[1 : N_FRIEND*5 : 5]
            fhp = vec[2 : N_FRIEND*5 : 5]
            f_alive = fhp > 0
            if (bx >= 0) and (by >= 0) and f_alive.any():
                dists = np.hypot(fx - bx, fy - by)
                vec[VEC_DIST] = float(dists[f_alive].min())
            else:
                vec[VEC_DIST] = 128.0
        else:
            vec[VEC_DIST] = 128.0

        vec[VEC_TIME]   = ob.game_loop[0] / 16.0
        vec[VEC_FCOUNT] = (vec[2 : N_FRIEND*5 : 5] > 0).sum()

        # ───────────── FULL ACTION MASKS ─────────────────────────
        fhp = vec[2 : N_FRIEND*5 : 5]
        who_mask = (fhp > 0).astype(np.int8)
        any_friend_alive = int(who_mask.any())

        # verb: noop always ok; move only if some friend alive
        verb_mask = np.zeros(2, np.int8)
        verb_mask[0] = 1
        verb_mask[1] = any_friend_alive

        # direction: 0 unused if moving impossible; else 1..8 allowed
        direction_mask = np.zeros(9, np.int8)
        if verb_mask[1]:             # move is state-feasible
            direction_mask[1:] = 1   # allow all 8 compass directions
        else:
            direction_mask[0] = 1    # only "unused" slot allowed

        action_mask = {
            "verb":      verb_mask,
            "who":       who_mask,
            "direction": direction_mask,
        }

        self._step_ctr += 1
        return {
            "screen":      np.asarray(ob.feature_screen,  np.uint8),
            "minimap":     np.asarray(ob.feature_minimap, np.uint8),
            "vector":      vec,
            "action_mask": action_mask
        }

    # -------------- shaped reward (NAV ONLY) -------------------
    def _shape_reward(self, vec, done, res):
        """
        Navigation-only reward:
          - nav_r: Δ-distance to beacon (per-step shaping)
          - term_r: terminal bonus/penalty for nav_win / timeout_loss
        """
        fx  = vec[0 : N_FRIEND*5 : 5]
        fy  = vec[1 : N_FRIEND*5 : 5]
        fhp = vec[2 : N_FRIEND*5 : 5]
        f_alive = fhp > 0

        bx, by = vec[VEC_BXY : VEC_BXY+2]

        # FIRST FRAME GUARD
        if self._prev_beacon_dists is None:
            if (bx >= 0) and (by >= 0):
                self._prev_beacon_dists = np.hypot(fx - bx, fy - by)
            else:
                self._prev_beacon_dists = None

            nav_dist = float(
                np.mean(np.hypot(fx - bx, fy - by))
            ) if f_alive.any() and (bx >= 0) and (by >= 0) else 0.0

            self._last_reward_components = {
                "nav_r":     0.0,
                "term_r":    0.0,
                "friend_hp": float(fhp.sum()),
                "nav_dist":  nav_dist,
            }
            return 0.0

        # NAVIGATION SHAPING
        nav_r = 0.0
        if (bx >= 0) and (by >= 0):
            beacon_dists = np.hypot(fx - bx, fy - by)
            if self._prev_beacon_dists is not None:
                diff = (self._prev_beacon_dists - beacon_dists)[f_alive]
                nav_r = diff.mean() if diff.size > 0 else 0.0
            self._prev_beacon_dists = beacon_dists

        # TERMINAL BONUS
        if done:
            if   res == "nav_win":
                term_r = NAV_WIN_BONUS
            elif res == "timeout_loss":
                term_r = NAV_TIMEOUT_PENALTY
            else:
                term_r = 0.0
        else:
            term_r = 0.0

        nav_dist = float(
            np.mean(np.hypot(fx - bx, fy - by))
        ) if f_alive.any() and (bx >= 0) and (by >= 0) else 0.0

        self._last_reward_components = {
            "nav_r":     float(nav_r),
            "term_r":    float(term_r),
            "friend_hp": float(fhp.sum()),
            "nav_dist":  nav_dist,
        }

        return float(nav_r + term_r)

    def get_reward_components(self):
        return getattr(self, "_last_reward_components", {})

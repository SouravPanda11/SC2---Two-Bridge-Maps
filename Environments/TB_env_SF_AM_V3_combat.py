import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ────────────────────── configuration ──────────────────────────
N_FRIEND = 5           # <-- keep 5 marines on our side
N_ENEMY  = 3           # <-- only 3 enemy marines now

# index helpers for the compact vector  ─────────────────────────
VEC_FRIEND  = 0
VEC_ENEMY   = VEC_FRIEND  + N_FRIEND*5
VEC_BXY     = VEC_ENEMY   + N_ENEMY *5          # 2 × float32
VEC_DIST    = VEC_BXY     + 2                   # 1 × float32
VEC_TIME    = VEC_DIST    + 1                   # 1 × float32
VEC_ECOUNT  = VEC_TIME    + 1                   # 1 × float32
VEC_SIZE    = VEC_ECOUNT  + 1

# ───────────────────── Map registration ───────────────────────
class TwoBridgeMap_V3_combat(lib.Map):
    name      = "TwoBridgeMap_V3_combat"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_V3_combat.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V3_combat", None)
lib.get_maps()["TwoBridgeMap_V3_combat"] = TwoBridgeMap_V3_combat()

# ───────────────────────── constants ───────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW              = actions.RAW_FUNCTIONS
MARINE_HP        = 45
BEACON_TYPE_ID   = 317
BEACON_RADIUS    = 2.0

STEP_MUL         = 8
FIVE_MIN_LOOPS   = 5*60*16
MAX_STEPS        = FIVE_MIN_LOOPS // STEP_MUL
STEP_PIX         = 2

SCR_RES          = 64
SCR_CH           = len(features.SCREEN_FEATURES)    # 17
MINI_CH          = len(features.MINIMAP_FEATURES)   # 7

MOVE_DIRS = [
    ( 0, -STEP_PIX), ( 0,  STEP_PIX), (-STEP_PIX, 0), ( STEP_PIX, 0),
    ( STEP_PIX,-STEP_PIX), (-STEP_PIX,-STEP_PIX),
    ( STEP_PIX, STEP_PIX), (-STEP_PIX, STEP_PIX)
]

# ─────────────────────── environment ───────────────────────────
class TwoBridgeEnv(gym.Env):
    """
    5 v 3 Two-Bridge – navigation & combat.
    Action space = {verb, who-mask, direction, enemy_idx}
    """
    metadata = {}

    # -------------- Gym spaces ---------------------------------
    action_space = spaces.Dict({
        "verb":      spaces.Discrete(3),            # 0 noop | 1 move | 2 atk
        "who":       spaces.MultiBinary(N_FRIEND),  # 5 marines
        "direction": spaces.Discrete(9),            # 0 unused | 1-8 compass
        "enemy_idx": spaces.Discrete(N_ENEMY + 1)   # 0 none | 1-3 enemy slot
    })

    observation_space = spaces.Dict({
        "screen":      spaces.Box(0, 255, (SCR_CH,  SCR_RES, SCR_RES), np.uint8),
        "minimap":     spaces.Box(0, 255, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
        "vector":      spaces.Box(0.0, np.inf, (VEC_SIZE,), np.float32),
        "action_mask": spaces.MultiBinary(3)        # verb-level mask only
    })

    # -------------- ctor / close -------------------------------
    def __init__(self, visualize: bool = False):
        super().__init__()
        self._env = sc2_env.SC2Env(
            map_name="TwoBridgeMap_V3_combat",
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
            visualize=visualize)

        # caches ────────────────────────────────────────────────
        self._my_tags     = np.zeros(N_FRIEND, np.int64)
        self._enemy_tags  = np.zeros(N_ENEMY, np.int64)
        self._enemy_alive = np.zeros(N_ENEMY, bool)
        self._fx = self._fy = np.zeros(N_FRIEND, np.float32)

        self._step_ctr          = 0
        self._prev_beacon_dist  = None
        self._prev_enemy_alive  = np.zeros(N_ENEMY, bool)
        self._prev_friend_alive = np.zeros(N_FRIEND, bool)

    def close(self): self._env.close()

    # -------------- Gym API ------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0
        self._prev_enemy_alive[:]  = False
        self._prev_friend_alive[:] = False
        self._prev_beacon_dist     = None
        ts = self._env.reset()[0]
        return self._build_obs(ts), {}

    def step(self, action):
        cmds = self._translate_actions(action)
        ts   = self._env.step(cmds)[0]
        obs  = self._build_obs(ts)

        # built-in win/loss
        if ts.last():
            res = ("victory" if ts.reward > 0 else
                   "defeat"  if ts.reward < 0 else
                   "tie")
            return obs, float(ts.reward), True, False, {"result": res}

        # custom termination
        friend_alive = (obs["vector"][2 : N_FRIEND*5 : 5] > 0).sum()
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
        return obs, reward, done, False, info

    # -------------- translate actions --------------------------
    def _translate_actions(self, act):
        verb      = int(act["verb"])
        who_bits  = act["who"].astype(bool)
        dir_id    = int(act["direction"])
        enemy_idx = int(act["enemy_idx"]) - 1   # shift so 0-based

        tags = [int(t) for t, b in zip(self._my_tags, who_bits) if b]

        # MOVE
        if verb == 1 and tags and 1 <= dir_id <= 8:
            dx, dy = MOVE_DIRS[dir_id-1]
            cx = np.mean(self._fx[who_bits]) + dx
            cy = np.mean(self._fy[who_bits]) + dy
            pt = (float(np.clip(cx, 0, SCR_RES-1)),
                  float(np.clip(cy, 0, SCR_RES-1)))
            return [RAW.Move_pt("now", tags, pt)]

        # ATTACK
        if (verb == 2 and tags and
                0 <= enemy_idx < N_ENEMY and self._enemy_alive[enemy_idx]):
            return [RAW.Attack_unit("now", tags,
                                    int(self._enemy_tags[enemy_idx]))]

        return [RAW.no_op()]

    # -------------- build observation --------------------------
    def _build_obs(self, ts):
        ob   = ts.observation
        ru   = ob.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        ene  = sorted([u for u in ru if u.owner == 2], key=lambda u: u.tag)
        bea  = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)

        # coords
        bx, by = (-1., -1.) if bea is None else (bea.x, bea.y)

        self._my_tags[:]     = 0
        self._enemy_tags[:]  = 0
        self._enemy_alive[:] = False
        self._fx[:] = self._fy[:] = 0

        for i, u in enumerate(fri[:N_FRIEND]):
            self._my_tags[i] = u.tag
            self._fx[i], self._fy[i] = u.x, u.y
        for i, u in enumerate(ene[:N_ENEMY]):
            self._enemy_tags[i]  = u.tag
            self._enemy_alive[i] = u.health > 0

        vec = np.zeros(VEC_SIZE, np.float32)
        # friend features
        for i, u in enumerate(fri[:N_FRIEND]):
            vec[i*5:(i+1)*5] = (u.x, u.y, u.health, u.weapon_cooldown, 1.)
        # enemy features
        base = VEC_ENEMY
        for i, u in enumerate(ene[:N_ENEMY]):
            vec[base+i*5 : base+(i+1)*5] = (
                u.x, u.y, u.health, u.weapon_cooldown, float(u.health > 0))

        # beacon / misc
        vec[VEC_BXY : VEC_BXY+2] = (bx, by)
        if fri and bea is not None:
            vec[VEC_DIST] = np.hypot(fri[0].x - bx, fri[0].y - by)
        else:
            vec[VEC_DIST] = 128.0
        vec[VEC_TIME]   = ob.game_loop[0] / 16.0
        vec[VEC_ECOUNT] = self._enemy_alive.sum()

        # verb-level mask
        mask = np.ones(3, np.int8)
        mask[1] = int((vec[2 : N_FRIEND*5 : 5] > 0).any())  # MOVE
        mask[2] = int(vec[VEC_ECOUNT] > 0)                  # ATTACK

        self._step_ctr += 1
        return {
            "screen":      np.asarray(ob.feature_screen,  np.uint8),
            "minimap":     np.asarray(ob.feature_minimap, np.uint8),
            "vector":      vec,
            "action_mask": mask
        }

    # -------------- shaped reward ------------------------------
    def _shape_reward(self, vec, done, res):
        enemy_now  = self._enemy_alive
        friend_now = (vec[2 : N_FRIEND*5 : 5] > 0)

        r  = (self._prev_enemy_alive.sum()  - enemy_now.sum()) \
           - (self._prev_friend_alive.sum() - friend_now.sum())

        d_now = vec[VEC_DIST]
        if self._prev_beacon_dist is not None:
            r += (self._prev_beacon_dist - d_now)
        self._prev_beacon_dist = d_now

        if done:
            if "win" in res:    r += 10.
            elif "loss" in res: r -= 10.

        self._prev_enemy_alive  = enemy_now.copy()
        self._prev_friend_alive = friend_now.copy()
        return float(r)

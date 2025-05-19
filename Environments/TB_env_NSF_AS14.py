# two_bridge_env.py  –  5 v 5 marines + beacon capture + 5-min limit
# Hybrid action space:
#   Move in 8 compass directions  |  Attack enemy-idx 0-4
#   (0=no-op → 14 discrete choices per marine)
import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ─────────────────────────────────────────  map registration  ───────────
class TwoBridgeMap_V2_Base(lib.Map):
    name      = "TwoBridgeMap_V2_Base"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_V2_Base.SC2Map"
    players   = 2                     # agent vs bot

lib.get_maps().pop("TwoBridgeMap_V2_Base", None)
lib.get_maps()["TwoBridgeMap_V2_Base"] = TwoBridgeMap_V2_Base()

# ────────────────────────────────────────────── constants  ──────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW             = actions.RAW_FUNCTIONS
MARINE_HP       = 45
BEACON_TYPE_ID  = 317
BEACON_RADIUS = 2.0           # ←  match the trigger radius
MAP_NAME        = "TwoBridgeMap_V2_Base"

FIVE_MIN_LOOPS  = 5 * 60 * 16         # 300 s × 16 loops/s = 4800
STEP_MUL        = 8                   # env.step advances 8 loops
MAX_STEPS       = FIVE_MIN_LOOPS // STEP_MUL     # 600 env steps
STEP_PIX        = 2                   # length of one move action (raw units)

# ────────────────────────────────────────────── environment  ────────────
class TwoBridgeEnv(gym.Env):
    """
    Observation (55 floats):
        friend[5] · enemy[5] · beacon-x,y · dist-beacon · game-time · #enemy-alive
    Action (MultiDiscrete [14]*5):
        0 no-op |
        1-8  move (N,S,W,E, NE,NW,SE,SW) by STEP_PIX |
        9-13 attack enemy index 0-4
    Reward:
        kill-loss delta  +  beacon-distance delta  ±10 on terminal win/loss
    """
    metadata = {}

    # ───── init ─────────────────────────────────────────────────────────
    def __init__(self, screen_res: int = 64, visualize: bool = False):
        super().__init__()
        self.screen = screen_res

        self._env = sc2_env.SC2Env(
            map_name  = MAP_NAME,
            players   = [sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot  (sc2_env.Race.terran,
                                       sc2_env.Difficulty.easy)],
            agent_interface_format = sc2_env.AgentInterfaceFormat(
                action_space      = actions.ActionSpace.RAW,
                use_raw_units     = True,
                feature_dimensions=features.Dimensions(screen=screen_res,
                                                       minimap=screen_res)),
            step_mul  = STEP_MUL,
            visualize = visualize)

        # 14 choices per marine
        self.action_space      = spaces.MultiDiscrete([14]*5)
        self.observation_space = spaces.Box(
            0.0,
            np.array([screen_res, screen_res, MARINE_HP, 15.0, 1.0]*10 +
                     [screen_res, screen_res, screen_res*2, 1e4, 5],
                     dtype=np.float32),
            dtype=np.float32)

        # per-episode caches ------------------------------------------------
        self._my_tags     = np.zeros(5, np.int64)
        self._enemy_tags  = np.zeros(5, np.int64)
        self._alive       = np.zeros(5, np.bool_)
        self._fx = self._fy = np.zeros(5, np.float32)
        self._step_ctr    = 0
        self._prev_beacon_dist = None

    # ───── Gym API ───────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0
        ts  = self._env.reset()[0]

        self._prev_enemy_alive  = np.zeros(5, np.bool_)
        self._prev_friend_alive = np.zeros(5, np.bool_)
        self._prev_beacon_dist  = None
        return self._build_obs(ts), {}

    def step(self, action):
        self._step_ctr += 1
        ts  = self._env.step(self._decode(action))[0]

        # built-in SC2 termination (Galaxy triggers / surrender / etc.)
        if ts.last():
            obs   = self._build_obs(ts)
            pr    = getattr(ts.observation, "player_result", [])
            lbl   = ("victory" if pr and pr[0].result == 1
                     else "defeat"  if pr and pr[0].result == 2
                     else "tie")
            info  = {"result": f"galaxy_{lbl}"}
            return obs, float(ts.reward), True, False, info

        obs  = self._build_obs(ts)
        info = {"result": None}

        # >>> NEW: how many of our marines are still alive?
        friend_alive = (obs[2:25:5] > 0).sum()   # hp > 0 in the 5 friend slots
        no_friend    = friend_alive == 0
        no_enemy     = obs[54] == 0
        beacon_win   = obs[52] < BEACON_RADIUS
        # ------------------------------------

        # decide termination --------------------------------------
        if beacon_win:
            info["result"] = "nav_win"
        elif no_enemy and no_friend:
            info["result"] = "tie"
        elif no_enemy:
            info["result"] = "combat_win"
        elif no_friend:
            info["result"] = "combat_loss"
            
        # episode time limit
        if self._step_ctr >= MAX_STEPS and info["result"] is None:
            info["result"] = "timeout_loss"

        done = info["result"] is not None

        # ── reward shaping ──────────────────────────────────────────────
        #   (a) kill‑loss delta
        reward  = (self._prev_enemy_alive.sum() - obs[54]) \
                - (self._prev_friend_alive.sum() - (obs[4::5] > 0).sum())

        #   (b) beacon distance delta
        d_now = obs[52]
        if self._prev_beacon_dist is not None:
            reward += (self._prev_beacon_dist - d_now)
        self._prev_beacon_dist = d_now

        #   (c) terminal bonus / penalty
        # +10 for any victory, –10 for any defeat (including timeout),
        #  0 for tie
        if done:
            if "win"  in info["result"]:
                reward += 10.0
            elif "loss" in info["result"]:      # combat_loss or timeout_loss
                reward -= 10.0
            # "tie" → no adjustment

        # update caches
        self._prev_enemy_alive  = self._alive.copy()
        self._prev_friend_alive = (obs[4::5] > 0).astype(bool)[:5]

        return obs, float(reward), done, False, info

    def close(self):       self._env.close()

    # ───── observation builder ─────────────────────────────────────────
    def _build_obs(self, ts):
        ru = ts.observation.raw_units
        ff = [u for u in ru if u.owner == 1]
        ee = [u for u in ru if u.owner == 2]
        ff.sort(key=lambda u: u.tag)
        ee.sort(key=lambda u: u.tag)

        bea = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)
        bx, by = (bea.x, bea.y) if bea is not None else (-1., -1.)

        self._my_tags[:] = 0;  self._enemy_tags[:] = 0;  self._alive[:] = False
        for i,u in enumerate(ff): self._my_tags[i]    = u.tag
        for i,u in enumerate(ee): self._enemy_tags[i] = u.tag
        for i,u in enumerate(ee): self._alive[i]      = (u.health > 0)
        self._fx[:len(ff)] = [u.x for u in ff]
        self._fy[:len(ff)] = [u.y for u in ff]

        vec = np.zeros(55, np.float32)
        for i,u in enumerate(ff):
            vec[i*5:(i+1)*5] = (u.x, u.y, u.health,
                                u.weapon_cooldown, 1.0)
        for i,u in enumerate(ee):
            vec[25+i*5:25+(i+1)*5] = (u.x, u.y, u.health,
                                      u.weapon_cooldown, float(u.health > 0))

        vec[50:52] = (bx, by)
        if len(ff) > 0 and bea is not None:
            vec[52] = np.hypot(ff[0].x - bx, ff[0].y - by)
        else:
            vec[52] = 128.0
        vec[53]    = ts.observation.game_loop[0] / 16.0
        vec[54]    = self._alive.sum()
        return vec

    # ───── RAW action decoder ──────────────────────────────────────────
    _DIRS = [  # (dx, dy) for choices 1-8
        ( 0, -STEP_PIX),      # N
        ( 0,  STEP_PIX),      # S
        (-STEP_PIX, 0),       # W
        ( STEP_PIX, 0),       # E
        ( STEP_PIX, -STEP_PIX),   # NE
        (-STEP_PIX, -STEP_PIX),   # NW
        ( STEP_PIX,  STEP_PIX),   # SE
        (-STEP_PIX,  STEP_PIX)    # SW
    ]

    def _decode(self, a_vec):
        cmds = []
        for i, a in enumerate(a_vec):
            tag = int(self._my_tags[i])
            if tag == 0 or a == 0:
                continue

            if 1 <= a <= 8:                        # MOVE 8-dir
                dx, dy = self._DIRS[a-1]
                x = float(np.clip(self._fx[i] + dx, 0, self.screen-1))
                y = float(np.clip(self._fy[i] + dy, 0, self.screen-1))
                cmds.append(RAW.Move_pt("now", tag, (x, y)))

            else:                                  # ATTACK idx
                ei = a - 9                         # 0-4
                if 0 <= ei < 5 and self._alive[ei]:
                    cmds.append(RAW.Attack_unit("now", tag,
                                                int(self._enemy_tags[ei])))
        return cmds or [RAW.no_op()]

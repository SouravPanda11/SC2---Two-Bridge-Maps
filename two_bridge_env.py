# two_bridge_env.py  –  5 v 5 marines + beacon capture
import gymnasium as gym, numpy as np
from gymnasium import spaces
from pysc2.env   import sc2_env
from pysc2.lib   import actions, features
from pysc2.maps  import lib
from absl        import flags

# ─── register custom map ────────────────────────────────────────────────
class TwoBridgeMap_Same(lib.Map):
    name      = "TwoBridgeMap_Same"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_Same.SC2Map"
    players   = 2                 # 1 agent vs 1 easy bot

lib.get_maps().pop("TwoBridgeMap_Same", None)
lib.get_maps()["TwoBridgeMap_Same"] = TwoBridgeMap_Same()

# ─── constants ──────────────────────────────────────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW            = actions.RAW_FUNCTIONS
MARINE_HP      = 45
BEACON_TYPE_ID = 317
BEACON_RADIUS  = 0.9
MAP_NAME       = "TwoBridgeMap_Same"

# ─────────────────────────────────────────────────────────────────────────
class TwoBridgeEnv(gym.Env):
    metadata = {}

    def __init__(self, screen_res: int = 64, step_mul: int = 8,
                 visualize: bool = False):
        super().__init__()
        self.screen = screen_res

        self._env = sc2_env.SC2Env(
            map_name = MAP_NAME,
            players  = [ sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot  (sc2_env.Race.terran,
                                       sc2_env.Difficulty.easy) ],
            agent_interface_format = sc2_env.AgentInterfaceFormat(
                action_space      = actions.ActionSpace.RAW,
                use_raw_units     = True,
                feature_dimensions=features.Dimensions(screen=screen_res,
                                                       minimap=screen_res)),
            step_mul = step_mul, visualize = visualize)

        # 5 marines → each chooses one of 10 integers
        # 0 noop | 1-4 move NSEW | 5-9 attack enemy-idx 0-4
        self.action_space = spaces.MultiDiscrete([10] * 5)     # (5,)

        self.observation_space = spaces.Box(
            0.0,
            np.array([screen_res, screen_res, MARINE_HP, 15.0, 1.0] * 10 +
                     [screen_res, screen_res, screen_res * 2, 1e4, 5],
                     dtype=np.float32),
            dtype=np.float32)

        # caches
        self._alive_cache = np.ones (5, np.bool_)
        self._my_tags     = np.zeros(5, np.int64)
        self._enemy_tags  = np.zeros(5, np.int64)
        self._fx = self._fy = np.zeros(5, np.float32)   # friend positions

    # ─── Gym API ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()[0]

        # clear per-episode caches
        self._my_tags  .fill(0)
        self._enemy_tags.fill(0)
        self._alive_cache.fill(False)
        self._prev_enemy_alive  = np.zeros(5, np.bool_)
        self._prev_friend_alive = np.zeros(5, np.bool_)

        return self._build_obs(ts), {}

    def step(self, action: np.ndarray):
        ts = self._env.step(self._decode(action))[0]

        # — 1) Galaxy trigger ended the match?
        if ts.last():
            obs = self._build_obs(ts)
            return obs, float(ts.reward), True, False, {}

        obs = self._build_obs(ts)

        no_enemy  = obs[54] == 0
        no_friend = (obs[4::5] > 0).sum() == 0
        beacon_ok = obs[52] < BEACON_RADIUS
        done      = no_enemy or no_friend or beacon_ok

        reward = (self._prev_enemy_alive.sum() - obs[54]) - \
                 (self._prev_friend_alive.sum() -
                  (obs[4::5] > 0).sum())
        if done:
            reward += 10.0 if (beacon_ok or no_enemy) else -10.0

        self._prev_enemy_alive  = self._alive_cache.copy()
        self._prev_friend_alive = (obs[4::5] > 0).astype(np.bool_)[:5]

        return obs, float(reward), done, False, {}

    def close(self): self._env.close()

    # ─── helpers ─────────────────────────────────────────────────────────
    def _build_obs(self, ts):
        ru   = ts.observation.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        ene  = sorted([u for u in ru if u.owner == 2], key=lambda u: u.tag)
        beacons = [u for u in ru if u.unit_type == BEACON_TYPE_ID]
        beac = beacons[0] if len(beacons) > 0 else None
        bx, by = (beac.x, beac.y) if beac is not None else (-1., -1.)


        self._my_tags     = np.array([u.tag for u in fri] + [0]* (5-len(fri)))
        self._enemy_tags  = np.array([u.tag for u in ene]+ [0]* (5-len(ene)))
        self._alive_cache = np.array([u.health>0 for u in ene] +
                                     [False]*(5-len(ene)))

        self._fx[:len(fri)] = [u.x for u in fri]
        self._fy[:len(fri)] = [u.y for u in fri]

        vec = np.zeros(55, np.float32)
        for i,u in enumerate(fri):
            vec[i*5:(i+1)*5] = (u.x, u.y, u.health,
                                u.weapon_cooldown, 1.)

        for i,u in enumerate(ene):
            vec[25+i*5:25+(i+1)*5] = (u.x, u.y, u.health,
                                      u.weapon_cooldown, float(u.health>0))

        vec[50:52] = (bx, by)
        # avoid indexing fri[0] when no marines alive
        if len(fri) and beac is not None:
            vec[52] = np.hypot(fri[0].x - bx, fri[0].y - by)
        else:
            vec[52] = 128.0                       # far away sentinel
        vec[53] = ts.observation.game_loop[0] / 16.0
        vec[54] = self._alive_cache.sum()

        return vec

    # ─── action decoder ──────────────────────────────────────────────────
    def _decode(self, a_vec):
        cmds=[]
        for i,a in enumerate(a_vec):
            if i >= len(self._my_tags): break
            tag=int(self._my_tags[i])
            if tag==0 or a==0: continue          # dead or NO-OP

            if 1 <= a <= 4:                      # MOVE
                dx,dy = [(0,-2),(0,2),(-2,0),(2,0)][a-1]
                x = float(np.clip(self._fx[i]+dx,0,self.screen-1))
                y = float(np.clip(self._fy[i]+dy,0,self.screen-1))
                cmds.append(RAW.Move_pt("now", tag, (x,y)))
            else:                                # ATTACK
                eidx = a-5
                if eidx < len(self._enemy_tags) and self._alive_cache[eidx]:
                    cmds.append(RAW.Attack_unit("now", tag,
                                                int(self._enemy_tags[eidx])))

        return cmds or [RAW.no_op()]

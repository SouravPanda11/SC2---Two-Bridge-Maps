# two_bridge_env.py  – 5 v 5 marines + beacon capture + 5-min limit
# one file, two observation variants ("vector" | "full"), common 20-RAW action set
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete

from pysc2.env   import sc2_env
from pysc2.lib   import actions, features, point
from pysc2.maps  import lib
from absl        import flags

# ───────────────────────── map registration ────────────────────────────
class TwoBridgeMap_Same(lib.Map):
    name      = "TwoBridgeMap_Same"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_Same.SC2Map"
    players   = 2                    # agent vs bot

lib.get_maps().pop("TwoBridgeMap_Same", None)
lib.get_maps()["TwoBridgeMap_Same"] = TwoBridgeMap_Same()

# ─────────────────────────── constants ─────────────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW_FUNCS = actions.RAW_FUNCTIONS

# ---------- keep only no-arg OR single-point-target RAW functions ------
POINT_FUNCS = [f.id for f in RAW_FUNCS
               if (len(f.args) == 0) or
                  (len(f.args) == 1 and f.args[0].name.endswith("point"))]
POINT_FUNCS.sort()
FUNC_ID2IDX = {fid: i for i, fid in enumerate(POINT_FUNCS)}
N_FUNCS = len(POINT_FUNCS)        # ≈ 20

MARINE_HP      = 45
BEACON_TYPE_ID = 317
BEACON_RADIUS  = 2.0

FIVE_MIN_LOOPS = 5 * 60 * 16      # 300 s × 16 loops/s
STEP_MUL       = 8                # env.step advances 8 loops
MAX_STEPS      = FIVE_MIN_LOOPS // STEP_MUL
SCREEN_RES     = 64

SCR_CH  = len(features.SCREEN_FEATURES)    # 17
MINI_CH = len(features.MINIMAP_FEATURES)   # 7
MAX_UNITS = 20                             # generous padding

# ──────────────────────── Env definition ───────────────────────────────
class TwoBridgeEnv(gym.Env):
    """
    Parameters
    ----------
    obs_type : {"vector", "full"}, default "full"
        * "vector" – returns a 55-float ndarray
        * "full"   – returns {"screen","minimap","vector"} dict
    """
    metadata = {}

    def __init__(self,
                 obs_type : str = "full",
                 screen_res: int = SCREEN_RES,
                 visualize : bool = False):
        super().__init__()
        assert obs_type in {"vector", "full"}
        self.obs_type  = obs_type
        self.screen    = screen_res

        # ---------- launch SC2 ------------------------------------------------
        self._env = sc2_env.SC2Env(
            map_name = TwoBridgeMap_Same.name,
            players  = [sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot  (sc2_env.Race.terran,
                                      sc2_env.Difficulty.easy)],
            step_mul = STEP_MUL,
            visualize = visualize,
            agent_interface_format = sc2_env.AgentInterfaceFormat(
                action_space      = actions.ActionSpace.RAW,
                use_raw_units     = True,
                feature_dimensions=features.Dimensions(
                                        screen = screen_res,
                                        minimap= screen_res))
        )

        # ---------- action / observation space --------------------------------
        self.action_space = MultiDiscrete([N_FUNCS, screen_res, screen_res])

        hi_vec = np.array(
            [screen_res, screen_res, MARINE_HP, 15.0, 1.0]*10 +
            [screen_res, screen_res, screen_res*2, 1e4, 5],
            dtype=np.float32)

        if obs_type == "vector":
            # 55 state floats + 20 legality bits  →  75-length vector
            hi_vec_vec = np.concatenate([hi_vec, np.ones(N_FUNCS, dtype=np.float32)])
            self.observation_space = spaces.Box(0.0, hi_vec_vec, dtype=np.float32)
        else:
            self.observation_space = spaces.Dict({
                "screen":  spaces.Box(0, 255,
                                      (SCR_CH, screen_res, screen_res),
                                      np.uint8),
                "minimap": spaces.Box(0, 255,
                                      (MINI_CH, screen_res, screen_res),
                                      np.uint8),
                "vector":  spaces.Box(0.0, hi_vec, dtype=np.float32),
            })

        # ---------- per-episode caches ----------------------------------------
        self._my_tags    = np.zeros(5, np.int64)
        self._enemy_tags = np.zeros(5, np.int64)
        self._alive      = np.zeros(5, np.bool_)
        self._fx = self._fy = np.zeros(5, np.float32)
        self._step_ctr   = 0
        self._prev_beacon_dist  = None
        self._prev_enemy_alive  = np.zeros(5, np.bool_)
        self._prev_friend_alive = np.zeros(5, np.bool_)

    # ──────────────────── Gym API ───────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0
        ts  = self._env.reset()[0]

        self._prev_enemy_alive.fill(False)
        self._prev_friend_alive.fill(False)
        self._prev_beacon_dist = None
        return self._build_obs(ts), {}

    def step(self, action):
        func_id, x, y = action
        raw_id = POINT_FUNCS[int(func_id)]

        if raw_id not in self._avail_last:
            sc_act = RAW_FUNCS.no_op()
        else:
            raw_f = RAW_FUNCS[raw_id]
            sc_act = ( raw_f("now", point.Point2((x, y)))
                       if raw_f.args else raw_f() )

        ts  = self._env.step([sc_act])[0]
        self._step_ctr += 1
        obs = self._build_obs(ts)

        # ------- check termination conditions --------------------------------
        if ts.last():
            pr  = getattr(ts.observation, "player_result", [])
            res = ("victory" if pr and pr[0].result == 1
                   else "defeat" if pr and pr[0].result == 2
                   else "tie")
            return obs, float(ts.reward), True, False, {"result": res}

        friend_alive = (obs["vector"][2:25:5] > 0).sum() if self.obs_type=="full" \
                       else (obs[2:25:5] > 0).sum()
        no_friend  = friend_alive == 0
        no_enemy   = int(obs["vector"][-1] if self.obs_type=="full" else obs[-1]) == 0
        beacon_win = (obs["vector"][-3] if self.obs_type=="full" else obs[-3]) < BEACON_RADIUS

        info = {"result": None}
        if beacon_win:                 info["result"] = "nav_win"
        elif no_enemy and no_friend:   info["result"] = "tie"
        elif no_enemy:                 info["result"] = "combat_win"
        elif no_friend:                info["result"] = "combat_loss"
        elif self._step_ctr >= MAX_STEPS:
            info["result"] = "timeout_loss"

        done   = info["result"] is not None
        reward = self._shape_reward(obs, done, info["result"])

        return obs, float(reward), done, False, info

    def close(self): self._env.close()

    # ───────────────── helper functions ────────────────────────────────
    def _build_obs(self, ts):
        raw_obs = ts.observation
        screen  = np.asarray(raw_obs["feature_screen"],  np.uint8)
        minimap = np.asarray(raw_obs["feature_minimap"], np.uint8)

        # ---------- legality mask (always build; used differently) ----------
        avail = np.zeros(N_FUNCS, np.bool_)
        raw_avail = raw_obs.get("available_actions", [])
        for rid in raw_avail:
            if rid in FUNC_ID2IDX:
                avail[FUNC_ID2IDX[rid]] = True
        self._avail_last = raw_avail

        # ---------- vector (55 floats) -------------------------------------
        vec = self._build_vector(raw_obs)

        if self.obs_type == "vector":
            # concat legality bits (float32 0/1)
            return np.concatenate([vec, avail.astype(np.float32)])

        # ---------- full observation (dict) --------------------------------
        # FULL observation (dict) – MUST RETURN
        return {
            "screen":        screen,
            "minimap":       minimap,
            "vector":        vec,
            "avail_actions": avail.astype(np.float32),
        }

    def _build_vector(self, raw_obs):
        ru = raw_obs["raw_units"]
        ff = [u for u in ru if u.owner == 1]
        ee = [u for u in ru if u.owner == 2]
        ff.sort(key=lambda u: u.tag);  ee.sort(key=lambda u: u.tag)

        bea = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)
        if bea is not None:
            bx, by = bea.x, bea.y
        else:
            bx, by = -1., -1.

        self._my_tags.fill(0);   self._enemy_tags.fill(0);   self._alive.fill(False)
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
        vec[53] = raw_obs.game_loop[0] / 16.0
        vec[54] = self._alive.sum()
        return vec

    def _shape_reward(self, obs, done, result):
        vec = obs["vector"] if self.obs_type == "full" else obs
        enemy_alive_now  = int(vec[-1])
        friend_alive_now = (vec[4::5] > 0).sum()

        reward  = (self._prev_enemy_alive.sum() - enemy_alive_now)  \
                - (self._prev_friend_alive.sum() - friend_alive_now)

        d_now = vec[-3]
        if self._prev_beacon_dist is not None:
            reward += (self._prev_beacon_dist - d_now)
        self._prev_beacon_dist = d_now

        if done:
            if "win"  in result: reward += 10.0
            elif "loss" in result: reward -= 10.0

        self._prev_enemy_alive  = self._alive.copy()
        self._prev_friend_alive = (vec[4::5] > 0).astype(bool)[:5]
        return float(reward)

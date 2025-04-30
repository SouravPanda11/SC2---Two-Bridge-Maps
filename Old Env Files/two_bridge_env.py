# two_bridge_env.py  –  self‑managed termination (Move / Attack only)
import numpy as np, gymnasium as gym
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions
from absl import flags
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

# ─────────── Map registration ───────────
from pysc2.maps import lib

class TwoBridgeMap_Same(lib.Map):
    name      = "TwoBridgeMap_Same"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_Same.SC2Map"
    players   = 1

lib.get_maps().pop("TwoBridgeMap_Same", None)
lib.get_maps()["TwoBridgeMap_Same"] = TwoBridgeMap_Same()

# ─────────── constants ──────────────────
RAW_SIZE       = 64
MAX_UNITS      = 5
BEACON_TYPE_ID = 317            # Terran Large Beacon

# ═══════════ ENV CLASS ═══════════════════
class TwoBridgeEnv(gym.Env):
    """
    Five Marines; each step choose Move_pt(x,y) or Attack_pt(x,y) for each unit.
    Termination detected in Python (no Galaxy End Game actions).
    """
    metadata = {"render_modes": []}

    # ----------------------------------------------------------
    def __init__(self, step_mul: int = 8, visualize=False):
        super().__init__()
        self._env = sc2_env.SC2Env(
            map_name="TwoBridgeMap_Same",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True, raw_resolution=RAW_SIZE),
            step_mul=step_mul, game_steps_per_episode=0,
            visualize=visualize)

        # ----- action & obs spaces -----------------------------
        self.action_space = spaces.Dict({
            "act_type": spaces.MultiBinary(MAX_UNITS),         # 0=Move, 1=Attack
            "target"  : spaces.Box(-1., RAW_SIZE, (MAX_UNITS,2), np.float32)
        })
        # [n_friend, n_enemy, mean_hp_F, mean_hp_E, dist_beacon]
        self.observation_space = spaces.Box(0, 500, (5,), np.float32)

        # caches
        self._beacon_xy   = None
        self._prev_enemy  = 0
        self._prev_friend = 0
        self._prev_dist   = 0.0

    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()[0]
        self._beacon_xy = self._locate_beacon(ts)
        obs = self._to_obs(ts)
        self._prev_enemy, self._prev_friend, self._prev_dist = obs[1], obs[0], obs[4]
        return obs, {}

    # ----------------------------------------------------------
    def step(self, act):
        cmds = self._translate(act)
        ts   = self._env.step(cmds)[0]

        obs = self._to_obs(ts)

        # ---------- termination check -------------------------
        no_enemy   = (obs[1] == 0)
        no_friend  = (obs[0] == 0)
        beacon_cap = (obs[4] < 1.0)          # dist < 1 raw unit
        done       = no_enemy or no_friend or beacon_cap
        truncated  = False
        info       = {}

        # ---------- reward ------------------------------------
        reward = 0.0
        # sparse terminal part
        if done:
            if beacon_cap or (no_enemy and not no_friend):
                reward = +10.0          # victory
            elif no_friend and not no_enemy:
                reward = -10.0          # defeat
            else:
                reward = 0.0            # tie

        # dense shaping (applies every step incl. terminal)
        reward += 0.2 * (self._prev_enemy  - obs[1])    # enemy kill bonus
        reward -= 0.2 * (self._prev_friend - obs[0])    # friendly loss penalty
        reward += 0.05* (self._prev_dist   - obs[4])    # beacon distance

        # update caches
        self._prev_enemy, self._prev_friend, self._prev_dist = obs[1], obs[0], obs[4]
        return obs, reward, done, truncated, info

    # ----------------------------------------------------------
    def close(self):
        self._env.close()

    # ================= helpers ===============================
    def _to_obs(self, ts):
        units  = ts.observation.raw_units
        friend = sorted([u for u in units if u.owner == 1], key=lambda u: u.tag)
        enemy  = [u for u in units if u.owner == 2]

        n_f, n_e = len(friend), len(enemy)
        mean_hp_f = np.mean([u.health for u in friend]) if friend else 0.
        mean_hp_e = np.mean([u.health for u in enemy ]) if enemy  else 0.

        dist = 0.0
        if friend and self._beacon_xy:
            dx = friend[0].x - self._beacon_xy[0]
            dy = friend[0].y - self._beacon_xy[1]
            dist = (dx*dx + dy*dy)**0.5

        self._friend_sorted = friend
        return np.array([n_f, n_e, mean_hp_f, mean_hp_e, dist], np.float32)

    # ---------- RAW command translator -----------------------
    def _translate(self, act):
        cmds=[]
        types, targets = act["act_type"], act["target"]
        for i,(flag,xy) in enumerate(zip(types, targets)):
            if i >= len(self._friend_sorted): break
            x,y = xy
            if x < 0 or y < 0: continue           # no‑op for this marine
            x = float(np.clip(x,0,RAW_SIZE-1));  y = float(np.clip(y,0,RAW_SIZE-1))
            tag = self._friend_sorted[i].tag
            fn  = actions.RAW_FUNCTIONS.Attack_pt if flag else actions.RAW_FUNCTIONS.Move_pt
            cmds.append(fn("now", tag, (x,y)))
        return cmds or [actions.RAW_FUNCTIONS.no_op()]

    # ---------- locate beacon once ---------------------------
    def _locate_beacon(self, ts):
        for u in ts.observation.raw_units:
            if u.unit_type == BEACON_TYPE_ID:
                return (u.x, u.y)
        return None


# ────────── smoke‑test ──────────
if __name__ == "__main__":
    env = TwoBridgeEnv()
    obs,_ = env.reset()
    done  = False
    while not done:
        a = {
            "act_type": env.action_space["act_type"].sample(),
            "target"  : env.action_space["target"].sample()
        }
        obs, r, done, _, _ = env.step(a)
        print(f"r={r:+.2f}  obs={obs[:3]}")
    env.close()

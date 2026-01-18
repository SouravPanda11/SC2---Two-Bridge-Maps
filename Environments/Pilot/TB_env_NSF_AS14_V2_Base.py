import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ─────────────────────────────────────────  map registration  ───────────
class TwoBridgeMap_V2_Base(lib.Map):
    name      = "TwoBridgeMap_V2_Base"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"
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
BEACON_RADIUS   = 5.0
MAP_NAME        = "TwoBridgeMap_V2_Base"

FIVE_MIN_LOOPS  = 5 * 60 * 16          # 4800
STEP_MUL        = 8                    # env.step advances 8 loops
MAX_STEPS       = FIVE_MIN_LOOPS // STEP_MUL  # 600 env steps
STEP_PIX        = 2

# Reward constants
TERM_WIN_BONUS   = 10.0
TERM_LOSS_PENALTY= -10.0
TERM_TIE_BONUS   = 0.0

# ────────────────────────────────────────────── environment  ────────────
class TwoBridgeEnv(gym.Env):
    """
    Observation (55 floats):
        friend[5] · enemy[5] · beacon-x,y · dist-beacon · game-time · #enemy-alive
    Action (MultiDiscrete [14]*5):
        0 no-op |
        1-8  move |
        9-13 attack enemy index 0-4
    """

    metadata = {}

    # ───── init ─────────────────────────────────────────────────────────
    def __init__(self,
                 screen_res: int = 64,
                 visualize: bool = False,
                 realtime: bool = False,
                 replay_dir: str = None,
                 save_replay_episodes: int = 0):
        super().__init__()
        self.screen = screen_res

        self._env = sc2_env.SC2Env(
            map_name  = MAP_NAME,
            players   = [sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot  (sc2_env.Race.terran,
                                       sc2_env.Difficulty.easy)],
            agent_interface_format = sc2_env.AgentInterfaceFormat(
                action_space       = actions.ActionSpace.RAW,
                use_raw_units      = True,
                feature_dimensions = features.Dimensions(screen=screen_res,
                                                         minimap=screen_res)),
            step_mul  = STEP_MUL,
            visualize = visualize,
            realtime  = realtime,
            save_replay_episodes=save_replay_episodes,
            replay_dir=replay_dir
        )

        # 14 choices per marine
        self.action_space      = spaces.MultiDiscrete([14]*5)

        self.observation_space = spaces.Box(
            0.0,
            np.array([screen_res, screen_res, MARINE_HP, 15.0, 1.0]*10 +
                     [screen_res, screen_res, screen_res*2, 1e4, 5],
                     dtype=np.float32),
            dtype=np.float32
        )

        # caches -----------------------------------------------------------
        self._my_tags     = np.zeros(5, np.int64)
        self._enemy_tags  = np.zeros(5, np.int64)
        self._enemy_alive = np.zeros(5, np.bool_)
        self._fx = self._fy = np.zeros(5, np.float32)

        self._step_ctr          = 0
        self._prev_beacon_dist  = None
        self._prev_enemy_alive  = np.zeros(5, np.bool_)
        self._prev_friend_alive = np.zeros(5, np.bool_)

        # instrumentation caches --------------------------------------------
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

    # ───── Gym API ───────────────────────────────────────────────────────
    def close(self):
        self._env.close()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0

        self._prev_enemy_alive[:]  = False
        self._prev_friend_alive[:] = False
        self._prev_beacon_dist     = None

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

    def step(self, action):
        self._step_ctr += 1
        ts = self._env.step(self._decode(action))[0]

        # Build obs first
        obs = self._build_obs(ts)

        # Built-in PySC2 termination
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
                # fallback
                res = "tie"

            shaped_reward = self._shape_reward(obs, done=True, res=res)
            info = {"result": res, "rew": self.get_reward_components()}
            return obs, float(shaped_reward), True, False, info

        # Custom termination logic
        friend_alive_ct = int((obs[2:25:5] > 0).sum())
        enemy_alive_ct  = int(obs[54])                   
        no_friend       = (friend_alive_ct == 0)
        no_enemy        = (enemy_alive_ct == 0)
        beacon_win      = (obs[52] < BEACON_RADIUS)

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

        shaped_reward = self._shape_reward(obs, done=done, res=res)
        info = {"result": res, "rew": self.get_reward_components()}
        return obs, float(shaped_reward), done, False, info

    # ───── observation builder ─────────────────────────────────────────
    def _build_obs(self, ts):
        ru = ts.observation.raw_units
        ff = [u for u in ru if u.owner == 1]
        ee = [u for u in ru if u.owner == 2]
        ff.sort(key=lambda u: u.tag)
        ee.sort(key=lambda u: u.tag)

        bea = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)
        bx, by = (bea.x, bea.y) if bea is not None else (-1., -1.)

        self._my_tags[:] = 0
        self._enemy_tags[:] = 0
        self._enemy_alive[:] = False

        # Friend tags / positions
        for i, u in enumerate(ff[:5]):
            self._my_tags[i] = u.tag
        self._fx[:] = 0.0
        self._fy[:] = 0.0
        self._fx[:len(ff[:5])] = [u.x for u in ff[:5]]
        self._fy[:len(ff[:5])] = [u.y for u in ff[:5]]

        # Enemy tags / alive
        for i, u in enumerate(ee[:5]):
            self._enemy_tags[i] = u.tag
            self._enemy_alive[i] = (u.health > 0)

        vec = np.zeros(55, np.float32)

        # friends: 5 slots
        for i, u in enumerate(ff[:5]):
            vec[i*5:(i+1)*5] = (u.x, u.y, u.health, u.weapon_cooldown, 1.0)

        # enemies: 5 slots
        for i, u in enumerate(ee[:5]):
            vec[25+i*5:25+(i+1)*5] = (u.x, u.y, u.health,
                                      u.weapon_cooldown, float(u.health > 0))

        # beacon xy
        vec[50:52] = (bx, by)

        # distance to beacon
        if len(ff) > 0 and bea is not None:
            vec[52] = np.hypot(ff[0].x - bx, ff[0].y - by)
        else:
            vec[52] = 128.0

        vec[53] = ts.observation.game_loop[0] / 16.0
        vec[54] = float(self._enemy_alive.sum())
        return vec

    # ───── reward shaping + instrumentation ─────────────────────────────
    def _shape_reward(self, obs_vec, done: bool, res: str):
        """
        - combat_r: kill/loss delta
        - nav_r: beacon distance delta
        - term_r: terminal bonus/penalty
        """
        # Current alive counts
        enemy_alive_ct  = float(obs_vec[54])
        friend_alive_ct = float((obs_vec[2:25:5] > 0).sum())

        # (a) kill-loss delta
        prev_enemy_ct  = float(self._prev_enemy_alive.sum())
        prev_friend_ct = float(self._prev_friend_alive.sum())

        combat_r = (prev_enemy_ct - enemy_alive_ct) - (prev_friend_ct - friend_alive_ct)

        # (b) beacon distance delta
        d_now = float(obs_vec[52])
        nav_r = 0.0
        if self._prev_beacon_dist is not None:
            nav_r = float(self._prev_beacon_dist - d_now)
        self._prev_beacon_dist = d_now

        # (c) terminal
        term_r = 0.0
        if done:
            if res in ("nav_win", "combat_win"):
                term_r = TERM_WIN_BONUS
            elif res in ("combat_loss", "timeout_loss"):
                term_r = TERM_LOSS_PENALTY
            elif res == "tie":
                term_r = TERM_TIE_BONUS
            else:
                term_r = 0.0

        # Update caches AFTER computing deltas
        self._prev_enemy_alive  = self._enemy_alive.copy()
        # friend alive derived from obs slots: presence flag is last dim in each friend 5-tuple (index 4)
        self._prev_friend_alive = (obs_vec[4::5] > 0).astype(bool)[:5]

        # Log component diagnostics (hp sums available from obs)
        friend_hp_sum = float(obs_vec[2:25:5].sum())
        enemy_hp_sum  = float(obs_vec[27:50:5].sum())

        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": friend_hp_sum,
            "enemy_hp": enemy_hp_sum,
            "nav_dist": float(d_now),
            "enemy_alive": float(enemy_alive_ct),
        }

        # Per-unit metrics keyed by tags
        friend_dict = {}
        for i in range(5):
            tag = int(self._my_tags[i])
            if tag != 0:
                hp = float(obs_vec[i*5 + 2])
                friend_dict[tag] = {"hp": hp}

        enemy_dict = {}
        for j in range(5):
            etag = int(self._enemy_tags[j])
            if etag != 0:
                hp = float(obs_vec[25 + j*5 + 2])
                enemy_dict[etag] = {"hp": hp}

        self._last_unit_metrics = {"friend": friend_dict, "enemy": enemy_dict}

        return float(combat_r + nav_r + term_r)

    # ───── RAW action decoder ──────────────────────────────────────────
    _DIRS = [
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

            if 1 <= a <= 8:  # MOVE
                dx, dy = self._DIRS[a-1]
                x = float(np.clip(self._fx[i] + dx, 0, self.screen-1))
                y = float(np.clip(self._fy[i] + dy, 0, self.screen-1))
                cmds.append(RAW.Move_pt("now", tag, (x, y)))

            else:            # ATTACK
                ei = a - 9
                if 0 <= ei < 5 and self._enemy_alive[ei]:
                    cmds.append(RAW.Attack_unit("now", tag, int(self._enemy_tags[ei])))

        return cmds or [RAW.no_op()]

    # ───────────────────────── instrumentation getters ─────────────────────────
    def get_reward_components(self):
        return getattr(self, "_last_reward_components", {})

    def get_friendly_tags(self):
        return self._my_tags.copy()

    def get_unit_metrics(self):
        return getattr(self, "_last_unit_metrics", {"friend": {}, "enemy": {}})

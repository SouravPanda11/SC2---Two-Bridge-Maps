import gymnasium as gym, numpy as np
from  gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ───────────────────── Map registration ──────────────────────────────
class TwoBridgeMap_V2_Combat(lib.Map):
    name      = "TwoBridgeMap_V2_Combat"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps"
    filename  = "TwoBridgeMap_V2_Combat.SC2Map"
    players   = 2                     # agent vs bot

lib.get_maps().pop("TwoBridgeMap_V2_Combat", None)
lib.get_maps()["TwoBridgeMap_V2_Combat"] = TwoBridgeMap_V2_Combat()

# ────────────────────────── constants ────────────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():  FLAGS([''])

RAW              = actions.RAW_FUNCTIONS
# MARINE_HP        = 45
BEACON_TYPE_ID   = 317
BEACON_RADIUS    = 2.0

STEP_MUL         = 8
FIVE_MIN_LOOPS   = 5*60*16
MAX_STEPS        = FIVE_MIN_LOOPS // STEP_MUL
STEP_PIX         = 2

SCR_RES          = 64
SCR_CH           = len(features.SCREEN_FEATURES)    # 17
MINI_CH          = len(features.MINIMAP_FEATURES)   # 7

MOVE_DIRS = [     # 8-dir unit steps
    ( 0, -STEP_PIX), ( 0,  STEP_PIX), (-STEP_PIX, 0), ( STEP_PIX, 0),
    ( STEP_PIX,-STEP_PIX), (-STEP_PIX,-STEP_PIX),
    ( STEP_PIX, STEP_PIX), (-STEP_PIX, STEP_PIX)
]

# ───────────────────── additional constants ───────────────────────────
KILL_BONUS   = 1.0      # shaped reward when a unit is destroyed
HP_SCALE     = 0.05      # 0.05 reward pt per 1 HP Δ (tune as you like)

# ───── terminal bonus scales ────────────────────────────────
NAV_WIN_BONUS       = 25.0
COMBAT_WIN_BONUS    = 10.0
COMBAT_LOSS_PENALTY = -10.0
NAV_TIMEOUT_PENALTY = -15.0
TIE_BONUS           = 0.0

# ──────────────────────── environment ────────────────────────────────
class TwoBridgeEnv(gym.Env):
    """
    5 v 5 Two-Bridge – navigation & combat only.
    Action space = {verb, who-mask, direction, enemy_idx}
    """
    metadata = {}

    # ---------- Gym spaces -------------------------------------------------
    action_space = spaces.Dict({
        "verb":      spaces.Discrete(3),    # 0 noop | 1 move | 2 attack
        "who":       spaces.MultiBinary(5), # which marines receive the order
        "direction": spaces.Discrete(9),    # 0 unused | 1-8 compass (for move)
        "enemy_idx": spaces.Discrete(6)     # 0 none | 1-5 enemy slot (for atk)
    })

    observation_space = spaces.Dict({
        "screen":      spaces.Box(0, 255, (SCR_CH,  SCR_RES, SCR_RES), np.uint8),
        "minimap":     spaces.Box(0, 255, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
        "vector":      spaces.Box(0.0, np.inf, (55,), np.float32),
        # "action_mask": spaces.MultiBinary(28)
        "action_mask": spaces.MultiBinary(3)                     
    })

    # ---------- ctor / close ----------------------------------------------
    def __init__(self,
                 screen_res: int = 64,
                 visualize: bool = False,
                 realtime: bool = False,
                 replay_dir: str = None,
                 save_replay_episodes: int = 0):
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
                feature_dimensions=features.Dimensions(
                    screen=SCR_RES, minimap=SCR_RES)),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes)

        # caches ───────────────────────────────────────────────────────────
        self._my_tags     = np.zeros(5, np.int64)
        self._enemy_tags  = np.zeros(5, np.int64)
        self._enemy_alive = np.zeros(5, bool)
        self._fx = self._fy = np.zeros(5, np.float32)

        self._step_ctr          = 0
        # self._prev_beacon_dist  = None
        # self._prev_enemy_alive  = np.zeros(5, bool)
        # self._prev_friend_alive = np.zeros(5, bool)
        # ─── NEW caches ────────────────────────────────────────────────
        self._prev_beacon_dists   = None          # ‖ 5-vector
        self._prev_centroid_dists = None          # ‖ 5-vector
        self._prev_enemy_hp       = np.zeros(5, np.float32)
        self._prev_friend_hp      = np.zeros(5, np.float32)
        
        self._last_act = {"verb": 0, "who_bits": np.zeros(5, bool), "enemy_idx": -1}

    def close(self): self._env.close()

    # ---------- Gym API ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0
        # self._prev_enemy_alive[:]  = False
        # self._prev_friend_alive[:] = False
        # self._prev_beacon_dist     = None
        self._prev_beacon_dists   = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp[:]    = 0.
        self._prev_friend_hp[:]   = 0.
        
        self._last_act = {"verb": 0, "who_bits": np.zeros(5, bool), "enemy_idx": -1}
        
        ts = self._env.reset()[0]
        return self._build_obs(ts), {}

    def step(self, action):
        cmds = self._translate_actions(action)
        ts   = self._env.step(cmds)[0]
        obs  = self._build_obs(ts)

        # built-in victory / defeat
        if ts.last():
            res = "victory" if ts.reward > 0 else "defeat" if ts.reward < 0 else "tie"
            return obs, float(ts.reward), True, False, {"result": res}

        # custom termination ----------------------------------------------
        friend_alive = (obs["vector"][2:25:5] > 0).sum()
        no_friend    = friend_alive == 0
        no_enemy     = obs["vector"][54] == 0
        beacon_win   = obs["vector"][52] < BEACON_RADIUS

        info = {"result": None}
        if beacon_win:                    info["result"] = "nav_win"
        elif no_enemy and no_friend:      info["result"] = "tie"
        elif no_enemy:                    info["result"] = "combat_win"
        elif no_friend:                   info["result"] = "combat_loss"

        if self._step_ctr >= MAX_STEPS and info["result"] is None:
            info["result"] = "timeout_loss"

        done   = info["result"] is not None
        reward = self._shape_reward(obs["vector"], done, info["result"])
        return obs, reward, done, False, info

    # ---------- action translation ----------------------------------------
    def _translate_actions(self, act):
        verb       = int(act["verb"])
        who_bits   = act["who"].astype(bool)
        dir_id     = int(act["direction"])
        enemy_idx  = int(act["enemy_idx"]) - 1     # shift so 0-4

        tags = [int(t) for t,bit in zip(self._my_tags, who_bits) if bit]
        
        self._last_act = {"verb": verb, "who_bits": who_bits.copy(), "enemy_idx": enemy_idx}

        # ───── MOVE ───────────────────────────────────────────────────────
        if verb == 1 and tags and 1 <= dir_id <= 8:
            dx, dy = MOVE_DIRS[dir_id-1]
            cx = np.mean(self._fx[who_bits]) + dx
            cy = np.mean(self._fy[who_bits]) + dy
            pt = (float(np.clip(cx, 0, SCR_RES-1)),
                  float(np.clip(cy, 0, SCR_RES-1)))
            return [RAW.Move_pt("now", tags, pt)]

        # ───── ATTACK ─────────────────────────────────────────────────────
        if verb == 2 and tags and 0 <= enemy_idx < 5 and self._enemy_alive[enemy_idx]:
            return [RAW.Attack_unit("now", tags,
                                    int(self._enemy_tags[enemy_idx]))]

        # default no-op
        return [RAW.no_op()]

    # ---------- observation builder ---------------------------------------
    def _build_obs(self, ts):
        ob   = ts.observation
        ru   = ob.raw_units
        fri  = sorted([u for u in ru if u.owner == 1], key=lambda u: u.tag)
        ene  = sorted([u for u in ru if u.owner == 2], key=lambda u: u.tag)
        bea  = next((u for u in ru if u.unit_type == BEACON_TYPE_ID), None)

        # coords
        if bea is not None:
            bx, by = bea.x, bea.y
        else:
            bx, by = -1., -1.

        self._my_tags[:]     = 0
        self._enemy_tags[:]  = 0
        self._enemy_alive[:] = False
        self._fx[:] = self._fy[:] = 0

        for i,u in enumerate(fri[:5]):
            self._my_tags[i] = u.tag
            self._fx[i], self._fy[i] = u.x, u.y
        for i,u in enumerate(ene[:5]):
            self._enemy_tags[i]  = u.tag
            self._enemy_alive[i] = u.health > 0

        vec = np.zeros(55, np.float32)
        for i,u in enumerate(fri[:5]):
            vec[i*5:(i+1)*5] = (u.x,u.y,u.health,u.weapon_cooldown,1.)
        for i,u in enumerate(ene[:5]):
            vec[25+i*5:25+(i+1)*5] = (
                u.x,u.y,u.health,u.weapon_cooldown,float(u.health>0))
        vec[50:52] = (bx,by)
        if fri and (bea is not None):
            vec[52] = np.hypot(fri[0].x - bx, fri[0].y - by)
        else:
            vec[52] = 128.0
        vec[53] = ob.game_loop[0]/16.0
        vec[54] = self._enemy_alive.sum()

        # env: _build_obs, delete the padding lines
        mask = np.ones(3, np.int8)
        mask[1] = int((vec[2:25:5] > 0).any())   # MOVE?
        mask[2] = int(vec[54] > 0)               # ATTACK?

        self._step_ctr += 1
        return {
            "screen":      np.asarray(ob.feature_screen,  np.uint8),
            "minimap":     np.asarray(ob.feature_minimap, np.uint8),
            "vector":      vec,
            "action_mask": mask
        }

    def _shape_reward(self, vec, done, res):
        """
        Team reward = navigation Δ-distance  +  combat (centroid Δ-distance + HP + kill/loss)
                    + terminal bonus (episode-level).
        Also emits per‑unit diagnostics (nav_r Δ, combat_r Δ, friendly HP, enemy HP) keyed by tags.
        """

        # ---------- unpack -------------------------------------------------
        fx, fy, fhp = vec[0:25:5], vec[1:25:5], vec[2:25:5]
        ex, ey, ehp = vec[25:50:5], vec[26:50:5], vec[27:50:5]
        f_alive, e_alive = fhp > 0, ehp > 0
        bx, by          = vec[50:52]

        # ---------- FIRST FRAME GUARD -------------------------------------
        if self._prev_enemy_hp.sum() == 0 and self._prev_friend_hp.sum() == 0:
            # initialize “previous-step” caches and return 0 so spawn state has no reward
            self._prev_beacon_dists   = np.hypot(fx - bx, fy - by) if (bx >= 0 and by >= 0) else None
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

        # ───────── NAVIGATION SHAPING + per‑unit deltas (compute BEFORE updating cache) ─────────
        nav_per = np.zeros(5, np.float32)
        nav_r   = 0.0
        if (bx >= 0) and (by >= 0):
            beacon_dists = np.hypot(fx - bx, fy - by)
            if self._prev_beacon_dists is not None:
                nav_per = self._prev_beacon_dists - beacon_dists      # + if moving closer
                sel = nav_per[f_alive]
                nav_r = sel.mean() if sel.size > 0 else 0.0
            # update cache AFTER computing per‑unit deltas
            self._prev_beacon_dists = beacon_dists
        else:
            self._prev_beacon_dists = None

        # ───────── COMBAT SHAPING + per‑unit deltas (compute BEFORE updating cache) ─────────────
        combat_per = np.zeros(5, np.float32)
        combat_r   = 0.0
        if e_alive.any():
            cx, cy = ex[e_alive].mean(), ey[e_alive].mean()
            centroid_dists = np.hypot(fx - cx, fy - cy)
            if self._prev_centroid_dists is not None:
                combat_per = self._prev_centroid_dists - centroid_dists  # + if moving closer
                sel = combat_per[f_alive]
                combat_r += sel.mean() if sel.size > 0 else 0.0
            # update cache AFTER computing per‑unit deltas
            self._prev_centroid_dists = centroid_dists
        else:
            self._prev_centroid_dists = None

        # HP shaping (team-level)
        combat_r +=  HP_SCALE * (self._prev_enemy_hp.sum()  - ehp.sum())
        combat_r += -HP_SCALE * (self._prev_friend_hp.sum() - fhp.sum())

        # kill / loss shaped bonuses (team-level)
        kills  = (~e_alive & (self._prev_enemy_hp > 0)).sum()
        losses = (~f_alive & (self._prev_friend_hp > 0)).sum()
        combat_r +=  KILL_BONUS * kills
        combat_r += -KILL_BONUS * losses

        # update HP caches (after using the previous values)
        self._prev_enemy_hp[:]  = ehp
        self._prev_friend_hp[:] = fhp

        # ───────────────────── TERMINAL BONUS (team-level) ───────────────────────
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

        # ---------- log team-level components for CSVs/dashboards ----------
        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": float(fhp.sum()),
            "enemy_hp": float(ehp.sum()),
            "nav_dist":    float(np.mean(np.hypot(fx - bx, fy - by))) if f_alive.any() else 0.0,
            "combat_dist": float(np.mean(np.hypot(fx - cx, fy - cy))) if (e_alive.any() and f_alive.any()) else 0.0,
        }

        # ---------- per‑unit diagnostics keyed by persistent tags ----------
        friend_dict = {}
        for i in range(5):
            tag = int(self._my_tags[i])
            if tag != 0:
                friend_dict[tag] = {
                    "nav_r":    float(nav_per[i]),
                    "combat_r": float(combat_per[i]),
                    "hp":       float(fhp[i]),
                }

        enemy_dict = {}
        for j in range(5):
            etag = int(self._enemy_tags[j])
            if etag != 0:
                enemy_dict[etag] = {"hp": float(ehp[j])}

        self._last_unit_metrics = {"friend": friend_dict, "enemy": enemy_dict}

        return float(nav_r + combat_r + term_r)

    
    def get_reward_components(self):
        return getattr(self, "_last_reward_components", {})
    
    # In TwoBridgeEnv
    def get_friendly_tags(self):
        # returns the 5-slot array of current friendly unit tags (0 means empty slot)
        return self._my_tags.copy()

    def get_unit_metrics(self):
        """
        Returns a dict with per‑unit values for this step:
        'friend': {tag: {'nav_r': float, 'combat_r': float, 'hp': float}}
        'enemy':  {etag: {'hp': float}}
        Missing tags simply won’t appear.
        """
        return getattr(self, "_last_unit_metrics", {"friend": {}, "enemy": {}})
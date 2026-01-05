import gymnasium as gym, numpy as np
from gymnasium import spaces

from pysc2.env  import sc2_env
from pysc2.lib  import actions, features
from pysc2.maps import lib
from absl       import flags

# ────────────────────── configuration ──────────────────────────
N_FRIEND = 5
N_ENEMY  = 3

# index helpers for the compact vector  ─────────────────────────
VEC_FRIEND  = 0
VEC_ENEMY   = VEC_FRIEND  + N_FRIEND*5
VEC_BXY     = VEC_ENEMY   + N_ENEMY *5      # 2 × float32
VEC_DIST    = VEC_BXY     + 2               # 1 × float32 (slot-0 → beacon)
VEC_TIME    = VEC_DIST    + 1               # 1 × float32
VEC_ECOUNT  = VEC_TIME    + 1               # 1 × float32
VEC_SIZE    = VEC_ECOUNT  + 1

# ───────────────────── Map registration ───────────────────────
class TwoBridgeMap_V1_Combat_Cam(lib.Map):
    name      = "TwoBridgeMap_V1_Combat_Cam"
    directory = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Lock"
    filename  = "TwoBridgeMap_V1_Combat_Cam.SC2Map"
    players   = 2

lib.get_maps().pop("TwoBridgeMap_V1_Combat_Cam", None)
lib.get_maps()["TwoBridgeMap_V1_Combat_Cam"] = TwoBridgeMap_V1_Combat_Cam()

# ───────────────────────── constants ───────────────────────────
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS([''])

RAW              = actions.RAW_FUNCTIONS
MARINE_HP        = 45
BEACON_TYPE_ID   = 317
BEACON_RADIUS    = 5.0

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

# ───────────────────── additional reward constants ────────────
KILL_BONUS   = 1.0     # shaped reward per unit destroyed
HP_SCALE     = 0.05    # reward per 1 HP delta

NAV_WIN_BONUS       = 25.0
COMBAT_WIN_BONUS    = 10.0
COMBAT_LOSS_PENALTY = -10.0
NAV_TIMEOUT_PENALTY = -15.0
TIE_BONUS           = 0.0

# ─────────────────────── environment ───────────────────────────
class TwoBridgeEnv(gym.Env):
    """
    5 v 8 Two-Bridge V1_Combat_Cam.
    Action space = {verb, who-mask, direction, enemy_idx}
    """
    metadata = {}

    # -------------- Gym spaces ---------------------------------
    action_space = spaces.Dict({
        "verb":      spaces.Discrete(3),            # 0 noop | 1 move | 2 atk
        "who":       spaces.MultiBinary(N_FRIEND),  # 5 friendly marines
        "direction": spaces.Discrete(9),            # 0 unused | 1-8 compass
        "enemy_idx": spaces.Discrete(N_ENEMY + 1)   # 0 none | 1-8 enemy slot
    })

    observation_space = spaces.Dict({
        "screen":  spaces.Box(0, 255, (SCR_CH,  SCR_RES, SCR_RES), np.uint8),
        "minimap": spaces.Box(0, 255, (MINI_CH, SCR_RES, SCR_RES), np.uint8),
        "vector":  spaces.Box(0.0, np.inf, (VEC_SIZE,), np.float32),

        # ── FULL MASKS PER BRANCH ───────────────────────────────
        "action_mask": spaces.Dict({
            "verb":      spaces.MultiBinary(3),
            "who":       spaces.MultiBinary(N_FRIEND),
            "direction": spaces.MultiBinary(9),
            "enemy_idx": spaces.MultiBinary(N_ENEMY + 1),
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
            map_name="TwoBridgeMap_V1_Combat_Cam",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot  (sc2_env.Race.terran,
                                   sc2_env.Difficulty.easy)],
            step_mul=STEP_MUL,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=SCR_RES,
                feature_dimensions=features.Dimensions(
                    screen=SCR_RES, 
                    minimap=SCR_RES)),
            visualize=visualize,
            realtime=realtime,
            replay_dir=replay_dir,
            save_replay_episodes=save_replay_episodes)

        # caches ────────────────────────────────────────────────
        self._my_tags     = np.zeros(N_FRIEND, np.int64)
        self._enemy_tags  = np.zeros(N_ENEMY, np.int64)
        self._enemy_alive = np.zeros(N_ENEMY, bool)
        self._fx = self._fy = np.zeros(N_FRIEND, np.float32)

        self._step_ctr            = 0
        self._prev_beacon_dists   = None          # shape (N_FRIEND,)
        self._prev_centroid_dists = None          # shape (N_FRIEND,)
        self._prev_enemy_hp       = np.zeros(N_ENEMY,  np.float32)
        self._prev_friend_hp      = np.zeros(N_FRIEND, np.float32)

    def close(self): self._env.close()

    # -------------- Gym API ------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_ctr = 0

        self._prev_beacon_dists   = None
        self._prev_centroid_dists = None
        self._prev_enemy_hp[:]    = 0.0
        self._prev_friend_hp[:]   = 0.0

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
        info["rew"] = self.get_reward_components()   # expose components (TB)
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

    # -------------- build observation + FULL MASKS --------------
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
            # ANY-UNIT distance: min over alive friendlies
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
        vec[VEC_ECOUNT] = self._enemy_alive.sum()

        # ───────────── FULL ACTION MASKS ─────────────────────────
        # Who: alive friendlies can be selected
        who_mask = (vec[2 : N_FRIEND*5 : 5] > 0).astype(np.int8)
        any_friend_alive = int(who_mask.any())

        # Verb: noop always ok; move if someone alive; attack if any enemy alive
        verb_mask = np.zeros(3, np.int8)
        verb_mask[0] = 1
        verb_mask[1] = any_friend_alive
        verb_mask[2] = int(vec[VEC_ECOUNT] > 0)

        # Direction:
        #   index 0 is the "unused" slot. Enable 1..8 only if moving is possible in principle.
        direction_mask = np.zeros(9, np.int8)
        if verb_mask[1]:             # move is state-feasible
            direction_mask[1:] = 1   # allow all 8 compass directions
        else:
            direction_mask[0] = 1    # only "unused" slot allowed

        # Enemy index:
        #   0 (none) always valid; alive enemies enabled when ATTACK is possible.
        enemy_mask = np.zeros(N_ENEMY + 1, np.int8)
        enemy_mask[0] = 1
        if verb_mask[2]:
            enemy_mask[1:1+N_ENEMY] = self._enemy_alive.astype(np.int8)

        action_mask = {
            "verb":      verb_mask,
            "who":       who_mask,
            "direction": direction_mask,
            "enemy_idx": enemy_mask
        }

        self._step_ctr += 1
        return {
            "screen":      np.asarray(ob.feature_screen,  np.uint8),
            "minimap":     np.asarray(ob.feature_minimap, np.uint8),
            "vector":      vec,
            "action_mask": action_mask
        }

    # -------------- shaped reward (rich) -----------------------
    def _shape_reward(self, vec, done, res):
        """
        Reward = navigation Δ-distance  +  combat (distance + HP + kill)  +  terminal.
        The first call after reset returns 0 to avoid spawn spikes.
        """
        # ---------- unpack -------------------------------------------------
        fx = vec[0 : N_FRIEND*5 : 5]
        fy = vec[1 : N_FRIEND*5 : 5]
        fhp = vec[2 : N_FRIEND*5 : 5]

        ex = vec[VEC_ENEMY   : VEC_ENEMY+N_ENEMY*5 : 5]
        ey = vec[VEC_ENEMY+1 : VEC_ENEMY+N_ENEMY*5 : 5]
        ehp = vec[VEC_ENEMY+2 : VEC_ENEMY+N_ENEMY*5 : 5]

        f_alive = fhp > 0
        e_alive = ehp > 0

        bx, by = vec[VEC_BXY : VEC_BXY+2]

        # ---------- FIRST FRAME GUARD -------------------------------------
        if self._prev_enemy_hp.sum() == 0 and self._prev_friend_hp.sum() == 0:
            self._prev_beacon_dists   = np.hypot(fx - bx, fy - by) if (bx >= 0 and by >= 0) else None
            if e_alive.any():
                cx, cy = ex[e_alive].mean(), ey[e_alive].mean()
                self._prev_centroid_dists = np.hypot(fx - cx, fy - cy)
            else:
                self._prev_centroid_dists = None
            self._prev_enemy_hp[:]  = ehp
            self._prev_friend_hp[:] = fhp
            self._last_reward_components = {"nav_r":0.0,"combat_r":0.0,"term_r":0.0}
            return 0.0

        # ───────── NAVIGATION SHAPING ─────────────────────
        nav_r = 0.0
        if (bx >= 0) and (by >= 0):
            beacon_dists = np.hypot(fx - bx, fy - by)         # (N_FRIEND,)
            diff = (self._prev_beacon_dists - beacon_dists)[f_alive] if self._prev_beacon_dists is not None else np.array([], dtype=np.float32)
            nav_r = diff.mean() if diff.size > 0 else 0.0
            self._prev_beacon_dists = beacon_dists

        # ───────── COMBAT SHAPING ─────────────────────────
        combat_r = 0.0
        cx = cy = np.nan
        if e_alive.any():
            cx, cy = ex[e_alive].mean(), ey[e_alive].mean()
            centroid_dists = np.hypot(fx - cx, fy - cy)
            if self._prev_centroid_dists is not None:
                diff = (self._prev_centroid_dists - centroid_dists)[f_alive]
                combat_r += diff.mean() if diff.size > 0 else 0.0
            self._prev_centroid_dists = centroid_dists
        else:
            self._prev_centroid_dists = None

        # HP shaping
        combat_r +=  HP_SCALE * (self._prev_enemy_hp.sum()  - ehp.sum())
        combat_r += -HP_SCALE * (self._prev_friend_hp.sum() - fhp.sum())

        # kill / loss bonuses
        kills  = (~e_alive & (self._prev_enemy_hp > 0)).sum()
        losses = (~f_alive & (self._prev_friend_hp > 0)).sum()
        combat_r +=  KILL_BONUS * kills
        combat_r += -KILL_BONUS * losses

        # update HP caches
        self._prev_enemy_hp[:]  = ehp
        self._prev_friend_hp[:] = fhp

        # ───────────────────── TERMINAL BONUS ───────────────────────
        if done:
            if   res == "nav_win":      term_r = NAV_WIN_BONUS
            elif res == "combat_win":   term_r = COMBAT_WIN_BONUS
            elif res == "combat_loss":  term_r = COMBAT_LOSS_PENALTY
            elif res == "timeout_loss": term_r = NAV_TIMEOUT_PENALTY
            elif res == "tie":          term_r = TIE_BONUS
            elif res == "victory":      term_r = COMBAT_WIN_BONUS
            elif res == "defeat":       term_r = COMBAT_LOSS_PENALTY
        else:
            term_r = 0.0

        # store components for logging
        self._last_reward_components = {
            "nav_r": float(nav_r),
            "combat_r": float(combat_r),
            "term_r": float(term_r),
            "friend_hp": float(fhp.sum()),
            "enemy_hp": float(ehp.sum()),
            "nav_dist": float(np.mean(np.hypot(fx - bx, fy - by))) if f_alive.any() and (bx>=0) and (by>=0) else 0.0,
            "combat_dist": float(np.mean(np.hypot(fx - cx, fy - cy))) if (e_alive.any() and f_alive.any()) else 0.0,
        }

        return float(nav_r + combat_r + term_r)

    def get_reward_components(self):
        return getattr(self, "_last_reward_components", {})

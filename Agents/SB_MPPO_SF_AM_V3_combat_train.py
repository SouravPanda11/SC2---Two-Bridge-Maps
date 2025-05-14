"""
Train MaskablePPO on the 5 v 3 Two-Bridge map.
"""
import sys, os, torch, numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from Environments.TB_env_SF_AM_V3_combat import TwoBridgeEnv, N_FRIEND, N_ENEMY
from sb3_contrib.common.wrappers import ActionMasker

# ──────────────────── FLATTEN-ACTION WRAPPER ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) →
    MultiDiscrete([3, 2×N_FRIEND, 9, N_ENEMY+1])
    """

    def __init__(self, env):
        super().__init__(env)

        # ------------- action space (unchanged) -----------------
        self.action_space = spaces.MultiDiscrete(
            [3] + [2]*N_FRIEND + [9] + [N_ENEMY + 1]
        )

        # bits beyond the verb-level mask that are always legal
        self._mask_template = np.ones(
            sum(self.action_space.nvec) - 3, dtype=np.int8
        )

        # ------------- **NEW**: advertise correct obs shape -----
        flat_len = 3 + len(self._mask_template)          # 26
        obs_spaces = dict(env.observation_space.spaces)  # shallow copy
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)
        # --------------------------------------------------------

    @staticmethod
    def _flatten(a_dict):
        return np.array([
            a_dict["verb"],
            *a_dict["who"],
            a_dict["direction"],
            a_dict["enemy_idx"]
        ], dtype=np.int64)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb":      int(a_vec[0]),
            "who":       np.asarray(a_vec[1 : 1+N_FRIEND], np.int8),
            "direction": int(a_vec[1+N_FRIEND]),
            "enemy_idx": int(a_vec[-1]),
        }

    # gym API overrides
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    # expand verb-mask → flat-mask
    def _convert_mask(self, obs):
        flat_mask = np.concatenate([obs["action_mask"], self._mask_template])
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask

# ───────────────────── hardware / logging ─────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

agent_name = "SB_MaskPPO_SF_AM_V3_combat"
save_dir   = f"./Agents/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# ───────────────── env + wrappers ─────────────────────────────
def mask_fn(env):                       # env is the FlattenActionWrapper
    return env.action_masks()

base_env = TwoBridgeEnv(visualize=False)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

# ───────────────────── model (Maskable) ───────────────────────
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    device=device,
    verbose=1,
    tensorboard_log=f"./tb_logs/{agent_name}/"
)

# ───────────────────── training loop ──────────────────────────
total_timesteps = 2_000_000
save_interval   =   400_000

for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")
    # model.save(f"{save_dir}{agent_name}_{(i + save_interval)}K")

model.save(f"{save_dir}{agent_name}_final")
env.close()

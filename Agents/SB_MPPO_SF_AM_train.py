"""
Train MaskablePPO on Two-Bridge env without creating a 3rd file.
The Dict → MultiDiscrete flattening wrapper lives in this script.
"""

import sys, os, torch, numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from Environments.TB_env_SF_AM import TwoBridgeEnv    # your Dict-action env
from sb3_contrib.common.wrappers import ActionMasker

# ──────────────────── FLATTEN-ACTION WRAPPER ────────────────────
class FlattenActionWrapper(Wrapper):
    """
    Convert Dict(verb, who, direction, enemy_idx) ->
    MultiDiscrete([3, 2,2,2,2,2, 9, 6]) so SB3 algorithms accept it.
    Also expands the 3-slot verb action_mask to flat mask.
    """

    def __init__(self, env):
        super().__init__(env)

        # MultiDiscrete([3,2,2,2,2,2,9,6])
        self.action_space = spaces.MultiDiscrete([3] + [2]*5 + [9] + [6])

        # template of always-legal bits (28-3 = 25)
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3,
                                      dtype=np.int8)

    # -------------- helper: flatten Dict -> list[ints] -------------
    @staticmethod
    def _flatten(a_dict):
        out = [a_dict["verb"]]
        out += list(a_dict["who"])             # five 0/1 ints
        out.append(a_dict["direction"])
        out.append(a_dict["enemy_idx"])
        return np.array(out, dtype=np.int64)

    # -------------- helper: unflatten list -> Dict ----------------
    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb":       int(a_vec[0]),
            "who":        np.asarray(a_vec[1:6], np.int8),
            "direction":  int(a_vec[6]),
            "enemy_idx":  int(a_vec[7]),
        }

    # ---------------- gym API overrides ---------------------------
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    # -------------- mask: verb(3) -> flat(3+rest) -----------------
    def _convert_mask(self, obs):
        verb_mask = obs["action_mask"]                    # (3,), dtype=int8
        flat_mask = np.concatenate([verb_mask, self._mask_template]).astype(np.int8)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        """Called by sb3-contrib to retrieve the current mask."""
        return self._last_mask


# ───────────────────── hardware / logging ───────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

agent_name = "SB_MaskPPO_SF_AM"
save_dir   = f"./Agents/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# ───────────────── env + wrappers ───────────────────────────────
def mask_fn(env):
    return env.action_masks()      # env is the FlattenActionWrapper

base_env = TwoBridgeEnv(visualize=False)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)


# ───────────────────── model (Maskable) ──────────────────────────
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    device=device,
    verbose=1,
    tensorboard_log=f"./tb_logs/{agent_name}/"
)

# ───────────────────── training loop ─────────────────────────────
total_timesteps = 2_000_000   # 2 M
save_interval   = 400_000   # every 400 k

for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")
    # model.save(f"{save_dir}{agent_name}_{(i + save_interval)}K")

# final save
model.save(f"{save_dir}{agent_name}_final")
env.close()

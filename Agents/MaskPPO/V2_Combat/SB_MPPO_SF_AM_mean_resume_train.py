import sys
import torch, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium import spaces, Wrapper
import numpy as np
from Environments.AM_RM_mean.TB_env_SF_AM_RM_mean_V2_Combat import TwoBridgeEnv



# ---- same wrapper & mask_fn as training ----
class FlattenActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([3] + [2]*5 + [9] + [6])
        self._mask_template = np.ones(int(self.action_space.nvec.sum()) - 3, dtype=np.int8)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(3 + len(self._mask_template))
        self.observation_space = spaces.Dict(obs_spaces)
    def step(self, a):
        obs, r, d, tr, info = self.env.step({
            "verb": int(a[0]),
            "who":  np.asarray(a[1:6], np.int8),
            "direction": int(a[6]),
            "enemy_idx": int(a[7]),
        })
        obs["action_mask"] = np.concatenate([obs["action_mask"], self._mask_template])
        self._last_mask = obs["action_mask"]
        return obs, r, d, tr, info
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        obs["action_mask"] = np.concatenate([obs["action_mask"], self._mask_template])
        self._last_mask = obs["action_mask"]
        return obs, info
    def action_masks(self): return self._last_mask

def mask_fn(e): return e.action_masks()

# ---- rebuild env exactly as before ----
base_env = TwoBridgeEnv(visualize=False)
env = ActionMasker(FlattenActionWrapper(base_env), mask_fn)

# ---- load last saved params ----
agent_name = "SB_MaskPPO_SF_AM_RM_mean"
map_name = "V2_Combat"
save_dir = f"./Agents/MaskPPO/{map_name}/saved_models/{agent_name}/"
model_path = os.path.join(save_dir, f"{agent_name}_final.zip")  # or 2000K checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskablePPO.load(model_path, env=env, device=device, tensorboard_log=f"./tb_logs/MaskPPO/{map_name}/{agent_name}/")

# ---- continue training: from 2M to 10M (= +8M) ----
remaining = 10_000_000 - int(model.num_timesteps)   # should be ~8_000_000
remaining = max(remaining, 0)
print(f"Continuing training for {remaining} timesteps...")

save_interval = 500_000
for i in range(0, remaining, save_interval):
    model.learn(total_timesteps=min(save_interval, remaining - i),
                reset_num_timesteps=False,  # <— continue counter
                progress_bar=True)
    model.save(os.path.join(save_dir, f"{agent_name}_{(model.num_timesteps // 1000)}K"))

# final save
model.save(os.path.join(save_dir, f"{agent_name}_final"))
env.close()

import sys, os, torch, numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# Environment imports
from Environments.TB_env_SF_AM_RM_mean_V1_Base import TwoBridgeEnv, N_FRIEND, N_ENEMY

# ──────────────────── FLATTEN-ACTION WRAPPER ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) →
    MultiDiscrete([3, 2×N_FRIEND, 9, N_ENEMY+1])
    """

    def __init__(self, env):
        super().__init__(env)

        # MultiDiscrete layout
        self.action_space = spaces.MultiDiscrete([3] + [2]*N_FRIEND + [9] + [N_ENEMY + 1])

        # bits beyond the verb-level mask that are always legal
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        # Advertise the flattened mask shape to SB3
        flat_len = 3 + len(self._mask_template)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

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

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        flat_mask = np.concatenate([obs["action_mask"], self._mask_template]).astype(np.int8)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask


# ───────────────────── TB CALLBACK (reward components) ─────────────────────
class TBRewardLogger(BaseCallback):
    """
    Logs env-provided reward components under 'rew/*' in TensorBoard.
    Requires env to set info['rew'] = env.get_reward_components() each step.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # info can be a list (VecEnv) or dict (non-vec); handle both
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and "rew" in info and self.logger is not None:
                    for k, v in info["rew"].items():
                        try:
                            self.logger.record(f"rew/{k}", float(v))
                        except Exception:
                            pass
        elif isinstance(infos, dict) and "rew" in infos and self.logger is not None:
            for k, v in infos["rew"].items():
                try:
                    self.logger.record(f"rew/{k}", float(v))
                except Exception:
                    pass
        return True


# ───────────────────── hardware / logging ─────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

agent_name = "SB_MaskPPO_SF_AM_RM_mean"
map_name = "V1_Base"   # tag it as reward‑modeling
save_dir = f"./Agents/MaskPPO/{map_name}/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

tb_log_dir = f"./tb_logs/MaskPPO/{map_name}/{agent_name}/"
os.makedirs(tb_log_dir, exist_ok=True)

# ───────────────── env + wrappers ─────────────────────────────
def mask_fn(env):  # env is the FlattenActionWrapper
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
    tensorboard_log=tb_log_dir
)

# ───────────────────── training loop ──────────────────────────
# total_timesteps = 2_000_000   # 2 M
# save_interval   = 500_000   # every 500 k
total_timesteps = 3
save_interval   = 1
tb_callback = TBRewardLogger()

for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval,
                reset_num_timesteps=False,
                callback=tb_callback, 
                progress_bar=True)
    # model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")
    model.save(f"{save_dir}{agent_name}_{(i + save_interval)}")

model.save(f"{save_dir}{agent_name}_final")
env.close()

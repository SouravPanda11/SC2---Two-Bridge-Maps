import sys, os, torch, numpy as np
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# Environment imports
from Environments.AM_RM_mean.TB_env_SF_AM_RM_mean_V2_Base import TwoBridgeEnv, N_FRIEND, N_ENEMY


# ───────────────────── Reproducibility ─────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ──────────────────── FLATTEN-ACTION WRAPPER ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) →
    MultiDiscrete([3, 2×N_FRIEND, 9, N_ENEMY+1])
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([3] + [2]*N_FRIEND + [9] + [N_ENEMY + 1])

        # bits beyond the verb-level mask that are always legal
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        # Advertise flattened mask to SB3
        flat_len = 3 + len(self._mask_template)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

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
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
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
print(f"Using device: {device} | SEED={SEED}")

agent_name = "SB_MaskPPO_SF_AM_RM_mean"
map_name   = "V2_Base"

save_dir = f"./Agents/MaskPPO/{map_name}/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

tb_log_dir = f"./tb_logs/MaskPPO/{map_name}/{agent_name}/"
os.makedirs(tb_log_dir, exist_ok=True)


# ───────────────── env + wrappers (MUST match training) ─────────────────
def mask_fn(env):
    return env.action_masks()

base_env = TwoBridgeEnv(visualize=False)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

env.reset(seed=SEED)


# ───────────────────── resume checkpoint ─────────────────────
resume_k = 3500  # in K timesteps
ckpt_path = f"{save_dir}{agent_name}_{resume_k}K.zip"

assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
print(f"Resuming from: {ckpt_path}")

model = MaskablePPO.load(
    ckpt_path,
    env=env,                  
    device=device,
    tensorboard_log=tb_log_dir,
    seed=SEED
)

# ───────────────────── continue training ─────────────────────
TOTAL_TIMESTEPS = 5_000_000
ALREADY_DONE    = resume_k * 1000
REMAINING       = TOTAL_TIMESTEPS - ALREADY_DONE

SAVE_INTERVAL   = 500_000
tb_callback     = TBRewardLogger()

print(f"Already done: {ALREADY_DONE} | Remaining: {REMAINING}")

steps_done = 0
while steps_done < REMAINING:
    step_chunk = min(SAVE_INTERVAL, REMAINING - steps_done)

    model.learn(
        total_timesteps=step_chunk,
        reset_num_timesteps=False,   # keep global timestep continuity
        callback=tb_callback,
        progress_bar=True
    )

    steps_done += step_chunk
    current_total = ALREADY_DONE + steps_done

    model.save(f"{save_dir}{agent_name}_{current_total // 1000}K")

model.save(f"{save_dir}{agent_name}_final")
env.close()
print("Done.")

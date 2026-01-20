import sys, os, torch, numpy as np
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# Environment imports (MUST match the training file)
from Environments.FAM_CAM.TB_env_FAM_V1_Combat_Cam import TwoBridgeEnv, N_FRIEND, N_ENEMY


# ───────────────────── Reproducibility (single seed) ─────────────────────
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ──────────────────── FLATTEN-ACTION WRAPPER (MUST match training) ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) →
    MultiDiscrete([3, 2×N_FRIEND, 9, N_ENEMY+1])

    Flattens the env's per-branch action_mask dict into a single boolean vector:
      flat_mask = [verb(3),
                   who_0(2), who_1(2), ..., who_{N-1}(2),
                   direction(9),
                   enemy_idx(N_ENEMY+1)]
    """
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete([3] + [2]*N_FRIEND + [9] + [N_ENEMY + 1])

        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

        self._last_mask = np.ones(flat_len, dtype=np.int8)

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

    def _convert_mask(self, obs):
        am = obs["action_mask"]
        verb_mask = np.asarray(am["verb"], dtype=np.int8)  # (3,)

        who_bits = np.asarray(am["who"], dtype=np.int8)    # (N_FRIEND,)
        who_pairs = []
        for b in who_bits:
            who_pairs.extend([1, int(b)])  # allow_0 always, allow_1 iff b==1

        direction_mask = np.asarray(am["direction"], dtype=np.int8)   # (9,)
        enemy_mask     = np.asarray(am["enemy_idx"], dtype=np.int8)   # (N_ENEMY+1,)

        flat_mask = np.concatenate([
            verb_mask,
            np.asarray(who_pairs, np.int8),
            direction_mask,
            enemy_mask
        ], dtype=np.int8)

        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

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

agent_name = "SB_MaskPPO_FAM_CAM"
map_name   = "V1_Combat_Cam"

save_dir   = f"./Agents/MaskPPO/{map_name}/saved_models/{agent_name}/"
tb_log_dir = f"./tb_logs/MaskPPO/{map_name}/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(tb_log_dir, exist_ok=True)


# ───────────────── env + wrappers (MUST match training) ─────────────────
def mask_fn(env):
    return env.action_masks()

base_env = TwoBridgeEnv(visualize=False)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

# Seed the environment RNG (Gymnasium style)
env.reset(seed=SEED)


# ───────────────────── resume checkpoint ─────────────────────
# Pick ONE of these patterns:

# (A) Resume by K-timestep checkpoint name (matches your save convention)
resume_k  = 1500  # e.g., 500, 1000, 1500, ..., 5000
ckpt_path = f"{save_dir}{agent_name}_{resume_k}K.zip"

# (B) Or point directly:
# ckpt_path = r"./Agents/MaskPPO/V1_Combat_Cam/saved_models/SB_MaskPPO_FAM_CAM/SB_MaskPPO_FAM_CAM_1500K.zip"
# resume_k  = int(os.path.basename(ckpt_path).split("_")[-1].replace("K.zip", ""))

assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
print(f"Resuming from: {ckpt_path}")

model = MaskablePPO.load(
    ckpt_path,
    env=env,                  # IMPORTANT: pass env again
    device=device,
    tensorboard_log=tb_log_dir,
    seed=SEED
)

# ───────────────────── continue training ─────────────────────
TOTAL_TIMESTEPS = 5_000_000
ALREADY_DONE    = resume_k * 1000
REMAINING       = TOTAL_TIMESTEPS - ALREADY_DONE
assert REMAINING > 0, f"Nothing to do. TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, ALREADY_DONE={ALREADY_DONE}"

SAVE_INTERVAL = 500_000
tb_callback   = TBRewardLogger()

print(f"Already done: {ALREADY_DONE} | Remaining: {REMAINING} | Save interval: {SAVE_INTERVAL}")

steps_done = 0
while steps_done < REMAINING:
    step_chunk = min(SAVE_INTERVAL, REMAINING - steps_done)

    model.learn(
        total_timesteps=step_chunk,
        reset_num_timesteps=False,   # keep global timestep continuity in TB + schedulers
        callback=tb_callback,
        progress_bar=True
    )

    steps_done += step_chunk
    current_total = ALREADY_DONE + steps_done

    # Save using TOTAL timesteps so filenames stay consistent
    model.save(f"{save_dir}{agent_name}_{current_total // 1000}K")
    print(f"Saved: {agent_name}_{current_total // 1000}K")

model.save(f"{save_dir}{agent_name}_final")
env.close()
print("Done.")

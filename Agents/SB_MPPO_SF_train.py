import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# train_two_bridge.py
import os, gymnasium as gym, torch
from sb3_contrib import MaskablePPO              # pip install sb3-contrib
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import NatureCNN
from sb3_contrib.common.maskable.utils import get_action_masks
from Environments.TB_env_SF_AS14 import TwoBridgeEnv  # new env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import CombinedExtractor

# ── 1. Mask adapter ────────────────────────────────────────────────────
class MaskInfoWrapper(gym.Wrapper):
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        info["action_mask"] = self._build_mask(obs["avail_actions"])
        return obs, info

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        info["action_mask"] = self._build_mask(obs["avail_actions"])
        return obs, rew, done, trunc, info

    # ← MaskablePPO calls this every sampling step
    def action_masks(self):
        return self._last_mask

    # ---------------------------------------------------------------
    def _build_mask(self, func_mask: np.ndarray) -> np.ndarray:
        """
        Build a single 1-D legality mask of length
            N_FUNCS  +  SCREEN_RES  +  SCREEN_RES
        = action_space.nvec.sum()
        """
        func_mask = func_mask.astype(np.uint8).ravel()         # (N_FUNCS,)

        nvec      = self.unwrapped.action_space.nvec           # [N_FUNCS, 64, 64]
        xy_len    = int(nvec[1] + nvec[2])                     # 64 + 64 = 128
        xy_mask   = np.ones(xy_len, dtype=np.uint8)            # (128,)

        self._last_mask = np.concatenate([func_mask, xy_mask]) # (N_FUNCS+128,)
        return self._last_mask


# ── 2. Vectorised env (single process using DummyVecEnv) ───────────────
def make_env():
    return MaskInfoWrapper(TwoBridgeEnv(visualize=False))

# ── 4. Policy kwargs ───────────────────────────────────────────────────
policy_kwargs = dict(
    features_extractor_class=CombinedExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=128),  # Output dimension for CNN features
)

if __name__ == "__main__":
    # Wrap the environment in DummyVecEnv
    vec_env = DummyVecEnv([make_env])  # Single environment

    # ── 5. Instantiate MaskablePPO ─────────────────────────────────────────
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs/SB_MPPO_SP/",
    )

    # ── 6. Train & checkpoint ──────────────────────────────────────────────
    total_steps = 2_000_000  # 2 million timesteps
    checkpoint_every = 500_000  # Save every 500k timesteps
    save_dir = "./Agents/saved_models/SB_MaskablePPO/"
    os.makedirs(save_dir, exist_ok=True)

    steps = 0
    print_interval = 100_000  # Print progress every 100k timesteps

    while steps < total_steps:
        model.learn(total_timesteps=checkpoint_every, reset_num_timesteps=False)
        steps += checkpoint_every
        model.save(f"{save_dir}maskppo_{steps}")
        
        # Print progress after every 100k timesteps
        if steps % print_interval == 0:
            print(f"Progress: {steps} timesteps completed")

    print(f"Training completed. Final model saved at {total_steps} timesteps.")
    model.save(f"{save_dir}maskppo_final")
    vec_env.close()

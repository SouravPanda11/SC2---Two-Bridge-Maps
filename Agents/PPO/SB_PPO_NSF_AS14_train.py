import sys, os, random
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import stable_baselines3 as sb3

# Environment import
from Environments.TB_env_NSF_AS14_V2_Base import TwoBridgeEnv

# ───────────────────── Reproducibility (single seed) ─────────────────────
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: cpu | SEED={SEED}")

# ───────────────────────────── logging / dirs ────────────────────────────
agent_name = "SB_PPO_NSF_AS14"
map_name   = "NSF_AS14_V2_Base"   # just for folder structure consistency

save_dir = f"./Agents/PPO/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

tb_log_dir = f"./tb_logs/PPO/{agent_name}/"
os.makedirs(tb_log_dir, exist_ok=True)

# ───────────────────────────── env + seed ───────────────────────────────
env = TwoBridgeEnv(visualize=True)

# Seed the environment RNG (Gymnasium style)
env.reset(seed=SEED)

# ───────────────────────────── model (PPO) ──────────────────────────────
model = sb3.PPO(
    "MlpPolicy",
    env,
    device="cpu",                 # non-spatial features
    verbose=1,
    tensorboard_log=tb_log_dir,
    seed=SEED                     # IMPORTANT: SB3 RNG seed
)

# ───────────────────────────── training loop ────────────────────────────
TOTAL_TIMESTEPS = 2_000_000
SAVE_INTERVAL   = 500_000

for i in range(0, TOTAL_TIMESTEPS, SAVE_INTERVAL):
    model.learn(
        total_timesteps=SAVE_INTERVAL,
        reset_num_timesteps=False,
        progress_bar=True
    )
    model.save(f"{save_dir}{agent_name}_{(i + SAVE_INTERVAL) // 1000}K")

model.save(f"{save_dir}{agent_name}_final")
env.close()

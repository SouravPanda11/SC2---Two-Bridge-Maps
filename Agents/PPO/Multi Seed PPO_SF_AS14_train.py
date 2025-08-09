import os, sys, random
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed

# ----- project imports -----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Environments.TB_env_SF_AS14_V2_Base import TwoBridgeEnv

# ----- config -----
agent_name = "SB_PPO_SF_AS14"
seeds = [0, 1, 2, 3, 4]

total_timesteps = 2_000_000
save_interval = 500_000  # save every 500K

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Root dirs
save_root = f"./Agents/PPO/saved_models/{agent_name}"
tb_root   = f"./tb_logs/{agent_name}"

os.makedirs(save_root, exist_ok=True)
os.makedirs(tb_root, exist_ok=True)

# Optional: nice, compact tqdm formatting
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def make_env(seed: int):
    """Create and seed the environment deterministically."""
    env = TwoBridgeEnv(visualize=False)
    # Gymnasium-style seeding
    try:
        env.reset(seed=seed)
    except TypeError:
        # If older Gym API:
        if hasattr(env, "seed"):
            env.seed(seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env

for seed in seeds:
    print(f"\n==== Training seed {seed} ====")
    # Global seeds
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Per-seed dirs
    seed_save_dir = os.path.join(save_root, f"seed_{seed}")
    seed_tb_dir   = os.path.join(tb_root,   f"seed_{seed}")
    os.makedirs(seed_save_dir, exist_ok=True)
    os.makedirs(seed_tb_dir,   exist_ok=True)

    # Env + model
    env = make_env(seed)
    model = sb3.PPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=1,
        seed=seed,
        tensorboard_log=seed_tb_dir,
    )

    # Interval training with progress bars
    intervals = range(0, total_timesteps, save_interval)
    outer_iter = tqdm(intervals, desc=f"Seed {seed} intervals", leave=True) if tqdm else intervals

    for i in outer_iter:
        # SB3 progress bar for the internal training loop
        pbar_cb = ProgressBarCallback()
        model.learn(
            total_timesteps=save_interval,
            reset_num_timesteps=False,
            callback=pbar_cb,
            progress_bar=False,  # SB3>=2.0 has progress_bar arg; keep False since we pass our callback
        )
        # Save checkpoint
        k = (i + save_interval) // 1000
        ckpt_path = os.path.join(seed_save_dir, f"{agent_name}_{k}K")
        model.save(ckpt_path)

    # Final save
    final_path = os.path.join(seed_save_dir, f"{agent_name}_final")
    model.save(final_path)
    env.close()

print("\nAll seeds finished.")
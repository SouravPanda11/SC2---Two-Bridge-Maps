import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import stable_baselines3 as sb3

# Import environment
from Environments.TB_env_SF_AS14_V2_Base import TwoBridgeEnv

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define agent name and create save directory
agent_name = "SB_PPO_SF_AS14"
save_dir = f"./Agents/PPO/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# Initialize environment and model
env = TwoBridgeEnv(visualize=False)
model = sb3.PPO(
    "MultiInputPolicy",  # Use MultiInputPolicy to handle spatial + vector obs
    env,
    device=device,
    verbose=1,
    tensorboard_log=f"./tb_logs/{agent_name}/"
)

# Training loop with interval saving
total_timesteps = 2_000_000  # 2 million timesteps
save_interval = 400_000  # Save every 400K timesteps
# total_timesteps = 10
# save_interval = 3
for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")
    # model.save(f"{save_dir}{agent_name}_{(i + save_interval)}")

# Final model save
model.save(f"{save_dir}{agent_name}_final")
env.close()
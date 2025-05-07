import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import stable_baselines3 as sb3
from Environments.TB_env_NSF import TwoBridgeEnv

# Check if GPU is available (optional)
if torch.cuda.is_available():
    print(f"GPU found, but we'll use CPU.")
else:
    print("GPU not available. Using CPU.")

# Define the agent name
agent_name = "SB_PPO_NSF"

# Create a folder to save models for this agent
save_dir = f"./Agents/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# Initialize the environment and model
env = TwoBridgeEnv(visualize=True)
model = sb3.PPO(
    "MlpPolicy",
    env,
    device="cpu",  # Force CPU as using non-spatial features
    verbose=1,
    tensorboard_log="./tb_logs/"
)

# Train the model and save at intervals
total_timesteps = 10
save_interval = 3
for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    # model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")
    model.save(f"{save_dir}{agent_name}_{(i + save_interval)}K")

# Save the final model
model.save(f"{save_dir}{agent_name}_final")
env.close()
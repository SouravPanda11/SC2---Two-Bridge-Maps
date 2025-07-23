import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import stable_baselines3 as sb3

# Import environment
from Environments.TB_env_NSF_AS14_V2_Base import TwoBridgeEnv

# Define the agent name
agent_name = "SB_PPO_NSF_AS14"

# Create a folder to save models for this agent
save_dir = f"./Agents/PPO/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# Initialize the environment and model
env = TwoBridgeEnv(visualize=True)
model = sb3.PPO(
    "MlpPolicy",
    env,
    device="cpu",  # Force CPU as using non-spatial features
    verbose=1,
    tensorboard_log=f"./tb_logs/{agent_name}/"
)

# Train the model and save at intervals
total_timesteps = 1_000_000  # 1 million timesteps
save_interval = 300_000  # Save every 300K timesteps
for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(f"{save_dir}{agent_name}_{(i + save_interval) // 1000}K")

# Save the final model
model.save(f"{save_dir}{agent_name}_final")
env.close()
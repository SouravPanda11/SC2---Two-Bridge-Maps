import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import stable_baselines3 as sb3
from Environments.TB_env_SF_AS14 import TwoBridgeEnv  # make sure this points to your spatial+vector env

# ------------------- Check Device -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------- Setup Save Path -------------------
agent_name = "SB_A2C_SF_AS14"
save_dir = f"./Agents/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# ------------------- Initialize Environment -------------------
env = TwoBridgeEnv(visualize=False)

# ------------------- Create A2C Model -------------------
model = sb3.A2C(
    policy="MultiInputPolicy",  # handles spatial (dict obs)
    env=env,
    device=device,
    verbose=1,
    tensorboard_log=f"./tb_logs/{agent_name}/",
    n_steps=8,                 # You can tune this
    gamma=0.99,
    learning_rate=7e-4         # A2C default, but tunable
)

# ------------------- Training Loop -------------------
total_timesteps = 2_000_000
save_interval = 400_000

for i in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(f"{save_dir}{agent_name}_{(i + save_interval)//1000}K")

# ------------------- Final Save -------------------
model.save(f"{save_dir}{agent_name}_final")
env.close()

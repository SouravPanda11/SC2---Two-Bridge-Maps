# train_a2c_nsf_as14.py
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import stable_baselines3 as sb3
from Environments.TB_env_NSF_AS14 import TwoBridgeEnv

# --------------------------- device ---------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # Force CPU for A2C
print(f"Using {device.upper()} for A2C")

# --------------------------- run label ------------------------
agent_name = "SB_A2C_NSF_AS14"
save_dir   = f"./Agents/saved_models/{agent_name}/"
os.makedirs(save_dir, exist_ok=True)

# --------------------------- env ------------------------------
env = TwoBridgeEnv(visualize=False)

# --------------------------- model ----------------------------
model = sb3.A2C(
    policy               = "MlpPolicy",
    env                  = env,
    n_steps              = 8,          # 8*env.step = 64 game loops
    gamma                = 0.99,
    learning_rate        = 2.5e-4,
    ent_coef             = 0.01,
    vf_coef              = 0.5,
    max_grad_norm        = 0.5,
    device               = device,
    tensorboard_log      = "./tb_logs/",
    verbose              = 1,
)

# --------------------------- training loop --------------------
total_timesteps = 2_000_000  # 2 million timesteps
save_interval   = 400_000  # Save every 400K timesteps

for step in range(0, total_timesteps, save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    model.save(f"{save_dir}{agent_name}_{(step + save_interval)//1000}K")
    # model.save(f"{save_dir}{agent_name}_{(step + save_interval)}")

# --------------------------- final save -----------------------
model.save(f"{save_dir}{agent_name}_final")
env.close()

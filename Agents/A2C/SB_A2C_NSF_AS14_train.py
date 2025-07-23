import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import stable_baselines3 as sb3

# Import environment
from Environments.TB_env_NSF_AS14_V2_Base import TwoBridgeEnv

# --------------------------- device ---------------------------
device = "cpu"  # Force CPU for A2C
print(f"Using {device.upper()} for A2C")

# --------------------------- run label ------------------------
agent_name = "SB_A2C_NSF_AS14"
save_dir   = f"./Agents/A2C/saved_models/{agent_name}/"
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
    tensorboard_log      = f"./tb_logs/{agent_name}/",
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

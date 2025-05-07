import torch
import stable_baselines3 as sb3
from Environments.TB_env_NSF import TwoBridgeEnv

# Check if GPU is available (optional)
if torch.cuda.is_available():
    print(f"GPU found, but we'll use CPU.")
else:
    print("GPU not available. Using CPU.")

# Initialize the environment and model
env = TwoBridgeEnv(visualize=True)
model = sb3.PPO(
    "MlpPolicy",
    env,
    device="cpu",  # Force CPU
    verbose=1,
    tensorboard_log="./tb_logs/"
)
model.learn(1000)
model.save("two_bridge_ppo")
env.close()

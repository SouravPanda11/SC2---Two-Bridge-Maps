from absl import flags ; flags.FLAGS([''])      # silence PySC2 flags
import gymnasium as gym
from Environments.TB_env_AS20 import TwoBridgeEnv

env = TwoBridgeEnv(obs_type="vector", visualize=False)
obs = env.reset()
print("Initial observation:", obs)
action = env.action_space.sample()
obs, reward, done, _, info = env.step(action)
print("Step result:", obs, reward, done, info)
env.close()

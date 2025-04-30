from absl import flags ; flags.FLAGS([''])      # silence PySC2 flags
import gymnasium as gym, two_bridge_env                   # just import env.py once

two_bridge_env = gym.make("TwoBridge-v0", visualize=False)
obs, _ = two_bridge_env.reset()

total_r = 0
while True:
    action = two_bridge_env.action_space.sample()
    obs, r, done, tr, _ = two_bridge_env.step(action)
    total_r += r
    if done: break

print("episode reward:", total_r)
two_bridge_env.close()

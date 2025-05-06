"""
Run trained agent for N episodes, collect win/loss stats and plot.
"""
import numpy as np, matplotlib.pyplot as plt, torch
import stable_baselines3 as sb3
from two_bridge_env import TwoBridgeEnv

MODEL_PATH   = "two_bridge_ppo.zip"
EPISODES     = 3            # evaluation batch size
RENDER       = False           # switch True if you want to watch

env   = TwoBridgeEnv(visualize=RENDER)
model = sb3.PPO.load(MODEL_PATH, env=env, device="cpu")

counters = {
    "nav_win"      : 0,
    "combat_win"   : 0,
    "combat_loss"  : 0,
    "timeout_loss" : 0,
    "tie"          : 0,
    "galaxy_10"    : 0,   # for completeness (0=undef, 1=defeat, 2=unknown, 3=win)
    "galaxy_1"     : 0,
    "galaxy_3"     : 0,
}

for ep in range(EPISODES):
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
    res = info.get("result", "tie")
    counters[res] = counters.get(res, 0) + 1
    if ep % 20 == 0:
        print(f"[{ep}/{EPISODES}] result:", res)

env.close()

# --- aggregate ----------------------------------------------------------
labels = ["nav_win","combat_win","combat_loss","timeout_loss","tie"]
values = [counters.get(k,0) for k in labels]

win_pct  = 100* (counters["nav_win"]+counters["combat_win"]) / EPISODES
print(f"\nTotal episodes: {EPISODES}")
print(f"Win rate       : {win_pct:5.1f}%")

# --- plot ---------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.bar(labels, values)
plt.ylabel("# episodes out of "+str(EPISODES))
plt.title("Agent performance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

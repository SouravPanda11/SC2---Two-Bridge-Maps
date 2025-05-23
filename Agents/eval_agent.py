import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np, matplotlib.pyplot as plt, torch
import stable_baselines3 as sb3
# from Environments.TB_env_SF_AS14 import TwoBridgeEnv
# from Environments.TB_env_NSF_AS14 import TwoBridgeEnv
from Environments.TB_env_SF_AM import TwoBridgeEnv

# AGENT_NAME = "SB_PPO_NSF"
# AGENT_NAME = "SB_PPO_SF_AS14"
# AGENT_NAME = "SB_A2C_SF_AS14"
# AGENT_NAME = "SB_A2C_NSF_AS14"
AGENT_NAME = "SB_MaskPPO_SF_AM"

# Absolute model path
# MODEL_PATH = os.path.join(project_root, "Agents", "saved_models", AGENT_NAME, f"{AGENT_NAME}_final.zip")
MODEL_PATH = os.path.join(project_root, "Agents", "saved_models", AGENT_NAME, f"{AGENT_NAME}_400k.zip")
EPISODES = 200
RENDER = False

# Initialize environment
env = TwoBridgeEnv(visualize=RENDER)

# Load model with error handling
try:
    model = sb3.PPO.load(MODEL_PATH, env=env, device="cpu")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")
    sys.exit(1)

# Initialize result counters
counters = {
    "nav_win": 0,
    "combat_win": 0,
    "combat_loss": 0,
    "timeout_loss": 0,
    "tie": 0,
    "galaxy_10": 0,  # for completeness (0=undef, 1=defeat, 2=unknown, 3=win)
    "galaxy_1": 0,
    "galaxy_3": 0,
}

# Run episodes
for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

    # Handle result safely
    res = info.get("result", "tie")
    if res not in counters:
        print(f"[WARN] Unexpected result: '{res}', defaulting to 'tie'")
        res = "tie"
    counters[res] += 1

    if ep % 20 == 0 or ep == EPISODES - 1:
        print(f"[{ep + 1}/{EPISODES}] result: {res}")

env.close()

# --- Aggregated Results ---
labels = ["nav_win", "combat_win", "combat_loss", "timeout_loss", "tie"]
values = [counters.get(k, 0) for k in labels]

win_pct = 100 * (counters["nav_win"] + counters["combat_win"]) / EPISODES
print(f"\nTotal episodes: {EPISODES}")
print(f"Win rate       : {win_pct:.1f}%")

# --- Plot Results ---
plt.figure(figsize=(7, 4))
plt.bar(labels, values)
plt.ylabel("# episodes out of " + str(EPISODES))
plt.title("Agent performance")
plt.xticks(rotation=30)
plt.tight_layout()

# Save the plot in the Agent Performance Charts folder
performance_folder = os.path.join(project_root, "Agents", "Agent Performance Charts")
os.makedirs(performance_folder, exist_ok=True)  # Ensure the folder exists
plot_path = os.path.join(performance_folder, f"{AGENT_NAME}_performance_{EPISODES}_ep.png")
plt.savefig(plot_path)
plt.show()

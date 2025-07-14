import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np, matplotlib.pyplot as plt, torch
import stable_baselines3 as sb3

from Environments.TB_env_NSF_AS14 import TwoBridgeEnv
# from Environments.TB_env_SF_AS14 import TwoBridgeEnv
# from Environments.TB_env_SF_AM import TwoBridgeEnv

import matplotlib
matplotlib.use('Agg')


AGENT_NAME = "SB_PPO_NSF"
# AGENT_NAME = "SB_PPO_SF_AS14"
# AGENT_NAME = "SB_A2C_SF_AS14"
# AGENT_NAME = "SB_A2C_NSF_AS14"
# AGENT_NAME = "SB_MaskPPO_SF_AM"

# Absolute model path
MODEL_PATH = os.path.join(project_root, "Agents", "saved_models", AGENT_NAME, f"{AGENT_NAME}_final.zip")
# MODEL_PATH = os.path.join(project_root, "Agents", "saved_models", AGENT_NAME, f"{AGENT_NAME}_400k.zip")
EPISODES = 3
RENDER = False

# replay_output_dir = os.path.join(project_root, "Agents", "Replays", AGENT_NAME)
replay_output_dir = os.path.abspath(
    os.path.join(project_root, "Agents", "Replays", AGENT_NAME)
)
print("Replay directory:", replay_output_dir)
os.makedirs(replay_output_dir, exist_ok=True)

env = TwoBridgeEnv(
    visualize=True,              # Show window
    realtime=True,               # Play in real time
    replay_dir=replay_output_dir,
    save_replay_episodes=1       # Save every episode
)

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

performance_folder = os.path.join(project_root, "Agents", "Agent Performance Charts")
os.makedirs(performance_folder, exist_ok=True)  # Ensure the folder exists

reward_value_plot_path = os.path.join(performance_folder, AGENT_NAME)
os.makedirs(reward_value_plot_path, exist_ok=True)  # Ensure the folder exists

# Run episodes
for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    
    episode_rewards = []
    value_predictions = []
    
    while not done:
        # Predict action and value
        action, _ = model.predict(obs, deterministic=True)

        # Get predicted value estimate for current state
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor.to(model.device)).cpu().item()

        # Step environment
        obs, reward, done, trunc, info = env.step(action)

        # Log actual and predicted rewards
        episode_rewards.append(reward)
        value_predictions.append(value)

    # Save the replay manually
    if hasattr(env, "_env") and hasattr(env._env, "save_replay"):
        env._env.save_replay(replay_output_dir, prefix=f"eval_ep_{ep+1}")
            
    # Handle result safely
    res = info.get("result", "tie")
    if res not in counters:
        print(f"[WARN] Unexpected result: '{res}', defaulting to 'tie'")
        res = "tie"
    counters[res] += 1
    
    print(f"[{ep+1}/{EPISODES}] result: {res} → Replay saved as eval_ep_{ep+1}.SC2Replay")

    # Plot: Environment reward vs Agent value prediction
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, label="Env Reward", marker='o', linestyle='--')
    plt.plot(value_predictions, label="Agent Value Estimate", marker='x', linestyle='-')
    plt.xlabel("Timestep")
    plt.ylabel("Reward / Value")
    plt.title(f"{AGENT_NAME} - Episode {ep+1}: Reward vs Value Estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the reward-vs-value plot
    reward_plot_path = os.path.join(reward_value_plot_path, f"{AGENT_NAME}_eval_ep_{ep+1}_rewards_vs_values.png")
    plt.savefig(reward_plot_path)
    plt.close()
    
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
plot_path = os.path.join(performance_folder, f"{AGENT_NAME}_performance_{EPISODES}_ep.png")
plt.savefig(plot_path)
plt.show()

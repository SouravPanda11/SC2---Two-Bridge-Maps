import sys
import os
import glob

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import numpy as np, matplotlib.pyplot as plt, torch
import stable_baselines3 as sb3
import matplotlib
matplotlib.use('Agg')

# Import environment
from Environments.TB_env_SF_AS14_V2_Base import TwoBridgeEnv

# Agent name
AGENT_NAME = "SB_PPO_SF_AS14"

# Absolute model path
MODEL_PATH = os.path.join(project_root, "Agents", "PPO", "saved_models", AGENT_NAME, f"{AGENT_NAME}_1600000.zip")
EPISODES = 3
RENDER = False

# Replay output directory
replay_output_dir = os.path.abspath(
    os.path.join(project_root, "Replays", "PPO", AGENT_NAME)
)
os.makedirs(replay_output_dir, exist_ok=True)

# Create environment with replay capabilities
env = TwoBridgeEnv(
    visualize=True,              # Show window
    realtime=False,               # Play in real time
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

# Create directory for performance charts
performance_folder = os.path.join(project_root, "Agent Performance Charts", "PPO", AGENT_NAME)
os.makedirs(performance_folder, exist_ok=True)  # Ensure the folder exists

# Create directory for reward vs value plots
reward_value_plot_path = os.path.join(performance_folder, "EpRds_vs_Values")
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

        # Convert obs dict to torch tensors and move to model device (GPU)
        obs_tensor = {
            k: torch.tensor(v).float().unsqueeze(0).to(model.device)
            for k, v in obs.items()
        }

        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor).cpu().item()

        # Step environment
        obs, reward, done, trunc, info = env.step(action)

        # Log actual and predicted rewards
        episode_rewards.append(reward)
        value_predictions.append(value)

    # # Save the replay manually
    # if hasattr(env, "_env") and hasattr(env._env, "save_replay"):
    #     env._env.save_replay(replay_output_dir, prefix=f"eval_ep_{ep+1}")
    
    # Save the replay manually
    if hasattr(env, "_env") and hasattr(env._env, "save_replay"):
        env._env.save_replay(replay_output_dir, prefix=f"eval_ep_{ep+1}")
        
        # Rename the most recent .SC2Replay file to remove timestamp
        matching_files = sorted(
            glob.glob(os.path.join(replay_output_dir, f"eval_ep_{ep+1}_*.SC2Replay")),
            key=os.path.getmtime,
            reverse=True
        )
        if matching_files:
            final_path = os.path.join(replay_output_dir, f"eval_ep_{ep+1}.SC2Replay")
            os.rename(matching_files[0], final_path)
            
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
    reward_plot_path = os.path.join(reward_value_plot_path, f"eval_ep_{ep+1}.png")
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

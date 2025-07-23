import sys, os, collections, numpy as np, matplotlib.pyplot as plt, torch
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# ─────────────────── SB3 / gym imports ────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ─────────────────── env + wrapper ────────────────
from Environments.TB_env_SF_AM_V3_Combat import (
    TwoBridgeEnv, N_FRIEND, N_ENEMY
)

class FlattenActionWrapper(Wrapper):
    """Dict(verb, who, dir, enemy_idx) → flat MultiDiscrete; expands mask."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete(
            [3] + [2]*N_FRIEND + [9] + [N_ENEMY + 1]
        )
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, np.int8)

        # correct mask length (26)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(
            3 + len(self._mask_template)
        )
        self.observation_space = spaces.Dict(obs_spaces)

    @staticmethod
    def _unflatten(vec):
        return {
            "verb":      int(vec[0]),
            "who":       np.asarray(vec[1 : 1 + N_FRIEND], np.int8),
            "direction": int(vec[1 + N_FRIEND]),
            "enemy_idx": int(vec[-1]),
        }

    def step(self, a):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(a))
        return self._expand_mask(obs), rew, term, trunc, info

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._expand_mask(obs), info

    def _expand_mask(self, obs):
        obs["action_mask"] = np.concatenate(
            [obs["action_mask"], self._mask_template]
        )
        self._last_mask = obs["action_mask"]
        return obs

    def action_masks(self):
        return self._last_mask

mask_fn = lambda e: e.action_masks()

# Agent name
AGENT_NAME = "SB_MaskPPO_SF_AM"
# Map name
map_name = "V3_Combat"

# Absolute path to the model file
MODEL_PATH = os.path.join(project_root, "Agents", "MaskPPO", map_name, "saved_models",
                          AGENT_NAME, f"{AGENT_NAME}_final.zip")

EPISODES = 3
RENDER   = False

# Replay output directory
replay_output_dir = os.path.abspath(
    os.path.join(project_root, "Replays", "MaskPPO", map_name, AGENT_NAME)
)
os.makedirs(replay_output_dir, exist_ok=True)
print("Replay directory:", replay_output_dir)

# ─────────────────── env / model ──────────────────
base_env = TwoBridgeEnv(visualize=RENDER)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

if not os.path.isfile(MODEL_PATH):
    sys.exit(f"[ERROR] Model file not found at: {MODEL_PATH}")

model = MaskablePPO.load(
    MODEL_PATH, env=env,
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

# ─────────────────── evaluation loop ─────────────
counters = collections.Counter({
    "nav_win": 0,
    "combat_win": 0,
    "combat_loss": 0,
    "timeout_loss": 0,
    "tie": 0,
})

# Create directory for performance charts
performance_folder = os.path.join(project_root, "Agent Performance Charts", "MaskPPO", map_name, AGENT_NAME)
os.makedirs(performance_folder, exist_ok=True)  # Ensure the folder exists

# Create directory for reward vs value plots
reward_value_plot_path = os.path.join(performance_folder, "EpRds_vs_Values")
os.makedirs(reward_value_plot_path, exist_ok=True)  # Ensure the folder exists

# --- Utility: unwrap to get the base TwoBridgeEnv (unwrapped SC2Env) ---
def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

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
    
    # --- Replay saving (corrected) ---
    base = unwrap_env(env)
    if hasattr(base, "_env") and hasattr(base._env, "save_replay"):
        base._env.save_replay(replay_output_dir, prefix=f"eval_ep_{ep+1}")

        # Rename the most recent .SC2Replay file to remove timestamp
        matching_files = sorted(
            glob.glob(os.path.join(replay_output_dir, f"eval_ep_{ep+1}_*.SC2Replay")),
            key=os.path.getmtime,
            reverse=True
        )
        if matching_files:
            final_path = os.path.join(replay_output_dir, f"eval_ep_{ep+1}.SC2Replay")
            os.rename(matching_files[0], final_path)
            print(f"[{ep+1}/{EPISODES}] result: {info.get('result', 'tie')} → Replay saved as eval_ep_{ep+1}.SC2Replay")
        else:
            print(f"[{ep+1}/{EPISODES}] result: {info.get('result', 'tie')} → [WARN] Replay file not found")
    else:
        print(f"[{ep+1}/{EPISODES}] result: {info.get('result', 'tie')} → [WARN] save_replay not accessible")

            
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

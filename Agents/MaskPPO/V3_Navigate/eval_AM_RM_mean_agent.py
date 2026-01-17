import sys, os, collections, numpy as np, matplotlib.pyplot as plt, torch
import pandas as pd
import json
from datetime import datetime
import random
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# ─────────────────── SB3 / gym imports ────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ─────────────────── env import (MODULAR) ─────────
from Environments.AM_RM_mean.TB_env_SF_AM_RM_mean_V3_Navigate import TwoBridgeEnv, N_FRIEND, N_ENEMY

# ==================================================
#                  CONFIG
# ==================================================
SEED = 0
EPISODES = 5
RENDER = False

AGENT_NAME = "SB_MaskPPO_SF_AM_RM_mean"
map_name = "V3_Navigate"

# ==================================================
#                  FEATURE FLAGS
# ==================================================
DO_OVERALL_PERF = True                 # overall performance counts + overall bar chart
SAVE_EPISODE_DETAILS = False           # per-episode decomposed CSV + reward-vs-value plot (by terminal condition)
SAVE_REPLAYS = False                   # save SC2Replay files (by terminal condition)
SAVE_OVERALL_SUMMARY_TO_FILE = True    # save Episode counts + Win rate to disk
SHOW_OVERALL_PLOT = False              # plt.show() for overall bar chart
# ==================================================

# ───────────────────── Reproducibility ─────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────── FLATTEN-ACTION WRAPPER (MODULAR) ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) →
    MultiDiscrete([3, 2×N_FRIEND, 9, N_ENEMY+1])
    """
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete([3] + [2] * N_FRIEND + [9] + [N_ENEMY + 1])

        # bits beyond the verb-level mask that are always legal
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        # Advertise flattened mask to SB3
        flat_len = 3 + len(self._mask_template)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb":      int(a_vec[0]),
            "who":       np.asarray(a_vec[1 : 1 + N_FRIEND], np.int8),
            "direction": int(a_vec[1 + N_FRIEND]),
            "enemy_idx": int(a_vec[-1]),
        }

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        flat_mask = np.concatenate([obs["action_mask"], self._mask_template]).astype(np.int8)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask

def mask_fn(env):
    return env.action_masks()

# ─────────────────── model path ───────────────────
MODEL_PATH = os.path.join(
    project_root, "Agents", "MaskPPO", map_name, "saved_models",
    AGENT_NAME, f"{AGENT_NAME}_final.zip"
)

if not os.path.isfile(MODEL_PATH):
    sys.exit(f"[ERROR] Model file not found at: {MODEL_PATH}")

# ---------- directories ---------------------------
performance_root = os.path.join(project_root, "Agent Performance Charts", "MaskPPO", map_name, AGENT_NAME)
os.makedirs(performance_root, exist_ok=True)

replay_root = os.path.join(project_root, "Replays", "MaskPPO", map_name, AGENT_NAME)

RESULT_KINDS = ["nav_win", "combat_win", "combat_loss", "timeout_loss", "tie"]

folders = {}
if SAVE_EPISODE_DETAILS or SAVE_REPLAYS:
    if SAVE_REPLAYS:
        os.makedirs(replay_root, exist_ok=True)

    for rk in RESULT_KINDS:
        perf_dir = os.path.join(performance_root, rk)
        folders[rk] = {}

        if SAVE_EPISODE_DETAILS:
            folders[rk]["plots"] = os.path.join(perf_dir, "EpRds_vs_Values")
            folders[rk]["csv"]   = os.path.join(perf_dir, "Decomposed_reward")
            os.makedirs(folders[rk]["plots"], exist_ok=True)
            os.makedirs(folders[rk]["csv"], exist_ok=True)

        if SAVE_REPLAYS:
            folders[rk]["replay"] = os.path.join(replay_root, rk)
            os.makedirs(folders[rk]["replay"], exist_ok=True)

# ─────────────────── env / model ──────────────────
base_env = TwoBridgeEnv(visualize=RENDER)
flat_env = FlattenActionWrapper(base_env)
env = ActionMasker(flat_env, mask_fn)

# Seed the environment RNG (Gymnasium style)
env.reset(seed=SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskablePPO.load(MODEL_PATH, env=env, device=device)

# ---------- evaluation loop -----------------------
counters = collections.Counter({k: 0 for k in RESULT_KINDS})

def unwrap_env(env_):
    while hasattr(env_, "env"):
        env_ = env_.env
    return env_

for ep in range(EPISODES):
    obs, _ = env.reset(seed=SEED + ep)  # optional: different episode seeds but still reproducible
    done = False

    logs = [] if SAVE_EPISODE_DETAILS else None
    ep_r = [] if SAVE_EPISODE_DETAILS else None
    ep_v = [] if SAVE_EPISODE_DETAILS else None

    while not done:
        act, _ = model.predict(obs, deterministic=True)

        # value estimate only if needed
        if SAVE_EPISODE_DETAILS:
            obs_tensor = {k: torch.tensor(v).float().unsqueeze(0).to(model.device) for k, v in obs.items()}
            with torch.no_grad():
                v_hat = model.policy.predict_values(obs_tensor).cpu().item()
        else:
            v_hat = None

        obs, rew, done, trunc, info = env.step(act)

        if SAVE_EPISODE_DETAILS:
            step = unwrap_env(env).get_reward_components()
            step.update({"reward": rew, "value_estimate": v_hat})
            logs.append(step)
            ep_r.append(rew)
            ep_v.append(v_hat)

    # -------- result & counters --------------------
    res = info.get("result", "tie")
    if res not in RESULT_KINDS:
        res = "tie"
    counters[res] += 1

    # -------- save episode details -----------------
    if SAVE_EPISODE_DETAILS:
        dest = folders[res]
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(dest["csv"], f"decomposed_ep_{ep+1}.csv"), index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(ep_r, label="Env Reward", marker="o", ls="--")
        plt.plot(ep_v, label="Value Estimate", marker="x")
        plt.xlabel("Timestep")
        plt.ylabel("Reward / Value")
        plt.title(f"{AGENT_NAME} – Episode {ep+1} ({res})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}.png"))
        plt.close()

    # -------- save replay --------------------------
    if SAVE_REPLAYS:
        dest = folders[res]
        base = unwrap_env(env)
        if hasattr(base, "_env") and hasattr(base._env, "save_replay"):
            base._env.save_replay(dest["replay"], prefix=f"ep_{ep+1}")
            newest = sorted(
                glob.glob(os.path.join(dest["replay"], f"ep_{ep+1}_*.SC2Replay")),
                key=os.path.getmtime, reverse=True
            )
            if newest:
                os.rename(newest[0], os.path.join(dest["replay"], f"ep_{ep+1}.SC2Replay"))

    print(f"[{ep+1}/{EPISODES}] result: {res}")

env.close()

# ---------- overall summary ------------------------
overall_plot_path = None
if DO_OVERALL_PERF:
    labels = RESULT_KINDS
    values = [counters[k] for k in RESULT_KINDS]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=30)
    plt.ylabel(f"# episodes out of {EPISODES}")
    plt.title("Agent performance")
    plt.tight_layout()

    overall_plot_path = os.path.join(performance_root, f"{AGENT_NAME}_performance_{EPISODES}_ep.png")
    plt.savefig(overall_plot_path)

    if SHOW_OVERALL_PLOT:
        plt.show()
    else:
        plt.close()

episode_counts = dict(counters)
win_pct = 100.0 * (counters["nav_win"] + counters["combat_win"]) / EPISODES

print("\nEpisode counts:", episode_counts)
print(f"Win rate: {win_pct:.1f}%")

if SAVE_OVERALL_SUMMARY_TO_FILE:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "agent": AGENT_NAME,
        "map": map_name,
        "episodes": EPISODES,
        "seed": SEED,
        "episode_counts": episode_counts,
        "win_rate_percent": round(win_pct, 1),
        "saved_overall_plot": (overall_plot_path if DO_OVERALL_PERF else None),
        "flags": {
            "DO_OVERALL_PERF": DO_OVERALL_PERF,
            "SAVE_EPISODE_DETAILS": SAVE_EPISODE_DETAILS,
            "SAVE_REPLAYS": SAVE_REPLAYS,
            "SHOW_OVERALL_PLOT": SHOW_OVERALL_PLOT,
        },
        "timestamp_local": ts,
    }

    json_path = os.path.join(performance_root, f"{AGENT_NAME}_summary_{EPISODES}ep_{ts}.json")
    txt_path  = os.path.join(performance_root, f"{AGENT_NAME}_summary_{EPISODES}ep_{ts}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Agent: {AGENT_NAME}\n")
        f.write(f"Map: {map_name}\n")
        f.write(f"Episodes: {EPISODES}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write(f"Episode counts: {episode_counts}\n")
        f.write(f"Win rate: {win_pct:.1f}%\n")
        if DO_OVERALL_PERF:
            f.write(f"Overall plot: {overall_plot_path}\n")
        f.write(f"\nFlags: {summary['flags']}\n")

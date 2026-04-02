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

# ─────────────────── env import (FAM_CAM) ─────────
from Environments.FAM_CAM.TB_env_FAM_V1_Navigate_Cam import TwoBridgeEnv, N_FRIEND, N_ENEMY

# ─────────────────── argparsing ───────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=10)
args = parser.parse_args()

EPISODES = args.episodes

# ==================================================
#                  CONFIG
# ==================================================
SEED = 0
# EPISODES = 10
RENDER = False

AGENT_NAME = "SB_MaskPPO_FAM_CAM"   
map_name   = "V1_Navigate"

# ==================================================
#                  FEATURE FLAGS
# ==================================================
DO_OVERALL_PERF = True
SAVE_EPISODE_DETAILS = False
SAVE_REPLAYS = True
SAVE_OVERALL_SUMMARY_TO_FILE = True
SHOW_OVERALL_PLOT = False
# ==================================================

# ───────────────────── Reproducibility ─────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────── FLATTEN-ACTION WRAPPER (FAM_CAM) ───────────────────
class FlattenActionWrapper(Wrapper):
    """
    Dict(verb, who, direction, enemy_idx) →
    MultiDiscrete([3, 2×N_FRIEND, 9, N_ENEMY+1])

    Also flattens the env's *dict* action_mask into a single MultiBinary vector:
      [verb(3) | who(N_FRIEND) | direction(9) | enemy_idx(N_ENEMY+1)]
    Then appends always-legal bits to match SB3's MultiDiscrete flattening.
    """
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.MultiDiscrete([3] + [2] * N_FRIEND + [9] + [N_ENEMY + 1])

        # bits beyond the verb-level mask that are always legal (SB3 expects these)
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        # Advertise flattened mask to SB3
        flat_len = 3 + len(self._mask_template)
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

        self._last_mask = np.ones(flat_len, dtype=np.int8)

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
        am = obs["action_mask"]

        # FAM_CAM env provides dict masks
        # expected keys: verb(3), who(N_FRIEND), direction(9), enemy_idx(N_ENEMY+1)
        if isinstance(am, dict):
            verb_m = np.asarray(am["verb"], dtype=np.int8).reshape(-1)
            who_m  = np.asarray(am["who"], dtype=np.int8).reshape(-1)
            dir_m  = np.asarray(am["direction"], dtype=np.int8).reshape(-1)
            ene_m  = np.asarray(am["enemy_idx"], dtype=np.int8).reshape(-1)

            if verb_m.size != 3:
                raise ValueError(f"verb mask size {verb_m.size} != 3")
            if who_m.size != N_FRIEND:
                raise ValueError(f"who mask size {who_m.size} != N_FRIEND={N_FRIEND}")
            if dir_m.size != 9:
                raise ValueError(f"direction mask size {dir_m.size} != 9")
            if ene_m.size != (N_ENEMY + 1):
                raise ValueError(f"enemy_idx mask size {ene_m.size} != N_ENEMY+1={N_ENEMY+1}")

            flat_head = np.concatenate([verb_m, who_m, dir_m, ene_m]).astype(np.int8)
        else:
            flat_head = np.asarray(am, dtype=np.int8).reshape(-1)

        flat_mask = np.concatenate([flat_head[:3], self._mask_template]).astype(np.int8)

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
    obs, _ = env.reset(seed=SEED + ep)
    done = False

    logs = [] if SAVE_EPISODE_DETAILS else None
    ep_r = [] if SAVE_EPISODE_DETAILS else None
    ep_v = [] if SAVE_EPISODE_DETAILS else None

    while not done:
        act, _ = model.predict(obs, deterministic=True)

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

    res = info.get("result", "tie")
    if res not in RESULT_KINDS:
        res = "tie"
    counters[res] += 1

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

    overall_plot_path = os.path.join(performance_root, f"performance_{EPISODES}_ep.png")
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

    json_path = os.path.join(performance_root, f"summary_{EPISODES}ep.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

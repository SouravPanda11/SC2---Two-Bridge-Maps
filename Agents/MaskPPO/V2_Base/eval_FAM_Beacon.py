import sys, os, collections, numpy as np, matplotlib.pyplot as plt, torch
import pandas as pd
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# ─────────────────── SB3 / gym imports ────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ─────────────────── env + wrapper (NAV-ONLY) ─────
# Use the SAME env + N_FRIEND as in the TRAIN script
from Environments.full_action_mask.TB_env_FAM_V2_Base_Beacon import (
    TwoBridgeEnv, N_FRIEND
)

class FlattenActionWrapper(Wrapper):
    """
    NAV-ONLY version:
    Dict(verb, who, direction) →
        MultiDiscrete([2, 2×N_FRIEND, 9])

    Flat action layout:
      [ verb(2),
        who_0(2), who_1(2), ..., who_{N_FRIEND-1}(2),
        direction(9) ]

    For each who_i, the two entries correspond to choices {0,1}.
    """

    def __init__(self, env):
        super().__init__(env)

        # MultiDiscrete layout: [verb] + [who bits] + [direction]
        self.action_space = spaces.MultiDiscrete(
            [2] + [2] * N_FRIEND + [9]
        )

        # Build the observation space: keep everything from env, but
        # advertise a *flat* action_mask whose length is sum(nvec)
        flat_len = int(np.sum(self.action_space.nvec))
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
        }

    def _convert_mask(self, obs):
        """
        Convert env's dict masks → flat vector compatible with ActionMasker.
        Expected obs["action_mask"] structure from env:
          {
            "verb":      (2,),
            "who":       (N_FRIEND,),
            "direction": (9,),
          }
        """
        am = obs["action_mask"]  # dict of np.int8 arrays

        verb_mask      = np.asarray(am["verb"], dtype=np.int8)      # (2,)
        who_bits       = np.asarray(am["who"], dtype=np.int8)       # (N_FRIEND,)
        direction_mask = np.asarray(am["direction"], dtype=np.int8) # (9,)

        # For each who_i (∈ {0,1}), make a 2-entry mask: [allow_0, allow_1]
        who_pairs = []
        for b in who_bits:
            # 0 (don't select) always valid; 1 valid iff unit alive (b==1)
            who_pairs.extend([1, int(b)])

        flat_mask = np.concatenate([
            verb_mask,                      # 2
            np.asarray(who_pairs, np.int8), # 2*N_FRIEND
            direction_mask,                 # 9
        ], dtype=np.int8)

        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(self._unflatten(action))
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    # ActionMasker will call this to fetch the current mask
    def action_masks(self):
        return self._last_mask


mask_fn = lambda e: e.action_masks()

# Agent name (must match TRAIN script)
AGENT_NAME = "SB_MaskPPO_FAM_Beacon"
map_name   = "V2_Base"

# Absolute path to the model file
MODEL_PATH = os.path.join(
    project_root, "Agents", "MaskPPO", map_name, "saved_models",
    AGENT_NAME, f"{AGENT_NAME}_final.zip"
)

EPISODES = 10
RENDER   = False

# ---------- directories ---------------------------
performance_root = os.path.join(
    project_root, "Agent Performance Charts", "MaskPPO", map_name, AGENT_NAME
)
replay_root = os.path.join(
    project_root, "Replays", "MaskPPO", map_name, AGENT_NAME
)
os.makedirs(performance_root, exist_ok=True)
os.makedirs(replay_root, exist_ok=True)

RESULT_KINDS = ["nav_win", "timeout_loss", "combat_win", "combat_loss", "tie"]

folders = {}
for rk in RESULT_KINDS:
    perf_dir = os.path.join(performance_root, rk)
    folders[rk] = {
        "plots":  os.path.join(perf_dir, "EpRds_vs_Values"),
        "csv":    os.path.join(perf_dir, "Decomposed_reward"),
        "replay": os.path.join(replay_root, rk),
    }
    for p in folders[rk].values():
        os.makedirs(p, exist_ok=True)

# ─────────────────── env / model ──────────────────
base_env = TwoBridgeEnv(visualize=RENDER)
flat_env = FlattenActionWrapper(base_env)
env      = ActionMasker(flat_env, mask_fn)

if not os.path.isfile(MODEL_PATH):
    sys.exit(f"[ERROR] Model file not found at: {MODEL_PATH}")

model = MaskablePPO.load(
    MODEL_PATH,
    env=env,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
)

# ---------- evaluation loop -----------------------
counters = collections.Counter({k: 0 for k in RESULT_KINDS})

def unwrap_env(env):
    """Unwrap down to the base TwoBridgeEnv (that has get_reward_components)."""
    while hasattr(env, "env"):
        env = env.env
    return env

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    logs, ep_r, ep_v = [], [], []

    while not done:
        act, _ = model.predict(obs, deterministic=True)

        # value estimate
        obs_tensor = {
            k: torch.tensor(v).float().unsqueeze(0).to(model.device)
            for k, v in obs.items()
        }
        with torch.no_grad():
            v_hat = model.policy.predict_values(obs_tensor).cpu().item()

        obs, rew, done, trunc, info = env.step(act)

        # collect metrics
        step = unwrap_env(env).get_reward_components()
        step.update({"reward": rew, "value_estimate": v_hat})
        logs.append(step)
        ep_r.append(rew)
        ep_v.append(v_hat)

    # -------- result & folder ---------------------
    res = info.get("result", "tie")
    if res not in RESULT_KINDS:
        res = "tie"
    counters[res] += 1
    dest = folders[res]

    # -------- save CSV ----------------------------
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(dest["csv"], f"decomposed_ep_{ep+1}.csv"), index=False)

    # -------- save reward-vs-value plot -----------
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

    # -------- save replay -------------------------
    base = unwrap_env(env)
    if hasattr(base, "_env") and hasattr(base._env, "save_replay"):
        base._env.save_replay(dest["replay"], prefix=f"ep_{ep+1}")
        # strip timestamp
        newest = sorted(
            glob.glob(os.path.join(dest["replay"], f"ep_{ep+1}_*.SC2Replay")),
            key=os.path.getmtime,
            reverse=True,
        )
        if newest:
            os.rename(
                newest[0],
                os.path.join(dest["replay"], f"ep_{ep+1}.SC2Replay"),
            )

    print(f"[{ep+1}/{EPISODES}] result: {res}")

env.close()

# ---------- summary bar chart ---------------------
labels, values = zip(*[(k, counters[k]) for k in RESULT_KINDS])
plt.figure(figsize=(7, 4))
plt.bar(labels, values)
plt.xticks(rotation=30)
plt.ylabel("# episodes out of " + str(EPISODES))
plt.title("Agent performance")
plt.tight_layout()
plt.savefig(
    os.path.join(performance_root, f"{AGENT_NAME}_performance_{EPISODES}_ep.png")
)
plt.show()

print("\nEpisode counts:", dict(counters))
win_pct = 100 * (counters["nav_win"]) / EPISODES
print(f"Nav-win rate: {win_pct:.1f}%")

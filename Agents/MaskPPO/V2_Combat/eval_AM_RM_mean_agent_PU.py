import sys, os, collections, numpy as np, matplotlib.pyplot as plt, torch
import pandas as pd  

import matplotlib.patches as mpatches

# Shared colors across dashboard & standalone
ACTION_COLORS = {
    -1: "#ffffff",  # Dead
     0: "#1f77b4",  # No-op
     1: "#ff7f0e",  # Move
     2: "#2ca02c",  # Attack
}

# Legend patches (Dead gets grey border so it shows up)
ACTION_LEGEND = [
    mpatches.Patch(facecolor=ACTION_COLORS[-1], edgecolor="#888888", label="Dead"),
    mpatches.Patch(color=ACTION_COLORS[0], label="No-op"),
    mpatches.Patch(color=ACTION_COLORS[1], label="Move"),
    mpatches.Patch(color=ACTION_COLORS[2], label="Attack"),
]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# ─────────────────── SB3 / gym imports ────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import glob

# ─────────────────── env + wrapper ────────────────
from Environments.TB_env_SF_AM_RM_mean_V2_Combat import TwoBridgeEnv

class FlattenActionWrapper(Wrapper):
    """Dict(verb, who, dir, enemy_idx) → flat MultiDiscrete; expands mask."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([3] + [2]*5 + [9] + [6])
        self._mask_template = np.ones(sum(self.action_space.nvec) - 3, dtype=np.int8)

        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(3 + len(self._mask_template))
        self.observation_space = spaces.Dict(obs_spaces)

    @staticmethod
    def _unflatten(vec):
        return {
            "verb":      int(vec[0]),
            "who":       np.asarray(vec[1 : 1 + 5], np.int8),
            "direction": int(vec[1 + 5]),
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
AGENT_NAME = "SB_MaskPPO_SF_AM_RM_mean"
# Map name
map_name = "V2_Combat"
# Absolute path to the model file
MODEL_PATH = os.path.join(project_root, "Agents", "MaskPPO", map_name, "saved_models",
                          AGENT_NAME, f"{AGENT_NAME}_final.zip")

EPISODES = 3
RENDER   = False

# ---------- directories ---------------------------
performance_root = os.path.join(project_root,"Agent Performance Charts","MaskPPO",map_name,AGENT_NAME)
replay_root      = os.path.join(project_root,"Replays","MaskPPO",map_name,AGENT_NAME)
os.makedirs(performance_root, exist_ok=True); os.makedirs(replay_root, exist_ok=True)

RESULT_KINDS = ["nav_win","combat_win","combat_loss","timeout_loss","tie"]
folders = {}
for rk in RESULT_KINDS:
    perf_dir = os.path.join(performance_root, rk)
    folders[rk] = {
        "plots"  : os.path.join(perf_dir,"EpRds_vs_Values"),
        "csv"    : os.path.join(perf_dir,"Decomposed_reward"),
        "replay" : os.path.join(replay_root, rk),
        "actions": os.path.join(perf_dir,"Actions")     # <-- NEW
    }
    for p in folders[rk].values(): os.makedirs(p, exist_ok=True)

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

# ---------- evaluation loop -----------------------
counters = collections.Counter({k:0 for k in RESULT_KINDS})

# --- Utility: unwrap to get the base TwoBridgeEnv (unwrapped SC2Env) ---
def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

# Run episodes
for ep in range(EPISODES):
    obs, _ = env.reset();  done=False
    logs, ep_r, ep_v = [], [], []

    while not done:
        act,_ = model.predict(obs, deterministic=True)

        # value estimate
        obs_tensor = {k:torch.tensor(v).float().unsqueeze(0).to(model.device) for k,v in obs.items()}
        with torch.no_grad():
            v_hat = model.policy.predict_values(obs_tensor).cpu().item()

        # ----- NEW: per‑timestep action logging from (obs_t, act_t) -----
        # We log what the policy *attempted* at time t. Dead units are ignored.
        vec = obs["vector"]                              # (55,)
        friend_alive = (vec[2:25:5] > 0)                 # 5 booleans (our units)
        enemy_alive  = (vec[27:50:5] > 0)                # 5 booleans (enemy slots)

        verb      = int(act[0])                          # 0=noop, 1=move, 2=attack
        who_bits  = np.array([int(x) for x in act[1:6]], dtype=np.int8).astype(bool)
        dir_id    = int(act[6])                          # 0 unused, 1..8 compass
        eidx_raw  = int(act[7])                          # 0=none, 1..5→enemy slot 0..4
        eidx      = eidx_raw - 1

        # validity at time t
        move_ok   = (verb == 1) and who_bits.any() and (1 <= dir_id <= 8)
        attack_ok = (verb == 2) and who_bits.any() and (0 <= eidx < 5) and enemy_alive[eidx]

        # per‑unit labels: -1=DEAD, 0=NOOP, 1=MOVE, 2=ATTACK
        unit_labels = np.full(5, -1, np.int8)            # default DEAD
        alive_mask  = friend_alive
        # alive but not selected → NOOP
        unit_labels[alive_mask & ~who_bits] = 0

        if move_ok:
            unit_labels[alive_mask & who_bits] = 1
        elif attack_ok:
            unit_labels[alive_mask & who_bits] = 2
        else:
            # selected but invalid verb/params → NOOP
            unit_labels[alive_mask & who_bits] = 0

        # counts for stacked bars (ignore DEAD)
        a_move   = int((unit_labels == 1).sum())
        a_attack = int((unit_labels == 2).sum())
        a_noop   = int((unit_labels == 0).sum())
        a_alive  = int(alive_mask.sum())
        # ---------------------------------------------------------------
        
        obs, rew, done, trunc, info = env.step(act)

        # collect metrics
        step = unwrap_env(env).get_reward_components()
        step.update({
            "reward": rew,
            "value_estimate": v_hat,

            # NEW: action counts at time t
            "a_move": a_move,
            "a_attack": a_attack,
            "a_noop": a_noop,
            "a_alive": a_alive,

            # NEW: per‑unit labels at time t (for future per‑unit timelines)
            "u0": int(unit_labels[0]),
            "u1": int(unit_labels[1]),
            "u2": int(unit_labels[2]),
            "u3": int(unit_labels[3]),
            "u4": int(unit_labels[4]),

            # (optional but handy)
            "A_verb": verb,
            "A_direction": dir_id,
            "A_enemy_idx": eidx_raw,
            "A_selected": int((friend_alive & who_bits).sum()),
        })
        logs.append(step);  ep_r.append(rew);  ep_v.append(v_hat)

    # -------- result & folder ---------------------
    res = info.get("result","tie")
    if res not in RESULT_KINDS: res="tie"
    counters[res]+=1;   dest = folders[res]

    # -------- save CSV ----------------------------
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(dest["csv"], f"decomposed_ep_{ep+1}.csv"), index=False)

    # -------- save reward-vs-value plot -----------
    plt.figure(figsize=(10,4))
    plt.plot(ep_r,label="Env Reward",marker='o',ls='--')
    plt.plot(ep_v,label="Value Estimate",marker='x')
    plt.xlabel("Timestep"); plt.ylabel("Reward / Value")
    plt.title(f"{AGENT_NAME} – Episode {ep+1} ({res})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(dest["plots"], f"ep_{ep+1}.png")); plt.close()

    # -------- save actions stacked bar (counts per timestep) -----------
    # expects the CSV row to include: a_noop, a_move, a_attack, a_alive
    t = np.arange(len(df))
    noop = df["a_noop"].to_numpy()   if "a_noop"   in df else np.zeros_like(t)
    move = df["a_move"].to_numpy()   if "a_move"   in df else np.zeros_like(t)
    atck = df["a_attack"].to_numpy() if "a_attack" in df else np.zeros_like(t)

    plt.figure(figsize=(10,3.0))
    b1 = plt.bar(t, noop, label="No-op", color=ACTION_COLORS[0])
    b2 = plt.bar(t, move, bottom=noop, label="Move", color=ACTION_COLORS[1])
    b3 = plt.bar(t, atck, bottom=noop+move, label="Attack", color=ACTION_COLORS[2])
    plt.xlabel("Timestep"); plt.ylabel("# Units")
    plt.title(f"{AGENT_NAME} – Actions (Episode {ep+1})")
    plt.legend(handles=ACTION_LEGEND, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(dest["actions"], f"ep_{ep+1}_actions_stacked.png"))
    plt.close()


    # -------- OPTIONAL: per‑unit ribbon (N/M/A/D over time) ------------
    # encodes per‑unit labels u0..u4 (-1=dead, 0=noop, 1=move, 2=attack)
    have_units = all(u in df.columns for u in ["u0","u1","u2","u3","u4"])
    if have_units:
        U = df[["u0","u1","u2","u3","u4"]].to_numpy().T  # (5, T)
        # Map to RGBA array using ACTION_COLORS
        cmap_array = np.empty((U.shape[0], U.shape[1], 3), dtype=float)
        for label, hexcol in ACTION_COLORS.items():
            rgb = plt.matplotlib.colors.to_rgb(hexcol)
            cmap_array[U == label] = rgb

        plt.figure(figsize=(10,2.8))
        plt.imshow(cmap_array, aspect="auto", interpolation="nearest")
        plt.yticks(range(5), [f"Unit {i}" for i in range(5)])
        plt.xlabel("Timestep")
        plt.title(f"{AGENT_NAME} – Per-Unit Actions (Episode {ep+1})")
        plt.legend(handles=ACTION_LEGEND, ncol=4, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(dest["actions"], f"ep_{ep+1}_actions_units.png"))
        plt.close()

    # -------- save replay -------------------------
    base = unwrap_env(env)
    if hasattr(base,"_env") and hasattr(base._env,"save_replay"):
        base._env.save_replay(dest["replay"], prefix=f"ep_{ep+1}")
        # strip timestamp
        newest = sorted(glob.glob(os.path.join(dest["replay"],f"ep_{ep+1}_*.SC2Replay")),
                        key=os.path.getmtime, reverse=True)
        if newest:
            os.rename(newest[0], os.path.join(dest["replay"],f"ep_{ep+1}.SC2Replay"))

    print(f"[{ep+1}/{EPISODES}] result: {res}")

env.close()

# ---------- summary bar chart ---------------------
labels, values = zip(*[(k,counters[k]) for k in RESULT_KINDS])
plt.figure(figsize=(7,4))
plt.bar(labels, values); plt.xticks(rotation=30)
plt.ylabel("# episodes out of "+str(EPISODES))
plt.title("Agent performance"); plt.tight_layout()
plt.savefig(os.path.join(performance_root,f"{AGENT_NAME}_performance_{EPISODES}_ep.png"))
plt.show()

print("\nEpisode counts:", dict(counters))
win_pct = 100*(counters["nav_win"]+counters["combat_win"])/EPISODES
print(f"Win rate: {win_pct:.1f}%")
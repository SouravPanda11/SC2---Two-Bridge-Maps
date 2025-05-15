"""
Evaluate a Maskable-PPO agent trained on the 5 v 3 Two-Bridge map.
File layout, prints, and plot are identical to your previous script.
"""

# ─────────────────── path setup ───────────────────
import sys, os, collections, numpy as np, matplotlib.pyplot as plt, torch
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ─────────────────── SB3 / gym imports ────────────
from gymnasium import Wrapper, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ─────────────────── env + wrapper ────────────────
from Environments.TB_env_SF_AM import (
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

# ─────────────────── user-config ──────────────────
AGENT_NAME = "SB_MaskPPO_SF_AM"
MODEL_PATH = os.path.join(project_root, "Agents", "saved_models",
                          AGENT_NAME, f"{AGENT_NAME}_final.zip")

EPISODES = 200
RENDER   = False

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

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)

    res = info.get("result", "tie")
    if res not in counters:
        print(f"[WARN] Unknown result '{res}' (added to summary)")
    counters[res] += 1

    if ep % 20 == 0 or ep == EPISODES - 1:
        print(f"[{ep+1}/{EPISODES}] result: {res}")

env.close()

# ─────────────────── summary + plot ──────────────
labels = ["nav_win", "combat_win", "combat_loss", "timeout_loss", "tie"]
values = [counters[k] for k in labels]

win_pct = 100 * (counters["nav_win"] + counters["combat_win"]) / EPISODES
print(f"\nTotal episodes: {EPISODES}")
print(f"Win rate       : {win_pct:.1f}%")

plt.figure(figsize=(7, 4))
plt.bar(labels, values)
plt.ylabel(f"# episodes out of {EPISODES}")
plt.title("Agent performance")
plt.xticks(rotation=30)
plt.tight_layout()

charts_dir = os.path.join(project_root,
                          "Agents", "Agent Performance Charts")
os.makedirs(charts_dir, exist_ok=True)
plot_path = os.path.join(
    charts_dir, f"{AGENT_NAME}_performance_{EPISODES}_ep.png"
)
plt.savefig(plot_path)
plt.show()

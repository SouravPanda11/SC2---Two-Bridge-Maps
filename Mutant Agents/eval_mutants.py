import os, sys, json, csv
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium import Wrapper, spaces

# ENV IMPORTS
from Environments.FAM_CAM.TB_env_FAM_V2_Base_Cam import (
    TwoBridgeEnv, N_FRIEND, N_ENEMY
)

# CONFIG
AGENT_NAME = "SB_MaskPPO_FAM_CAM"
map_name   = "V2_Base"

MUTANT_NET = "mutate_policy_action_net"  # Options: mutate_policy_net, mutate_action_net, mutate_policy_action_net, mutate_none

MUTANTS_DIR = os.path.join("Mutant Agents", f"{map_name}_mutants", MUTANT_NET)
OUT_ROOT    = os.path.join("Mutant Agents", "performance", f"{map_name}", MUTANT_NET)

N_EVAL_EPISODES = 5          
SEED_BASE = 4242

os.makedirs(OUT_ROOT, exist_ok=True)

# WRAPPERS
class FlattenActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete(
            [3] + [2]*N_FRIEND + [9] + [N_ENEMY + 1]
        )

        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)

        self._last_mask = np.ones(flat_len, dtype=np.int8)

    @staticmethod
    def _unflatten(a_vec):
        return {
            "verb":      int(a_vec[0]),
            "who":       np.asarray(a_vec[1 : 1+N_FRIEND], np.int8),
            "direction": int(a_vec[1+N_FRIEND]),
            "enemy_idx": int(a_vec[-1]),
        }

    def _convert_mask(self, obs):
        am = obs["action_mask"]

        verb_mask = np.asarray(am["verb"], dtype=np.int8)
        who_bits  = np.asarray(am["who"], dtype=np.int8)

        who_pairs = []
        for b in who_bits:
            who_pairs.extend([1, int(b)])

        direction_mask = np.asarray(am["direction"], dtype=np.int8)
        enemy_mask     = np.asarray(am["enemy_idx"], dtype=np.int8)

        flat_mask = np.concatenate(
            [verb_mask, np.asarray(who_pairs, np.int8), direction_mask, enemy_mask],
            dtype=np.int8
        )

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

    def action_masks(self):
        return self._last_mask

def make_eval_env(seed: int):
    def mask_fn(e):
        return e.action_masks()

    base_env = TwoBridgeEnv(visualize=False)
    flat_env = FlattenActionWrapper(base_env)
    env      = ActionMasker(flat_env, mask_fn)

    env.reset(seed=seed)
    return env

# MODEL LOADING
def load_model_clean(path: str):
    return MaskablePPO.load(
        path,
        device="cpu",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )

# TERMINAL OUTCOME PARSING
def extract_outcome(info: dict, terminated: bool, truncated: bool) -> str:
    if isinstance(info, dict):
        for k in ["outcome", "terminal_outcome", "result"]:
            if k in info and isinstance(info[k], str):
                return info[k]

        for k in ["nav_win", "combat_win", "combat_loss", "timeout_loss", "tie"]:
            if k in info and bool(info[k]):
                return k

    if truncated:
        return "timeout"
    if terminated:
        return "terminated"
    return "unknown"

# EVALUATION
def eval_one_model(model_path: str, out_dir: str, n_episodes: int):
    os.makedirs(out_dir, exist_ok=True)

    model = load_model_clean(model_path)

    env = make_eval_env(seed=SEED_BASE)

    returns  = []
    lengths  = []
    outcomes = []
    rows     = []

    for ep_idx in range(n_episodes):
        env_seed = SEED_BASE + ep_idx

        obs, info = env.reset(seed=env_seed)

        done = False
        ep_return = 0.0
        ep_len = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs)  
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            ep_len += 1
            last_info = info

            done = bool(terminated or truncated)

        outcome = extract_outcome(last_info, terminated, truncated)

        returns.append(ep_return)
        lengths.append(ep_len)
        outcomes.append(outcome)

        rows.append({
            "episode": ep_idx,
            "seed": env_seed,
            "return": ep_return,
            "length": ep_len,
            "outcome": outcome,
        })

    env.close()

    # Save per-episode CSV
    with open(os.path.join(out_dir, "episode_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "seed", "return", "length", "outcome"]
        )
        writer.writeheader()
        writer.writerows(rows)

    outcome_counts = dict(Counter(outcomes))

    summary = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "return_mean": float(np.mean(returns)),
        "return_std":  float(np.std(returns)),
        "len_mean":    float(np.mean(lengths)),
        "len_std":     float(np.std(lengths)),
        "outcome_counts": outcome_counts,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "returns.png"))
    plt.close()

    plt.figure()
    plt.bar(outcome_counts.keys(), outcome_counts.values())
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "outcomes.png"))
    plt.close()

    return summary

def find_mutant_zips(mutants_dir: str):
    files = [
        fn for fn in os.listdir(mutants_dir)
        if fn.startswith("mutant_") and fn.endswith(".zip")
    ]
    files.sort()
    return [os.path.join(mutants_dir, fn) for fn in files]

def main():
    mutant_paths = find_mutant_zips(MUTANTS_DIR)
    if not mutant_paths:
        raise RuntimeError(f"No mutant_*.zip found in {MUTANTS_DIR}")

    all_summaries = []

    for mp in mutant_paths:
        mutant_name = os.path.splitext(os.path.basename(mp))[0]
        out_dir = os.path.join(OUT_ROOT, mutant_name)

        print(f"\n=== Evaluating {mutant_name} ===")
        summary = eval_one_model(mp, out_dir, N_EVAL_EPISODES)
        all_summaries.append(summary)

    # Aggregate CSV
    with open(os.path.join(OUT_ROOT, "all_mutants_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mutant", "return_mean", "return_std", "len_mean", "len_std"])

        for s in all_summaries:
            mutant = os.path.splitext(os.path.basename(s["model_path"]))[0]
            writer.writerow([
                mutant,
                s["return_mean"],
                s["return_std"],
                s["len_mean"],
                s["len_std"],
            ])

    print(f"\n[DONE] Results saved under: {OUT_ROOT}")

if __name__ == "__main__":
    main()
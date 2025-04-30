"""
evaluate_two_bridge.py – run a trained PPO agent in TwoBridgeEnv
"""
import argparse, pathlib, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from two_bridge_env import TwoBridgeEnv     # ← your custom env

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="two_bridge_ppo.zip",
                    help="Path to SB3 .zip model (default: two_bridge_ppo.zip)")
parser.add_argument("--episodes", type=int, default=10,
                    help="How many episodes to play")
parser.add_argument("--render", action="store_true",
                    help="Show SC2 window (slower)")
parser.add_argument("--record", metavar="NPZ",
                    help="Save a single episode as .npz (obs, act, rew)")
args = parser.parse_args()

# ── load model ──────────────────────────────────────────────────────────
print(f"Loading model from {args.model} …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = PPO.load(args.model, device=device)

# ── make environment (no Monitor wrapper needed for testing) ────────────
env = TwoBridgeEnv(visualize=args.render)

# ── evaluation helper (vectorised in SB3) ───────────────────────────────
mean_r, std_r = evaluate_policy(model, env,
                                n_eval_episodes=args.episodes,
                                deterministic=True, render=False,
                                return_episode_rewards=False)
print(f"\n=== SB3 evaluate_policy ===\n"
      f"  mean reward {mean_r:.1f} ± {std_r:.1f}\n")

# ── optional detailed loop (prints each episode) ────────────────────────
ep_returns, ep_lengths = [], []
record_buffer = []          # for --record option

for ep in range(args.episodes):
    obs, _ = env.reset()
    done   = False
    ep_r   = 0.0
    steps  = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _, _ = env.step(action)
        if args.record and ep == 0:          # keep first episode
            record_buffer.append((obs.copy(), action.copy(), reward))
        obs   = next_obs
        ep_r += reward
        steps += 1
        if args.render:
            time.sleep(0.02)                 # slow down for humans
    ep_returns.append(ep_r)
    ep_lengths.append(steps)
    print(f"Episode {ep:2d}: reward={ep_r:6.1f}  steps={steps}")

# ── summary ─────────────────────────────────────────────────────────────
print("\n── summary ─────────────────────────────────────────────")
print(f"episodes:        {len(ep_returns)}")
print(f"reward  mean±sd: {np.mean(ep_returns):.1f} ± {np.std(ep_returns):.1f}")
print(f"length  mean±sd: {np.mean(ep_lengths):.1f} ± {np.std(ep_lengths):.1f}")

# ── save recording ──────────────────────────────────────────────────────
if args.record:
    rec_path = pathlib.Path(args.record).with_suffix(".npz")
    np.savez_compressed(rec_path,
                        obs   = np.array([r[0] for r in record_buffer]),
                        act   = np.array([r[1] for r in record_buffer]),
                        rew   = np.array([r[2] for r in record_buffer]))
    print(f"Episode 0 trajectory saved → {rec_path.absolute()}")

env.close()

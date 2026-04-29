import argparse
import atexit
import collections
import importlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import Wrapper, spaces
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================================
# Variant config.
# Copy this file for other variants and only update these constants.
# ============================================================================
AGENT_NAME = "MaskPPO_NS"
MAP_NAME = "V2_Base"
ENV_MODULE = "Environments.NS_AM_RM_mean.V2_Base_NS"
DEFAULT_EVAL_EPISODES = 10
DEFAULT_NUM_EVAL_ENVS = 5
DEFAULT_DETERMINISTIC = True
CHECKPOINT_NAME_RE = re.compile(
    rf"^{re.escape(AGENT_NAME)}_(\d+)([KMB]?)\.zip$",
    re.IGNORECASE,
)
env_module = importlib.import_module(ENV_MODULE)
TwoBridgeEnv = env_module.TwoBridgeEnv
N_FRIEND = env_module.N_FRIEND


PLOT_COLORS = {
    "nav_win": "#2A9D8F",
    "combat_win": "#E76F51",
    "total_win": "#1D3557",
    "mean_win": "#111111",
}

RAW_OUTCOME_COLUMNS = [
    "raw_nav_win",
    "raw_combat_win",
    "raw_combat_loss",
    "raw_timeout_loss",
    "raw_tie",
    "raw_victory",
    "raw_defeat",
]


class FlattenActionWrapper(Wrapper):
    """
    Keeps the NS env's per-friendly MultiDiscrete action space and flattens
    the 2D action mask so MaskablePPO can consume it.
    """

    def __init__(self, env):
        super().__init__(env)

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise TypeError("Expected MultiDiscrete action_space from NS env")
        self.action_space = env.action_space
        self._n_agents = int(len(self.action_space.nvec))
        if self._n_agents != N_FRIEND:
            raise ValueError(
                f"Unexpected action_space rank {self._n_agents}; expected N_FRIEND={N_FRIEND}"
            )
        if np.unique(self.action_space.nvec).size != 1:
            raise ValueError("Expected identical per-friendly action dimensions")
        self._n_unit_actions = int(self.action_space.nvec[0])

        flat_len = int(np.sum(self.action_space.nvec))
        obs_spaces = dict(env.observation_space.spaces)
        obs_spaces["action_mask"] = spaces.MultiBinary(flat_len)
        self.observation_space = spaces.Dict(obs_spaces)
        self._last_mask = np.ones(flat_len, dtype=np.int8)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        obs, rew, term, trunc, info = self.env.step(action_arr)
        obs = self._convert_mask(obs)
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._convert_mask(obs)
        return obs, info

    def _convert_mask(self, obs):
        am = np.asarray(obs["action_mask"], dtype=np.int8)
        expected_shape = (self._n_agents, self._n_unit_actions)
        if am.shape != expected_shape:
            raise ValueError(f"action_mask shape {am.shape} != expected {expected_shape}")

        flat_mask = am.reshape(-1).astype(np.int8, copy=True)
        obs["action_mask"] = flat_mask
        self._last_mask = flat_mask
        return obs

    def action_masks(self):
        return self._last_mask

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every saved MaskPPO checkpoint for each seed and plot "
            "win rate vs timesteps for V2_Base."
        )
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument(
        "--num-eval-envs",
        type=int,
        default=DEFAULT_NUM_EVAL_ENVS,
        help="Number of parallel SC2 eval environments to run per checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing cached CSV for this episode count and re-run all checkpoints.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic prediction.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return device_name


def validate_args(args):
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1.")
    if args.num_eval_envs < 1:
        raise ValueError("--num-eval-envs must be at least 1.")


def get_agent_save_root():
    return project_root / "Agents" / "MaskPPO" / MAP_NAME / "saved_models" / AGENT_NAME


def get_output_root():
    return (
        project_root
        / "Agent Performance Charts"
        / "MaskPPO"
        / MAP_NAME
        / AGENT_NAME
        / "checkpoint_sweep"
    )


def load_latest_seed_manifest():
    manifest_path = get_agent_save_root() / "latest_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Seed manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    raw_seeds = manifest.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise RuntimeError(
            f"Seed manifest {manifest_path} is missing a non-empty 'seeds' list."
        )

    seeds = []
    seen = set()
    for raw_seed in raw_seeds:
        seed = int(raw_seed)
        if seed in seen:
            raise RuntimeError(f"Seed manifest {manifest_path} contains duplicate seed {seed}.")
        seen.add(seed)
        seeds.append(seed)

    manifest["seeds"] = tuple(seeds)
    return manifest_path, manifest


def parse_checkpoint_steps(checkpoint_path):
    checkpoint_name = Path(checkpoint_path).name
    if "_final" in checkpoint_name:
        return None

    match = CHECKPOINT_NAME_RE.fullmatch(checkpoint_name)
    if match is None:
        return None

    value = int(match.group(1))
    suffix = match.group(2).upper()
    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers[suffix]


def collect_seed_checkpoints(seed):
    seed_dir = get_agent_save_root() / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    checkpoints = []
    for entry in seed_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".zip":
            continue
        checkpoint_steps = parse_checkpoint_steps(entry)
        if checkpoint_steps is None:
            continue
        checkpoints.append((checkpoint_steps, entry))

    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    if not checkpoints:
        raise RuntimeError(f"No step checkpoints found under {seed_dir}")
    return checkpoints


def make_env(rank, base_seed):
    def _init():
        worker_seed = int(base_seed) + int(rank)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        # Give each worker a private temp root to avoid Windows SC2 collisions.
        worker_tmp_dir = tempfile.mkdtemp(prefix=f"tbm-eval-worker-{rank}-")
        atexit.register(lambda path=worker_tmp_dir: shutil.rmtree(path, ignore_errors=True))
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = worker_tmp_dir

        # Stagger startup slightly so SC2 ports/temp files settle.
        time.sleep(0.5 * rank)

        env = TwoBridgeEnv(visualize=False, realtime=False)
        env = FlattenActionWrapper(env)
        return env

    return _init


def create_vec_env(num_envs, seed):
    env_fns = [make_env(rank=rank, base_seed=seed) for rank in range(num_envs)]
    return SubprocVecEnv(env_fns, start_method="spawn")


def get_vec_action_masks(vec_env):
    masks = vec_env.env_method("action_masks")
    return np.asarray(masks, dtype=np.int8)


def canonical_outcome_label(raw_result):
    if raw_result is None:
        return "unknown"
    return str(raw_result).strip().lower()


def collapse_outcomes(raw_counts, episodes):
    nav_win = int(raw_counts.get("nav_win", 0))
    combat_win = int(raw_counts.get("combat_win", 0) + raw_counts.get("victory", 0))
    combat_loss = int(raw_counts.get("combat_loss", 0) + raw_counts.get("defeat", 0))
    nav_loss = int(raw_counts.get("timeout_loss", 0))
    accounted = nav_win + combat_win + combat_loss + nav_loss
    unexpected_count = max(int(episodes) - accounted, 0)
    return {
        "nav_win": nav_win,
        "combat_win": combat_win,
        "combat_loss": combat_loss,
        "nav_loss": nav_loss,
        "unexpected_count": unexpected_count,
    }


def evaluate_checkpoint(vec_env, checkpoint_path, eval_episodes, deterministic, device, seed):
    model = MaskablePPO.load(str(checkpoint_path), env=vec_env, device=device)
    raw_counts = collections.Counter()
    episodes_assigned = 0
    episodes_finished = 0
    num_eval_envs = int(vec_env.num_envs)

    # Evaluate in seeded batches so each counted episode still has an explicit
    # reset seed, while the env workers remain alive across checkpoints.
    while episodes_finished < eval_episodes:
        batch_size = min(num_eval_envs, eval_episodes - episodes_assigned)
        if batch_size <= 0:
            break

        vec_env.seed(int(seed) + episodes_assigned)
        obs = vec_env.reset()
        episodes_assigned += batch_size

        active_envs = np.zeros(num_eval_envs, dtype=bool)
        active_envs[:batch_size] = True
        completed_envs = ~active_envs.copy()

        while not np.all(completed_envs):
            action_masks = get_vec_action_masks(vec_env)
            actions, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            actions = np.asarray(actions, dtype=np.int64)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            actions[completed_envs] = 0

            obs, _, dones, infos = vec_env.step(actions)
            for env_idx, done in enumerate(dones):
                if not active_envs[env_idx] or completed_envs[env_idx] or not done:
                    continue
                raw_result = canonical_outcome_label(infos[env_idx].get("result"))
                raw_counts[raw_result] += 1
                completed_envs[env_idx] = True
                episodes_finished += 1

    if episodes_finished != eval_episodes:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} finished {episodes_finished} eval episodes, "
            f"expected {eval_episodes}."
        )

    return raw_counts


def build_record(seed, checkpoint_steps, checkpoint_path, eval_episodes, raw_counts):
    collapsed = collapse_outcomes(raw_counts, eval_episodes)
    total_episodes = int(sum(raw_counts.values()))
    if total_episodes != eval_episodes:
        raise RuntimeError(
            f"Expected {eval_episodes} episodes for {checkpoint_path}, got {total_episodes}."
        )

    unexpected = {
        key: int(value)
        for key, value in sorted(raw_counts.items())
        if key not in {"nav_win", "combat_win", "combat_loss", "timeout_loss", "tie", "victory", "defeat"}
    }

    nav_win = collapsed["nav_win"]
    combat_win = collapsed["combat_win"]
    combat_loss = collapsed["combat_loss"]
    nav_loss = collapsed["nav_loss"]
    win_rate = 100.0 * (nav_win + combat_win) / total_episodes
    nav_win_rate = 100.0 * nav_win / total_episodes
    combat_win_rate = 100.0 * combat_win / total_episodes

    record = {
        "seed": int(seed),
        "checkpoint_name": Path(checkpoint_path).name,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_steps": int(checkpoint_steps),
        "episodes": int(total_episodes),
        "nav_win": nav_win,
        "combat_win": combat_win,
        "combat_loss": combat_loss,
        "nav_loss": nav_loss,
        "unexpected_count": int(collapsed["unexpected_count"]),
        "win_rate_percent": round(win_rate, 4),
        "nav_win_rate_percent": round(nav_win_rate, 4),
        "combat_win_rate_percent": round(combat_win_rate, 4),
        "unexpected_outcomes_json": json.dumps(unexpected, sort_keys=True),
    }

    record["raw_nav_win"] = int(raw_counts.get("nav_win", 0))
    record["raw_combat_win"] = int(raw_counts.get("combat_win", 0))
    record["raw_combat_loss"] = int(raw_counts.get("combat_loss", 0))
    record["raw_timeout_loss"] = int(raw_counts.get("timeout_loss", 0))
    record["raw_tie"] = int(raw_counts.get("tie", 0))
    record["raw_victory"] = int(raw_counts.get("victory", 0))
    record["raw_defeat"] = int(raw_counts.get("defeat", 0))
    return record


def normalize_results_df(df):
    if df.empty:
        return df

    int_columns = [
        "seed",
        "checkpoint_steps",
        "episodes",
        "nav_win",
        "combat_win",
        "combat_loss",
        "nav_loss",
        "unexpected_count",
        *RAW_OUTCOME_COLUMNS,
    ]
    float_columns = [
        "win_rate_percent",
        "nav_win_rate_percent",
        "combat_win_rate_percent",
    ]

    for column in int_columns:
        if column in df.columns:
            df[column] = df[column].fillna(0).astype(int)
    for column in float_columns:
        if column in df.columns:
            df[column] = df[column].astype(float)

    if "unexpected_outcomes_json" in df.columns:
        df["unexpected_outcomes_json"] = df["unexpected_outcomes_json"].fillna("{}")

    sort_cols = [col for col in ("seed", "checkpoint_steps") if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def load_existing_results(csv_path, overwrite):
    if overwrite or not csv_path.is_file():
        return pd.DataFrame()
    return normalize_results_df(pd.read_csv(csv_path))


def save_results_csv(df, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_results_df(df).to_csv(csv_path, index=False)


def format_timestep_label(value, _pos=None):
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def compute_bar_width(x_values):
    if len(x_values) <= 1:
        return 25_000
    diffs = np.diff(np.sort(np.asarray(x_values, dtype=float)))
    min_gap = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 25_000
    return max(min_gap * 0.55, 10_000)


def plot_seed(df, seed, eval_episodes, output_path):
    sub = df[df["seed"] == seed].sort_values("checkpoint_steps")
    if sub.empty:
        return

    x = sub["checkpoint_steps"].to_numpy()
    nav_rate = sub["nav_win_rate_percent"].to_numpy()
    combat_rate = sub["combat_win_rate_percent"].to_numpy()
    total_rate = sub["win_rate_percent"].to_numpy()
    bar_width = compute_bar_width(x)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar(
        x,
        nav_rate,
        width=bar_width,
        color=PLOT_COLORS["nav_win"],
        alpha=0.9,
        label="Navigation win %",
        zorder=1,
    )
    ax.bar(
        x,
        combat_rate,
        width=bar_width,
        bottom=nav_rate,
        color=PLOT_COLORS["combat_win"],
        alpha=0.9,
        label="Combat win %",
        zorder=1,
    )
    ax.plot(
        x,
        total_rate,
        color=PLOT_COLORS["total_win"],
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        label="Total win rate",
        zorder=3,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_xlabel("Timesteps")
    ax.set_title(
        f"{AGENT_NAME} {MAP_NAME} | seed_{seed} | {eval_episodes} eval episodes / checkpoint"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep_label))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.legend(loc="upper left", ncol=3, frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined(df, eval_episodes, output_path):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6.5))
    seed_order = sorted(df["seed"].unique())

    for seed in seed_order:
        sub = df[df["seed"] == seed].sort_values("checkpoint_steps")
        ax.plot(
            sub["checkpoint_steps"],
            sub["win_rate_percent"],
            marker="o",
            markersize=4.2,
            linewidth=1.9,
            label=f"seed_{seed}",
        )

    mean_df = (
        df.groupby("checkpoint_steps", as_index=False)["win_rate_percent"]
        .mean()
        .sort_values("checkpoint_steps")
    )
    ax.plot(
        mean_df["checkpoint_steps"],
        mean_df["win_rate_percent"],
        color=PLOT_COLORS["mean_win"],
        marker="o",
        markersize=4.8,
        linewidth=3.0,
        label="mean",
        zorder=4,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_xlabel("Timesteps")
    ax.set_title(f"{AGENT_NAME} {MAP_NAME} | seed win-rate curves + mean")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep_label))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.legend(loc="upper left", frameon=False, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_metadata(metadata_path, manifest_path, manifest, args, device, csv_path, df):
    payload = {
        "agent_name": AGENT_NAME,
        "map_name": MAP_NAME,
        "env_module": ENV_MODULE,
        "episodes_per_checkpoint": int(args.episodes),
        "num_eval_envs": int(args.num_eval_envs),
        "deterministic": bool(not args.stochastic),
        "device": device,
        "manifest_path": str(manifest_path.resolve()),
        "manifest": manifest,
        "results_csv": str(csv_path.resolve()),
        "rows": int(len(df)),
        "seeds": [int(seed) for seed in sorted(df["seed"].unique())] if not df.empty else [],
        "checkpoint_steps": (
            [int(step) for step in sorted(df["checkpoint_steps"].unique())] if not df.empty else []
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    validate_args(args)
    deterministic = not args.stochastic
    device = resolve_device(args.device)
    manifest_path, manifest = load_latest_seed_manifest()
    seeds = manifest["seeds"]
    mode_tag = "det" if deterministic else "stoch"
    run_tag = f"{args.episodes}ep_{mode_tag}_nenv{args.num_eval_envs}"

    output_root = get_output_root()
    csv_path = output_root / f"checkpoint_metrics_{run_tag}.csv"
    metadata_path = output_root / f"checkpoint_eval_metadata_{run_tag}.json"

    results_df = load_existing_results(csv_path, overwrite=args.overwrite)
    existing_keys = {
        (int(row.seed), int(row.checkpoint_steps))
        for row in results_df.itertuples(index=False)
    }

    print(f"Agent: {AGENT_NAME}")
    print(f"Map: {MAP_NAME}")
    print(f"Env module: {ENV_MODULE}")
    print(f"Device: {device}")
    print(f"Episodes per checkpoint: {args.episodes}")
    print(f"Parallel eval envs per checkpoint: {args.num_eval_envs}")
    print(f"Deterministic policy: {deterministic}")
    print(f"Manifest: {manifest_path}")
    print(f"Seeds: {list(seeds)}")
    print(f"Cached rows found: {len(existing_keys)}")

    all_records = results_df.to_dict("records")

    for seed_idx, seed in enumerate(seeds, start=1):
        checkpoints = collect_seed_checkpoints(seed)
        print(
            f"\nSeed {seed_idx}/{len(seeds)} | seed_{seed} | "
            f"checkpoint_count={len(checkpoints)}"
        )

        env = create_vec_env(num_envs=args.num_eval_envs, seed=int(seed))
        try:
            env.seed(int(seed))
            obs = env.reset()
            print(
                "  Eval env ready | "
                f"num_envs={env.num_envs} | "
                f"minimap={obs['minimap'].shape} | "
                f"vector={obs['vector'].shape} | "
                f"action_mask={obs['action_mask'].shape}"
            )
            for checkpoint_idx, (checkpoint_steps, checkpoint_path) in enumerate(
                checkpoints, start=1
            ):
                key = (int(seed), int(checkpoint_steps))
                if key in existing_keys:
                    print(
                        f"  [{checkpoint_idx}/{len(checkpoints)}] "
                        f"skip cached | step={checkpoint_steps} | file={checkpoint_path.name}"
                    )
                    continue

                print(
                    f"  [{checkpoint_idx}/{len(checkpoints)}] "
                    f"eval start | step={checkpoint_steps} | file={checkpoint_path.name}"
                )
                raw_counts = evaluate_checkpoint(
                    vec_env=env,
                    checkpoint_path=checkpoint_path,
                    eval_episodes=args.episodes,
                    deterministic=deterministic,
                    device=device,
                    seed=int(seed),
                )
                record = build_record(
                    seed=seed,
                    checkpoint_steps=checkpoint_steps,
                    checkpoint_path=checkpoint_path,
                    eval_episodes=args.episodes,
                    raw_counts=raw_counts,
                )
                all_records.append(record)
                existing_keys.add(key)

                results_df = normalize_results_df(pd.DataFrame(all_records))
                save_results_csv(results_df, csv_path)

                print(
                    f"  [{checkpoint_idx}/{len(checkpoints)}] "
                    f"eval done | step={checkpoint_steps} | "
                    f"win_rate={record['win_rate_percent']:.2f}% | "
                    f"nav_win={record['nav_win']} | combat_win={record['combat_win']} | "
                    f"combat_loss={record['combat_loss']} | nav_loss={record['nav_loss']} | "
                    f"unexpected={record['unexpected_count']}"
                )
        finally:
            env.close()

    results_df = normalize_results_df(pd.DataFrame(all_records))
    save_results_csv(results_df, csv_path)
    if results_df.empty:
        raise RuntimeError("No checkpoint evaluation rows were produced.")

    for seed in sorted(results_df["seed"].unique()):
        seed_plot_path = (
            output_root / f"seed_{seed}_winrate_vs_timesteps_{run_tag}.png"
        )
        plot_seed(results_df, int(seed), args.episodes, seed_plot_path)

    combined_plot_path = (
        output_root / f"all_seeds_winrate_vs_timesteps_{run_tag}.png"
    )
    plot_combined(results_df, args.episodes, combined_plot_path)

    write_metadata(metadata_path, manifest_path, manifest, args, device, csv_path, results_df)

    print("\nOutputs")
    print(f"  Results CSV: {csv_path}")
    print(f"  Metadata JSON: {metadata_path}")
    print(f"  Combined plot: {combined_plot_path}")
    for seed in sorted(results_df["seed"].unique()):
        print(
            f"  Seed plot: "
            f"{output_root / f'seed_{int(seed)}_winrate_vs_timesteps_{run_tag}.png'}"
        )


if __name__ == "__main__":
    main()

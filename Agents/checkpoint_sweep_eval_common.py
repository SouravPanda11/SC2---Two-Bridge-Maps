import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


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


def load_latest_seed_manifest(save_root: Path):
    manifest_path = save_root / "latest_run_manifest.json"
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


def parse_checkpoint_steps(agent_name: str, checkpoint_path: Path):
    checkpoint_name = checkpoint_path.name
    if checkpoint_name == f"{agent_name}_final.pt":
        return None

    prefix = f"{agent_name}_"
    suffix = ".pt"
    if not checkpoint_name.startswith(prefix) or not checkpoint_name.endswith(suffix):
        return None

    label = checkpoint_name[len(prefix) : -len(suffix)]
    if label.endswith(".replay"):
        return None

    import re

    match = re.fullmatch(r"(\d+)([KMB]?)", label, flags=re.IGNORECASE)
    if match is None:
        return None

    value = int(match.group(1))
    scale = match.group(2).upper()
    multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers[scale]


def collect_seed_checkpoints(save_root: Path, agent_name: str, seed: int):
    seed_dir = save_root / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    checkpoints = []
    for entry in seed_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".pt":
            continue
        checkpoint_steps = parse_checkpoint_steps(agent_name, entry)
        if checkpoint_steps is None:
            continue
        checkpoints.append((checkpoint_steps, entry))

    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    if not checkpoints:
        raise RuntimeError(f"No step checkpoints found under {seed_dir}")
    return checkpoints


def canonical_outcome_label(raw_result):
    if raw_result is None:
        return "unknown"
    return str(raw_result).strip().lower()


def collapse_outcomes(raw_counts, episodes: int):
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


def build_record(seed, checkpoint_steps, checkpoint_path, eval_episodes, raw_counts):
    collapsed = collapse_outcomes(raw_counts, eval_episodes)
    total_episodes = int(sum(raw_counts.values()))
    if total_episodes != eval_episodes:
        raise RuntimeError(
            f"Expected {eval_episodes} episodes for {checkpoint_path}, got {total_episodes}."
        )

    expected_raw = {
        "nav_win",
        "combat_win",
        "combat_loss",
        "timeout_loss",
        "tie",
        "victory",
        "defeat",
    }
    unexpected = {
        key: int(value)
        for key, value in sorted(raw_counts.items())
        if key not in expected_raw
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


def load_existing_results(csv_path: Path, overwrite: bool):
    if overwrite or not csv_path.is_file():
        return pd.DataFrame()
    return normalize_results_df(pd.read_csv(csv_path))


def save_results_csv(df, csv_path: Path):
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


def choose_best_seed(df):
    if df.empty:
        return None
    summary = (
        df.groupby("seed", as_index=False)
        .agg(
            best_win_rate=("win_rate_percent", "max"),
            mean_win_rate=("win_rate_percent", "mean"),
        )
        .sort_values(
            ["best_win_rate", "mean_win_rate", "seed"],
            ascending=[False, False, True],
        )
    )
    return int(summary.iloc[0]["seed"])


def plot_combined(df, agent_name: str, map_name: str, output_path: Path):
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
    ax.set_title(f"{agent_name} {map_name} | seed win-rate curves + mean")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep_label))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.legend(loc="upper left", frameon=False, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_seed_stacked(df, agent_name: str, map_name: str, output_path: Path):
    best_seed = choose_best_seed(df)
    if best_seed is None:
        return None

    sub = df[df["seed"] == best_seed].sort_values("checkpoint_steps")
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

    best_row = sub.sort_values(
        ["win_rate_percent", "checkpoint_steps"], ascending=[False, True]
    ).iloc[0]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.set_xlabel("Timesteps")
    ax.set_title(
        f"{agent_name} {map_name} | best seed_{best_seed} "
        f"(peak {best_row['win_rate_percent']:.1f}% at "
        f"{format_timestep_label(best_row['checkpoint_steps'])})"
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
    return best_seed


def write_metadata(
    metadata_path: Path,
    agent_name: str,
    map_name: str,
    manifest_path: Path,
    manifest: dict,
    args,
    device: str,
    csv_path: Path,
    df,
    combined_plot_path: Path,
    best_seed_plot_path: Path,
    best_seed,
):
    payload = {
        "agent_name": agent_name,
        "map_name": map_name,
        "episodes_per_checkpoint": int(args.episodes),
        "num_eval_envs": int(args.num_eval_envs),
        "deterministic": bool(not getattr(args, "stochastic", False)),
        "device": device,
        "manifest_path": str(manifest_path.resolve()),
        "manifest": manifest,
        "results_csv": str(csv_path.resolve()),
        "combined_plot": str(combined_plot_path.resolve()),
        "best_seed_stacked_plot": str(best_seed_plot_path.resolve()),
        "best_seed": int(best_seed) if best_seed is not None else None,
        "rows": int(len(df)),
        "seeds": [int(seed) for seed in sorted(df["seed"].unique())] if not df.empty else [],
        "checkpoint_steps": (
            [int(step) for step in sorted(df["checkpoint_steps"].unique())] if not df.empty else []
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

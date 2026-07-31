from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


CHART_ROOT = Path(__file__).resolve().parent
OUT_DIR = CHART_ROOT / "Terminal Reward Ablation Plots"

TARGET_STEPS = 2_000_000
MAX_PLOT_STEPS = 2_125_000
EVAL_ENVS = 8
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260727

MAPS = ("V1_Base", "V2_Navigate")
OUTCOMES = ("nav_win", "combat_win", "combat_loss", "nav_loss")
OUTCOME_LABELS = {
    "nav_win": "Navigation win",
    "combat_win": "Combat win",
    "combat_loss": "Combat loss",
    "nav_loss": "Timeout loss",
}

AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 15
LINE_WIDTH = 2.5


@dataclass(frozen=True)
class AgentSpec:
    label: str
    root_dir: str
    model_name: str
    color: str


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    model_suffix: str


AGENTS = (
    AgentSpec("QMIX", "Qmix_reduced", "QMIX_reduced", "#1f77b4"),
    AgentSpec("MaskPPO", "MaskPPO", "MaskPPO_NS_reduced", "#9467bd"),
    AgentSpec("MAPPO", "MAPPO_reduced", "MAPPO_reduced", "#8c564b"),
)

EXPERIMENTS = (
    ExperimentSpec(
        "reward_swap",
        "Terminal rewards swapped",
        "_terminal_reward_swap",
    ),
    ExperimentSpec(
        "equal_terminal_reward_25",
        "Equal terminal rewards (= 25)",
        "_equal_terminal_reward_25",
    ),
)


def format_timestep_label(value, _pos=None) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def eval_env_count(csv_path: Path) -> int | None:
    match = re.search(r"_nenv(\d+)(?:\D|$)", csv_path.name)
    return int(match.group(1)) if match else None


def checkpoint_max_step(csv_path: Path) -> int | None:
    try:
        steps = pd.read_csv(csv_path, usecols=["checkpoint_steps"])["checkpoint_steps"]
    except Exception:
        return None
    steps = pd.to_numeric(steps, errors="coerce").dropna()
    return int(steps.max()) if not steps.empty else None


def select_checkpoint_csv(
    agent: AgentSpec,
    experiment: ExperimentSpec,
    map_name: str,
) -> Path:
    sweep_dir = (
        CHART_ROOT
        / agent.root_dir
        / map_name
        / f"{agent.model_name}{experiment.model_suffix}"
        / "checkpoint_sweep"
    )
    csvs = [
        path
        for path in sorted(sweep_dir.glob("checkpoint_metrics_*.csv"))
        if eval_env_count(path) == EVAL_ENVS
    ]
    if not csvs:
        raise FileNotFoundError(
            f"No nenv{EVAL_ENVS} checkpoint metrics CSV found in {sweep_dir}"
        )

    candidates = []
    for path in csvs:
        max_step = checkpoint_max_step(path)
        if max_step is not None and max_step >= TARGET_STEPS:
            candidates.append((abs(max_step - TARGET_STEPS), max_step, -path.stat().st_mtime, path))
    if not candidates:
        available = [checkpoint_max_step(path) for path in csvs]
        raise RuntimeError(
            f"No CSV in {sweep_dir} reaches {format_timestep_label(TARGET_STEPS)} "
            f"steps; available maxima: {available}"
        )

    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3]


def load_eligible_rows(csv_path: Path, required_columns: set[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    numeric_columns = sorted(required_columns)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=numeric_columns)
    if df.empty:
        raise ValueError(f"{csv_path} has no usable rows")

    df["checkpoint_steps"] = df["checkpoint_steps"].astype(int)
    df["seed"] = df["seed"].astype(int)
    seed_max_steps = df.groupby("seed")["checkpoint_steps"].max()
    eligible_seeds = seed_max_steps[seed_max_steps >= TARGET_STEPS].index
    df = df[
        df["seed"].isin(eligible_seeds)
        & (df["checkpoint_steps"] <= MAX_PLOT_STEPS)
    ]
    if df.empty:
        raise ValueError(f"{csv_path} has no seeds reaching {TARGET_STEPS} steps")

    return df.drop_duplicates(subset=["seed", "checkpoint_steps"], keep="last")


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean

    bootstrap_means = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, values.size),
        replace=True,
    ).mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return mean, float(low), float(high)


def load_win_curve(
    csv_path: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = load_eligible_rows(
        csv_path,
        {"checkpoint_steps", "seed", "win_rate_percent"},
    )

    rows = []
    for checkpoint_steps, checkpoint_df in df.groupby("checkpoint_steps", sort=True):
        mean, ci_low, ci_high = bootstrap_mean_ci(
            checkpoint_df["win_rate_percent"].to_numpy(),
            rng,
        )
        rows.append(
            {
                "checkpoint_steps": int(checkpoint_steps),
                "mean_win_rate_percent": mean,
                "ci95_low_percent": ci_low,
                "ci95_high_percent": ci_high,
                "seed_count": int(checkpoint_df["seed"].nunique()),
                "seeds": ";".join(
                    str(seed) for seed in sorted(checkpoint_df["seed"].unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def load_final_outcomes(csv_path: Path) -> dict:
    df = load_eligible_rows(
        csv_path,
        {"checkpoint_steps", "seed", "episodes", *OUTCOMES},
    )
    final_rows = (
        df.sort_values(["seed", "checkpoint_steps"])
        .drop_duplicates(subset=["seed"], keep="last")
        .reset_index(drop=True)
    )

    total_episodes = int(final_rows["episodes"].sum())
    if total_episodes <= 0:
        raise ValueError(f"{csv_path} has zero final evaluation episodes")

    counts = {outcome: int(final_rows[outcome].sum()) for outcome in OUTCOMES}
    percentages = {
        outcome: 100.0 * counts[outcome] / total_episodes
        for outcome in OUTCOMES
    }
    return {
        "counts": counts,
        "percentages": percentages,
        "total_episodes": total_episodes,
        "seed_count": int(final_rows["seed"].nunique()),
        "seeds": sorted(final_rows["seed"].unique().tolist()),
        "final_checkpoint_steps": sorted(
            final_rows["checkpoint_steps"].unique().tolist()
        ),
        "unclassified_count": total_episodes - sum(counts.values()),
    }


def collect_data() -> tuple[dict, dict, list[str]]:
    curves = {}
    final_outcomes = {}
    warnings = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for experiment in EXPERIMENTS:
        for map_name in MAPS:
            for agent in AGENTS:
                csv_path = select_checkpoint_csv(agent, experiment, map_name)
                curve = load_win_curve(csv_path, rng)
                outcome = load_final_outcomes(csv_path)
                if curve["seed_count"].min() < 2:
                    warnings.append(
                        f"{experiment.key} {map_name} {agent.label}: "
                        "at least one checkpoint has fewer than two seeds"
                    )
                if outcome["unclassified_count"]:
                    warnings.append(
                        f"{experiment.key} {map_name} {agent.label}: "
                        f"{outcome['unclassified_count']} final episodes are not in "
                        f"{', '.join(OUTCOMES)}"
                    )

                metadata = {
                    "agent": agent,
                    "experiment": experiment,
                    "map_name": map_name,
                    "csv_path": csv_path,
                    "eval_envs": eval_env_count(csv_path),
                    "source_max_step": checkpoint_max_step(csv_path),
                }
                curves[(experiment.key, map_name, agent.label)] = {
                    **metadata,
                    "curve": curve,
                }
                final_outcomes[(experiment.key, map_name, agent.label)] = {
                    **metadata,
                    **outcome,
                }

    return curves, final_outcomes, warnings


def save_win_source(curves: dict) -> Path:
    rows = []
    for payload in curves.values():
        for row in payload["curve"].itertuples(index=False):
            rows.append(
                {
                    "experiment": payload["experiment"].key,
                    "experiment_label": payload["experiment"].label,
                    "map_name": payload["map_name"],
                    "agent": payload["agent"].label,
                    "csv_path": str(payload["csv_path"].relative_to(CHART_ROOT)),
                    "eval_envs": payload["eval_envs"],
                    "source_max_checkpoint_steps": payload["source_max_step"],
                    "checkpoint_steps": int(row.checkpoint_steps),
                    "mean_win_rate_percent": round(
                        float(row.mean_win_rate_percent), 6
                    ),
                    "ci95_low_percent": round(float(row.ci95_low_percent), 6),
                    "ci95_high_percent": round(float(row.ci95_high_percent), 6),
                    "seed_count": int(row.seed_count),
                    "seeds": row.seeds,
                    "ci_method": (
                        f"percentile bootstrap across seeds "
                        f"({BOOTSTRAP_SAMPLES} resamples)"
                    ),
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "terminal_reward_ablation_mean_winrate_source.csv"
    pd.DataFrame(rows).sort_values(
        ["experiment", "map_name", "agent", "checkpoint_steps"]
    ).to_csv(path, index=False)
    return path


def save_outcome_source(final_outcomes: dict) -> Path:
    rows = []
    for payload in final_outcomes.values():
        row = {
            "experiment": payload["experiment"].key,
            "experiment_label": payload["experiment"].label,
            "map_name": payload["map_name"],
            "agent": payload["agent"].label,
            "csv_path": str(payload["csv_path"].relative_to(CHART_ROOT)),
            "eval_envs": payload["eval_envs"],
            "source_max_checkpoint_steps": payload["source_max_step"],
            "final_checkpoint_steps": ";".join(
                str(step) for step in payload["final_checkpoint_steps"]
            ),
            "seed_count": payload["seed_count"],
            "seeds": ";".join(str(seed) for seed in payload["seeds"]),
            "total_episodes": payload["total_episodes"],
            "unclassified_count": payload["unclassified_count"],
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_count"] = payload["counts"][outcome]
            row[f"{outcome}_percent"] = round(
                payload["percentages"][outcome], 6
            )
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "terminal_reward_ablation_final_outcomes_source.csv"
    pd.DataFrame(rows).sort_values(
        ["experiment", "map_name", "agent"]
    ).to_csv(path, index=False)
    return path


def make_axes() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        len(EXPERIMENTS),
        len(MAPS),
        figsize=(14.8, 8.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    return fig, axes


def plot_mean_winrates(curves: dict) -> Path:
    fig, axes = make_axes()
    legend_handles = {}

    for row_idx, experiment in enumerate(EXPERIMENTS):
        for col_idx, map_name in enumerate(MAPS):
            ax = axes[row_idx, col_idx]
            for agent in AGENTS:
                curve = curves[
                    (experiment.key, map_name, agent.label)
                ]["curve"]
                x = curve["checkpoint_steps"].to_numpy()
                mean = curve["mean_win_rate_percent"].to_numpy()
                ci_low = curve["ci95_low_percent"].to_numpy()
                ci_high = curve["ci95_high_percent"].to_numpy()

                ax.fill_between(
                    x,
                    ci_low,
                    ci_high,
                    color=agent.color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=1,
                )
                (line,) = ax.plot(
                    x,
                    mean,
                    color=agent.color,
                    linewidth=LINE_WIDTH,
                    marker="o",
                    markersize=3.8,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    label=agent.label,
                    zorder=2,
                )
                legend_handles.setdefault(agent.label, line)

            ax.set_title(map_name, fontsize=16, pad=8)
            ax.text(
                0.015,
                0.96,
                experiment.label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
                color="#333333",
            )
            ax.set_xlim(0, MAX_PLOT_STEPS)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.24)
            ax.grid(axis="x", alpha=0.10)
            ax.xaxis.set_major_formatter(FuncFormatter(format_timestep_label))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

            if col_idx == 0:
                ax.set_ylabel(
                    "Mean win rate (%)",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            if row_idx == len(EXPERIMENTS) - 1:
                ax.set_xlabel("Timesteps", fontsize=AXIS_LABEL_FONTSIZE)

    fig.legend(
        [legend_handles[agent.label] for agent in AGENTS],
        [agent.label for agent in AGENTS],
        loc="lower center",
        ncol=len(AGENTS),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.6,
        columnspacing=2.0,
    )
    fig.suptitle(
        "Terminal-reward ablations: seed-averaged win rate",
        fontsize=18,
        y=0.992,
    )
    fig.text(
        0.5,
        0.052,
        "Shading shows the 95% percentile-bootstrap CI across the two training seeds.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.93,
        bottom=0.14,
        wspace=0.08,
        hspace=0.18,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "terminal_reward_ablation_mean_winrate.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_final_outcomes(final_outcomes: dict) -> Path:
    fig, axes = make_axes()
    y = np.arange(len(OUTCOMES), dtype=float)
    bar_height = 0.23
    offsets = np.linspace(-bar_height, bar_height, len(AGENTS))
    legend_handles = {}

    for row_idx, experiment in enumerate(EXPERIMENTS):
        for col_idx, map_name in enumerate(MAPS):
            ax = axes[row_idx, col_idx]
            for agent_idx, agent in enumerate(AGENTS):
                payload = final_outcomes[
                    (experiment.key, map_name, agent.label)
                ]
                values = [
                    payload["percentages"][outcome]
                    for outcome in OUTCOMES
                ]
                bars = ax.barh(
                    y + offsets[agent_idx],
                    values,
                    height=bar_height,
                    color=agent.color,
                    label=agent.label,
                    zorder=2,
                )
                legend_handles.setdefault(agent.label, bars[0])

            ax.set_title(map_name, fontsize=16, pad=8)
            ax.text(
                0.985,
                0.96,
                experiment.label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
                color="#333333",
            )
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.set_yticks(y)
            ax.set_yticklabels(
                [OUTCOME_LABELS[outcome] for outcome in OUTCOMES],
                fontsize=TICK_LABEL_FONTSIZE,
            )
            ax.set_ylim(len(OUTCOMES) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.16)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)

            if col_idx == 0:
                ax.set_ylabel("Terminal outcome", fontsize=AXIS_LABEL_FONTSIZE)
            if row_idx == len(EXPERIMENTS) - 1:
                ax.set_xlabel(
                    "Final evaluation episodes (%)",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )

    fig.legend(
        [legend_handles[agent.label] for agent in AGENTS],
        [agent.label for agent in AGENTS],
        loc="lower center",
        ncol=len(AGENTS),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.4,
        columnspacing=2.0,
    )
    fig.suptitle(
        "Terminal-reward ablations: final terminal-outcome distribution",
        fontsize=18,
        y=0.992,
    )
    episode_totals = sorted(
        {payload["total_episodes"] for payload in final_outcomes.values()}
    )
    totals_text = "/".join(str(total) for total in episode_totals)
    fig.text(
        0.5,
        0.052,
        f"Each distribution pools the final checkpoint evaluations across two seeds "
        f"({totals_text} episodes per agent).",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.99,
        top=0.93,
        bottom=0.14,
        wspace=0.10,
        hspace=0.18,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "terminal_reward_ablation_final_outcomes.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    curves, final_outcomes, warnings = collect_data()
    win_source = save_win_source(curves)
    outcome_source = save_outcome_source(final_outcomes)
    win_plot = plot_mean_winrates(curves)
    outcome_plot = plot_final_outcomes(final_outcomes)

    print(f"Saved mean win-rate source: {win_source}")
    print(f"Saved mean win-rate plot: {win_plot}")
    print(f"Saved mean win-rate PDF: {win_plot.with_suffix('.pdf')}")
    print(f"Saved final-outcome source: {outcome_source}")
    print(f"Saved final-outcome plot: {outcome_plot}")
    print(f"Saved final-outcome PDF: {outcome_plot.with_suffix('.pdf')}")
    print(f"Series plotted: {len(curves)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()

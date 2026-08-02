"""Create new reduced-agent comparison plots with a fixed scripted baseline.

The existing plotting scripts and their outputs are deliberately read-only.
This script reuses their data-selection functions, adds the 32-episode
scripted-oracle results, and writes to a separate output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


CHART_ROOT = Path(__file__).resolve().parent
AGENT_ROOT = CHART_ROOT.parent / "Agents"
DEFAULT_SCRIPTED_DIR = (
    CHART_ROOT / "Scripted" / "scripted_32ep_seed0_20260731"
)
DEFAULT_SCRIPTED_SUMMARY = DEFAULT_SCRIPTED_DIR / "scripted_summary.csv"
DEFAULT_SCRIPTED_EPISODES = DEFAULT_SCRIPTED_DIR / "scripted_episodes.csv"
DEFAULT_OUTPUT_DIR = CHART_ROOT / "Reduced Agent + Scripted Aggregate Plots"

WIN_MODULE_PATH = CHART_ROOT / "reduced_agents_multiplot.py"
TERMINAL_MODULE_PATH = CHART_ROOT / "reduced_agents_terminal_outcomes.py"

VERSIONS = ("V1", "V2", "V3")
VARIANTS = ("Base", "Combat", "Navigate")
TERMINAL_OUTCOMES = ("nav_win", "combat_win", "combat_loss", "timeout_loss")
TERMINAL_LABELS = {
    "nav_win": "nav_win",
    "combat_win": "combat_win",
    "combat_loss": "combat_loss",
    "timeout_loss": "timeout_loss",
}
VERSION_COLORS = {
    "V1": "green",
    "V2": "orange",
    "V3": "red",
}

SCRIPTED_LABEL = "Scripted oracle (32 ep)"
SCRIPTED_ROW_LABEL = "Scripted oracle"
SCRIPTED_COLOR = "#111111"
SCRIPTED_LINESTYLE = (0, (6, 3))
EXPECTED_2M_TRAINING_SEEDS = 5
TWO_MILLION_CHECKPOINT_TOLERANCE = 2_000


def load_local_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(CHART_ROOT))
    except ValueError:
        return str(path.resolve())


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_scripted_results(
    summary_path: Path,
    episodes_path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing scripted summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    required = {"variant", "episodes", "outcomes", "any_win_rate"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"{summary_path} is missing required columns: {sorted(missing)}"
        )

    sc2_seeds: dict[str, list[int]] = {}
    if episodes_path is not None and episodes_path.is_file():
        episodes = pd.read_csv(episodes_path)
        episode_required = {"variant", "sc2_seed"}
        episode_missing = episode_required.difference(episodes.columns)
        if episode_missing:
            raise ValueError(
                f"{episodes_path} is missing required columns: "
                f"{sorted(episode_missing)}"
            )
        for map_name, rows in episodes.groupby("variant"):
            seeds = pd.to_numeric(rows["sc2_seed"], errors="raise").astype(int)
            sc2_seeds[str(map_name)] = sorted(seeds.unique().tolist())

    results: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary.itertuples(index=False):
        map_name = str(row.variant)
        try:
            version, variant = map_name.split("_", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid scripted map name: {map_name!r}") from exc
        if version not in VERSIONS or variant not in VARIANTS:
            raise ValueError(f"Unexpected scripted variant: {map_name}")

        episode_count = int(row.episodes)
        outcome_counts = {
            str(key): int(value)
            for key, value in json.loads(str(row.outcomes)).items()
        }
        unsupported = {
            key: value
            for key, value in outcome_counts.items()
            if key not in TERMINAL_OUTCOMES and value
        }
        if unsupported:
            raise ValueError(
                f"Cannot place unsupported scripted outcomes for {map_name}: "
                f"{unsupported}"
            )
        if sum(outcome_counts.values()) != episode_count:
            raise ValueError(
                f"Scripted outcomes for {map_name} sum to "
                f"{sum(outcome_counts.values())}, expected {episode_count}."
            )

        terminal_counts = {
            outcome: outcome_counts.get(outcome, 0)
            for outcome in TERMINAL_OUTCOMES
        }
        total_wins = terminal_counts["nav_win"] + terminal_counts["combat_win"]
        measured_rate = total_wins / episode_count if episode_count else 0.0
        reported_rate = float(row.any_win_rate)
        if not math.isclose(measured_rate, reported_rate, abs_tol=1e-12):
            raise ValueError(
                f"Scripted win-rate mismatch for {map_name}: "
                f"outcomes give {measured_rate}, summary gives {reported_rate}."
            )
        ci_low, ci_high = wilson_interval(total_wins, episode_count)
        seeds = sc2_seeds.get(map_name, [])
        if seeds and len(seeds) != episode_count:
            raise ValueError(
                f"Expected one distinct SC2 seed per scripted episode for "
                f"{map_name}; found {len(seeds)} for {episode_count} episodes."
            )

        results[(version, variant)] = {
            "map_name": map_name,
            "version": version,
            "variant": variant,
            "episodes": episode_count,
            "counts": terminal_counts,
            "percentages": {
                outcome: 100.0 * count / episode_count
                for outcome, count in terminal_counts.items()
            },
            "win_count": total_wins,
            "win_rate_percent": 100.0 * measured_rate,
            "win_rate_ci95_percent": (100.0 * ci_low, 100.0 * ci_high),
            "sc2_seeds": seeds,
            "summary_path": summary_path.resolve(),
            "episodes_path": episodes_path.resolve()
            if episodes_path is not None and episodes_path.is_file()
            else None,
        }

    expected = {(version, variant) for version in VERSIONS for variant in VARIANTS}
    missing_results = expected.difference(results)
    if missing_results:
        raise ValueError(
            "Scripted summary is missing variants: "
            + ", ".join(f"{v}_{m}" for v, m in sorted(missing_results))
        )
    return results


def format_timestep_label(value, _pos=None) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def win_output_basename(config) -> str:
    suffix = "_10m_nenv8" if config.key == "10m" else ""
    return f"reduced_agents_mean_winrate_grid{suffix}_with_scripted"


def terminal_output_basename(config) -> str:
    suffix = "_10m_nenv8" if config.key == "10m" else ""
    return f"reduced_agents_final_terminal_outcome_grid{suffix}_with_scripted"


def save_win_source(
    curves: dict[tuple[str, str, str], dict],
    scripted: dict[tuple[str, str], dict[str, Any]],
    config,
    output_dir: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for (version, variant, agent_label), payload in sorted(curves.items()):
        for row in payload["mean_df"].itertuples(index=False):
            rows.append(
                {
                    "grid": config.key,
                    "grid_label": config.label,
                    "series_kind": "checkpoint_mean",
                    "agent": agent_label,
                    "version": version,
                    "variant": variant,
                    "map_name": payload["map_name"],
                    "source_path": relative_source(payload["csv_path"]),
                    "checkpoint_steps": int(row.checkpoint_steps),
                    "mean_win_rate_percent": round(
                        float(row.mean_win_rate_percent), 6
                    ),
                    "ci95_low_percent": "",
                    "ci95_high_percent": "",
                    "min_win_rate_percent": round(
                        float(row.min_win_rate_percent), 6
                    ),
                    "max_win_rate_percent": round(
                        float(row.max_win_rate_percent), 6
                    ),
                    "std_win_rate_percent": round(
                        float(row.std_win_rate_percent), 6
                    )
                    if pd.notna(row.std_win_rate_percent)
                    else "",
                    "training_seed_count": int(row.seed_count),
                    "evaluation_episodes": "",
                    "notes": "Mean across eligible training seeds at this checkpoint.",
                }
            )

    for version in VERSIONS:
        for variant in VARIANTS:
            result = scripted[(version, variant)]
            ci_low, ci_high = result["win_rate_ci95_percent"]
            rows.append(
                {
                    "grid": config.key,
                    "grid_label": config.label,
                    "series_kind": "fixed_baseline",
                    "agent": SCRIPTED_ROW_LABEL,
                    "version": version,
                    "variant": variant,
                    "map_name": result["map_name"],
                    "source_path": relative_source(result["summary_path"]),
                    "checkpoint_steps": "",
                    "mean_win_rate_percent": round(
                        result["win_rate_percent"], 6
                    ),
                    "ci95_low_percent": round(ci_low, 6),
                    "ci95_high_percent": round(ci_high, 6),
                    "min_win_rate_percent": "",
                    "max_win_rate_percent": "",
                    "std_win_rate_percent": "",
                    "training_seed_count": 0,
                    "evaluation_episodes": result["episodes"],
                    "notes": (
                        "Fixed non-learning policy; horizontal reference across "
                        "the checkpoint axis, not repeated checkpoint evaluations."
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{win_output_basename(config)}_source.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_win_grid(
    curves: dict[tuple[str, str, str], dict],
    scripted: dict[tuple[str, str], dict[str, Any]],
    config,
    win_module: ModuleType,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(VERSIONS),
        len(VARIANTS),
        figsize=(15.5, 10.7),
        sharex=True,
        sharey=True,
    )

    legend_handles: dict[str, Any] = {}
    global_max_step = max(
        int(payload["mean_df"]["checkpoint_steps"].max())
        for payload in curves.values()
    )

    for row_idx, version in enumerate(VERSIONS):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            map_name = f"{version}_{variant}"

            for agent in win_module.AGENTS:
                payload = curves.get((version, variant, agent.label))
                if payload is None:
                    continue
                mean_df = payload["mean_df"]
                (line,) = ax.plot(
                    mean_df["checkpoint_steps"].to_numpy(),
                    mean_df["mean_win_rate_percent"].to_numpy(),
                    color=agent.color,
                    linestyle=agent.linestyle,
                    linewidth=win_module.LINE_WIDTH,
                    marker="o",
                    markersize=win_module.MARKER_SIZE,
                    markeredgecolor="white",
                    markeredgewidth=win_module.MARKER_EDGE_WIDTH,
                    label=agent.label,
                    zorder=2,
                )
                legend_handles.setdefault(agent.label, line)

            result = scripted[(version, variant)]
            ci_low, ci_high = result["win_rate_ci95_percent"]
            ax.axhspan(
                ci_low,
                ci_high,
                color=SCRIPTED_COLOR,
                alpha=0.075,
                linewidth=0,
                zorder=0,
            )
            scripted_line = ax.axhline(
                result["win_rate_percent"],
                color=SCRIPTED_COLOR,
                linestyle=SCRIPTED_LINESTYLE,
                linewidth=2.8,
                label=SCRIPTED_LABEL,
                zorder=5,
            )
            legend_handles.setdefault(SCRIPTED_LABEL, scripted_line)

            ax.set_ylim(-2.5, 102.5)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", alpha=0.10)
            ax.xaxis.set_major_formatter(FuncFormatter(format_timestep_label))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="both", labelsize=win_module.TICK_LABEL_FONTSIZE)

            if col_idx == 0:
                ax.set_ylabel(
                    "Mean win rate (%)",
                    fontsize=win_module.AXIS_LABEL_FONTSIZE,
                )
            if row_idx == len(VERSIONS) - 1:
                ax.set_xlabel(
                    "Timesteps",
                    fontsize=win_module.AXIS_LABEL_FONTSIZE,
                )

            ax.text(
                0.015,
                0.965,
                map_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=win_module.PANEL_LABEL_FONTSIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
                zorder=6,
            )

    for ax in axes.ravel():
        ax.set_xlim(0, global_max_step * 1.02)

    ordered_labels = [
        *(agent.label for agent in win_module.AGENTS),
        SCRIPTED_LABEL,
    ]
    fig.legend(
        [legend_handles[label] for label in ordered_labels if label in legend_handles],
        [label for label in ordered_labels if label in legend_handles],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=15,
        handlelength=2.8,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(
        "Checkpoint mean win rate with fixed scripted-oracle reference",
        fontsize=18,
        y=0.995,
    )
    fig.text(
        0.5,
        0.014,
        "Scripted oracle is one fixed 32-episode evaluation; shaded region is its Wilson 95% CI.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.955,
        bottom=0.14,
        wspace=0.08,
        hspace=0.16,
    )

    basename = win_output_basename(config)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def save_terminal_source(
    learned: dict[tuple[str, str, str], dict],
    scripted: dict[tuple[str, str], dict[str, Any]],
    config,
    output_dir: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for (agent_label, version, variant), payload in sorted(learned.items()):
        row: dict[str, Any] = {
            "grid": config.key,
            "grid_label": config.label,
            "agent": agent_label,
            "agent_type": "learned_final_checkpoint",
            "version": version,
            "variant": variant,
            "map_name": payload["map_name"],
            "source_path": relative_source(payload["csv_path"]),
            "total_episodes": payload["total_episodes"],
            "training_seed_count": payload["seed_count"],
            "evaluation_sc2_seed_count": "",
            "final_checkpoint_steps": ";".join(
                str(step) for step in payload["final_checkpoint_steps"]
            ),
            "notes": "Pooled final eligible checkpoint across training seeds.",
        }
        for outcome in TERMINAL_OUTCOMES:
            learned_key = "nav_loss" if outcome == "timeout_loss" else outcome
            row[f"{outcome}_count"] = payload["counts"][learned_key]
            row[f"{outcome}_percent"] = round(
                payload["percentages"][learned_key], 6
            )
        rows.append(row)

    for version in VERSIONS:
        for variant in VARIANTS:
            result = scripted[(version, variant)]
            row = {
                "grid": config.key,
                "grid_label": config.label,
                "agent": SCRIPTED_ROW_LABEL,
                "agent_type": "fixed_non_learning_policy",
                "version": version,
                "variant": variant,
                "map_name": result["map_name"],
                "source_path": relative_source(result["summary_path"]),
                "total_episodes": result["episodes"],
                "training_seed_count": 0,
                "evaluation_sc2_seed_count": len(result["sc2_seeds"]),
                "final_checkpoint_steps": "",
                "notes": "One fixed-policy evaluation; no training checkpoint.",
            }
            for outcome in TERMINAL_OUTCOMES:
                row[f"{outcome}_count"] = result["counts"][outcome]
                row[f"{outcome}_percent"] = round(
                    result["percentages"][outcome], 6
                )
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{terminal_output_basename(config)}_source.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_terminal_grid(
    learned: dict[tuple[str, str, str], dict],
    scripted: dict[tuple[str, str], dict[str, Any]],
    config,
    terminal_module: ModuleType,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    learned_seed_counts = sorted(
        {int(payload["seed_count"]) for payload in learned.values()}
    )
    learned_episode_counts = sorted(
        {int(payload["total_episodes"]) for payload in learned.values()}
    )
    if len(learned_seed_counts) == len(learned_episode_counts) == 1:
        learned_summary = (
            f"{learned_seed_counts[0]} training seeds "
            f"({learned_episode_counts[0]} episodes/map)"
        )
    else:
        learned_summary = (
            f"the available training seeds ({learned_seed_counts[0]}-"
            f"{learned_seed_counts[-1]} seeds; {learned_episode_counts[0]}-"
            f"{learned_episode_counts[-1]} episodes/map)"
        )
    agent_rows = [agent.label for agent in terminal_module.AGENTS] + [
        SCRIPTED_ROW_LABEL
    ]
    fig, axes = plt.subplots(
        len(agent_rows),
        len(VARIANTS),
        figsize=(15.8, 13.0),
        sharex=True,
        sharey=False,
    )

    y = np.arange(len(TERMINAL_OUTCOMES), dtype=float)
    bar_height = 0.22
    offsets = np.linspace(-bar_height, bar_height, len(VERSIONS))
    legend_handles: dict[str, Any] = {}

    for row_idx, agent_label in enumerate(agent_rows):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            for version_idx, version in enumerate(VERSIONS):
                if agent_label == SCRIPTED_ROW_LABEL:
                    result = scripted[(version, variant)]
                    values = [
                        result["percentages"][outcome]
                        for outcome in TERMINAL_OUTCOMES
                    ]
                else:
                    payload = learned.get((agent_label, version, variant))
                    if payload is None:
                        continue
                    values = [
                        payload["percentages"][
                            "nav_loss" if outcome == "timeout_loss" else outcome
                        ]
                        for outcome in TERMINAL_OUTCOMES
                    ]

                bars = ax.barh(
                    y + offsets[version_idx],
                    values,
                    height=bar_height,
                    color=VERSION_COLORS[version],
                    label=version,
                )
                legend_handles.setdefault(version, bars[0])

            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.grid(axis="x", alpha=0.12)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=12.5)
            ax.set_yticks(y)
            ax.invert_yaxis()

            if col_idx == 0:
                ax.set_yticklabels(
                    [TERMINAL_LABELS[outcome] for outcome in TERMINAL_OUTCOMES]
                )
            else:
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(variant, fontsize=15, pad=8)
            if row_idx == len(agent_rows) - 1:
                ax.set_xlabel("Outcome frequency (%)", fontsize=13.5)

            for spine in ax.spines.values():
                spine.set_linewidth(1.3)

    fig.suptitle(
        "Final terminal outcomes with scripted-oracle comparison",
        fontsize=18,
        y=0.995,
    )
    fig.text(
        0.5,
        0.972,
        (
            f"Learned rows pool final checkpoints across {learned_summary}; "
            "Scripted oracle is one fixed 32-episode evaluation/map."
        ),
        ha="center",
        va="top",
        fontsize=10.5,
        color="#444444",
    )
    fig.legend(
        [legend_handles[version] for version in VERSIONS],
        list(VERSIONS),
        loc="lower center",
        ncol=len(VERSIONS),
        frameon=False,
        fontsize=15,
        handlelength=2.2,
        columnspacing=2.0,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.99,
        top=0.94,
        bottom=0.09,
        wspace=0.08,
        hspace=0.18,
    )

    for row_idx, agent_label in enumerate(agent_rows):
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            0.018,
            (bbox.y0 + bbox.y1) / 2.0,
            agent_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold" if agent_label == SCRIPTED_ROW_LABEL else "normal",
        )

    basename = terminal_output_basename(config)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def find_config(module: ModuleType, key: str):
    return next(config for config in module.GRID_CONFIGS if config.key == key)


def active_training_seed_ids(agent, map_name: str) -> list[int]:
    model_dir = (
        AGENT_ROOT
        / agent.root_dir
        / map_name
        / "saved_models"
        / agent.model_dir
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing trained-model directory: {model_dir}")

    seeds: list[int] = []
    for path in model_dir.glob("seed_*"):
        if not path.is_dir():
            continue
        try:
            seeds.append(int(path.name.removeprefix("seed_")))
        except ValueError:
            continue
    seeds.sort()
    if len(seeds) != EXPECTED_2M_TRAINING_SEEDS:
        raise ValueError(
            f"Expected {EXPECTED_2M_TRAINING_SEEDS} active training seeds for "
            f"{agent.label} {map_name}, found {len(seeds)} in {model_dir}: {seeds}"
        )
    return seeds


def filter_win_curves_to_active_seeds(
    curves: dict[tuple[str, str, str], dict],
    config,
) -> None:
    for payload in curves.values():
        seeds = active_training_seed_ids(payload["agent"], payload["map_name"])
        df = pd.read_csv(payload["csv_path"])
        required = {"checkpoint_steps", "seed", "win_rate_percent"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"{payload['csv_path']} is missing required columns: "
                f"{sorted(missing)}"
            )

        df = df.copy()
        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=list(required))
        df["checkpoint_steps"] = df["checkpoint_steps"].astype(int)
        df["seed"] = df["seed"].astype(int)
        df = df[df["seed"].isin(seeds)]

        found_seeds = sorted(df["seed"].unique().tolist())
        if found_seeds != seeds:
            raise ValueError(
                f"The 8-env metrics for {payload['agent'].label} "
                f"{payload['map_name']} contain active seeds {found_seeds}; "
                f"expected {seeds}."
            )
        seed_max_steps = df.groupby("seed")["checkpoint_steps"].max()
        short_seeds = seed_max_steps[seed_max_steps < config.target_steps]
        if not short_seeds.empty:
            raise ValueError(
                f"Active seeds do not reach {config.target_steps} for "
                f"{payload['agent'].label} {payload['map_name']}: "
                f"{short_seeds.to_dict()}"
            )

        common_max_step = min(
            int(seed_max_steps.min()),
            int(config.max_plot_steps),
        )
        df = df[df["checkpoint_steps"] <= common_max_step]
        df = df.drop_duplicates(
            subset=["seed", "checkpoint_steps"], keep="last"
        )
        mean_df = (
            df.groupby("checkpoint_steps", as_index=False)
            .agg(
                mean_win_rate_percent=("win_rate_percent", "mean"),
                min_win_rate_percent=("win_rate_percent", "min"),
                max_win_rate_percent=("win_rate_percent", "max"),
                std_win_rate_percent=("win_rate_percent", "std"),
                seed_count=("seed", "nunique"),
            )
            .sort_values("checkpoint_steps")
        )
        incomplete = mean_df[
            mean_df["seed_count"] != EXPECTED_2M_TRAINING_SEEDS
        ]
        if not incomplete.empty:
            raise ValueError(
                f"Incomplete checkpoints for {payload['agent'].label} "
                f"{payload['map_name']}: "
                f"{incomplete[['checkpoint_steps', 'seed_count']].to_dict('records')}"
            )
        payload["mean_df"] = mean_df
        payload["training_seeds"] = seeds


def filter_terminal_outcomes_to_active_seeds(
    learned: dict[tuple[str, str, str], dict],
    config,
    terminal_module: ModuleType,
) -> None:
    for payload in learned.values():
        seeds = active_training_seed_ids(payload["agent"], payload["map_name"])
        df = pd.read_csv(payload["csv_path"])
        required = {
            "checkpoint_steps",
            "seed",
            "episodes",
            *terminal_module.OUTCOMES,
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"{payload['csv_path']} is missing required columns: "
                f"{sorted(missing)}"
            )

        df = df.copy()
        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=list(required))
        for column in required:
            df[column] = df[column].astype(int)
        df = df[df["seed"].isin(seeds)]

        found_seeds = sorted(df["seed"].unique().tolist())
        if found_seeds != seeds:
            raise ValueError(
                f"The 8-env terminal metrics for {payload['agent'].label} "
                f"{payload['map_name']} contain active seeds {found_seeds}; "
                f"expected {seeds}."
            )
        seed_max_steps = df.groupby("seed")["checkpoint_steps"].max()
        short_seeds = seed_max_steps[seed_max_steps < config.target_steps]
        if not short_seeds.empty:
            raise ValueError(
                f"Active seeds do not reach {config.target_steps} for "
                f"{payload['agent'].label} {payload['map_name']}: "
                f"{short_seeds.to_dict()}"
            )

        common_max_step = min(
            int(seed_max_steps.min()),
            int(config.max_plot_steps),
        )
        final_rows = (
            df[df["checkpoint_steps"] <= common_max_step]
            .sort_values(["seed", "checkpoint_steps"])
            .drop_duplicates(subset=["seed"], keep="last")
            .reset_index(drop=True)
        )
        if sorted(final_rows["seed"].tolist()) != seeds:
            raise ValueError(
                f"Could not select one final 2M checkpoint for every active "
                f"seed of {payload['agent'].label} {payload['map_name']}."
            )
        total_episodes = int(final_rows["episodes"].sum())
        counts = {
            outcome: int(final_rows[outcome].sum())
            for outcome in terminal_module.OUTCOMES
        }
        payload.update(
            {
                "final_rows": final_rows,
                "counts": counts,
                "percentages": {
                    outcome: 100.0 * count / total_episodes
                    for outcome, count in counts.items()
                },
                "total_episodes": total_episodes,
                "final_checkpoint_steps": sorted(
                    final_rows["checkpoint_steps"].unique().tolist()
                ),
                "seeds": seeds,
                "seed_count": len(seeds),
            }
        )


def run_grid(
    key: str,
    scripted: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    win_module: ModuleType,
    terminal_module: ModuleType,
) -> list[Path]:
    win_config = find_config(win_module, key)
    terminal_config = find_config(terminal_module, key)
    if key == "2m":
        win_config = replace(
            win_config,
            label="2M, 8 eval envs, 5 training seeds",
            eval_envs=8,
            target_tolerance_steps=TWO_MILLION_CHECKPOINT_TOLERANCE,
        )
        terminal_config = replace(
            terminal_config,
            label="2M, 8 eval envs, 5 training seeds",
            eval_envs=8,
            target_tolerance_steps=TWO_MILLION_CHECKPOINT_TOLERANCE,
        )

    curves, win_warnings = win_module.collect_curves(win_config)
    if not curves:
        raise RuntimeError(f"No learned win-rate curves found for grid {key}.")
    learned_outcomes, terminal_warnings = (
        terminal_module.collect_final_terminal_outcomes(terminal_config)
    )
    if not learned_outcomes:
        raise RuntimeError(f"No learned terminal outcomes found for grid {key}.")
    if key == "2m":
        filter_win_curves_to_active_seeds(curves, win_config)
        filter_terminal_outcomes_to_active_seeds(
            learned_outcomes,
            terminal_config,
            terminal_module,
        )

    win_source = save_win_source(curves, scripted, win_config, output_dir)
    win_png, win_pdf = plot_win_grid(
        curves,
        scripted,
        win_config,
        win_module,
        output_dir,
    )
    terminal_source = save_terminal_source(
        learned_outcomes,
        scripted,
        terminal_config,
        output_dir,
    )
    terminal_png, terminal_pdf = plot_terminal_grid(
        learned_outcomes,
        scripted,
        terminal_config,
        terminal_module,
        output_dir,
    )

    print(f"\nGrid: {key}")
    for path in (
        win_source,
        win_png,
        win_pdf,
        terminal_source,
        terminal_png,
        terminal_pdf,
    ):
        print(f"Saved: {path}")
    for warning in (*win_warnings, *terminal_warnings):
        print(f"Warning: {warning}")
    return [
        win_source,
        win_png,
        win_pdf,
        terminal_source,
        terminal_png,
        terminal_pdf,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create new learned-agent comparison plots with a fixed scripted "
            "oracle, without changing existing plots."
        )
    )
    parser.add_argument(
        "--grid",
        choices=("2m", "10m", "all"),
        default="10m",
        help="Checkpoint dataset to plot. Default: 10m.",
    )
    parser.add_argument(
        "--scripted-summary",
        type=Path,
        default=DEFAULT_SCRIPTED_SUMMARY,
    )
    parser.add_argument(
        "--scripted-episodes",
        type=Path,
        default=DEFAULT_SCRIPTED_EPISODES,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    scripted = load_scripted_results(
        args.scripted_summary.expanduser().resolve(),
        args.scripted_episodes.expanduser().resolve(),
    )
    win_module = load_local_module(
        "_existing_reduced_agents_multiplot",
        WIN_MODULE_PATH,
    )
    terminal_module = load_local_module(
        "_existing_reduced_agents_terminal_outcomes",
        TERMINAL_MODULE_PATH,
    )
    keys = ("2m", "10m") if args.grid == "all" else (args.grid,)
    paths: list[Path] = []
    for key in keys:
        paths.extend(
            run_grid(
                key,
                scripted,
                args.output_dir.expanduser().resolve(),
                win_module,
                terminal_module,
            )
        )
    return paths


if __name__ == "__main__":
    main()

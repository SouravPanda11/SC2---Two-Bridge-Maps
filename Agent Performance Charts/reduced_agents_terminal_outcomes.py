from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHART_ROOT = Path(__file__).resolve().parent
OUT_DIR = CHART_ROOT / "Reduced Agent Aggregate Plots"

VERSIONS = ("V1", "V2", "V3")
VARIANTS = ("Base", "Combat", "Navigate")
OUTCOMES = ("nav_win", "combat_win", "combat_loss", "nav_loss")
OUTCOME_LABELS = {
    "nav_win": "nav_win",
    "combat_win": "combat_win",
    "combat_loss": "combat_loss",
    "nav_loss": "timeout_loss",
}
VERSION_COLORS = {
    "V1": "green",
    "V2": "orange",
    "V3": "red",
}

TICK_FONTSIZE = 16


@dataclass(frozen=True)
class AgentSpec:
    label: str
    root_dir: str
    model_dir: str


@dataclass(frozen=True)
class GridConfig:
    key: str
    label: str
    output_suffix: str
    grid_basename: str
    source_csv: str
    target_steps: int
    eval_envs: int
    target_tolerance_steps: int = 125_000

    @property
    def max_plot_steps(self) -> int:
        return self.target_steps + self.target_tolerance_steps


AGENTS = (
    AgentSpec("QMIX", "Qmix_reduced", "QMIX_reduced"),
    AgentSpec("MaskPPO", "MaskPPO", "MaskPPO_NS_reduced"),
    AgentSpec("MAPPO", "MAPPO_reduced", "MAPPO_reduced"),
)

GRID_CONFIGS = (
    GridConfig(
        key="2m",
        label="2M, 16 eval envs",
        output_suffix="",
        grid_basename="reduced_agents_final_terminal_outcome_grid",
        source_csv="reduced_agents_final_terminal_outcome_source.csv",
        target_steps=2_000_000,
        eval_envs=16,
    ),
    GridConfig(
        key="10m",
        label="10M, 8 eval envs",
        output_suffix="_10m_nenv8",
        grid_basename="reduced_agents_final_terminal_outcome_grid_10m_nenv8",
        source_csv="reduced_agents_final_terminal_outcome_10m_nenv8_source.csv",
        target_steps=10_000_000,
        eval_envs=8,
    ),
)


def format_timestep_label(value: int | float) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def checkpoint_max_step(csv_path: Path) -> int | None:
    try:
        df = pd.read_csv(csv_path, usecols=["checkpoint_steps"])
    except Exception:
        return None

    steps = pd.to_numeric(df["checkpoint_steps"], errors="coerce").dropna()
    if steps.empty:
        return None
    return int(steps.max())


def eval_env_count(csv_path: Path) -> int | None:
    match = re.search(r"_nenv(\d+)(?:\D|$)", csv_path.name)
    return int(match.group(1)) if match else None


def select_checkpoint_csv(
    agent: AgentSpec,
    map_name: str,
    config: GridConfig,
) -> tuple[Path | None, str | None]:
    sweep_dir = CHART_ROOT / agent.root_dir / map_name / agent.model_dir / "checkpoint_sweep"
    csvs = sorted(sweep_dir.glob("checkpoint_metrics_*.csv"))
    if not csvs:
        return None, "no checkpoint metrics CSV found"

    matching_envs = [path for path in csvs if eval_env_count(path) == config.eval_envs]
    if not matching_envs:
        available_envs = sorted({env for path in csvs if (env := eval_env_count(path)) is not None})
        suffix = f"; available nenv values: {available_envs}" if available_envs else ""
        return None, f"no nenv{config.eval_envs} checkpoint metrics CSV found{suffix}"

    candidates = []
    for path in matching_envs:
        max_step = checkpoint_max_step(path)
        if max_step is None or max_step < config.target_steps:
            continue
        candidates.append((max_step, path.stat().st_mtime, path))

    if not candidates:
        available_steps = [
            max_step
            for path in matching_envs
            if (max_step := checkpoint_max_step(path)) is not None
        ]
        max_available = max(available_steps) if available_steps else "none"
        return None, (
            f"no nenv{config.eval_envs} CSV reaches {format_timestep_label(config.target_steps)} "
            f"(max available: {max_available})"
        )

    candidates.sort(key=lambda item: (abs(item[0] - config.target_steps), item[0], -item[1]))
    return candidates[0][2], None


def load_final_terminal_outcomes(
    csv_path: Path,
    target_steps: int,
    max_plot_steps: int,
) -> dict:
    df = pd.read_csv(csv_path)
    required = {"checkpoint_steps", "seed", "episodes", *OUTCOMES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    numeric_columns = ["checkpoint_steps", "seed", "episodes", *OUTCOMES]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=numeric_columns)
    if df.empty:
        raise ValueError(f"{csv_path} has no usable terminal outcome rows")

    df["checkpoint_steps"] = df["checkpoint_steps"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["episodes"] = df["episodes"].astype(int)
    for outcome in OUTCOMES:
        df[outcome] = df[outcome].astype(int)

    seed_max_steps = df.groupby("seed")["checkpoint_steps"].max()
    eligible_seeds = seed_max_steps[seed_max_steps >= target_steps].index
    df = df[df["seed"].isin(eligible_seeds)]
    df = df[df["checkpoint_steps"] <= max_plot_steps]
    if df.empty:
        raise ValueError(f"{csv_path} has no seeds reaching {target_steps} steps")

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
        outcome: 100.0 * count / total_episodes
        for outcome, count in counts.items()
    }

    return {
        "final_rows": final_rows,
        "counts": counts,
        "percentages": percentages,
        "total_episodes": total_episodes,
        "final_checkpoint_steps": sorted(final_rows["checkpoint_steps"].unique().tolist()),
        "seeds": sorted(final_rows["seed"].unique().tolist()),
        "seed_count": int(final_rows["seed"].nunique()),
    }


def collect_final_terminal_outcomes(
    config: GridConfig,
) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    outcomes: dict[tuple[str, str, str], dict] = {}
    warnings: list[str] = []

    for agent in AGENTS:
        for version in VERSIONS:
            for variant in VARIANTS:
                map_name = f"{version}_{variant}"
                csv_path, skip_reason = select_checkpoint_csv(agent, map_name, config)
                if csv_path is None:
                    warnings.append(f"{agent.label} {map_name}: {skip_reason}")
                    continue
                try:
                    payload = load_final_terminal_outcomes(
                        csv_path,
                        config.target_steps,
                        config.max_plot_steps,
                    )
                except Exception as exc:
                    warnings.append(f"{agent.label} {map_name}: skipped {csv_path} ({exc})")
                    continue
                payload.update(
                    {
                        "agent": agent,
                        "map_name": map_name,
                        "version": version,
                        "variant": variant,
                        "csv_path": csv_path,
                        "eval_envs": eval_env_count(csv_path),
                        "source_max_step": checkpoint_max_step(csv_path),
                    }
                )
                outcomes[(agent.label, version, variant)] = payload

    return outcomes, warnings


def save_final_outcome_source(
    outcomes: dict[tuple[str, str, str], dict],
    config: GridConfig,
) -> Path:
    rows = []
    for (agent_label, version, variant), payload in sorted(outcomes.items()):
        row = {
            "grid": config.key,
            "grid_label": config.label,
            "agent": agent_label,
            "version": version,
            "variant": variant,
            "map_name": payload["map_name"],
            "csv_path": str(payload["csv_path"].relative_to(CHART_ROOT)),
            "eval_envs": payload["eval_envs"],
            "source_max_checkpoint_steps": payload["source_max_step"],
            "total_episodes": payload["total_episodes"],
            "seed_count": payload["seed_count"],
            "seeds": ";".join(str(seed) for seed in payload["seeds"]),
            "final_checkpoint_steps": ";".join(
                str(step) for step in payload["final_checkpoint_steps"]
            ),
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_count"] = payload["counts"][outcome]
            row[f"{outcome}_percent"] = round(payload["percentages"][outcome], 6)
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / config.source_csv
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def plot_terminal_outcomes_for_agent(
    agent: AgentSpec,
    outcomes: dict[tuple[str, str, str], dict],
    config: GridConfig,
) -> Path | None:
    if not any(key[0] == agent.label for key in outcomes):
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(VARIANTS),
        figsize=(13.8, 4.2),
        sharex=True,
        sharey=True,
    )

    y = np.arange(len(OUTCOMES), dtype=float)
    bar_height = 0.25
    offsets = np.linspace(-bar_height, bar_height, len(VERSIONS))
    legend_handles = {}

    for col_idx, variant in enumerate(VARIANTS):
        ax = axes[col_idx]
        plotted = False
        ax.set_title(variant, fontsize=15, pad=8)

        for version_idx, version in enumerate(VERSIONS):
            payload = outcomes.get((agent.label, version, variant))
            if payload is None:
                continue
            values = [payload["percentages"][outcome] for outcome in OUTCOMES]
            bars = ax.barh(
                y + offsets[version_idx],
                values,
                height=bar_height,
                color=VERSION_COLORS[version],
                label=version,
            )
            legend_handles.setdefault(version, bars[0])
            plotted = True

        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No CSV found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666666",
                fontsize=12,
            )

        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(axis="x", alpha=0.12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.set_yticks(y)
        ax.set_yticklabels([OUTCOME_LABELS[outcome] for outcome in OUTCOMES])
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

    if legend_handles:
        ordered_versions = [version for version in VERSIONS if version in legend_handles]
        fig.legend(
            [legend_handles[version] for version in ordered_versions],
            ordered_versions,
            loc="lower center",
            ncol=len(ordered_versions),
            frameon=False,
            fontsize=14,
            handlelength=2.2,
            columnspacing=1.8,
        )

    fig.suptitle(f"{agent.label} final terminal outcomes ({config.label})", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.22, wspace=0.10)

    safe_agent = agent.label.lower().replace(" ", "_")
    png_path = OUT_DIR / f"{safe_agent}_final_terminal_outcomes{config.output_suffix}.png"
    pdf_path = OUT_DIR / f"{safe_agent}_final_terminal_outcomes{config.output_suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_individual_terminal_outcomes(
    outcomes: dict[tuple[str, str, str], dict],
    config: GridConfig,
) -> list[Path]:
    paths = []
    for agent in AGENTS:
        path = plot_terminal_outcomes_for_agent(agent, outcomes, config)
        if path is not None:
            paths.append(path)
    return paths


def plot_terminal_outcome_grid(
    outcomes: dict[tuple[str, str, str], dict],
    config: GridConfig,
) -> Path | None:
    if not outcomes:
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(AGENTS),
        len(VARIANTS),
        figsize=(15.5, 10.2),
        sharex=True,
        sharey=False,
    )

    y = np.arange(len(OUTCOMES), dtype=float)
    bar_height = 0.22
    offsets = np.linspace(-bar_height, bar_height, len(VERSIONS))
    legend_handles = {}

    for row_idx, agent in enumerate(AGENTS):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            plotted = False

            for version_idx, version in enumerate(VERSIONS):
                payload = outcomes.get((agent.label, version, variant))
                if payload is None:
                    continue

                values = [payload["percentages"][outcome] for outcome in OUTCOMES]
                bars = ax.barh(
                    y + offsets[version_idx],
                    values,
                    height=bar_height,
                    color=VERSION_COLORS[version],
                    label=version,
                )
                legend_handles.setdefault(version, bars[0])
                plotted = True

            if not plotted:
                ax.text(
                    0.5,
                    0.5,
                    "No CSV found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#666666",
                    fontsize=10,
                )

            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.grid(axis="x", alpha=0.12)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=13)
            ax.set_yticks(y)
            ax.tick_params(axis="y", labelleft=False)
            ax.invert_yaxis()

            if col_idx == 0:
                ax.set_yticklabels([])
                for outcome_idx, outcome in enumerate(OUTCOMES):
                    ax.text(
                        -0.04,
                        y[outcome_idx],
                        OUTCOME_LABELS[outcome],
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="center",
                        fontsize=13,
                        clip_on=False,
                    )
            else:
                ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_linewidth(1.3)

    if legend_handles:
        ordered_versions = [version for version in VERSIONS if version in legend_handles]
        fig.legend(
            [legend_handles[version] for version in ordered_versions],
            ordered_versions,
            loc="lower center",
            ncol=len(ordered_versions),
            frameon=False,
            fontsize=16,
            handlelength=2.2,
            columnspacing=2.0,
        )

    fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.11, wspace=0.08, hspace=0.18)

    png_path = OUT_DIR / f"{config.grid_basename}.png"
    pdf_path = OUT_DIR / f"{config.grid_basename}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reduced-agent final terminal outcomes.")
    parser.add_argument(
        "--grid",
        choices=[*(config.key for config in GRID_CONFIGS), "all"],
        default="10m",
        help="Which terminal-outcome grid to generate. Default: 10m.",
    )
    return parser.parse_args()


def run_grid(config: GridConfig) -> None:
    outcomes, warnings = collect_final_terminal_outcomes(config)
    if not outcomes:
        raise RuntimeError(
            f"No {config.label} checkpoint_metrics_*.csv files found for terminal outcomes "
            f"under {CHART_ROOT}"
        )

    source_csv = save_final_outcome_source(outcomes, config)
    individual_paths = plot_individual_terminal_outcomes(outcomes, config)
    grid_path = plot_terminal_outcome_grid(outcomes, config)

    print(f"\nGrid: {config.label}")
    print(f"Saved final outcome source: {source_csv}")
    for path in individual_paths:
        print(f"Saved final outcome plot: {path}")
        print(f"Saved final outcome PDF: {path.with_suffix('.pdf')}")
    if grid_path is not None:
        print(f"Saved final outcome grid: {grid_path}")
        print(f"Saved final outcome grid PDF: {grid_path.with_suffix('.pdf')}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


def main() -> None:
    args = parse_args()
    configs = GRID_CONFIGS if args.grid == "all" else tuple(
        config for config in GRID_CONFIGS if config.key == args.grid
    )
    for config in configs:
        run_grid(config)


if __name__ == "__main__":
    main()

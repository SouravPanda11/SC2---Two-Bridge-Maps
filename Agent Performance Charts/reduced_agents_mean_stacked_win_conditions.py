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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


CHART_ROOT = Path(__file__).resolve().parent
OUT_DIR = CHART_ROOT / "Reduced Agent Aggregate Plots"

VERSIONS = ("V1", "V2", "V3")
VARIANTS = ("Base", "Combat", "Navigate")
WIN_COMPONENTS = ("nav_win_rate_percent", "combat_win_rate_percent")

COMPONENT_LABELS = {
    "nav_win_rate_percent": "Navigation win %",
    "combat_win_rate_percent": "Combat win %",
}
COMPONENT_COLORS = {
    "nav_win_rate_percent": "#2A9D8F",
    "combat_win_rate_percent": "#E76F51",
}
TOTAL_WIN_COLOR = "#1D3557"

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 16


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
        source_csv="reduced_agents_mean_stacked_win_conditions_source.csv",
        target_steps=2_000_000,
        eval_envs=16,
    ),
    GridConfig(
        key="10m",
        label="10M, 8 eval envs",
        output_suffix="_10m_nenv8",
        source_csv="reduced_agents_mean_stacked_win_conditions_10m_nenv8_source.csv",
        target_steps=10_000_000,
        eval_envs=8,
    ),
)


def format_timestep_label(value, _pos=None) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def compute_bar_width(x_values) -> float:
    if len(x_values) <= 1:
        return 25_000.0
    diffs = np.diff(np.sort(np.asarray(x_values, dtype=float)))
    positive_diffs = diffs[diffs > 0]
    min_gap = float(np.min(positive_diffs)) if positive_diffs.size else 25_000.0
    return max(min_gap * 0.55, 10_000.0)


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
        if max_step is None:
            continue
        if max_step < config.target_steps:
            continue
        candidates.append((max_step, path.stat().st_mtime, path))

    if not candidates:
        available_steps = [
            checkpoint_max_step(path)
            for path in matching_envs
            if checkpoint_max_step(path) is not None
        ]
        max_available = max(available_steps) if available_steps else "none"
        return None, (
            f"no nenv{config.eval_envs} CSV reaches {format_timestep_label(config.target_steps)} "
            f"(max available: {max_available})"
        )

    candidates.sort(key=lambda item: (abs(item[0] - config.target_steps), item[0], -item[1]))
    return candidates[0][2], None


def load_mean_stacked_curve(
    csv_path: Path,
    target_steps: int,
    max_plot_steps: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "checkpoint_steps",
        "seed",
        "episodes",
        "win_rate_percent",
        *WIN_COMPONENTS,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    numeric_columns = ["checkpoint_steps", "seed", "episodes", "win_rate_percent", *WIN_COMPONENTS]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=numeric_columns)
    if df.empty:
        return pd.DataFrame()

    df["checkpoint_steps"] = df["checkpoint_steps"].astype(int)
    df["seed"] = df["seed"].astype(int)
    seed_max_steps = df.groupby("seed")["checkpoint_steps"].max()
    eligible_seeds = seed_max_steps[seed_max_steps >= target_steps].index
    df = df[df["seed"].isin(eligible_seeds)]
    df = df[df["checkpoint_steps"] <= max_plot_steps]
    if df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset=["seed", "checkpoint_steps"], keep="last")

    grouped = (
        df.groupby("checkpoint_steps", as_index=False)
        .agg(
            mean_nav_win_rate_percent=("nav_win_rate_percent", "mean"),
            mean_combat_win_rate_percent=("combat_win_rate_percent", "mean"),
            mean_win_rate_percent=("win_rate_percent", "mean"),
            seed_count=("seed", "nunique"),
            mean_episodes=("episodes", "mean"),
        )
        .sort_values("checkpoint_steps")
    )
    return grouped


def collect_curves(config: GridConfig) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    curves: dict[tuple[str, str, str], dict] = {}
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
                    mean_df = load_mean_stacked_curve(
                        csv_path,
                        config.target_steps,
                        config.max_plot_steps,
                    )
                except Exception as exc:
                    warnings.append(f"{agent.label} {map_name}: skipped {csv_path} ({exc})")
                    continue
                if mean_df.empty:
                    warnings.append(f"{agent.label} {map_name}: skipped empty curve from {csv_path}")
                    continue
                curves[(agent.label, version, variant)] = {
                    "agent": agent,
                    "version": version,
                    "variant": variant,
                    "map_name": map_name,
                    "csv_path": csv_path,
                    "eval_envs": eval_env_count(csv_path),
                    "source_max_step": checkpoint_max_step(csv_path),
                    "mean_df": mean_df,
                }

    return curves, warnings


def save_source_csv(curves: dict[tuple[str, str, str], dict], config: GridConfig) -> Path:
    rows = []
    for (agent_label, version, variant), payload in sorted(curves.items()):
        mean_df = payload["mean_df"]
        for row in mean_df.itertuples(index=False):
            rows.append(
                {
                    "grid": config.key,
                    "grid_label": config.label,
                    "agent": agent_label,
                    "version": version,
                    "variant": variant,
                    "map_name": payload["map_name"],
                    "csv_path": str(payload["csv_path"].relative_to(CHART_ROOT)),
                    "eval_envs": payload["eval_envs"],
                    "source_max_checkpoint_steps": payload["source_max_step"],
                    "checkpoint_steps": int(row.checkpoint_steps),
                    "mean_nav_win_rate_percent": round(float(row.mean_nav_win_rate_percent), 6),
                    "mean_combat_win_rate_percent": round(float(row.mean_combat_win_rate_percent), 6),
                    "mean_win_rate_percent": round(float(row.mean_win_rate_percent), 6),
                    "seed_count": int(row.seed_count),
                    "mean_episodes": round(float(row.mean_episodes), 6),
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / config.source_csv
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def plot_agent_grid(
    agent: AgentSpec,
    curves: dict[tuple[str, str, str], dict],
    config: GridConfig,
) -> Path | None:
    if not any(key[0] == agent.label for key in curves):
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(VERSIONS),
        len(VARIANTS),
        figsize=(18.0, 11.0),
        sharex=True,
        sharey=True,
    )

    max_step = 0
    legend_handles = [
        Patch(
            facecolor=COMPONENT_COLORS["nav_win_rate_percent"],
            label=COMPONENT_LABELS["nav_win_rate_percent"],
        ),
        Patch(
            facecolor=COMPONENT_COLORS["combat_win_rate_percent"],
            label=COMPONENT_LABELS["combat_win_rate_percent"],
        ),
    ]
    total_line_handle = None

    for row_idx, version in enumerate(VERSIONS):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            payload = curves.get((agent.label, version, variant))

            if payload is None:
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
            else:
                mean_df = payload["mean_df"]
                x = mean_df["checkpoint_steps"].to_numpy()
                nav_rate = mean_df["mean_nav_win_rate_percent"].to_numpy()
                combat_rate = mean_df["mean_combat_win_rate_percent"].to_numpy()
                total_rate = mean_df["mean_win_rate_percent"].to_numpy()
                bar_width = compute_bar_width(x)
                max_step = max(max_step, int(mean_df["checkpoint_steps"].max()))

                ax.bar(
                    x,
                    nav_rate,
                    width=bar_width,
                    color=COMPONENT_COLORS["nav_win_rate_percent"],
                    alpha=0.92,
                    zorder=1,
                )
                ax.bar(
                    x,
                    combat_rate,
                    width=bar_width,
                    bottom=nav_rate,
                    color=COMPONENT_COLORS["combat_win_rate_percent"],
                    alpha=0.92,
                    zorder=1,
                )
                (total_line,) = ax.plot(
                    x,
                    total_rate,
                    color=TOTAL_WIN_COLOR,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.8,
                    label="Mean total win rate",
                    zorder=3,
                )
                if total_line_handle is None:
                    total_line_handle = total_line

            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.23)
            ax.grid(axis="x", alpha=0.08)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(FuncFormatter(format_timestep_label))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

            if col_idx == 0:
                ax.set_ylabel("Mean win rate (%)", fontsize=AXIS_LABEL_FONTSIZE)
            if row_idx == len(VERSIONS) - 1:
                ax.set_xlabel("Timesteps", fontsize=AXIS_LABEL_FONTSIZE)

            ax.text(
                0.015,
                0.965,
                f"{version}_{variant}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
                color="#333333",
            )

    if max_step > 0:
        for ax in axes.ravel():
            ax.set_xlim(0, max_step * 1.02)

    if total_line_handle is not None:
        fig.legend(
            [*legend_handles, total_line_handle],
            [
                COMPONENT_LABELS["nav_win_rate_percent"],
                COMPONENT_LABELS["combat_win_rate_percent"],
                "Mean total win rate",
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            handlelength=2.4,
            columnspacing=1.8,
        )

    fig.suptitle(
        f"{agent.label} mean stacked win conditions by map variant",
        fontsize=18,
        y=0.992,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.955, bottom=0.11, wspace=0.07, hspace=0.14)

    safe_agent = agent.label.lower().replace(" ", "_")
    png_path = OUT_DIR / f"{safe_agent}_mean_stacked_win_conditions_grid{config.output_suffix}.png"
    pdf_path = OUT_DIR / f"{safe_agent}_mean_stacked_win_conditions_grid{config.output_suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_all_agent_grids(
    curves: dict[tuple[str, str, str], dict],
    config: GridConfig,
) -> list[Path]:
    paths = []
    for agent in AGENTS:
        path = plot_agent_grid(agent, curves, config)
        if path is not None:
            paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reduced-agent stacked win-condition grids.")
    parser.add_argument(
        "--grid",
        choices=[*(config.key for config in GRID_CONFIGS), "all"],
        default="10m",
        help="Which grid to generate. Default: 10m.",
    )
    return parser.parse_args()


def run_grid(config: GridConfig) -> None:
    curves, warnings = collect_curves(config)
    if not curves:
        raise RuntimeError(
            f"No {config.label} checkpoint_metrics_*.csv files found for "
            f"{', '.join(a.label for a in AGENTS)} under {CHART_ROOT}"
        )

    source_csv = save_source_csv(curves, config)
    plot_paths = plot_all_agent_grids(curves, config)

    print(f"\nGrid: {config.label}")
    print(f"Saved mean stacked source: {source_csv}")
    for path in plot_paths:
        print(f"Saved mean stacked plot: {path}")
        print(f"Saved mean stacked PDF: {path.with_suffix('.pdf')}")
    print(f"Curves plotted: {len(curves)}")

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

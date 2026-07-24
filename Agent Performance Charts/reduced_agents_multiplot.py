from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


CHART_ROOT = Path(__file__).resolve().parent
OUT_DIR = CHART_ROOT / "Reduced Agent Aggregate Plots"

VERSIONS = ("V1", "V2", "V3")
VARIANTS = ("Base", "Combat", "Navigate")

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 17
PANEL_LABEL_FONTSIZE = 12


@dataclass(frozen=True)
class AgentSpec:
    label: str
    root_dir: str
    model_dir: str
    color: str
    linestyle: str = "-"


@dataclass(frozen=True)
class GridConfig:
    key: str
    label: str
    output_basename: str
    target_steps: int
    eval_envs: int
    target_tolerance_steps: int = 125_000

    @property
    def max_plot_steps(self) -> int:
        return self.target_steps + self.target_tolerance_steps


AGENTS = (
    AgentSpec("QMIX", "Qmix_reduced", "QMIX_reduced", "#1f77b4"),
    AgentSpec("MaskPPO", "MaskPPO", "MaskPPO_NS_reduced", "#9467bd"),
    AgentSpec("MAPPO", "MAPPO_reduced", "MAPPO_reduced", "#8c564b"),
)

GRID_CONFIGS = (
    GridConfig(
        key="2m",
        label="2M, 16 eval envs",
        output_basename="reduced_agents_mean_winrate_grid",
        target_steps=2_000_000,
        eval_envs=16,
    ),
    GridConfig(
        key="10m",
        label="10M, 8 eval envs",
        output_basename="reduced_agents_mean_winrate_grid_10m_nenv8",
        target_steps=10_000_000,
        eval_envs=8,
    ),
)

LINE_WIDTH = 2.6
MARKER_SIZE = 4.4
MARKER_EDGE_WIDTH = 0.4


def format_timestep_label(value, _pos=None) -> str:
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


def load_mean_win_curve(
    csv_path: Path,
    target_steps: int,
    max_plot_steps: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"checkpoint_steps", "seed", "win_rate_percent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["checkpoint_steps"] = pd.to_numeric(df["checkpoint_steps"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["win_rate_percent"] = pd.to_numeric(df["win_rate_percent"], errors="coerce")
    df = df.dropna(subset=["checkpoint_steps", "seed", "win_rate_percent"])
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

    return (
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


def collect_curves(config: GridConfig) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    curves: dict[tuple[str, str, str], dict] = {}
    warnings: list[str] = []

    for version in VERSIONS:
        for variant in VARIANTS:
            map_name = f"{version}_{variant}"
            for agent in AGENTS:
                csv_path, skip_reason = select_checkpoint_csv(agent, map_name, config)
                if csv_path is None:
                    warnings.append(f"{agent.label} {map_name}: {skip_reason}")
                    continue
                try:
                    mean_df = load_mean_win_curve(
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
                curves[(version, variant, agent.label)] = {
                    "agent": agent,
                    "map_name": map_name,
                    "csv_path": csv_path,
                    "eval_envs": eval_env_count(csv_path),
                    "source_max_step": checkpoint_max_step(csv_path),
                    "mean_df": mean_df,
                }

    return curves, warnings


def save_source_csv(curves: dict[tuple[str, str, str], dict], config: GridConfig) -> Path:
    rows = []
    for (version, variant, agent_label), payload in sorted(curves.items()):
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
                    "mean_win_rate_percent": round(float(row.mean_win_rate_percent), 6),
                    "min_win_rate_percent": round(float(row.min_win_rate_percent), 6),
                    "max_win_rate_percent": round(float(row.max_win_rate_percent), 6),
                    "std_win_rate_percent": (
                        round(float(row.std_win_rate_percent), 6)
                        if pd.notna(row.std_win_rate_percent)
                        else ""
                    ),
                    "seed_count": int(row.seed_count),
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = OUT_DIR / f"{config.output_basename}_source.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False)
    return source_path


def plot_grid(curves: dict[tuple[str, str, str], dict], config: GridConfig) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(VERSIONS),
        len(VARIANTS),
        figsize=(15.5, 10.2),
        sharex=True,
        sharey=True,
    )

    legend_handles = {}
    max_step = 0

    for row_idx, version in enumerate(VERSIONS):
        for col_idx, variant in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            map_name = f"{version}_{variant}"
            plotted = False

            for agent in AGENTS:
                payload = curves.get((version, variant, agent.label))
                if payload is None:
                    continue
                mean_df = payload["mean_df"]
                max_step = max(max_step, int(mean_df["checkpoint_steps"].max()))
                x = mean_df["checkpoint_steps"].to_numpy()
                mean_rate = mean_df["mean_win_rate_percent"].to_numpy()
                (line,) = ax.plot(
                    x,
                    mean_rate,
                    color=agent.color,
                    linestyle=agent.linestyle,
                    linewidth=LINE_WIDTH,
                    marker="o",
                    markersize=MARKER_SIZE,
                    markeredgecolor="white",
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    label=agent.label,
                    zorder=2,
                )
                legend_handles.setdefault(agent.label, line)
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

            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", alpha=0.10)
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
                map_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
                color="#333333",
            )

    if max_step > 0:
        for ax in axes.ravel():
            ax.set_xlim(0, max_step * 1.02)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            handlelength=2.6,
            columnspacing=1.8,
        )

    fig.subplots_adjust(left=0.075, right=0.99, top=0.985, bottom=0.125, wspace=0.08, hspace=0.16)

    png_path = OUT_DIR / f"{config.output_basename}.png"
    pdf_path = OUT_DIR / f"{config.output_basename}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reduced-agent mean win-rate grids.")
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

    source_path = save_source_csv(curves, config)
    png_path = plot_grid(curves, config)
    print(f"\nGrid: {config.label}")
    print(f"Saved source: {source_path}")
    print(f"Saved plot: {png_path}")
    print(f"Saved PDF: {png_path.with_suffix('.pdf')}")
    print(f"Curves plotted: {len(curves)}")

    by_map: dict[str, list[str]] = {}
    for (version, variant, agent_label), payload in sorted(curves.items()):
        map_name = f"{version}_{variant}"
        csv_path = payload["csv_path"].relative_to(CHART_ROOT)
        max_step = format_timestep_label(payload["source_max_step"])
        by_map.setdefault(map_name, []).append(
            f"{agent_label} (nenv{payload['eval_envs']}, max {max_step}, {csv_path})"
        )

    print("\nAvailable curves:")
    for version in VERSIONS:
        for variant in VARIANTS:
            map_name = f"{version}_{variant}"
            agents = by_map.get(map_name, [])
            if agents:
                print(f"  {map_name}:")
                for item in agents:
                    print(f"    - {item}")
            else:
                print(f"  {map_name}: none")

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

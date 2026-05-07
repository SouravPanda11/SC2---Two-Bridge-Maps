from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


CHART_ROOT = Path(__file__).resolve().parent
OUT_DIR = CHART_ROOT / "Reduced Agent Aggregate Plots"
OUT_BASENAME = "reduced_agents_mean_winrate_grid"

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


AGENTS = (
    AgentSpec("QMIX", "Qmix_reduced", "QMIX_reduced", "#1f77b4"),
    AgentSpec("MaskPPO", "MaskPPO", "MaskPPO_NS_reduced", "#9467bd"),
    AgentSpec("MAPPO", "MAPPO_reduced", "MAPPO_reduced", "#8c564b"),
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


def latest_checkpoint_csv(agent: AgentSpec, map_name: str) -> Path | None:
    sweep_dir = CHART_ROOT / agent.root_dir / map_name / agent.model_dir / "checkpoint_sweep"
    csvs = sorted(
        sweep_dir.glob("checkpoint_metrics_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return csvs[0] if csvs else None


def load_mean_win_curve(csv_path: Path) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["checkpoint_steps", "mean_win_rate_percent"])

    df["checkpoint_steps"] = df["checkpoint_steps"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df = df.drop_duplicates(subset=["seed", "checkpoint_steps"], keep="last")

    return (
        df.groupby("checkpoint_steps", as_index=False)["win_rate_percent"]
        .mean()
        .rename(columns={"win_rate_percent": "mean_win_rate_percent"})
        .sort_values("checkpoint_steps")
    )


def collect_curves() -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    curves: dict[tuple[str, str, str], dict] = {}
    warnings: list[str] = []

    for version in VERSIONS:
        for variant in VARIANTS:
            map_name = f"{version}_{variant}"
            for agent in AGENTS:
                csv_path = latest_checkpoint_csv(agent, map_name)
                if csv_path is None:
                    continue
                try:
                    mean_df = load_mean_win_curve(csv_path)
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
                    "mean_df": mean_df,
                }

    return curves, warnings


def plot_grid(curves: dict[tuple[str, str, str], dict]) -> Path:
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
                (line,) = ax.plot(
                    mean_df["checkpoint_steps"],
                    mean_df["mean_win_rate_percent"],
                    color=agent.color,
                    linestyle=agent.linestyle,
                    linewidth=LINE_WIDTH,
                    marker="o",
                    markersize=MARKER_SIZE,
                    markeredgecolor="white",
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    label=agent.label,
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

    png_path = OUT_DIR / f"{OUT_BASENAME}.png"
    pdf_path = OUT_DIR / f"{OUT_BASENAME}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    curves, warnings = collect_curves()
    if not curves:
        raise RuntimeError(
            f"No checkpoint_metrics_*.csv files found for {', '.join(a.label for a in AGENTS)} "
            f"under {CHART_ROOT}"
        )

    png_path = plot_grid(curves)
    print(f"Saved plot: {png_path}")
    print(f"Saved PDF: {png_path.with_suffix('.pdf')}")
    print(f"Curves plotted: {len(curves)}")

    by_map: dict[str, list[str]] = {}
    for (version, variant, agent_label), payload in sorted(curves.items()):
        map_name = f"{version}_{variant}"
        csv_path = payload["csv_path"].relative_to(CHART_ROOT)
        by_map.setdefault(map_name, []).append(f"{agent_label} ({csv_path})")

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


if __name__ == "__main__":
    main()

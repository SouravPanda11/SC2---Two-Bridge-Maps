from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHART_ROOT = Path(__file__).resolve().parent
OUT_DIR = CHART_ROOT / "Reduced Agent Aggregate Plots"
FINAL_OUTCOME_GRID_BASENAME = "reduced_agents_final_terminal_outcome_grid"
FINAL_OUTCOME_SOURCE_CSV = "reduced_agents_final_terminal_outcome_source.csv"

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


AGENTS = (
    AgentSpec("QMIX", "Qmix_reduced", "QMIX_reduced"),
    AgentSpec("MaskPPO", "MaskPPO", "MaskPPO_NS_reduced"),
    AgentSpec("MAPPO", "MAPPO_reduced", "MAPPO_reduced"),
)


def latest_checkpoint_csv(agent: AgentSpec, map_name: str) -> Path | None:
    sweep_dir = CHART_ROOT / agent.root_dir / map_name / agent.model_dir / "checkpoint_sweep"
    csvs = sorted(
        sweep_dir.glob("checkpoint_metrics_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return csvs[0] if csvs else None


def load_final_terminal_outcomes(csv_path: Path) -> dict:
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
    }


def collect_final_terminal_outcomes() -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    outcomes: dict[tuple[str, str, str], dict] = {}
    warnings: list[str] = []

    for agent in AGENTS:
        for version in VERSIONS:
            for variant in VARIANTS:
                map_name = f"{version}_{variant}"
                csv_path = latest_checkpoint_csv(agent, map_name)
                if csv_path is None:
                    warnings.append(f"{agent.label} {map_name}: no checkpoint metrics CSV found")
                    continue
                try:
                    payload = load_final_terminal_outcomes(csv_path)
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
                    }
                )
                outcomes[(agent.label, version, variant)] = payload

    return outcomes, warnings


def save_final_outcome_source(outcomes: dict[tuple[str, str, str], dict]) -> Path:
    rows = []
    for (agent_label, version, variant), payload in sorted(outcomes.items()):
        row = {
            "agent": agent_label,
            "version": version,
            "variant": variant,
            "map_name": payload["map_name"],
            "csv_path": str(payload["csv_path"].relative_to(CHART_ROOT)),
            "total_episodes": payload["total_episodes"],
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
    csv_path = OUT_DIR / FINAL_OUTCOME_SOURCE_CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def plot_terminal_outcomes_for_agent(
    agent: AgentSpec,
    outcomes: dict[tuple[str, str, str], dict],
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

    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.22, wspace=0.10)

    safe_agent = agent.label.lower().replace(" ", "_")
    png_path = OUT_DIR / f"{safe_agent}_final_terminal_outcomes.png"
    pdf_path = OUT_DIR / f"{safe_agent}_final_terminal_outcomes.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_individual_terminal_outcomes(
    outcomes: dict[tuple[str, str, str], dict],
) -> list[Path]:
    paths = []
    for agent in AGENTS:
        path = plot_terminal_outcomes_for_agent(agent, outcomes)
        if path is not None:
            paths.append(path)
    return paths


def plot_terminal_outcome_grid(
    outcomes: dict[tuple[str, str, str], dict],
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

    fig.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.11, wspace=0.08, hspace=0.18)

    png_path = OUT_DIR / f"{FINAL_OUTCOME_GRID_BASENAME}.png"
    pdf_path = OUT_DIR / f"{FINAL_OUTCOME_GRID_BASENAME}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    outcomes, warnings = collect_final_terminal_outcomes()
    if not outcomes:
        raise RuntimeError(
            f"No checkpoint_metrics_*.csv files found for terminal outcomes under {CHART_ROOT}"
        )

    source_csv = save_final_outcome_source(outcomes)
    individual_paths = plot_individual_terminal_outcomes(outcomes)
    grid_path = plot_terminal_outcome_grid(outcomes)

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


if __name__ == "__main__":
    main()

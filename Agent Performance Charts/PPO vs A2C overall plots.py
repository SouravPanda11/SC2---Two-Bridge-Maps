import os, glob, json, re
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────── config ─────────────────────
BASE_ROOT = "Agent Performance Charts"
ALGOS = ["PPO", "A2C"]

TARGET_MAP = "V2_Base"

# ✅ remove tie from plotting
TERMINAL_OUTCOMES = ["nav_win", "combat_win", "combat_loss", "timeout_loss"]

RECIPES = ["NSF", "SF"]  # order on the plot

OUT_DIR = os.path.join(BASE_ROOT, "PPO vs A2C Aggregate Plots")
os.makedirs(OUT_DIR, exist_ok=True)

RECIPE_COLORS = {"NSF": "tab:blue", "SF": "tab:orange"}

# ───────────────── helper parsing ─────────────────
def infer_algo_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "Agent Performance Charts" in parts:
        i = parts.index("Agent Performance Charts")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "Unknown"

def infer_recipe(agent_name: str) -> str:
    # e.g., SB_PPO_NSF_AS14 / SB_A2C_SF_AS14
    m = re.search(r"_([A-Z]{2,3})_AS14\b", agent_name)
    if m:
        return m.group(1)
    if "_NSF_" in agent_name:
        return "NSF"
    if "_SF_" in agent_name:
        return "SF"
    return "Unknown"

# ───────────────── load summaries ─────────────────
json_paths = []
for algo in ALGOS:
    root = os.path.join(BASE_ROOT, algo)
    json_paths.extend(glob.glob(os.path.join(root, "**", "*_summary_*.json"), recursive=True))

if not json_paths:
    raise FileNotFoundError(
        f"No summary jsons found under {BASE_ROOT}/PPO or {BASE_ROOT}/A2C.\n"
        f"Run from repo root and confirm eval wrote *_summary_*.json."
    )

rows = []
for p in json_paths:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    algo = infer_algo_from_path(p)
    agent = d.get("agent", "")
    recipe = infer_recipe(agent)
    map_name = d.get("map", "")

    if algo not in ALGOS:
        continue
    if recipe not in RECIPES:
        continue
    if map_name != TARGET_MAP:
        continue

    counts = d.get("episode_counts", {}) or {}
    row = {
        "path": p,
        "algo": algo,
        "recipe": recipe,
        "agent": agent,
        "map": map_name,
        "episodes": int(d.get("episodes", 0) or 0),
        "seed": d.get("seed", None),
    }
    for k in TERMINAL_OUTCOMES:
        row[k] = int(counts.get(k, 0) or 0)
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(
        f"Found summary jsons, but none matched TARGET_MAP='{TARGET_MAP}' "
        f"and recipes {RECIPES} under algos {ALGOS}."
    )

# ───────────── aggregate across runs ─────────────
agg = (
    df.groupby(["algo", "recipe"], as_index=False)[TERMINAL_OUTCOMES + ["episodes"]]
      .sum()
)

# ✅ optional sanity check: outcomes should sum to episodes (ignoring tie by design)
sum_outcomes = agg[TERMINAL_OUTCOMES].sum(axis=1)
bad = sum_outcomes != agg["episodes"]
if bad.any():
    print("WARNING: outcome counts (without tie) do not sum to episodes for:")
    print(agg.loc[bad, ["algo", "recipe", "episodes"] + TERMINAL_OUTCOMES].to_string(index=False))
    print("Note: If tie exists in the underlying json, episodes will be larger than the plotted outcomes.")

# ✅ shared y-axis limit from evaluation episode totals
y_lim_top = int(agg["episodes"].max())

# # ───────────────── plotting ─────────────────
# def plot_algo(algo: str) -> str:
#     sub = agg[agg["algo"] == algo].copy()

#     # ensure both recipes appear even if one is missing
#     for r in RECIPES:
#         if r not in set(sub["recipe"]):
#             sub = pd.concat(
#                 [sub, pd.DataFrame([{
#                     "algo": algo, "recipe": r, "episodes": 0,
#                     **{k: 0 for k in TERMINAL_OUTCOMES}
#                 }])],
#                 ignore_index=True
#             )

#     sub = sub.set_index("recipe").loc[RECIPES].reset_index()

#     plot_df = pd.DataFrame({"outcome": TERMINAL_OUTCOMES})
#     for r in RECIPES:
#         plot_df[r] = [int(sub.loc[sub["recipe"] == r, k].values[0]) for k in TERMINAL_OUTCOMES]

#     plot_mat = plot_df.set_index("outcome")[RECIPES]
#     ax = plot_mat.plot(kind="bar", color=[RECIPE_COLORS[r] for r in RECIPES])

#     ax.set_title(f"{algo} | NSF vs SF | {TARGET_MAP}", fontsize=15, fontweight="bold")
#     ax.set_xlabel("")
#     ax.set_ylabel("Episode count", fontsize=12)

#     # lock y-axis across plots
#     ax.set_ylim(0, y_lim_top)

#     ax.tick_params(axis="x", labelsize=12)
#     ax.tick_params(axis="y", labelsize=12)

#     ax.legend([f"{r} recipe" for r in RECIPES], title="Training recipe",
#               fontsize=12, title_fontsize=12)

#     plt.xticks(rotation=0)
#     plt.tight_layout()

#     out_path = os.path.join(OUT_DIR, f"{algo}_{TARGET_MAP}_NSF_vs_SF.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     return out_path

# out_files = [plot_algo("A2C"), plot_algo("PPO")]

# ───────────────── combined vertical plotting ─────────────────
def plot_vertical_algos(algos=("PPO", "A2C")) -> str:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7, 8),
        sharey=True
    )

    for ax, algo in zip(axes, algos):
        sub = agg[agg["algo"] == algo].copy()

        # ensure both recipes appear even if one is missing
        for r in RECIPES:
            if r not in set(sub["recipe"]):
                sub = pd.concat(
                    [sub, pd.DataFrame([{
                        "algo": algo, "recipe": r, "episodes": 0,
                        **{k: 0 for k in TERMINAL_OUTCOMES}
                    }])],
                    ignore_index=True
                )

        sub = sub.set_index("recipe").loc[RECIPES].reset_index()

        plot_df = pd.DataFrame({"outcome": TERMINAL_OUTCOMES})
        for r in RECIPES:
            plot_df[r] = [
                int(sub.loc[sub["recipe"] == r, k].values[0])
                for k in TERMINAL_OUTCOMES
            ]

        plot_mat = plot_df.set_index("outcome")[RECIPES]
        plot_mat.plot(
            kind="bar",
            ax=ax,
            color=[RECIPE_COLORS[r] for r in RECIPES],
            width=0.75
        )

        ax.set_title(f"{algo} | NSF vs SF | {TARGET_MAP}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, y_lim_top)

        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="x", labelrotation=0)
        ax.tick_params(axis="y", labelsize=11)

        ax.legend(
            [f"{r} recipe" for r in RECIPES],
            title="Training recipe",
            fontsize=11,
            title_fontsize=11
        )

    # axes[-1].set_xlabel("Terminal outcome", fontsize=12)
    # plt.xticks(rotation=0)

    fig.tight_layout()

    out_path = os.path.join(
        OUT_DIR, f"PPO_A2C_{TARGET_MAP}_NSF_vs_SF_vertical.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

out_file = plot_vertical_algos(("PPO", "A2C"))
print("Saved plot:", out_file)

print("\nAggregated counts (tie not plotted):")
print(agg.sort_values(["algo", "recipe"]).to_string(index=False))
print(f"\nShared y-axis max (episodes): {y_lim_top}")

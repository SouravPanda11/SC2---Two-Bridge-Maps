import os, glob, json
import pandas as pd
import matplotlib.pyplot as plt

# ---- config ----
ROOT = "Agent Performance Charts"
OUT_DIR = os.path.join(ROOT, "MPPO Aggregate Plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ choose which model folder to plot (change only this)
MODEL_NAME = "SB_MaskPPO_FAM_CAM"
# MODEL_NAME = "SB_MaskPPO_SF_AM_RM_mean"

# ✅ remove tie for plotting/aggregation
TERMINAL_OUTCOMES = ["nav_win", "combat_win", "combat_loss", "timeout_loss"]

VERSIONS = ["V1", "V2", "V3"]
VARIANTS = ["Base", "Combat", "Navigate"]

VERSION_COLORS = {"V1": "green", "V2": "orange", "V3": "red"}
VERSION_LABELS = {"V1": "V1 (easy)", "V2": "V2 (medium)", "V3": "V3 (hard)"}

def classify_variant(map_str: str) -> str:
    parts = map_str.split("_", 1)
    return parts[1] if len(parts) == 2 else "Unknown"

def classify_version(map_str: str) -> str:
    return map_str.split("_", 1)[0]

# ---- load all summary jsons ----
json_paths = glob.glob(os.path.join(ROOT, "**", "summary_*ep.json"), recursive=True)
if not json_paths:
    raise FileNotFoundError(
        f"No summary jsons found under: {ROOT}\n"
        f"Expected files named like summary_20ep.json"
    )

rows = []
for p in json_paths:
    # model folder is the parent dir of summary json
    model_dir = os.path.basename(os.path.dirname(p))

    # ✅ filter by selected model folder name
    if MODEL_NAME and model_dir != MODEL_NAME:
        continue

    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    map_name = d.get("map", "")
    ver = classify_version(map_name)
    var = classify_variant(map_name)

    if ver not in VERSIONS or var not in VARIANTS:
        continue

    counts = d.get("episode_counts", {}) or {}
    row = {
        "path": p,
        "agent": d.get("agent", ""),
        "model": model_dir,  # ✅ track model folder
        "map": map_name,
        "version": ver,
        "variant": var,
        "episodes": int(d.get("episodes", 0) or 0),
        "seed": d.get("seed", None),
    }
    for k in TERMINAL_OUTCOMES:
        row[k] = int(counts.get(k, 0) or 0)
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(
        "No summary_*.json matched your filters.\n"
        f"Check MODEL_NAME='{MODEL_NAME}' and that summary files exist under that folder."
    )

# ---- enforce exactly one evaluation per (variant, version, model) ----
dup = df.groupby(["variant", "version", "model"]).size()
bad = dup[dup > 1]
if not bad.empty:
    msg = ["Multiple evaluation summaries found for the same (variant, version, model):", bad.to_string()]
    for (variant, version, model), _ in bad.items():
        sub = df[(df["variant"] == variant) & (df["version"] == version) & (df["model"] == model)]
        msg.append(f"\n--- Files for {version}_{variant} ({model}) ---")
        msg.append(sub[["path", "episodes", "seed", "agent"]].to_string(index=False))
    raise RuntimeError("\n".join(msg))

agg = df.copy()

# ✅ optional sanity check: outcomes should sum to episodes
sum_outcomes = agg[TERMINAL_OUTCOMES].sum(axis=1)
bad = sum_outcomes != agg["episodes"]
if bad.any():
    print("WARNING: outcome counts do not sum to episodes for:")
    print(agg.loc[bad, ["variant", "version", "episodes"] + TERMINAL_OUTCOMES].to_string(index=False))

# ✅ shared y-axis limit from eval episode totals (per-map, should be 10/20/etc.)
y_lim_top = int(agg["episodes"].max())

def plot_variant(variant: str):
    sub = agg[agg["variant"] == variant].copy()

    # Ensure all versions appear (even if missing)
    for v in VERSIONS:
        if v not in set(sub["version"]):
            sub = pd.concat(
                [sub, pd.DataFrame([{
                    "variant": variant, "version": v, "episodes": 0, "model": MODEL_NAME,
                    **{k: 0 for k in TERMINAL_OUTCOMES}
                }])],
                ignore_index=True
            )

    sub = sub.set_index("version").loc[VERSIONS].reset_index()

    plot_df = pd.DataFrame({"outcome": TERMINAL_OUTCOMES})
    for v in VERSIONS:
        plot_df[v] = [int(sub.loc[sub["version"] == v, k].values[0]) for k in TERMINAL_OUTCOMES]

    plot_mat = plot_df.set_index("outcome")[VERSIONS]
    ax = plot_mat.plot(kind="bar", color=[VERSION_COLORS[v] for v in VERSIONS])

    # ax.set_title(f"{variant}: V1 vs V2 vs V3", fontsize=15, fontweight="bold")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Episode count", fontsize=12)

    # lock y-axis across all plots to avoid misinterpretation
    ax.set_ylim(0, y_lim_top)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.legend([VERSION_LABELS[v] for v in VERSIONS],
              title="Difficulty", fontsize=12, title_fontsize=12)

    plt.xticks(rotation=0)
    plt.tight_layout()

    # ✅ output inside a model-named subfolder
    model_out_dir = os.path.join(OUT_DIR, MODEL_NAME)
    os.makedirs(model_out_dir, exist_ok=True)

    out_path = os.path.join(model_out_dir, f"{variant}_V1_V2_V3_terminal_outcomes.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

out_files = [plot_variant(var) for var in VARIANTS]

print("Saved plots:")
for f in out_files:
    print(" -", f)

print("\nAggregated counts:")
print(agg.sort_values(["variant", "version"]).to_string(index=False))
print(f"\nShared y-axis max (episodes): {y_lim_top}")

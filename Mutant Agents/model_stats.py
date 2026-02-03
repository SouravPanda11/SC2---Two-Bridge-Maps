import os, sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO

AGENT_NAME = "SB_MaskPPO_FAM_CAM"
map_name   = "V2_Base"
MODEL_PATH = os.path.join(
    "Agents", "MaskPPO", map_name, "saved_models",
    AGENT_NAME, f"{AGENT_NAME}_final.zip"
)

OUT_DIR = os.path.join("Mutant Agents", f"{map_name}_mutants", "weight_stats")
os.makedirs(OUT_DIR, exist_ok=True)

# Which actor parts to inspect
MUTATE_POLICY_NET = True     # mlp_extractor.policy_net.*
MUTATE_ACTION_NET = True     # action_net.*

def is_actor_param(name: str) -> bool:
    if MUTATE_POLICY_NET and name.startswith("mlp_extractor.policy_net"):
        return True
    if MUTATE_ACTION_NET and name.startswith("action_net"):
        return True
    return False

if not os.path.isfile(MODEL_PATH):
    sys.exit(f"[ERROR] Model file not found at: {MODEL_PATH}")

model = MaskablePPO.load(
    MODEL_PATH,
    device="cpu",
    custom_objects={"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
)
policy = model.policy

rows = []
all_selected = []

for name, p in policy.named_parameters():
    if not is_actor_param(name):
        continue
    if not p.requires_grad:
        continue
    if p.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        continue

    x = p.detach().float().cpu().numpy().ravel()
    all_selected.append(x)

    q = np.percentile(x, [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100])

    rows.append({
        "name": name,
        "shape": tuple(p.shape),
        "dtype": str(p.dtype),
        "n": x.size,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "p0": float(q[0]),
        "p0.1": float(q[1]),
        "p1": float(q[2]),
        "p5": float(q[3]),
        "p50": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "p99.9": float(q[7]),
        "p100": float(q[8]),
    })

df = pd.DataFrame(rows).sort_values("std", ascending=False)
csv_path = os.path.join(OUT_DIR, "actor_param_stats.csv")
df.to_csv(csv_path, index=False)

print("\n=== TOP 15 by std (selected actor params) ===")
print(df[["name","shape","mean","std","p1","p99"]].head(15).to_string(index=False))

if len(all_selected) > 0:
    big = np.concatenate(all_selected)
    print("\n=== GLOBAL (all selected actor params) ===")
    print(f"count={big.size}")
    print(f"mean={big.mean():.6g}  std={big.std():.6g}")
    print(f"p1={np.percentile(big,1):.6g}  p99={np.percentile(big,99):.6g}")

    # Optional: global histogram
    plt.figure()
    plt.hist(big, bins=200)
    plt.title("Histogram: all selected actor parameters")
    plt.xlabel("value"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "actor_params_hist.png"), dpi=200)

    # Optional: per-layer histograms (can be many files)
    for _, r in df.iterrows():
        name = r["name"]
        # sanitize filename
        fname = name.replace(".", "_").replace("/", "_")
        # grab weights again
        p = dict(policy.named_parameters())[name]
        x = p.detach().float().cpu().numpy().ravel()

        plt.figure()
        plt.hist(x, bins=200)
        plt.title(f"Histogram: {name}\nstd={x.std():.4g}  p1={np.percentile(x,1):.4g}  p99={np.percentile(x,99):.4g}")
        plt.xlabel("value"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{fname}.png"), dpi=200)
        plt.close()

print(f"\n[DONE] Wrote: {csv_path}")
print(f"[DONE] Plots in: {OUT_DIR}")

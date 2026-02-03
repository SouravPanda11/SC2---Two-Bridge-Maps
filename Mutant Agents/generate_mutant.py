import os
import csv
import copy
import numpy as np
import torch
from sb3_contrib import MaskablePPO

# Config
AGENT_NAME = "SB_MaskPPO_FAM_CAM"
map_name   = "V2_Base"

BASE_MODEL_PATH = os.path.join(
    "Agents", "MaskPPO", map_name, "saved_models",
    AGENT_NAME, f"{AGENT_NAME}_final.zip"
)
MUTANT_DIR = os.path.join("Mutant Agents", f"{map_name}_mutants")

N_MUTANTS = 5

# Mutation settings
SIGMA = 10                 # noise scale
MODE  = "absolute"           # "absolute" or "relative"

# Which actor parts to mutate
MUTATE_POLICY_NET = True     # mlp_extractor.policy_net.*
MUTATE_ACTION_NET = True     # action_net.*

if MUTATE_ACTION_NET and MUTATE_POLICY_NET:
    OUT_DIR = os.path.join(MUTANT_DIR, "mutate_policy_action_net")
elif MUTATE_POLICY_NET:
    OUT_DIR = os.path.join(MUTANT_DIR, "mutate_policy_net")
elif MUTATE_ACTION_NET:
    OUT_DIR = os.path.join(MUTANT_DIR, "mutate_action_net")
else:
    OUT_DIR = os.path.join(MUTANT_DIR, "mutate_none")

# Reproducibility of mutation
MUTATION_SEED_BASE = 1000

os.makedirs(OUT_DIR, exist_ok=True)

# Helpers
def is_actor_param(name: str) -> bool:
    """Select which parameters are considered 'actor-side'."""
    if (MUTATE_POLICY_NET and name.startswith("mlp_extractor.policy_net")):
        return True
    if (MUTATE_ACTION_NET and name.startswith("action_net")):
        return True
    return False

def should_mutate_param(name: str, param: torch.Tensor) -> bool:
    if not is_actor_param(name):
        return False
    if not param.requires_grad:
        return False
    if param.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        return False
    return True

@torch.no_grad()
def mutate_policy_inplace(policy, sigma: float, mode: str, rng: np.random.Generator):
    """
    Adds Gaussian noise to selected parameters of policy in-place.

    mode:
      - "absolute": param += Normal(0, sigma)
      - "relative": param += Normal(0, sigma * std(param))
    """
    mutated = []
    skipped = []

    for name, param in policy.named_parameters():
        if not should_mutate_param(name, param):
            skipped.append(name)
            continue

        # compute scale
        if mode == "absolute":
            noise_scale = sigma
        elif mode == "relative":
            # scale by parameter tensor std (avoid zero std)
            p_std = float(param.std().cpu().item())
            noise_scale = sigma * (p_std if p_std > 1e-12 else 1.0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        noise_np = rng.normal(loc=0.0, scale=noise_scale, size=tuple(param.shape))
        noise = torch.as_tensor(noise_np, dtype=param.dtype, device=param.device)
        param.add_(noise)

        mutated.append((name, noise_scale))

    return mutated, skipped

def load_model_clean(path: str):
    """
    Load without SB3 schedule deserialization issues.
    This does NOT affect weights (only replaces schedules).
    """
    return MaskablePPO.load(
        path,
        device="cpu",
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )

# Generate mutants
base_model = load_model_clean(BASE_MODEL_PATH)

manifest_path = os.path.join(OUT_DIR, "mutants_manifest.csv")
with open(manifest_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mutant_id", "seed", "sigma", "mode", "mutated_param_count", "example_param", "example_scale", "saved_path"])

    for i in range(N_MUTANTS):
        seed = MUTATION_SEED_BASE + i
        rng = np.random.default_rng(seed)

        mutant = copy.deepcopy(base_model)

        mutated, _ = mutate_policy_inplace(mutant.policy, sigma=SIGMA, mode=MODE, rng=rng)

        out_path = os.path.join(OUT_DIR, f"mutant_{i+1:02d}.zip")
        mutant.save(out_path)

        example_name = mutated[0][0] if mutated else ""
        example_scale = mutated[0][1] if mutated else ""
        writer.writerow([i+1, seed, SIGMA, MODE, len(mutated), example_name, example_scale, out_path])

print(f"[DONE] Saved {N_MUTANTS} mutants to: {OUT_DIR}")
print(f"[DONE] Manifest: {manifest_path}")
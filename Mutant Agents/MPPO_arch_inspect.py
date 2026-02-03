from sb3_contrib import MaskablePPO
import os, sys
import torch.nn as nn

AGENT_NAME = "SB_MaskPPO_FAM_CAM"
map_name   = "V2_Base"

MODEL_PATH = os.path.join(
    "Agents", "MaskPPO", map_name, "saved_models",
    AGENT_NAME, f"{AGENT_NAME}_final.zip"
)

if not os.path.isfile(MODEL_PATH):
    sys.exit(f"[ERROR] Model file not found at: {MODEL_PATH}")

# Silence FloatSchedule deserialization issues (only schedules, not weights)
model = MaskablePPO.load(
    MODEL_PATH,
    device="cpu",
    custom_objects={
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    },
)
policy = model.policy

print("\n===== POLICY SUMMARY =====")
print(policy)

def print_module_tree(module, indent=0):
    pad = " " * indent
    for name, child in module.named_children():
        print(f"{pad}- {name}: {child.__class__.__name__}")
        print_module_tree(child, indent + 2)

print("\n===== MODULE TREE =====")
print_module_tree(policy)

print("\n===== TRAINABLE PARAMETERS =====")
for name, param in policy.named_parameters():
    shape_str = str(tuple(param.shape))
    print(f"{name:70s} shape={shape_str:18s} requires_grad={param.requires_grad}")

print("\n===== FEATURE EXTRACTOR =====")
print(policy.features_extractor.__class__.__name__)

def summarize_module(m: nn.Module, max_depth=3, _depth=0, prefix=""):
    if _depth > max_depth:
        return
    for name, child in m.named_children():
        print(f"{prefix}{name}: {child.__class__.__name__}")
        summarize_module(child, max_depth=max_depth, _depth=_depth+1, prefix=prefix+"  ")

if hasattr(policy.features_extractor, "extractors"):
    print("\n===== PER-KEY EXTRACTORS (CLEAN) =====")
    for k, extractor in policy.features_extractor.extractors.items():
        print(f"\n--- key: {k} | {extractor.__class__.__name__} ---")
        summarize_module(extractor, max_depth=4)

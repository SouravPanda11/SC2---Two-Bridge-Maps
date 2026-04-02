import subprocess
import sys
from pathlib import Path

# Folder that contains your variant folders (V1_Base, V1_Combat, ...)
ROOT = Path(__file__).parent

# ---- CONFIG: choose one or more eval filenames to run ----
TARGET_NAMES = [
    # "eval_AM_RM_mean_agent.py",
    "eval_FAM_CAM_agent.py",
]

# Optional: restrict to specific variant folders by prefix
ALLOWED_FOLDER_PREFIXES = [
    "V1_",
    "V2_",
    "V3_",
]

# ---- ONLY override passed to eval scripts ----
EPISODES_OVERRIDE = 50   # change once, affects all evals

def is_allowed_variant_folder(path: Path) -> bool:
    if not ALLOWED_FOLDER_PREFIXES:
        return True
    try:
        variant_folder = path.relative_to(ROOT).parts[0]  # e.g., V1_Base
    except Exception:
        return False
    return any(variant_folder.startswith(p) for p in ALLOWED_FOLDER_PREFIXES)

def main():
    # Collect matches for each target name
    matches = []
    for name in TARGET_NAMES:
        found = list(ROOT.rglob(name))
        found = [p for p in found if is_allowed_variant_folder(p)]
        matches.extend(found)

    # Sort for stable order: by filename then by folder path
    matches = sorted(matches, key=lambda p: (p.name, str(p.parent)))

    if not matches:
        print("❌ No matching eval files found.")
        print(f"   ROOT = {ROOT}")
        print(f"   TARGET_NAMES = {TARGET_NAMES}")
        if ALLOWED_FOLDER_PREFIXES:
            print(f"   ALLOWED_FOLDER_PREFIXES = {ALLOWED_FOLDER_PREFIXES}")
        sys.exit(1)

    print(f"🔍 Will run {len(matches)} eval scripts:\n")
    for p in matches:
        print(f" - {p.relative_to(ROOT)}")
    print(f"\n➡ Passing to each eval: --episodes {EPISODES_OVERRIDE}\n")

    for i, eval_path in enumerate(matches, 1):
        rel = eval_path.relative_to(ROOT)
        cmd = [
            sys.executable,
            str(eval_path),
            "--episodes",
            str(EPISODES_OVERRIDE),
        ]

        print("=" * 90)
        print(f"[{i}/{len(matches)}] ▶ Running: {rel}")
        print(f"CMD: {' '.join(cmd)}")
        print("=" * 90)

        result = subprocess.run(cmd, cwd=eval_path.parent)

        if result.returncode != 0:
            print(f"\n❌ FAILED: {rel} (exit code {result.returncode})")
            sys.exit(result.returncode)

        print(f"\n✅ DONE: {rel}\n")

    print("🎉 All selected eval scripts completed successfully.")

if __name__ == "__main__":
    main()

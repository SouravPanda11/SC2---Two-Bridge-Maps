import os
import sys
import subprocess
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1] 

RUN_A2C = True
RUN_PPO = True

# If empty -> run all eval_*.py found in those folders.
# Otherwise list exact filenames, e.g. ["eval_A2C_SF_AS14_agent.py", "eval_PPO_NSF_AS14_agent.py"]
ONLY_THESE = []

# - Set to None to disable
TARGET_EPISODES = 100
EPISODES_FLAG = "--episodes"

# If True, keep going even if one script fails
CONTINUE_ON_ERROR = False

# -----------------------------
# HELPERS
# -----------------------------
def discover_eval_scripts(folder: Path) -> list[Path]:
    """Find eval_*.py directly under folder (non-recursive)."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("eval_*.py") if p.is_file()])

def filter_scripts(scripts: list[Path], only_these: list[str]) -> list[Path]:
    if not only_these:
        return scripts
    want = set(only_these)
    out = [p for p in scripts if p.name in want]
    missing = want - set(p.name for p in out)
    if missing:
        print("WARNING: These requested scripts were not found:")
        for m in sorted(missing):
            print("  -", m)
    return out

def run_script(script_path: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(script_path), *extra_args]
    print("\n" + "=" * 80)
    print(f"RUN: {script_path.relative_to(PROJECT_ROOT)}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    return proc.returncode


def main():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    extra_args = []
    if TARGET_EPISODES is not None:
        extra_args = [EPISODES_FLAG, str(TARGET_EPISODES)]

    jobs: list[Path] = []

    if RUN_A2C:
        a2c_dir = PROJECT_ROOT / "Agents" / "A2C"
        jobs += discover_eval_scripts(a2c_dir)

    if RUN_PPO:
        ppo_dir = PROJECT_ROOT / "Agents" / "PPO"
        jobs += discover_eval_scripts(ppo_dir)

    jobs = filter_scripts(jobs, ONLY_THESE)

    if not jobs:
        raise FileNotFoundError(
            "No eval scripts found.\n"
            "Expected something like Agents/A2C/eval_*.py and/or Agents/PPO/eval_*.py"
        )

    failures = []
    for script in jobs:
        rc = run_script(script, extra_args)
        if rc != 0:
            failures.append((script, rc))
            print(f"\n❌ FAILED ({rc}): {script.name}")
            if not CONTINUE_ON_ERROR:
                break
        else:
            print(f"\n✅ OK: {script.name}")

    print("\n" + "-" * 80)
    print("DONE.")
    if TARGET_EPISODES is not None:
        print(f"Episodes override attempted: {EPISODES_FLAG} {TARGET_EPISODES}")
        print("(If your eval scripts don't accept this flag, set TARGET_EPISODES=None.)")

    if failures:
        print("\nFailures:")
        for s, rc in failures:
            print(f" - {s.relative_to(PROJECT_ROOT)} (rc={rc})")
        sys.exit(1)

    print("All eval scripts completed successfully.")

if __name__ == "__main__":
    main()
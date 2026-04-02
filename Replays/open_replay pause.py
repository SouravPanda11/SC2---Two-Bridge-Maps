import argparse
import ctypes
import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "TBMsc2").is_dir()
)
REPLAYS_ROOT = SCRIPT_DIR

DEFAULT_TBMSC2_PYTHON = PROJECT_ROOT / "TBMsc2" / "Scripts" / "python.exe"

# Edit this path if you want to launch a specific replay without passing it on
# the command line. Use a path relative to Replays/PPO.
# INLINE_REPLAY = r"PPO/SB_PPO_SF_AS14/combat_loss/ep_23.SC2Replay"
# INLINE_REPLAY = r"PPO/SB_PPO_SF_AS14/nav_win/ep_22.SC2Replay"
# INLINE_REPLAY = r"PPO/SB_PPO_SF_AS14/timeout_loss/ep_18.SC2Replay"
# INLINE_REPLAY = r"A2C/SB_A2C_SF_AS14/combat_loss/ep_5.SC2Replay"
# INLINE_REPLAY = r"A2C/SB_A2C_SF_AS14/nav_win/ep_14.SC2Replay"
# INLINE_REPLAY = r"A2C/SB_A2C_SF_AS14/timeout_loss/ep_1.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V1_Base/SB_MaskPPO_SF_AM_RM_mean/combat_win/ep_1.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V1_Base/SB_MaskPPO_SF_AM_RM_mean/nav_win/ep_17.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V1_Navigate/SB_MaskPPO_SF_AM_RM_mean/nav_win/ep_18.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Base/SB_MaskPPO_SF_AM_RM_mean/combat_loss/ep_7.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Base/SB_MaskPPO_SF_AM_RM_mean/combat_win/ep_3.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Base/SB_MaskPPO_SF_AM_RM_mean/nav_win/ep_1.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Base/SB_MaskPPO_SF_AM_RM_mean/timeout_loss/ep_8.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Combat/SB_MaskPPO_SF_AM_RM_mean/combat_loss/ep_2.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Combat/SB_MaskPPO_SF_AM_RM_mean/combat_win/ep_20.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Combat/SB_MaskPPO_SF_AM_RM_mean/nav_win/ep_1.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Navigate/SB_MaskPPO_SF_AM_RM_mean/combat_win/ep_36.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V2_Navigate/SB_MaskPPO_SF_AM_RM_mean/nav_win/ep_3.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V3_Base/SB_MaskPPO_SF_AM_RM_mean/combat_loss/ep_1.SC2Replay"
# INLINE_REPLAY = r"MaskPPO/V3_Base/SB_MaskPPO_SF_AM_RM_mean/combat_loss/ep_1.SC2Replay"
INLINE_REPLAY = r"MaskPPO/V3_Navigate/SB_MaskPPO_SF_AM_RM_mean/nav_win/ep_2.SC2Replay"

# Replay speed controls. Keep FPS at 22.4 for normal SC2 timing and raise
# STEP_MUL to speed up playback: 2 for about 2x, 3 for about 3x.
INLINE_REPLAY_FPS = 22.4
INLINE_REPLAY_STEP_MUL = 2
# Wait after the replay window opens, before stepping playback forward.
INLINE_PRE_START_DELAY_SECONDS = 10.0
INLINE_POST_END_DELAY_SECONDS = INLINE_PRE_START_DELAY_SECONDS

SC2_FULLSCREEN = False
SC2_REPLAY_WINDOW_SIZE = (1280, 720)
SC2_REPLAY_WINDOW_LOC = (640, 360)
SC2_CAMERA_RENDER_SIZE = (256, 128)
SC2_MINIMAP_RENDER_SIZE = (100, 100)
PAUSED_POLL_INTERVAL_SECONDS = 0.05
PAUSE_TOGGLE_VK_CODES = (0x20, 0x50)  # Space, P

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a saved PPO replay in the built-in SC2 replay window only. "
            "This keeps the 50/50 minimap-plus-camera layout and does not open "
            "the separate PySC2 Starcraft Viewer window."
        )
    )
    parser.add_argument(
        "replay",
        nargs="?",
        help=(
            "Replay path, absolute or relative to Replays/PPO. If omitted, the "
            "script uses INLINE_REPLAY first, then falls back to the latest replay."
        ),
    )
    parser.add_argument(
        "--agent",
        help="Restrict replay search to a single PPO agent folder, e.g. SB_PPO_NSF_AS14.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest replay under Replays/PPO or under --agent.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent replays and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many replays to show with --list. Default: 20.",
    )
    parser.add_argument(
        "--observed-player",
        type=int,
        default=1,
        help="Replay player id for PySC2. Default: 1.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=INLINE_REPLAY_FPS,
        help="PySC2 replay FPS. Default comes from INLINE_REPLAY_FPS.",
    )
    parser.add_argument(
        "--step-mul",
        type=int,
        default=INLINE_REPLAY_STEP_MUL,
        help="Game loops per PySC2 observation. Default comes from INLINE_REPLAY_STEP_MUL.",
    )
    parser.add_argument(
        "--render-sync",
        action="store_true",
        help="Unused in SC2-only replay mode.",
    )
    parser.add_argument(
        "--map-path",
        help="Optional replay map override for PySC2.",
    )
    parser.add_argument(
        "--pysc2-python",
        help="Explicit path to the Python interpreter that should run the internal replay launcher.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the chosen replay and launch commands without opening windows.",
    )
    parser.add_argument(
        "--internal-pysc2",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.replay and args.latest:
        parser.error("Use either a replay path or --latest, not both.")

    return args


def replay_search_root(agent: str | None) -> Path:
    if not agent:
        return REPLAYS_ROOT
    return REPLAYS_ROOT / agent


def find_replays(agent: str | None = None) -> list[Path]:
    root = replay_search_root(agent)
    if not root.exists():
        return []
    return sorted(root.rglob("*.SC2Replay"), key=lambda path: path.stat().st_mtime, reverse=True)


def print_replays(agent: str | None, limit: int) -> int:
    replays = find_replays(agent)
    if not replays:
        root = replay_search_root(agent)
        print(f"No replays found under: {root}")
        return 1

    for replay_path in replays[:limit]:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(replay_path.stat().st_mtime))
        print(f"{ts}  {replay_path}")
    return 0


def replay_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def replay_relpaths() -> list[str]:
    relpaths = []
    for replay_path in find_replays():
        relpaths.append(replay_path.relative_to(REPLAYS_ROOT).as_posix())
    return relpaths


def find_normalized_match(raw_path: str) -> Path | None:
    target = replay_key(raw_path)
    if not target:
        return None

    matches = []
    for relpath in replay_relpaths():
        if replay_key(relpath) == target or replay_key(Path(relpath).name) == target:
            matches.append((REPLAYS_ROOT / relpath).resolve())

    if len(matches) == 1:
        return matches[0]
    return None


def missing_replay_message(raw_path: str) -> str:
    relpaths = replay_relpaths()
    suggestions = difflib.get_close_matches(raw_path.replace("\\", "/"), relpaths, n=3, cutoff=0.45)
    normalized_match = find_normalized_match(raw_path)

    lines = [f"Replay not found: {raw_path}"]
    if normalized_match is not None:
        rel = normalized_match.relative_to(REPLAYS_ROOT).as_posix()
        lines.append(f"Did you mean: {rel}")
        return "\n".join(lines)

    if suggestions:
        lines.append("Closest matches:")
        lines.extend(f"  - {item}" for item in suggestions)
    return "\n".join(lines)


def resolve_replay_path(raw_path: str) -> Path:
    direct = Path(raw_path).expanduser()
    if direct.is_file():
        return direct.resolve()

    relative = (REPLAYS_ROOT / raw_path).resolve()
    if relative.is_file():
        return relative

    normalized_match = find_normalized_match(raw_path)
    if normalized_match is not None:
        return normalized_match

    raise FileNotFoundError(missing_replay_message(raw_path))


def choose_replay(args: argparse.Namespace) -> Path:
    if args.replay:
        return resolve_replay_path(args.replay)

    if not args.latest and INLINE_REPLAY:
        return resolve_replay_path(INLINE_REPLAY)

    replays = find_replays(args.agent)
    if not replays:
        root = replay_search_root(args.agent)
        raise FileNotFoundError(f"No replay files found under: {root}")

    return replays[0]


def pysc2_python_prefix(explicit_path: str | None) -> list[str]:
    if explicit_path:
        explicit = Path(explicit_path).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"PySC2 Python interpreter not found: {explicit}")
        return [str(explicit)]

    if DEFAULT_TBMSC2_PYTHON.is_file():
        return [str(DEFAULT_TBMSC2_PYTHON)]

    for candidate in ("python.exe", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]

    return [sys.executable]


def build_pysc2_command(
    replay_path: Path,
    args: argparse.Namespace,
    prefix: list[str],
) -> list[str]:
    cmd = list(prefix)
    cmd.extend(
        [
            str(Path(__file__).resolve()),
            "--internal-pysc2",
            str(replay_path),
            f"--observed-player={args.observed_player}",
            f"--fps={args.fps}",
            f"--step-mul={args.step_mul}",
        ]
    )

    if args.render_sync:
        cmd.append("--render-sync")
    if args.map_path:
        cmd.append(f"--map-path={args.map_path}")

    return cmd


def sc2_window_settings() -> tuple[tuple[int, int], tuple[int, int]]:
    return SC2_REPLAY_WINDOW_SIZE, SC2_REPLAY_WINDOW_LOC


def pause_toggle_requested(user32, key_states: dict[int, bool]) -> bool:
    if user32 is None:
        return False

    toggled = False
    for vk_code in PAUSE_TOGGLE_VK_CODES:
        is_down = bool(user32.GetAsyncKeyState(vk_code) & 0x8000)
        was_down = key_states.get(vk_code, False)
        if is_down and not was_down:
            toggled = True
        key_states[vk_code] = is_down

    return toggled


def run_pysc2_replay(args: argparse.Namespace) -> int:
    from absl import flags
    from pysc2 import run_configs
    from pysc2.lib import point
    from pysc2.lib import replay as replay_lib
    from s2clientprotocol import sc2api_pb2 as sc_pb

    flags.FLAGS([sys.argv[0]])

    replay_path = choose_replay(args)
    replay_data = replay_path.read_bytes()
    replay_version = replay_lib.get_replay_version(replay_data)

    run_config = run_configs.get(version="latest")
    build_dir = Path(run_config.data_dir) / "Versions" / f"Base{replay_version.build_version:05d}"
    if not build_dir.is_dir():
        raise FileNotFoundError(
            f"Replay build Base{replay_version.build_version:05d} is not installed under {build_dir.parent}."
        )

    run_config.version = run_config.version._replace(
        build_version=replay_version.build_version,
        data_version=replay_version.data_version,
    )
    user32 = ctypes.windll.user32 if sys.platform == "win32" else None
    pause_key_states: dict[int, bool] = {}
    paused = False

    interface = sc_pb.InterfaceOptions()
    interface.raw = True
    interface.raw_affects_selection = True
    interface.raw_crop_to_playable_area = False
    interface.score = True
    interface.show_cloaked = True
    interface.show_burrowed_shadows = True
    interface.show_placeholders = True

    point.Point(84, 84).assign_to(interface.feature_layer.resolution)
    point.Point(64, 64).assign_to(interface.feature_layer.minimap_resolution)
    interface.feature_layer.width = 24
    interface.feature_layer.crop_to_playable_area = False
    interface.feature_layer.allow_cheating_layers = True

    point.Point(*SC2_CAMERA_RENDER_SIZE).assign_to(interface.render.resolution)
    point.Point(*SC2_MINIMAP_RENDER_SIZE).assign_to(interface.render.minimap_resolution)
    interface.render.crop_to_playable_area = False

    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=False,
        observed_player_id=args.observed_player,
    )

    sc2_window_size, sc2_window_loc = sc2_window_settings()
    with run_config.start(
        full_screen=SC2_FULLSCREEN,
        window_size=sc2_window_size,
        window_loc=sc2_window_loc,
        want_rgb=True,
    ) as controller:
        info = controller.replay_info(replay_data)
        map_path = args.map_path or info.local_map_path
        if map_path:
            start_replay.map_data = run_config.map_data(map_path, len(info.player_info))
        controller.start_replay(start_replay)
        controller.observe()
        print("Controls: press Space or P to toggle replay pause/play.")
        if INLINE_PRE_START_DELAY_SECONDS > 0:
            time.sleep(INLINE_PRE_START_DELAY_SECONDS)

        try:
            while True:
                frame_start = time.time()
                if pause_toggle_requested(user32, pause_key_states):
                    paused = not paused
                    print("Replay paused." if paused else "Replay resumed.")

                if not paused:
                    controller.step(args.step_mul)
                obs = controller.observe()
                if obs.player_result:
                    if INLINE_POST_END_DELAY_SECONDS > 0:
                        time.sleep(INLINE_POST_END_DELAY_SECONDS)
                    break
                if paused:
                    time.sleep(PAUSED_POLL_INTERVAL_SECONDS)
                else:
                    time.sleep(max(0.0, frame_start + 1.0 / args.fps - time.time()))
        except KeyboardInterrupt:
            pass

    return 0


def main() -> int:
    args = parse_args()

    if args.internal_pysc2:
        return run_pysc2_replay(args)

    if args.list:
        return print_replays(args.agent, args.limit)

    replay_path = choose_replay(args)
    prefix = pysc2_python_prefix(args.pysc2_python)
    pysc2_cmd = build_pysc2_command(replay_path, args, prefix)

    print(f"Replay: {replay_path}")
    print("Replay launch:")
    print(" ".join(f'"{part}"' if " " in part else part for part in pysc2_cmd))
    print("Mode: single windowed SC2 replay window only; no separate Starcraft Viewer.")

    if args.dry_run:
        return 0

    subprocess.Popen(pysc2_cmd, cwd=str(PROJECT_ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

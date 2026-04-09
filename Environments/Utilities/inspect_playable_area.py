from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_MAP_NAME = "TwoBridgeMap_V1_Base"
DEFAULT_MAP_FILE = "TwoBridgeMap_V1_Base.SC2Map"
DEFAULT_MAP_DIR = r"C:/Program Files (x86)/StarCraft II/Maps/Strategy Maps/Camera Free"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a StarCraft II map through PySC2 and print the map bounds "
            "reported by start_raw.map_size and start_raw.playable_area."
        )
    )
    parser.add_argument(
        "--map-name",
        default=DEFAULT_MAP_NAME,
        help=f"PySC2 map registry name. Default: {DEFAULT_MAP_NAME}",
    )
    parser.add_argument(
        "--map-file",
        default=DEFAULT_MAP_FILE,
        help=f"Map filename inside --map-dir. Default: {DEFAULT_MAP_FILE}",
    )
    parser.add_argument(
        "--map-dir",
        default=DEFAULT_MAP_DIR,
        help=f"Directory containing the .SC2Map file. Default: {DEFAULT_MAP_DIR}",
    )
    parser.add_argument(
        "--map-path",
        help="Optional full path to a .SC2Map file. Overrides --map-file and --map-dir.",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Player count declared in the PySC2 map registration. Default: 2",
    )
    parser.add_argument(
        "--screen-res",
        type=int,
        default=64,
        help="Feature screen resolution to request from PySC2. Default: 64",
    )
    parser.add_argument(
        "--minimap-res",
        type=int,
        default=64,
        help="Feature minimap resolution to request from PySC2. Default: 64",
    )
    parser.add_argument(
        "--raw-resolution",
        type=int,
        default=64,
        help="Raw action/observation discretization resolution. Default: 64",
    )
    parser.add_argument(
        "--step-mul",
        type=int,
        default=8,
        help="Game loops per environment step. Default: 8",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=3,
        help="How many no_op steps to take after reset before reading bounds again. Default: 3",
    )
    parser.add_argument(
        "--agent-race",
        default="terran",
        choices=("random", "terran", "zerg", "protoss"),
        help="Agent race. Default: terran",
    )
    parser.add_argument(
        "--bot-race",
        default="terran",
        choices=("random", "terran", "zerg", "protoss"),
        help="Bot race. Default: terran",
    )
    parser.add_argument(
        "--bot-difficulty",
        default="easy",
        choices=(
            "very_easy",
            "easy",
            "medium",
            "medium_hard",
            "hard",
            "harder",
            "very_hard",
            "cheat_vision",
            "cheat_money",
            "cheat_insane",
        ),
        help="Bot difficulty. Default: easy",
    )
    parser.add_argument(
        "--crop-to-playable-area",
        action="store_true",
        help="Request cropped feature-layer observations.",
    )
    parser.add_argument(
        "--raw-crop-to-playable-area",
        action="store_true",
        help="Request cropped raw-unit observations.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the SC2 visualizer window.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run the SC2 episode in realtime mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON.",
    )
    return parser.parse_args()


def resolve_map_registration(args: argparse.Namespace) -> dict[str, object]:
    map_name = args.map_name
    map_dir = args.map_dir
    map_file = args.map_file

    if args.map_path:
        map_path = Path(args.map_path).expanduser().resolve()
        map_name = map_path.stem
        map_dir = str(map_path.parent)
        map_file = map_path.name

    return {
        "map_name": map_name,
        "map_dir": map_dir,
        "map_file": map_file,
        "players": args.players,
        "map_exists": Path(map_dir, map_file).exists(),
    }


def import_pysc2_modules():
    try:
        from absl import flags
        from pysc2.env import sc2_env
        from pysc2.lib import actions, features
        from pysc2.maps import lib
    except Exception as exc:
        raise RuntimeError(
            "Failed to import PySC2. Run this with the project SC2 environment "
            "(for example `TBMsc2\\Scripts\\python.exe`). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    parsed_flags = flags.FLAGS
    if not parsed_flags.is_parsed():
        parsed_flags([sys.argv[0]])

    return sc2_env, actions, features, lib


def register_map(lib_module, map_name: str, map_dir: str, map_file: str, players: int) -> None:
    existing_maps = lib_module.get_maps()
    if map_name in existing_maps:
        return

    type(
        map_name,
        (lib_module.Map,),
        {
            "directory": map_dir,
            "filename": map_file,
            "players": players,
        },
    )


def point_dict(point_obj) -> dict[str, float]:
    return {"x": float(point_obj.x), "y": float(point_obj.y)}


def describe_game_info(info) -> dict[str, object]:
    report: dict[str, object] = {"has_start_raw": bool(info.HasField("start_raw"))}
    if not info.HasField("start_raw"):
        return report

    start_raw = info.start_raw
    p0 = start_raw.playable_area.p0
    p1 = start_raw.playable_area.p1
    map_size = start_raw.map_size

    report["map_size"] = point_dict(map_size)
    report["playable_area"] = {
        "p0": point_dict(p0),
        "p1": point_dict(p1),
        "width": float(p1.x - p0.x),
        "height": float(p1.y - p0.y),
    }
    report["playable_matches_full_map"] = bool(
        p0.x == 0 and p0.y == 0 and p1.x == map_size.x and p1.y == map_size.y
    )
    if info.options.HasField("feature_layer"):
        fl = info.options.feature_layer
        report["feature_layer_options"] = {
            "screen_resolution": point_dict(fl.resolution),
            "minimap_resolution": point_dict(fl.minimap_resolution),
            "width": float(fl.width),
            "crop_to_playable_area": bool(fl.crop_to_playable_area),
        }
    return report


def describe_observation_proto(obs) -> dict[str, object]:
    raw_units = list(obs.observation.raw_data.units)
    report: dict[str, object] = {
        "game_loop": int(obs.observation.game_loop),
        "raw_unit_count": len(raw_units),
    }

    if not raw_units:
        report["raw_unit_world_bounds"] = None
        return report

    xs = [float(unit.pos.x) for unit in raw_units]
    ys = [float(unit.pos.y) for unit in raw_units]
    report["raw_unit_world_bounds"] = {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }
    report["raw_unit_sample"] = [
        {
            "tag": int(unit.tag),
            "owner": int(unit.owner),
            "unit_type": int(unit.unit_type),
            "x": float(unit.pos.x),
            "y": float(unit.pos.y),
            "health": float(unit.health),
        }
        for unit in raw_units[:8]
    ]
    return report


def _bbox_from_mask(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def describe_feature_minimap(ts, info, minimap_features) -> dict[str, object]:
    feature_minimap = np.asarray(ts.observation["feature_minimap"])
    report: dict[str, object] = {
        "tensor_shape": tuple(int(v) for v in feature_minimap.shape),
        "channel_count": int(feature_minimap.shape[0]),
    }

    if info.options.HasField("feature_layer"):
        fl = info.options.feature_layer
        report["requested_resolution"] = {
            "x": int(fl.minimap_resolution.x),
            "y": int(fl.minimap_resolution.y),
        }
        report["crop_to_playable_area"] = bool(fl.crop_to_playable_area)

    if info.HasField("start_raw"):
        sr = info.start_raw
        if info.options.HasField("feature_layer") and info.options.feature_layer.crop_to_playable_area:
            world_w = float(sr.playable_area.p1.x - sr.playable_area.p0.x)
            world_h = float(sr.playable_area.p1.y - sr.playable_area.p0.y)
            origin = {
                "x": float(sr.playable_area.p0.x),
                "y": float(sr.playable_area.p0.y),
            }
            world_area_source = "playable_area"
        else:
            world_w = float(sr.map_size.x)
            world_h = float(sr.map_size.y)
            origin = {"x": 0.0, "y": 0.0}
            world_area_source = "full_map"

        report["represented_world_area"] = {
            "source": world_area_source,
            "origin": origin,
            "width": world_w,
            "height": world_h,
        }
        report["world_units_per_pixel"] = {
            "x": world_w / feature_minimap.shape[2],
            "y": world_h / feature_minimap.shape[1],
        }

    channel_summaries = []
    for idx, feat in enumerate(minimap_features):
        channel = feature_minimap[idx]
        nonzero = channel != 0
        channel_summaries.append({
            "index": idx,
            "name": feat.name,
            "min": int(channel.min()),
            "max": int(channel.max()),
            "nonzero_count": int(nonzero.sum()),
            "nonzero_bbox": _bbox_from_mask(nonzero),
        })
    report["channels"] = channel_summaries
    return report


def build_report(args: argparse.Namespace) -> dict[str, object]:
    registration = resolve_map_registration(args)
    sc2_env, actions, features, lib = import_pysc2_modules()

    race_map = {
        "random": sc2_env.Race.random,
        "terran": sc2_env.Race.terran,
        "zerg": sc2_env.Race.zerg,
        "protoss": sc2_env.Race.protoss,
    }
    difficulty_map = {
        "very_easy": sc2_env.Difficulty.very_easy,
        "easy": sc2_env.Difficulty.easy,
        "medium": sc2_env.Difficulty.medium,
        "medium_hard": sc2_env.Difficulty.medium_hard,
        "hard": sc2_env.Difficulty.hard,
        "harder": sc2_env.Difficulty.harder,
        "very_hard": sc2_env.Difficulty.very_hard,
        "cheat_vision": sc2_env.Difficulty.cheat_vision,
        "cheat_money": sc2_env.Difficulty.cheat_money,
        "cheat_insane": sc2_env.Difficulty.cheat_insane,
    }

    register_map(
        lib,
        map_name=registration["map_name"],
        map_dir=registration["map_dir"],
        map_file=registration["map_file"],
        players=registration["players"],
    )

    env = None
    try:
        env = sc2_env.SC2Env(
            map_name=registration["map_name"],
            players=[
                sc2_env.Agent(race_map[args.agent_race]),
                sc2_env.Bot(race_map[args.bot_race], difficulty_map[args.bot_difficulty]),
            ],
            step_mul=args.step_mul,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=args.raw_resolution,
                feature_dimensions=features.Dimensions(
                    screen=args.screen_res,
                    minimap=args.minimap_res,
                ),
                crop_to_playable_area=args.crop_to_playable_area,
                raw_crop_to_playable_area=args.raw_crop_to_playable_area,
            ),
            visualize=args.visualize,
            realtime=args.realtime,
        )

        controller = env._controllers[0]
        cached_env_info = env.game_info[0]
        live_info_before_reset = controller.game_info()

        ts_after_reset = env.reset()[0]
        live_info_after_reset = controller.game_info()
        obs_after_reset = controller.observe()

        ts_after_probe = ts_after_reset
        for _ in range(args.probe_steps):
            ts_after_probe = env.step([actions.RAW_FUNCTIONS.no_op()])[0]

        live_info_after_probe = controller.game_info()
        obs_after_probe = controller.observe()

        return {
            "map_registration": registration,
            "interface_request": {
                "action_space": "RAW",
                "use_raw_units": True,
                "screen_res": args.screen_res,
                "minimap_res": args.minimap_res,
                "raw_resolution": args.raw_resolution,
                "crop_to_playable_area": args.crop_to_playable_area,
                "raw_crop_to_playable_area": args.raw_crop_to_playable_area,
                "step_mul": args.step_mul,
                "probe_steps": args.probe_steps,
            },
            "cached_env_game_info_before_reset": describe_game_info(cached_env_info),
            "live_controller_game_info_before_reset": describe_game_info(live_info_before_reset),
            "live_controller_game_info_after_reset": describe_game_info(live_info_after_reset),
            "feature_minimap_after_reset": describe_feature_minimap(
                ts_after_reset, live_info_after_reset, features.MINIMAP_FEATURES
            ),
            "raw_observation_after_reset": describe_observation_proto(obs_after_reset),
            "live_controller_game_info_after_probe_steps": describe_game_info(live_info_after_probe),
            "feature_minimap_after_probe_steps": describe_feature_minimap(
                ts_after_probe, live_info_after_probe, features.MINIMAP_FEATURES
            ),
            "raw_observation_after_probe_steps": describe_observation_proto(obs_after_probe),
            "interpretation_hint": (
                "If playable_area is smaller than map_size here, PySC2 extracted a "
                "playable rectangle. For feature_minimap, the tensor shape is usually "
                "still the requested resolution; what changes is the world area that "
                "gets mapped into that tensor. If only the live controller shows the "
                "expected value, prefer controller.game_info() over the cached env.game_info."
            ),
        }
    finally:
        if env is not None:
            env.close()


def print_report(report: dict[str, object]) -> None:
    def print_block(title: str, payload: dict[str, object]) -> None:
        print(title)
        print(json.dumps(payload, indent=2))
        print()

    print_block("Map Registration", report["map_registration"])
    print_block("Interface Request", report["interface_request"])
    print_block("Cached env.game_info before reset", report["cached_env_game_info_before_reset"])
    print_block("Live controller.game_info before reset", report["live_controller_game_info_before_reset"])
    print_block("Live controller.game_info after reset", report["live_controller_game_info_after_reset"])
    print_block("Feature minimap after reset", report["feature_minimap_after_reset"])
    print_block("Raw observation after reset", report["raw_observation_after_reset"])
    print_block(
        "Live controller.game_info after probe steps",
        report["live_controller_game_info_after_probe_steps"],
    )
    print_block("Feature minimap after probe steps", report["feature_minimap_after_probe_steps"])
    print_block("Raw observation after probe steps", report["raw_observation_after_probe_steps"])
    print("Interpretation")
    print(report["interpretation_hint"])


def main() -> int:
    args = parse_args()

    try:
        report = build_report(args)
    except Exception as exc:
        print(f"Failed to inspect playable area: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

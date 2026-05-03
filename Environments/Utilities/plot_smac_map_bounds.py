from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Environments.Utilities.plot_v2_base_agent_minimap_observation import (
    DEFAULT_MAP_DIR as DEFAULT_REFERENCE_MAP_DIR,
    DEFAULT_MAP_FILE as DEFAULT_REFERENCE_MAP_FILE,
    DEFAULT_MAP_NAME as DEFAULT_REFERENCE_MAP_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESOLUTION,
    DEFAULT_STEP_MUL,
    build_rgb_composite,
    import_pysc2_modules,
    install_runtime_compat,
    make_jsonable,
    minimap_feature_indices,
    player_relative_labels,
    unique_counts,
)


DEFAULT_SMAC_DIR = Path(r"C:/Program Files (x86)/StarCraft II/Maps/SMAC_Maps")
DEFAULT_SMACV2_DIR = Path(r"C:/Program Files (x86)/StarCraft II/Maps/SMAC_V2_Maps")
DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_DIR / "smac_map_bounds"


@dataclass(frozen=True)
class MapSpec:
    group: str
    path: Path
    registry_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load every .SC2Map in the SMAC and SMAC v2 map folders through PySC2, "
            "record map_size/playable_area bounds, compare them with TwoBridgeMap_V2_Base, "
            "and write one bounds plot per map."
        )
    )
    parser.add_argument(
        "--smac-dir",
        type=Path,
        default=DEFAULT_SMAC_DIR,
        help=f"Directory containing SMAC .SC2Map files. Default: {DEFAULT_SMAC_DIR}",
    )
    parser.add_argument(
        "--smacv2-dir",
        type=Path,
        default=DEFAULT_SMACV2_DIR,
        help=f"Directory containing SMAC v2 .SC2Map files. Default: {DEFAULT_SMACV2_DIR}",
    )
    parser.add_argument(
        "--include",
        choices=("smac", "smacv2", "both"),
        default="both",
        help="Which map folder(s) to inspect. Default: both",
    )
    parser.add_argument(
        "--map-glob",
        default="*.SC2Map",
        help="Glob used inside each selected map directory. Default: *.SC2Map",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of maps to inspect after sorting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_SUBDIR,
        help=f"Directory for generated plots and summaries. Default: {DEFAULT_OUTPUT_SUBDIR}",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Plot format to write. Default: png",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Raster figure DPI. Default: 250")
    parser.add_argument(
        "--screen-res",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Feature screen resolution requested from PySC2. Default: 64",
    )
    parser.add_argument(
        "--minimap-res",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Feature minimap resolution requested from PySC2. Default: 64",
    )
    parser.add_argument(
        "--raw-resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Raw action/observation resolution requested from PySC2. Default: 64",
    )
    parser.add_argument(
        "--step-mul",
        type=int,
        default=DEFAULT_STEP_MUL,
        help=f"SC2 game loops per step. Default: {DEFAULT_STEP_MUL}",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=0,
        help="Optional no-op steps after reset before capturing minimap channels. Default: 0",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Player count used for direct PySC2 map registration. Default: 2",
    )
    parser.add_argument(
        "--reference-map-name",
        default=DEFAULT_REFERENCE_MAP_NAME,
        help=f"Reference PySC2 map registry name. Default: {DEFAULT_REFERENCE_MAP_NAME}",
    )
    parser.add_argument(
        "--reference-map-dir",
        default=DEFAULT_REFERENCE_MAP_DIR,
        help=f"Reference map directory. Default: {DEFAULT_REFERENCE_MAP_DIR}",
    )
    parser.add_argument(
        "--reference-map-file",
        default=DEFAULT_REFERENCE_MAP_FILE,
        help=f"Reference map filename. Default: {DEFAULT_REFERENCE_MAP_FILE}",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the SC2 visualizer while inspecting maps.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run SC2 in realtime while inspecting maps.",
    )
    parser.add_argument(
        "--origin",
        choices=("upper", "lower"),
        default="upper",
        help="imshow origin for minimap plots. Default: upper",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first map load failure instead of recording the error and continuing.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def collect_map_specs(args: argparse.Namespace) -> list[MapSpec]:
    groups: list[tuple[str, Path]] = []
    if args.include in ("smac", "both"):
        groups.append(("smac", args.smac_dir))
    if args.include in ("smacv2", "both"):
        groups.append(("smacv2", args.smacv2_dir))

    specs = []
    for group, directory in groups:
        directory = directory.expanduser()
        for path in sorted(directory.glob(args.map_glob), key=lambda item: item.name.lower()):
            if path.is_file():
                specs.append(
                    MapSpec(
                        group=group,
                        path=path,
                        registry_name=f"{group}_{path.stem}",
                    )
                )

    if args.limit is not None:
        specs = specs[: max(0, int(args.limit))]
    return specs


def register_map(lib_module, map_name: str, map_dir: str, map_file: str, players: int) -> None:
    map_cls = type(
        map_name,
        (lib_module.Map,),
        {
            "name": map_name,
            "directory": map_dir,
            "filename": map_file,
            "players": int(players),
        },
    )
    lib_module.get_maps().pop(map_name, None)
    lib_module.get_maps()[map_name] = map_cls()


def point_dict(point_obj) -> dict[str, float]:
    return {"x": float(point_obj.x), "y": float(point_obj.y)}


def rect_from_proto(rect_obj) -> dict[str, float]:
    p0 = rect_obj.p0
    p1 = rect_obj.p1
    return {
        "x0": float(p0.x),
        "y0": float(p0.y),
        "x1": float(p1.x),
        "y1": float(p1.y),
        "width": float(p1.x - p0.x),
        "height": float(p1.y - p0.y),
    }


def describe_game_info(info) -> dict[str, object]:
    if not info.HasField("start_raw"):
        return {"has_start_raw": False}

    start_raw = info.start_raw
    map_size = point_dict(start_raw.map_size)
    playable_area = rect_from_proto(start_raw.playable_area)
    return {
        "has_start_raw": True,
        "map_size": map_size,
        "playable_area": playable_area,
        "playable_matches_full_map": bool(
            playable_area["x0"] == 0
            and playable_area["y0"] == 0
            and playable_area["x1"] == map_size["x"]
            and playable_area["y1"] == map_size["y"]
        ),
    }


def compare_bounds(bounds: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    if not bounds.get("has_start_raw") or not reference.get("has_start_raw"):
        return {"matches_reference": False, "reason": "missing start_raw"}

    map_size = bounds["map_size"]
    ref_map_size = reference["map_size"]
    playable = bounds["playable_area"]
    ref_playable = reference["playable_area"]
    map_size_delta = {
        "x": float(map_size["x"] - ref_map_size["x"]),
        "y": float(map_size["y"] - ref_map_size["y"]),
    }
    playable_delta = {
        key: float(playable[key] - ref_playable[key])
        for key in ("x0", "y0", "x1", "y1", "width", "height")
    }
    return {
        "matches_reference": bool(
            map_size_delta["x"] == 0
            and map_size_delta["y"] == 0
            and all(value == 0 for value in playable_delta.values())
        ),
        "map_size_delta": map_size_delta,
        "playable_area_delta": playable_delta,
    }


def create_env(
    map_name: str,
    sc2_env,
    actions,
    features_module,
    args: argparse.Namespace,
):
    return sc2_env.SC2Env(
        map_name=map_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy),
        ],
        step_mul=args.step_mul,
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=args.raw_resolution,
            raw_crop_to_playable_area=True,
            feature_dimensions=features_module.Dimensions(
                screen=args.screen_res,
                minimap=args.minimap_res,
            ),
        ),
        visualize=args.visualize,
        realtime=args.realtime,
    )


def inspect_registered_map(
    map_name: str,
    sc2_env,
    actions,
    features_module,
    args: argparse.Namespace,
) -> dict[str, object]:
    env = None
    try:
        env = create_env(map_name, sc2_env, actions, features_module, args)
        controller = env._controllers[0]
        cached_before_reset = describe_game_info(env.game_info[0])
        live_before_reset = describe_game_info(controller.game_info())

        ts = env.reset()[0]
        for _ in range(args.probe_steps):
            ts = env.step([actions.RAW_FUNCTIONS.no_op()])[0]
            if ts.last():
                break

        live_after_reset = describe_game_info(controller.game_info())
        pathable_idx, player_relative_idx = minimap_feature_indices(features_module)
        feature_minimap = np.asarray(ts.observation.feature_minimap, dtype=np.uint8)
        minimap = feature_minimap[[pathable_idx, player_relative_idx]]

        return {
            "cached_before_reset": cached_before_reset,
            "live_before_reset": live_before_reset,
            "live_after_reset": live_after_reset,
            "selected_bounds": live_after_reset,
            "minimap": minimap,
            "minimap_unique_counts": {
                "pathable": unique_counts(minimap[0]),
                "player_relative": unique_counts(minimap[1]),
            },
        }
    finally:
        if env is not None:
            env.close()


def inspect_map_spec(
    spec: MapSpec,
    reference_bounds: dict[str, object],
    sc2_env,
    actions,
    features_module,
    lib_module,
    args: argparse.Namespace,
) -> dict[str, object]:
    register_map(lib_module, spec.registry_name, str(spec.path.parent), spec.path.name, args.players)
    result = inspect_registered_map(spec.registry_name, sc2_env, actions, features_module, args)
    bounds = result["selected_bounds"]
    return {
        "group": spec.group,
        "map_name": spec.path.stem,
        "map_file": spec.path.name,
        "map_path": str(spec.path),
        "registry_name": spec.registry_name,
        "bounds": bounds,
        "reference_comparison": compare_bounds(bounds, reference_bounds),
        "minimap_unique_counts": result["minimap_unique_counts"],
        "minimap": result["minimap"],
    }


def rect_to_minimap(rect: dict[str, float], map_size: dict[str, float], shape: tuple[int, int]):
    height, width = shape
    map_w = max(float(map_size["x"]), 1.0)
    map_h = max(float(map_size["y"]), 1.0)
    return {
        "x0": rect["x0"] * width / map_w,
        "y0": rect["y0"] * height / map_h,
        "x1": rect["x1"] * width / map_w,
        "y1": rect["y1"] * height / map_h,
    }


def add_rect(axis, rect: dict[str, float], color: str, label: str, linestyle="-", linewidth=1.8):
    from matplotlib.patches import Rectangle

    axis.add_patch(
        Rectangle(
            (rect["x0"] - 0.5, rect["y0"] - 0.5),
            rect["x1"] - rect["x0"],
            rect["y1"] - rect["y0"],
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    )


def plot_map_result(
    result: dict[str, object],
    reference: dict[str, object],
    features_module,
    args: argparse.Namespace,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle

    minimap = result["minimap"]
    pathable = minimap[0]
    player_relative = minimap[1]
    bounds = result["bounds"]
    comparison = result["reference_comparison"]
    labels = player_relative_labels(features_module)

    path_cmap = ListedColormap(["#1b1f23", "#f0f0ea"])
    path_norm = BoundaryNorm([-0.5, 0.5, 1.5], path_cmap.N)
    pr_cmap = ListedColormap(["#161a1d", "#0d5be1", "#2ca25f", "#e0b91a", "#d33f32"])
    pr_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], pr_cmap.N)
    composite = build_rgb_composite(pathable, player_relative)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)
    fig.suptitle(f"{result['group']} / {result['map_name']} map bounds", fontsize=13)

    axes[0].imshow(pathable, cmap=path_cmap, norm=path_norm, interpolation="nearest", origin=args.origin)
    axes[0].set_title("Pathable + playable bounds")

    axes[1].imshow(player_relative, cmap=pr_cmap, norm=pr_norm, interpolation="nearest", origin=args.origin)
    axes[1].set_title("player_relative")

    axes[2].set_title("World bounds vs V2_Base")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("world x")
    axes[2].set_ylabel("world y")

    if bounds.get("has_start_raw"):
        mm_rect = rect_to_minimap(bounds["playable_area"], bounds["map_size"], pathable.shape)
        add_rect(axes[0], mm_rect, "#00c853", "playable_area", linewidth=2.2)

        map_size = bounds["map_size"]
        playable = bounds["playable_area"]
        axes[2].add_patch(
            Rectangle(
                (0, 0),
                map_size["x"],
                map_size["y"],
                fill=False,
                edgecolor="#1f77b4",
                linewidth=2.0,
                label="map full",
            )
        )
        axes[2].add_patch(
            Rectangle(
                (playable["x0"], playable["y0"]),
                playable["width"],
                playable["height"],
                fill=False,
                edgecolor="#00c853",
                linewidth=2.0,
                label="map playable",
            )
        )

    if reference.get("has_start_raw"):
        ref_size = reference["map_size"]
        ref_playable = reference["playable_area"]
        axes[2].add_patch(
            Rectangle(
                (0, 0),
                ref_size["x"],
                ref_size["y"],
                fill=False,
                edgecolor="#d62728",
                linestyle="--",
                linewidth=1.6,
                label="V2_Base full",
            )
        )
        axes[2].add_patch(
            Rectangle(
                (ref_playable["x0"], ref_playable["y0"]),
                ref_playable["width"],
                ref_playable["height"],
                fill=False,
                edgecolor="#ffb000",
                linestyle="--",
                linewidth=1.6,
                label="V2_Base playable",
            )
        )

    max_x = 1.0
    max_y = 1.0
    for item in (bounds, reference):
        if item.get("has_start_raw"):
            max_x = max(max_x, float(item["map_size"]["x"]))
            max_y = max(max_y, float(item["map_size"]["y"]))
    axes[2].set_xlim(-2, max_x + 2)
    axes[2].set_ylim(-2, max_y + 2)
    axes[2].grid(color="#d0d0d0", linewidth=0.4, alpha=0.7)
    axes[2].legend(loc="upper right", fontsize=7)

    pr_values = sorted(int(value) for value in np.unique(player_relative))
    handles = [
        Patch(facecolor="#f0f0ea", edgecolor="black", label="pathable"),
        Patch(facecolor="#1b1f23", edgecolor="black", label="blocked"),
    ]
    for value in pr_values:
        if value == 0:
            continue
        handles.append(
            Patch(
                facecolor=pr_cmap(pr_norm(value)),
                edgecolor="black",
                label=f"{value} {labels.get(value, 'unknown')}",
            )
        )
    axes[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=7)

    axes[0].legend(loc="upper right", fontsize=7)
    for axis in axes[:2]:
        axis.set_xlabel("minimap x pixel")
        axis.set_ylabel("minimap y pixel")
        axis.set_xticks(np.arange(0, minimap.shape[2], 8))
        axis.set_yticks(np.arange(0, minimap.shape[1], 8))
        axis.tick_params(labelsize=7)
        axis.grid(color="white", alpha=0.08, linewidth=0.3)

    if bounds.get("has_start_raw"):
        map_size_text = f"map_size=({bounds['map_size']['x']:.0f}, {bounds['map_size']['y']:.0f})"
        playable_text = (
            "playable="
            f"({bounds['playable_area']['x0']:.0f}, {bounds['playable_area']['y0']:.0f})"
            f"-({bounds['playable_area']['x1']:.0f}, {bounds['playable_area']['y1']:.0f})"
        )
    else:
        map_size_text = "map_size=unavailable"
        playable_text = "playable=unavailable"
    match_text = f"matches V2_Base={comparison.get('matches_reference', False)}"
    fig.text(0.5, -0.02, f"{map_size_text}; {playable_text}; {match_text}", ha="center", fontsize=9)

    return fig


def write_map_plot(
    result: dict[str, object],
    reference: dict[str, object],
    features_module,
    args: argparse.Namespace,
) -> list[Path]:
    group_dir = args.output_dir / result["group"]
    group_dir.mkdir(parents=True, exist_ok=True)
    prefix = sanitize_filename(str(result["map_name"]))
    fig = plot_map_result(result, reference, features_module, args)
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    written = []
    for file_format in formats:
        path = group_dir / f"{prefix}_bounds.{file_format}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        written.append(path)

    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def flatten_row(result: dict[str, object]) -> dict[str, object]:
    bounds = result.get("bounds", {})
    comparison = result.get("reference_comparison", {})
    map_size = bounds.get("map_size", {})
    playable = bounds.get("playable_area", {})
    map_delta = comparison.get("map_size_delta", {})
    playable_delta = comparison.get("playable_area_delta", {})
    return {
        "group": result.get("group"),
        "map_name": result.get("map_name"),
        "map_file": result.get("map_file"),
        "status": result.get("status", "ok"),
        "error": result.get("error", ""),
        "map_size_x": map_size.get("x"),
        "map_size_y": map_size.get("y"),
        "playable_x0": playable.get("x0"),
        "playable_y0": playable.get("y0"),
        "playable_x1": playable.get("x1"),
        "playable_y1": playable.get("y1"),
        "playable_width": playable.get("width"),
        "playable_height": playable.get("height"),
        "matches_reference": comparison.get("matches_reference"),
        "map_size_delta_x": map_delta.get("x"),
        "map_size_delta_y": map_delta.get("y"),
        "playable_delta_x0": playable_delta.get("x0"),
        "playable_delta_y0": playable_delta.get("y0"),
        "playable_delta_x1": playable_delta.get("x1"),
        "playable_delta_y1": playable_delta.get("y1"),
        "map_path": result.get("map_path"),
    }


def write_summary(
    results: list[dict[str, object]],
    errors: list[dict[str, object]],
    reference_bounds: dict[str, object],
    written_plots: list[Path],
    args: argparse.Namespace,
) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_results = []
    for result in results:
        clean = {key: value for key, value in result.items() if key != "minimap"}
        serializable_results.append(clean)

    payload = {
        "reference": {
            "map_name": args.reference_map_name,
            "map_dir": args.reference_map_dir,
            "map_file": args.reference_map_file,
            "bounds": reference_bounds,
        },
        "map_count": len(results),
        "error_count": len(errors),
        "results": serializable_results,
        "errors": errors,
        "plots": [str(path) for path in written_plots],
    }
    json_path = args.output_dir / "smac_map_bounds_summary.json"
    json_path.write_text(json.dumps(make_jsonable(payload), indent=2) + "\n", encoding="utf-8")

    rows = [flatten_row(result) for result in results]
    rows.extend(flatten_row({**error, "status": "error"}) for error in errors)
    csv_path = args.output_dir / "smac_map_bounds_summary.csv"
    fieldnames = list(flatten_row({}).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return [json_path, csv_path]


def main() -> int:
    args = parse_args()
    install_runtime_compat()
    sc2_env, actions, features_module, lib_module = import_pysc2_modules()

    specs = collect_map_specs(args)
    if not specs:
        print("No .SC2Map files found for the selected folders.", file=sys.stderr)
        return 1

    reference_path = Path(args.reference_map_dir, args.reference_map_file)
    register_map(
        lib_module,
        args.reference_map_name,
        str(reference_path.parent),
        reference_path.name,
        args.players,
    )
    reference_result = inspect_registered_map(
        args.reference_map_name,
        sc2_env,
        actions,
        features_module,
        args,
    )
    reference_bounds = reference_result["selected_bounds"]

    results = []
    errors = []
    written_plots = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] Inspecting {spec.group}/{spec.path.name} ...")
        try:
            result = inspect_map_spec(
                spec,
                reference_bounds,
                sc2_env,
                actions,
                features_module,
                lib_module,
                args,
            )
            results.append(result)
            written_plots.extend(write_map_plot(result, reference_bounds, features_module, args))
            comparison = result["reference_comparison"]
            bounds = result["bounds"]
            print(
                "  "
                f"map_size=({bounds['map_size']['x']:.0f}, {bounds['map_size']['y']:.0f}), "
                f"playable=({bounds['playable_area']['width']:.0f}, {bounds['playable_area']['height']:.0f}), "
                f"matches_reference={comparison['matches_reference']}"
            )
        except Exception as exc:
            error = {
                "group": spec.group,
                "map_name": spec.path.stem,
                "map_file": spec.path.name,
                "map_path": str(spec.path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"  failed: {error['error']}", file=sys.stderr)
            if args.stop_on_error:
                raise

    summary_paths = write_summary(results, errors, reference_bounds, written_plots, args)

    matching = sum(
        1 for result in results if result["reference_comparison"].get("matches_reference")
    )
    print()
    print(f"Inspected {len(results)} maps; {len(errors)} failures.")
    print(f"Maps matching {args.reference_map_name}: {matching}/{len(results)}")
    print("Wrote:")
    for path in summary_paths + written_plots:
        print(f"  {path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

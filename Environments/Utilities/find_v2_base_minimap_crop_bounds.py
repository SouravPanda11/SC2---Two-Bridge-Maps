from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Environments.Utilities.plot_v2_base_agent_minimap_observation import (
    DEFAULT_MAP_DIR,
    DEFAULT_MAP_FILE,
    DEFAULT_MAP_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESOLUTION,
    DEFAULT_STEP_MUL,
    build_rgb_composite,
    import_pysc2_modules,
    install_runtime_compat,
    make_jsonable,
    minimap_feature_indices,
    player_relative_labels,
    register_direct_map,
    resolve_direct_map,
    unique_counts,
)


N_FRIEND = 5


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample V2_Base minimap observations and evaluate square crop "
            "bounds for cutting unused 64x64 pixels from both pathable and "
            "player_relative channels."
        )
    )
    parser.add_argument(
        "--source",
        choices=("ns", "direct", "tensor"),
        default="ns",
        help=(
            "Where to get minimap observations. 'tensor' loads --tensor-path. "
            "Default: ns"
        ),
    )
    parser.add_argument(
        "--tensor-path",
        type=Path,
        help="Optional .npy tensor with shape (2,H,W) or (N,2,H,W). Used by --source tensor.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of reset observations to sample. Default: 20",
    )
    parser.add_argument(
        "--steps-after-reset",
        type=int,
        default=0,
        help="Number of no-op steps after each reset before capture. Default: 0",
    )
    parser.add_argument(
        "--bounds-source",
        choices=("pathable", "player_relative", "pathable_or_player_relative"),
        default="pathable_or_player_relative",
        help=(
            "Mask used to compute the base bounding box. Coverage for both "
            "channels is still reported. Default: pathable_or_player_relative"
        ),
    )
    parser.add_argument(
        "--candidate-sides",
        default="24,28,32,36,40,44,48,52,56,60,64",
        help="Comma-separated square side lengths to test. Default: 24..64 by common values.",
    )
    parser.add_argument(
        "--candidate-margins",
        default="0,2,4,6,8,10,12",
        help=(
            "Comma-separated margins around the measured bounding box. Each "
            "margin creates one square side candidate. Default: 0,2,4,6,8,10,12"
        ),
    )
    parser.add_argument(
        "--center-x",
        type=float,
        help="Optional crop center x pixel. Default: measured bounding-box center.",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        help="Optional crop center y pixel. Default: measured bounding-box center.",
    )
    parser.add_argument(
        "--require-player-relative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require candidate recommendations to include all nonzero player_relative pixels. Default: true",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default="v2_base_minimap_crop_bounds",
        help="Output filename prefix. Default: v2_base_minimap_crop_bounds",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Plot format to write. Default: png",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster figure DPI. Default: 300")
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
    parser.add_argument("--map-name", default=DEFAULT_MAP_NAME)
    parser.add_argument("--map-file", default=DEFAULT_MAP_FILE)
    parser.add_argument("--map-dir", default=DEFAULT_MAP_DIR)
    parser.add_argument("--map-path", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--origin",
        choices=("upper", "lower"),
        default="upper",
        help="imshow origin. Default: upper, which matches tensor row/column indexing.",
    )
    return parser.parse_args()


def noop_steps_ns(env, steps: int):
    obs = None
    info = {}
    action = np.zeros(N_FRIEND, dtype=np.int64)
    for _ in range(steps):
        obs, _, done, _, info = env.step(action)
        if done:
            break
    return obs, info


def load_tensor_samples(path: Path) -> np.ndarray:
    tensor = np.load(path)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim != 4 or tensor.shape[1] != 2:
        raise ValueError(
            f"Expected tensor shape (2,H,W) or (N,2,H,W), received {tensor.shape}."
        )
    return np.asarray(tensor, dtype=np.uint8)


def sample_ns(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    from Environments.NS_AM_RM_mean.V2_Base_NS import TwoBridgeEnv

    env = TwoBridgeEnv(
        screen_res=args.screen_res,
        visualize=args.visualize,
        realtime=args.realtime,
    )
    samples = []
    last_info = {}
    try:
        for _ in range(args.samples):
            obs, _ = env.reset()
            step_obs, last_info = noop_steps_ns(env, args.steps_after_reset)
            if step_obs is not None:
                obs = step_obs
            samples.append(np.asarray(obs["minimap"], dtype=np.uint8))
    finally:
        env.close()

    metadata = {
        "source": "ns",
        "environment": "Environments.NS_AM_RM_mean.V2_Base_NS.TwoBridgeEnv",
        "map_name": DEFAULT_MAP_NAME,
        "samples": int(args.samples),
        "steps_after_reset": int(args.steps_after_reset),
        "last_info": make_jsonable(last_info),
    }
    return np.stack(samples, axis=0), metadata


def sample_direct(args: argparse.Namespace, sc2_env, actions, features_module, lib_module):
    map_name, map_dir, map_file = resolve_direct_map(args)
    register_direct_map(lib_module, map_name, map_dir, map_file)

    env = sc2_env.SC2Env(
        map_name=map_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy),
        ],
        step_mul=DEFAULT_STEP_MUL,
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
    samples = []
    try:
        pathable_idx, player_relative_idx = minimap_feature_indices(features_module)
        for _ in range(args.samples):
            ts = env.reset()[0]
            for _ in range(args.steps_after_reset):
                ts = env.step([actions.RAW_FUNCTIONS.no_op()])[0]
                if ts.last():
                    break
            feature_minimap = np.asarray(ts.observation.feature_minimap, dtype=np.uint8)
            samples.append(feature_minimap[[pathable_idx, player_relative_idx]])
    finally:
        env.close()

    metadata = {
        "source": "direct",
        "environment": "pysc2.env.sc2_env.SC2Env",
        "map_name": map_name,
        "map_dir": map_dir,
        "map_file": map_file,
        "samples": int(args.samples),
        "steps_after_reset": int(args.steps_after_reset),
    }
    return np.stack(samples, axis=0), metadata


def collect_samples(args: argparse.Namespace):
    install_runtime_compat()
    sc2_env, actions, features_module, lib_module = import_pysc2_modules()

    if args.source == "tensor":
        if args.tensor_path is None:
            raise ValueError("--source tensor requires --tensor-path.")
        samples = load_tensor_samples(args.tensor_path)
        metadata = {
            "source": "tensor",
            "tensor_path": str(args.tensor_path),
            "samples": int(samples.shape[0]),
        }
    elif args.source == "ns":
        samples, metadata = sample_ns(args)
    else:
        samples, metadata = sample_direct(args, sc2_env, actions, features_module, lib_module)

    pathable_idx, player_relative_idx = minimap_feature_indices(features_module)
    metadata.update(
        {
            "tensor_shape_per_sample": tuple(int(v) for v in samples.shape[1:]),
            "channel_0": {"name": "pathable", "pysc2_minimap_feature_index": pathable_idx},
            "channel_1": {
                "name": "player_relative",
                "pysc2_minimap_feature_index": player_relative_idx,
            },
        }
    )
    return samples, metadata, features_module


def bounding_box(mask: np.ndarray) -> dict[str, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Cannot compute bounds from an empty mask.")
    return {
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()) + 1,
        "y1": int(ys.max()) + 1,
    }


def crop_mask(shape: tuple[int, int], crop: dict[str, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[crop["y0"] : crop["y1"], crop["x0"] : crop["x1"]] = True
    return mask


def clamp_square(center_x: float, center_y: float, side: int, width: int, height: int):
    side = int(side)
    if side < 1 or side > min(width, height):
        return None
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    return {"x0": x0, "y0": y0, "x1": x0 + side, "y1": y0 + side, "side": side}


def selected_bounds_mask(args, pathable_union: np.ndarray, pr_union: np.ndarray) -> np.ndarray:
    if args.bounds_source == "pathable":
        return pathable_union
    if args.bounds_source == "player_relative":
        return pr_union
    return pathable_union | pr_union


def evaluate_candidates(
    args: argparse.Namespace,
    pathable_union: np.ndarray,
    pr_union: np.ndarray,
):
    height, width = pathable_union.shape
    bounds_mask = selected_bounds_mask(args, pathable_union, pr_union)
    base_bbox = bounding_box(bounds_mask)

    bbox_width = base_bbox["x1"] - base_bbox["x0"]
    bbox_height = base_bbox["y1"] - base_bbox["y0"]
    center_x = (
        float(args.center_x)
        if args.center_x is not None
        else (base_bbox["x0"] + base_bbox["x1"] - 1) / 2.0
    )
    center_y = (
        float(args.center_y)
        if args.center_y is not None
        else (base_bbox["y0"] + base_bbox["y1"] - 1) / 2.0
    )

    sides = set(parse_int_list(args.candidate_sides))
    for margin in parse_int_list(args.candidate_margins):
        sides.add(max(bbox_width, bbox_height) + 2 * int(margin))
    sides = sorted(side for side in sides if 1 <= side <= min(width, height))

    path_total = int(pathable_union.sum())
    pr_total = int(pr_union.sum())
    bounds_total = int(bounds_mask.sum())
    candidates = []
    seen = set()
    for side in sides:
        crop = clamp_square(center_x, center_y, side, width, height)
        if crop is None:
            continue
        key = (crop["x0"], crop["y0"], crop["x1"], crop["y1"])
        if key in seen:
            continue
        seen.add(key)

        inside = crop_mask((height, width), crop)
        path_inside = int((pathable_union & inside).sum())
        pr_inside = int((pr_union & inside).sum())
        bounds_inside = int((bounds_mask & inside).sum())
        candidate = {
            **crop,
            "input_shape_after_crop": [2, int(side), int(side)],
            "area_pixels": int(side * side),
            "area_fraction_of_original": float((side * side) / (height * width)),
            "compute_reduction_fraction": float(1.0 - (side * side) / (height * width)),
            "pathable_inside": path_inside,
            "pathable_total": path_total,
            "pathable_coverage": float(path_inside / path_total) if path_total else 1.0,
            "player_relative_inside": pr_inside,
            "player_relative_total": pr_total,
            "player_relative_coverage": float(pr_inside / pr_total) if pr_total else 1.0,
            "bounds_inside": bounds_inside,
            "bounds_total": bounds_total,
            "bounds_coverage": float(bounds_inside / bounds_total) if bounds_total else 1.0,
            "slice": f"minimap[:, {crop['y0']}:{crop['y1']}, {crop['x0']}:{crop['x1']}]",
        }
        candidates.append(candidate)

    recommended = None
    for candidate in candidates:
        if candidate["pathable_coverage"] < 1.0 or candidate["bounds_coverage"] < 1.0:
            continue
        if args.require_player_relative and candidate["player_relative_coverage"] < 1.0:
            continue
        recommended = candidate
        break

    if recommended is None and candidates:
        recommended = max(
            candidates,
            key=lambda item: (
                item["bounds_coverage"],
                item["player_relative_coverage"],
                -item["area_pixels"],
            ),
        )

    return {
        "base_bbox": base_bbox,
        "base_bbox_width": bbox_width,
        "base_bbox_height": bbox_height,
        "center": {"x": center_x, "y": center_y},
        "candidates": candidates,
        "recommended": recommended,
    }


def write_csv(path: Path, candidates: list[dict[str, object]]) -> None:
    if not candidates:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "side",
        "x0",
        "x1",
        "y0",
        "y1",
        "area_pixels",
        "area_fraction_of_original",
        "compute_reduction_fraction",
        "pathable_coverage",
        "player_relative_coverage",
        "bounds_coverage",
        "slice",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({key: candidate[key] for key in fieldnames})


def plot_bounds(
    samples: np.ndarray,
    pathable_union: np.ndarray,
    pr_union: np.ndarray,
    eval_result: dict[str, object],
    features_module,
    args: argparse.Namespace,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle

    labels = player_relative_labels(features_module)
    pr_display = np.max(samples[:, 1], axis=0)
    composite = build_rgb_composite(pathable_union.astype(np.uint8), pr_display.astype(np.uint8))
    path_cmap = ListedColormap(["#1b1f23", "#f0f0ea"])
    path_norm = BoundaryNorm([-0.5, 0.5, 1.5], path_cmap.N)
    pr_cmap = ListedColormap(["#161a1d", "#0d5be1", "#2ca25f", "#e0b91a", "#d33f32"])
    pr_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], pr_cmap.N)

    candidates = eval_result["candidates"]
    recommended = eval_result["recommended"]
    bbox = eval_result["base_bbox"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.20, wspace=0.22)
    fig.suptitle("V2_Base minimap crop candidate bounds", fontsize=13)

    axes[0].imshow(pathable_union, cmap=path_cmap, norm=path_norm, interpolation="nearest", origin=args.origin)
    axes[0].set_title("Pathable union")

    axes[1].imshow(pr_display, cmap=pr_cmap, norm=pr_norm, interpolation="nearest", origin=args.origin)
    axes[1].set_title("player_relative union")

    axes[2].imshow(composite, interpolation="nearest", origin=args.origin)
    axes[2].set_title("Composite + recommended crop")

    for axis in axes:
        add_rect(axis, bbox, "#ffffff", "measured bbox", linewidth=1.3, linestyle="--")
        for candidate in candidates:
            color = "#808080"
            linewidth = 0.7
            alpha = 0.35
            if recommended is not None and same_crop(candidate, recommended):
                color = "#00c853"
                linewidth = 2.4
                alpha = 1.0
            add_rect(axis, candidate, color, None, linewidth=linewidth, alpha=alpha)
        axis.set_xlabel("minimap x pixel")
        axis.set_ylabel("minimap y pixel")
        axis.set_xticks(np.arange(0, samples.shape[3], 8))
        axis.set_yticks(np.arange(0, samples.shape[2], 8))
        axis.tick_params(labelsize=7)
        axis.grid(color="white", alpha=0.08, linewidth=0.3)

    pr_values = sorted(int(v) for v in np.unique(pr_display.astype(np.uint8)))
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
    handles.append(Patch(facecolor="none", edgecolor="#00c853", linewidth=2, label="recommended"))
    axes[2].legend(handles=handles, loc="upper right", frameon=True, fontsize=7)

    if recommended:
        subtitle = (
            f"recommended: {recommended['slice']} -> {recommended['input_shape_after_crop']} "
            f"({recommended['compute_reduction_fraction']:.1%} fewer pixels)"
        )
    else:
        subtitle = "no candidate fully covered the selected bounds"
    fig.text(0.5, 0.06, subtitle, ha="center", va="center", fontsize=9)

    return fig


def add_rect(axis, crop: dict[str, int], color: str, label: str | None, linewidth=1.0, linestyle="-", alpha=1.0):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (crop["x0"] - 0.5, crop["y0"] - 0.5),
        crop["x1"] - crop["x0"],
        crop["y1"] - crop["y0"],
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )
    axis.add_patch(rect)


def same_crop(left: dict[str, object], right: dict[str, object]) -> bool:
    keys = ("x0", "y0", "x1", "y1")
    return all(int(left[key]) == int(right[key]) for key in keys)


def write_outputs(samples, metadata, eval_result, features_module, args) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pathable_union = np.any(samples[:, 0] > 0, axis=0)
    pr_union = np.any(samples[:, 1] > 0, axis=0)
    pr_union_values = np.max(samples[:, 1], axis=0)

    summary = {
        **metadata,
        "bounds_source": args.bounds_source,
        "require_player_relative": bool(args.require_player_relative),
        "sample_count": int(samples.shape[0]),
        "pathable_union_unique_counts": unique_counts(pathable_union.astype(np.uint8)),
        "player_relative_union_unique_counts": unique_counts(pr_union_values.astype(np.uint8)),
        "latest_pathable_unique_counts": unique_counts(samples[-1, 0]),
        "latest_player_relative_unique_counts": unique_counts(samples[-1, 1]),
        "player_relative_labels": {
            str(key): value for key, value in player_relative_labels(features_module).items()
        },
        **eval_result,
    }

    written = []
    json_path = args.output_dir / f"{args.prefix}_summary.json"
    json_path.write_text(json.dumps(make_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    written.append(json_path)

    csv_path = args.output_dir / f"{args.prefix}_candidates.csv"
    write_csv(csv_path, eval_result["candidates"])
    written.append(csv_path)

    npy_path = args.output_dir / f"{args.prefix}_sampled_tensors.npy"
    np.save(npy_path, samples)
    written.append(npy_path)

    fig = plot_bounds(samples, pathable_union, pr_union, eval_result, features_module, args)
    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    for file_format in formats:
        figure_path = args.output_dir / f"{args.prefix}.{file_format}"
        fig.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
        written.append(figure_path)

    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def main() -> int:
    args = parse_args()
    samples, metadata, features_module = collect_samples(args)
    pathable_union = np.any(samples[:, 0] > 0, axis=0)
    pr_union = np.any(samples[:, 1] > 0, axis=0)
    eval_result = evaluate_candidates(args, pathable_union, pr_union)
    written = write_outputs(samples, metadata, eval_result, features_module, args)

    recommended = eval_result["recommended"]
    print("Evaluated V2_Base minimap crop candidates.")
    print(f"Samples: {samples.shape[0]}")
    print(f"Per-sample tensor shape: {tuple(int(v) for v in samples.shape[1:])}")
    print(f"Measured bbox: {eval_result['base_bbox']}")
    if recommended:
        print(f"Recommended crop: {recommended['slice']}")
        print(f"Output shape: {recommended['input_shape_after_crop']}")
        print(f"Pixel reduction: {recommended['compute_reduction_fraction']:.1%}")
        print(f"Pathable coverage: {recommended['pathable_coverage']:.1%}")
        print(f"Player-relative coverage: {recommended['player_relative_coverage']:.1%}")
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

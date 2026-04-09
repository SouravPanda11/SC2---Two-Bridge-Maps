from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


def find_pysc2_root() -> Path:
    spec = importlib.util.find_spec("pysc2")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Could not locate an installed 'pysc2' package.")
    return Path(next(iter(spec.submodule_search_locations)))


def load_module_source(path: Path) -> tuple[str, ast.AST]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def find_class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"Class '{name}' not found.")


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Function '{name}' not found.")


def find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Method '{name}' not found in class '{class_node.name}'.")


def find_class_with_method(tree: ast.AST, method_name: str) -> tuple[ast.ClassDef, ast.FunctionDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return node, item
    raise ValueError(f"No class with method '{method_name}' found.")


def find_assignment_call(tree: ast.AST, target_name: str) -> ast.Call:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    if isinstance(node.value, ast.Call):
                        return node.value
                    raise ValueError(f"Assignment for '{target_name}' is not a call.")
    raise ValueError(f"Assignment for '{target_name}' not found.")


def extract_namedtuple_fields(class_node: ast.ClassDef) -> list[str]:
    for base in class_node.bases:
        if not isinstance(base, ast.Call) or len(base.args) < 2:
            continue
        fields_arg = base.args[1]
        if isinstance(fields_arg, (ast.List, ast.Tuple)):
            fields = []
            for elt in fields_arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    fields.append(elt.value)
            if fields:
                return fields
    raise ValueError(f"Could not extract namedtuple fields for '{class_node.name}'.")


def extract_enum_members(class_node: ast.ClassDef) -> list[dict[str, str]]:
    members = []
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            members.append({
                "name": node.targets[0].id,
                "value": ast.unparse(node.value),
            })
    return members


def extract_call_keyword_rows(call_node: ast.Call, tuple_labels: list[str]) -> list[dict[str, str]]:
    rows = []
    for idx, kw in enumerate(call_node.keywords):
        if kw.arg is None:
            continue
        row = OrderedDict()
        row["index"] = idx
        row["name"] = kw.arg
        if isinstance(kw.value, ast.Tuple):
            for label, value_node in zip(tuple_labels, kw.value.elts):
                row[label] = ast.unparse(value_node)
        else:
            row["value"] = ast.unparse(kw.value)
        rows.append(row)
    return rows


def extract_parameters(fn_node: ast.FunctionDef, skip: set[str] | None = None) -> list[dict[str, str]]:
    skip = skip or set()
    params: list[dict[str, str]] = []

    positional = list(fn_node.args.args)
    positional_defaults = [None] * (len(positional) - len(fn_node.args.defaults)) + list(fn_node.args.defaults)
    for arg_node, default_node in zip(positional, positional_defaults):
        if arg_node.arg in skip:
            continue
        params.append({
            "name": arg_node.arg,
            "kind": "positional_or_keyword",
            "default": "<required>" if default_node is None else ast.unparse(default_node),
        })

    if fn_node.args.vararg is not None:
        params.append({
            "name": "*" + fn_node.args.vararg.arg,
            "kind": "vararg",
            "default": "",
        })

    for arg_node, default_node in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
        if arg_node.arg in skip:
            continue
        params.append({
            "name": arg_node.arg,
            "kind": "keyword_only",
            "default": "<required>" if default_node is None else ast.unparse(default_node),
        })

    if fn_node.args.kwarg is not None:
        params.append({
            "name": "**" + fn_node.args.kwarg.arg,
            "kind": "varkw",
            "default": "",
        })

    return params


def summarize_docstring(node: ast.AST, max_lines: int = 3) -> list[str]:
    doc = ast.get_docstring(node)
    if not doc:
        return []
    lines = [line.strip() for line in doc.splitlines() if line.strip()]
    return lines[:max_lines]


def extract_obs_spec(method_node: ast.FunctionDef) -> dict[str, object]:
    always_keys: list[str] = []
    conditional: OrderedDict[str, list[str]] = OrderedDict()

    def obs_spec_key(target: ast.AST) -> str | None:
        if not isinstance(target, ast.Subscript):
            return None
        if not isinstance(target.value, ast.Name) or target.value.id != "obs_spec":
            return None
        slice_node = target.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        return None

    def record(key: str, condition: str | None) -> None:
        if condition is None:
            if key not in always_keys:
                always_keys.append(key)
            return
        keys = conditional.setdefault(condition, [])
        if key not in keys:
            keys.append(key)

    def walk(statements: list[ast.stmt], conditions: list[str]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    key = obs_spec_key(target)
                    if key is not None:
                        record(key, " and ".join(conditions) if conditions else None)
                if (
                    any(isinstance(target, ast.Name) and target.id == "obs_spec" for target in stmt.targets)
                    and isinstance(stmt.value, ast.Call)
                    and stmt.value.args
                    and isinstance(stmt.value.args[0], ast.Dict)
                ):
                    for dict_key in stmt.value.args[0].keys:
                        if isinstance(dict_key, ast.Constant) and isinstance(dict_key.value, str):
                            record(dict_key.value, None)
            elif isinstance(stmt, ast.If):
                cond = ast.unparse(stmt.test)
                walk(stmt.body, conditions + [cond])
                if stmt.orelse:
                    walk(stmt.orelse, conditions + [f"not ({cond})"])

    walk(method_node.body, [])
    return {
        "always": always_keys,
        "conditional": conditional,
    }


def try_runtime_import() -> dict[str, str]:
    try:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        __import__("pysc2.lib.features")
        __import__("pysc2.lib.actions")
        __import__("pysc2.env.sc2_env")
        return {"ok": "true", "message": "Runtime import succeeded."}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {"ok": "false", "message": "Runtime import failed; report generated from installed source files.", "error": message}


def build_report() -> dict[str, object]:
    pysc2_root = find_pysc2_root()
    version = importlib.metadata.version("pysc2")
    runtime_status = try_runtime_import()

    features_path = pysc2_root / "lib" / "features.py"
    actions_path = pysc2_root / "lib" / "actions.py"
    sc2_env_path = pysc2_root / "env" / "sc2_env.py"

    _, features_tree = load_module_source(features_path)
    _, actions_tree = load_module_source(actions_path)
    _, sc2_env_tree = load_module_source(sc2_env_path)

    action_space_class = find_class(actions_tree, "ActionSpace")
    dimensions_class = find_class(features_tree, "Dimensions")
    aif_class = find_class(features_tree, "AgentInterfaceFormat")
    parse_aif_fn = find_function(features_tree, "parse_agent_interface_format")
    sc2_env_class = find_class(sc2_env_tree, "SC2Env")
    sc2_env_init = find_method(sc2_env_class, "__init__")
    features_class, observation_spec_method = find_class_with_method(features_tree, "observation_spec")

    screen_features_class = find_class(features_tree, "ScreenFeatures")
    minimap_features_class = find_class(features_tree, "MinimapFeatures")
    screen_features_call = find_assignment_call(features_tree, "SCREEN_FEATURES")
    minimap_features_call = find_assignment_call(features_tree, "MINIMAP_FEATURES")

    return {
        "python": sys.version.split()[0],
        "pysc2_version": version,
        "pysc2_root": str(pysc2_root),
        "source_files": {
            "features.py": str(features_path),
            "actions.py": str(actions_path),
            "sc2_env.py": str(sc2_env_path),
        },
        "runtime_import": runtime_status,
        "action_space": extract_enum_members(action_space_class),
        "dimensions_summary": summarize_docstring(dimensions_class, max_lines=4),
        "agent_interface_format": {
            "summary": summarize_docstring(aif_class, max_lines=2),
            "parameters": extract_parameters(find_method(aif_class, "__init__"), skip={"self"}),
        },
        "parse_agent_interface_format": {
            "summary": summarize_docstring(parse_aif_fn, max_lines=6),
            "parameters": extract_parameters(parse_aif_fn),
        },
        "sc2_env": {
            "summary": summarize_docstring(sc2_env_init, max_lines=6),
            "parameters": extract_parameters(sc2_env_init, skip={"self"}),
        },
        "observation_spec": {
            "class": features_class.name,
            "always": extract_obs_spec(observation_spec_method)["always"],
            "conditional": extract_obs_spec(observation_spec_method)["conditional"],
        },
        "screen_features": {
            "fields": extract_namedtuple_fields(screen_features_class),
            "entries": extract_call_keyword_rows(
                screen_features_call,
                tuple_labels=["scale", "type", "palette", "clip"],
            ),
        },
        "minimap_features": {
            "fields": extract_namedtuple_fields(minimap_features_class),
            "entries": extract_call_keyword_rows(
                minimap_features_call,
                tuple_labels=["scale", "type", "palette"],
            ),
        },
        "source_notes": [
            "AgentInterfaceFormat allows raw-only setups because use_raw_units can satisfy the constructor even when feature_dimensions and rgb_dimensions are None.",
            "The feature-layer path in Dimensions requires both screen and minimap sizes together.",
            "observation_spec shows that feature_screen and feature_minimap are both emitted whenever aif.feature_dimensions is enabled.",
        ],
    }


def print_parameters(title: str, params: list[dict[str, str]]) -> None:
    print(title)
    for param in params:
        if param["default"]:
            print(f"  - {param['name']} ({param['kind']}): {param['default']}")
        else:
            print(f"  - {param['name']} ({param['kind']})")


def print_feature_table(title: str, entries: list[dict[str, str]]) -> None:
    print(title)
    for entry in entries:
        pieces = [f"[{entry['index']}] {entry['name']}"]
        for key, value in entry.items():
            if key in {"index", "name"}:
                continue
            pieces.append(f"{key}={value}")
        print("  - " + " | ".join(pieces))


def print_report(report: dict[str, object]) -> None:
    print("PySC2 Inspection Report")
    print(f"Python: {report['python']}")
    print(f"PySC2 version: {report['pysc2_version']}")
    print(f"PySC2 root: {report['pysc2_root']}")
    print(f"features.py: {report['source_files']['features.py']}")
    print(f"actions.py: {report['source_files']['actions.py']}")
    print(f"sc2_env.py: {report['source_files']['sc2_env.py']}")
    print()

    runtime = report["runtime_import"]
    print("Runtime import status")
    print(f"  - ok: {runtime['ok']}")
    print(f"  - message: {runtime['message']}")
    if "error" in runtime:
        print(f"  - error: {runtime['error']}")
    print()

    print("ActionSpace enum")
    for member in report["action_space"]:
        print(f"  - {member['name']} = {member['value']}")
    print()

    print("Dimensions summary")
    for line in report["dimensions_summary"]:
        print(f"  - {line}")
    print()

    print_parameters("AgentInterfaceFormat.__init__ parameters", report["agent_interface_format"]["parameters"])
    print()
    for line in report["parse_agent_interface_format"]["summary"]:
        print(f"parse_agent_interface_format note: {line}")
    print_parameters("parse_agent_interface_format parameters", report["parse_agent_interface_format"]["parameters"])
    print()

    for line in report["sc2_env"]["summary"]:
        print(f"SC2Env.__init__ note: {line}")
    print_parameters("SC2Env.__init__ parameters", report["sc2_env"]["parameters"])
    print()

    print("Observation keys always present in observation_spec")
    for key in report["observation_spec"]["always"]:
        print(f"  - {key}")
    print()
    print("Observation keys gated by flags")
    for condition, keys in report["observation_spec"]["conditional"].items():
        print(f"  - if {condition}: {', '.join(keys)}")
    print()

    screen_entries = report["screen_features"]["entries"]
    minimap_entries = report["minimap_features"]["entries"]
    print(f"Screen feature layers ({len(screen_entries)})")
    print(f"  - field order: {', '.join(report['screen_features']['fields'])}")
    print_feature_table("  entries", screen_entries)
    print()

    print(f"Minimap feature layers ({len(minimap_entries)})")
    print(f"  - field order: {', '.join(report['minimap_features']['fields'])}")
    print_feature_table("  entries", minimap_entries)
    print()

    print("Source-derived notes")
    for line in report["source_notes"]:
        print(f"  - {line}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the locally installed PySC2 package and print feature-layer and interface options."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of formatted text.",
    )
    args = parser.parse_args()

    try:
        report = build_report()
    except Exception as exc:
        print(f"Failed to inspect PySC2: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

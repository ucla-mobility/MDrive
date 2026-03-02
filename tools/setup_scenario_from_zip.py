#!/usr/bin/env python3
"""Extract scenario-builder ZIP exports and stage them for the leaderboard runner."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import shlex

__all__ = ["prepare_routes_from_zip", "parse_route_metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="ZIP produced by scenario_builder.")
    parser.add_argument(
        "--scenario-name",
        help="Optional name for the extracted scenario directory. Defaults to the ZIP stem.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation/leaderboard/data/CustomRoutes"),
        help="Root directory where routes will be created.",
    )
    parser.add_argument(
        "--scenario-parameter",
        default="simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_no_npc.yaml",
        help="Scenario parameter YAML to use when launching the evaluator.",
    )
    parser.add_argument(
        "--agent",
        default="simulation/leaderboard/team_code/colmdriver_agent.py",
        help="Path to the evaluation agent.",
    )
    parser.add_argument(
        "--agent-config",
        default="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
        help="Agent configuration file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing scenario directory.",
    )
    return parser.parse_args()


def parse_route_metadata(xml_bytes: bytes) -> Tuple[str, str, str]:
    root = ET.fromstring(xml_bytes)
    route_node = root.find("route")
    if route_node is None:
        raise ValueError("Missing <route> element.")
    route_id = route_node.get("id", "0")
    town = route_node.get("town", "Town01")
    role = route_node.get("role", "ego")
    return route_id, town, role


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Destination {path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_routes_from_zip(
    zip_path: Path,
    scenario_name: str | None = None,
    output_root: Path = Path("simulation/leaderboard/data/CustomRoutes"),
    overwrite: bool = False,
) -> Dict[str, object]:
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    scenario_name = scenario_name or zip_path.stem
    output_root = Path(output_root).expanduser().resolve()
    dest_dir = output_root / scenario_name
    ensure_clean_dir(dest_dir, overwrite=overwrite)

    role_dirs: Dict[str, Path] = {"ego": dest_dir}
    extracted: List[Tuple[Path, str, str, str]] = []
    actors_manifest: Dict[str, List[Dict[str, str]]] = {}

    behavior_bytes = None
    with zipfile.ZipFile(zip_path) as archive:
        behavior_member = None
        for member in archive.infolist():
            if Path(member.filename).name == "actors_behavior.json" and not member.is_dir():
                behavior_member = member
                break
        if behavior_member is not None:
            behavior_bytes = archive.read(behavior_member.filename)
        members = [
            m
            for m in archive.infolist()
            if (
                not m.is_dir()
                and m.filename.lower().endswith(".xml")
                and "__MACOSX/" not in m.filename
                and not Path(m.filename).name.startswith("._")
            )
        ]
        if not members:
            raise RuntimeError("ZIP does not contain any XML route files.")
        for member in members:
            data = archive.read(member.filename)
            route_id, town, role = parse_route_metadata(data)
            model_attr = None
            speed_attr = None
            length_attr = None
            width_attr = None
            try:
                xml_root = ET.fromstring(data)
                route_node = xml_root.find("route")
            except ET.ParseError:
                route_node = None
            if route_node is not None:
                model_attr = route_node.get("model")
                speed_attr = route_node.get("speed")
                length_attr = route_node.get("length")
                width_attr = route_node.get("width")
            target_dir = role_dirs.get(role)
            if target_dir is None:
                target_dir = dest_dir / "actors" / role
                target_dir.mkdir(parents=True, exist_ok=True)
                role_dirs[role] = target_dir
            target_path = target_dir / Path(member.filename).name
            target_path.write_bytes(data)
            extracted.append((target_path, role, route_id, town))
            rel_path = target_path.relative_to(dest_dir).as_posix()
            actors_manifest.setdefault(role, []).append(
                {
                    "file": rel_path,
                    "route_id": route_id,
                    "town": town,
                    "name": Path(member.filename).stem,
                    "kind": role,
                    **(
                        {
                            k: v
                            for k, v in (
                                ("model", model_attr),
                                ("speed", speed_attr),
                                ("length", length_attr),
                                ("width", width_attr),
                            )
                            if v is not None
                        }
                    ),
                }
            )

    ego_count = sum(1 for _, role, _, _ in extracted if role == "ego")
    if ego_count == 0:
        raise RuntimeError("No routes with role='ego' discovered.")

    behavior_path = None
    if behavior_bytes is not None:
        behavior_path = dest_dir / "actors_behavior.json"
        behavior_path.write_bytes(behavior_bytes)

    manifest_path = dest_dir / "actors_manifest.json"
    manifest_path.write_text(json.dumps(actors_manifest, indent=2), encoding="utf-8")

    return {
        "scenario_name": scenario_name,
        "output_dir": dest_dir,
        "routes": extracted,
        "ego_count": ego_count,
        "route_ids": sorted({route_id for _, _, route_id, _ in extracted}),
        "towns": sorted({town for _, _, _, town in extracted}),
        "manifest_path": manifest_path,
        "actors_manifest": actors_manifest,
        "behavior_path": behavior_path,
    }


def main() -> None:
    args = parse_args()
    summary = prepare_routes_from_zip(
        zip_path=args.zip_path,
        scenario_name=args.scenario_name,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )

    dest_dir = summary["output_dir"]
    extracted = summary["routes"]
    ego_count = summary["ego_count"]
    manifest_path = summary["manifest_path"]
    actors_manifest = summary["actors_manifest"]

    print(f"\nRoutes extracted to: {dest_dir}")
    print("Actors:")
    for path, role, route_id, town in extracted:
        print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={path.name}")
    print(f"\nUnique route IDs discovered: {', '.join(summary['route_ids'])}")
    print(f"Towns referenced: {', '.join(summary['towns'])}")
    print(f"Ego vehicles: {ego_count}")
    non_ego_roles = [(role, len(entries)) for role, entries in actors_manifest.items() if role != "ego"]
    if non_ego_roles:
        print(f"\nActor manifest saved to: {manifest_path}")
        for role, count in sorted(non_ego_roles):
            print(f"  - {role}: {count} file(s)")

    quoted_cmd = " ".join(
        [
            "python",
            "simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py",
            "--routes_dir",
            shlex.quote(str(dest_dir)),
            "--ego-num",
            str(ego_count),
            "--scenario_parameter",
            shlex.quote(args.scenario_parameter),
            "--agent",
            shlex.quote(args.agent),
            "--agent-config",
            shlex.quote(args.agent_config),
        ]
    )

    print("\nSuggested launch command (adjust flags as needed):")
    print(f"  {quoted_cmd}\n")


if __name__ == "__main__":
    main()

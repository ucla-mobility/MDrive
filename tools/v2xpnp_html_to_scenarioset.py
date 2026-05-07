"""Export a pipeline-runtime HTML dataset into the scenarioset/v2xpnp/<scenario>/
layout (route XMLs + manifest + control config).

Reads the embedded `<script id="dataset" type="application/json">` block from a
pipeline_runtime HTML, applies the PKL→CARLA coordinate transform, and writes:

  <out>/<scenario>/
      ucla_v2_custom_ego_vehicle_<i>.xml      # ego routes (one per ego)
      actors/npc/ucla_v2_custom_Vehicle_<id>_npc.xml
      actors/walker/ucla_v2_custom_Walker_<id>_walker.xml
      actors/static/ucla_v2_custom_Vehicle_<id>_static.xml
      actors_manifest.json
      carla_control_config.json

The format mirrors the reference 21 already-converted v2xpnp scenarios. Z,
pitch, and roll are set to 0.0 — run carla_ground_align.py separately to set
real ground elevation.

Usage:
    python3 -m tools.v2xpnp_html_to_scenarioset \
        --html /tmp/eval_v3/2023-03-17-15-53-02_1_0.html \
        --out scenarioset/v2xpnp \
        --map-offset-json v2xpnp/map/map_offset_carla.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.v2xpnp_smoothness_postprocess import (
    smooth_track_waypoints,
    reject_speed_anomalies,
    trim_end_anomalies,
)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def parse_html_dataset(html_path: Path) -> Dict[str, Any]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'<script[^>]+id="dataset"[^>]*type="application/json"[^>]*>(.*?)</script>',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"No embedded dataset found in {html_path}")
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Coordinate transform: PKL → CARLA
# ---------------------------------------------------------------------------
def load_pkl_to_carla_transform(json_path: Optional[Path]) -> Dict[str, float]:
    """The map-offset JSON describes PKL→CARLA. Returns a dict with `tx`, `ty`,
    `flip_y`, and `theta_deg` (currently only theta=0 supported)."""
    out: Dict[str, float] = {"tx": 0.0, "ty": 0.0, "flip_y": False, "theta_deg": 0.0}
    if json_path is None or not json_path.exists():
        return out
    cfg = json.loads(json_path.read_text(encoding="utf-8"))
    out["tx"] = float(cfg.get("tx", 0.0))
    out["ty"] = float(cfg.get("ty", 0.0))
    out["flip_y"] = bool(cfg.get("flip_y", False))
    out["theta_deg"] = float(cfg.get("theta_deg", 0.0))
    return out


def pkl_to_carla(x: float, y: float, yaw_deg: float, t: Dict[str, float]) -> Tuple[float, float, float]:
    """Apply the inverse transform (PKL → CARLA) — note JSON describes
    CARLA→PKL so we negate translations as in `trajectory_ingest`."""
    tx = -t["tx"]
    ty = t["ty"] if t["flip_y"] else -t["ty"]
    if t["flip_y"]:
        x_out = x + tx
        y_out = -y + ty
        yaw_out = -yaw_deg  # mirror y → flip yaw sign
    else:
        x_out = x + tx
        y_out = y + ty
        yaw_out = yaw_deg
    if abs(t["theta_deg"]) > 1e-9:
        # rotate xy + adjust yaw
        c = math.cos(math.radians(-t["theta_deg"]))
        s = math.sin(math.radians(-t["theta_deg"]))
        x_out, y_out = x_out * c - y_out * s, x_out * s + y_out * c
        yaw_out -= t["theta_deg"]
    return x_out, y_out, yaw_out


# ---------------------------------------------------------------------------
# Actor classification
# ---------------------------------------------------------------------------
def classify_actor(track: Dict[str, Any]) -> str:
    """Returns one of: 'ego', 'npc', 'walker', 'static'."""
    role = str(track.get("role", "")).strip().lower()
    if role == "ego":
        return "ego"
    if role == "walker":
        # walkers that don't move much — call them walker_static
        # for now treat all as walker
        return "walker"
    # vehicle role: split into npc / static
    if bool(track.get("low_motion_vehicle", False)):
        return "static"
    # Check raw displacement
    fr = track.get("frames") or []
    if len(fr) < 2:
        return "static"
    x0, y0 = _safe_float(fr[0].get("x")), _safe_float(fr[0].get("y"))
    xN, yN = _safe_float(fr[-1].get("x")), _safe_float(fr[-1].get("y"))
    if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(xN) and math.isfinite(yN):
        if math.hypot(xN - x0, yN - y0) < 2.0:
            return "static"
    return "npc"


# ---------------------------------------------------------------------------
# XML writers
# ---------------------------------------------------------------------------
def _write_xml(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_ego_route(
    path: Path,
    route_id: str,
    town: str,
    waypoints: List[Tuple[float, float, float, float]],  # x, y, yaw_deg, t
) -> None:
    """Ego XML format (matches reference)."""
    root = ET.Element("routes", {"actor_control_mode": "replay", "log_replay_actors": "true"})
    route = ET.SubElement(root, "route", {
        "id": route_id,
        "role": "ego",
        "snap_to_road": "false",
        "town": town,
    })
    for x, y, yaw, _t in waypoints:
        ET.SubElement(route, "waypoint", {
            "pitch": "0.000000",
            "roll": "0.000000",
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "yaw": f"{yaw:.6f}",
            "z": "0.000000",
        })
    _write_xml(root, path)


def write_actor_route(
    path: Path,
    route_id: str,
    town: str,
    role: str,
    model: str,
    waypoints: List[Tuple[float, float, float, float]],  # x, y, yaw_deg, t
    control_mode: str = "replay",
) -> None:
    """NPC / walker / static XML format (matches reference)."""
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", {
        "control_mode": control_mode,
        "id": route_id,
        "model": model,
        "role": role,
        "snap_to_road": "false",
        "town": town,
    })
    for x, y, yaw, t in waypoints:
        ET.SubElement(route, "waypoint", {
            "pitch": "0.000000",
            "roll": "0.000000",
            "time": f"{t:.6f}",
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "yaw": f"{yaw:.6f}",
            "z": "0.000000",
        })
    _write_xml(root, path)


# ---------------------------------------------------------------------------
# Model name guessing — matches the conventions used by trajectory_ingest
# ---------------------------------------------------------------------------
DEFAULT_VEHICLE_MODEL = "vehicle.tesla.model3"
DEFAULT_EGO_MODEL = "vehicle.lincoln.mkz_2020"
DEFAULT_WALKER_MODEL = "walker.pedestrian.0001"


def actor_model(track: Dict[str, Any], kind: str) -> str:
    if kind == "walker":
        return str(track.get("model") or DEFAULT_WALKER_MODEL)
    if kind == "ego":
        return str(track.get("model") or DEFAULT_EGO_MODEL)
    return str(track.get("model") or DEFAULT_VEHICLE_MODEL)


def actor_filename(scenario_name: str, town: str, track_id: str, kind: str) -> str:
    actor_type = "Walker" if kind == "walker" else "Vehicle"
    return f"{town.lower()}_custom_{actor_type}_{track_id}_{kind}.xml"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_dataset(
    dataset: Dict[str, Any],
    scenario_name: str,
    out_root: Path,
    transform: Dict[str, float],
    town: str = "ucla_v2",
) -> Dict[str, Any]:
    """Convert one HTML dataset to a scenarioset XML directory.

    Returns the manifest dict that was written.
    """
    out_dir = out_root / scenario_name
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("npc", "walker", "static"):
        (actors_dir / sub).mkdir(parents=True, exist_ok=True)

    tracks = dataset.get("tracks") or []
    ego_entries: List[Dict[str, Any]] = []
    actors_by_kind: Dict[str, List[Dict[str, Any]]] = {"npc": [], "walker": [], "static": []}

    # Pre-compute lane segments for off-road filtering
    _lane_segs = []
    _lines = (dataset.get("carla_map") or {}).get("lines") or []
    for _line in _lines:
        _pts = _line.get("polyline") or []
        for k in range(len(_pts) - 1):
            try:
                x0, y0 = float(_pts[k][0]), float(_pts[k][1])
                x1, y1 = float(_pts[k + 1][0]), float(_pts[k + 1][1])
            except (TypeError, ValueError):
                continue
            sdx = x1 - x0; sdy = y1 - y0
            l2 = sdx * sdx + sdy * sdy
            if l2 < 1e-9:
                continue
            _lane_segs.append((x0, y0, sdx, sdy, l2))

    def _min_lane_dist(px: float, py: float) -> float:
        best = float("inf")
        for x0, y0, sdx, sdy, l2 in _lane_segs:
            t = ((px - x0) * sdx + (py - y0) * sdy) / l2
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
            cx = x0 + t * sdx; cy = y0 + t * sdy
            d = math.hypot(cx - px, cy - py)
            if d < best:
                best = d
        return best if math.isfinite(best) else float("inf")

    ego_count = 0
    n_dropped_off_road = 0
    for track in tracks:
        tid = str(track.get("id", "?"))
        role = str(track.get("role", "")).strip().lower()
        if role not in ("ego", "vehicle", "walker"):
            continue
        kind = classify_actor(track)
        frames = track.get("frames") or []
        if not frames:
            continue
        # Drop vehicle tracks that are ENTIRELY off-road (every frame > 4 m from
        # nearest drivable lane). These are perception-noise detections of
        # objects in parking lots, driveways, or interior parts of properties
        # — replaying them in CARLA would put a "vehicle" inside a building.
        # Walkers and ego are exempt (walkers can be on sidewalks; ego is the
        # target route and we trust it).
        if role == "vehicle" and kind != "ego" and _lane_segs:
            min_dists = []
            for f in frames:
                cx = _safe_float(f.get("cx"))
                cy = _safe_float(f.get("cy"))
                if math.isfinite(cx) and math.isfinite(cy):
                    min_dists.append(_min_lane_dist(cx, cy))
                if len(min_dists) >= 20:
                    break  # enough samples to decide
            if min_dists and min(min_dists) > 4.0:
                n_dropped_off_road += 1
                continue
        # Build raw and snap waypoint candidates; pick whichever is smoother
        # for moving vehicles (raw often beats snap for fast NPCs because the
        # snap pipeline introduces stuck/teleport frames that raw doesn't have).
        snap_wps: List[Tuple[float, float, float, float]] = []
        raw_wps: List[Tuple[float, float, float, float]] = []
        for f in frames:
            cx = _safe_float(f.get("cx"))
            cy = _safe_float(f.get("cy"))
            cyaw = _safe_float(f.get("cyaw"), default=_safe_float(f.get("yaw")))
            t = _safe_float(f.get("t"), 0.0)
            rx = _safe_float(f.get("x"))
            ry = _safe_float(f.get("y"))
            ryaw = _safe_float(f.get("yaw"))
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cyaw)):
                cx, cy, cyaw = rx, ry, ryaw
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cyaw)):
                continue
            xc, yc, yawc = pkl_to_carla(cx, cy, cyaw, transform)
            snap_wps.append((xc, yc, yawc, t))
            if math.isfinite(rx) and math.isfinite(ry) and math.isfinite(ryaw):
                rxc, ryc, ryawc = pkl_to_carla(rx, ry, ryaw, transform)
                raw_wps.append((rxc, ryc, ryawc, t))
            else:
                raw_wps.append((xc, yc, yawc, t))

        def _track_jerk_score(wps: List[Tuple[float, float, float, float]]) -> float:
            """Total jerk-like roughness of a waypoint sequence."""
            if len(wps) < 4:
                return 0.0
            speeds = []
            for i in range(1, len(wps)):
                dt = max(wps[i][3] - wps[i - 1][3], 1e-6)
                speeds.append(math.hypot(wps[i][0] - wps[i - 1][0], wps[i][1] - wps[i - 1][1]) / dt)
            jerks = 0.0
            for i in range(1, len(speeds) - 1):
                jerks += abs(speeds[i + 1] - 2 * speeds[i] + speeds[i - 1])
            return jerks

        # For non-static, prefer the smoother of raw vs snap. (Snap is usually
        # better for low-motion / parked / lane-following; raw is usually better
        # for fast moving NPCs where snap introduces stuck-and-teleport.)
        waypoints = snap_wps
        if kind not in ("static", "ego") and len(snap_wps) >= 5 and len(raw_wps) >= 5:
            j_snap = _track_jerk_score(snap_wps)
            j_raw = _track_jerk_score(raw_wps)
            if j_raw < j_snap * 0.85:  # raw is meaningfully smoother
                waypoints = raw_wps
        if not waypoints:
            continue
        # Trim "anchor frame" artifacts: tracks may have an isolated early frame
        # at t=0 followed by a multi-second gap to the real trajectory. Replaying
        # that creates a spawn-then-teleport pattern with absurd jerk. If any
        # consecutive pair has dt > 1.0s, drop everything before that gap.
        # Apply to non-static actors (static keep their fixed pose anyway).
        if kind != "static" and len(waypoints) >= 2:
            cut_at = 0
            for k in range(1, len(waypoints)):
                if waypoints[k][3] - waypoints[k - 1][3] > 1.0:
                    cut_at = k
            if cut_at > 0:
                waypoints = waypoints[cut_at:]
                if frames is not None and len(frames) >= cut_at:
                    frames = frames[cut_at:]
        if not waypoints:
            continue
        # Re-base time so first waypoint is at t=0 (CARLA expects this)
        t0 = waypoints[0][3]
        if abs(t0) > 1e-6:
            waypoints = [(x, y, yaw, t - t0) for (x, y, yaw, t) in waypoints]
        # For static actors, freeze pose at first waypoint (avoid replaying jitter)
        if kind == "static" and len(waypoints) > 1:
            anchor = waypoints[0]
            waypoints = [(anchor[0], anchor[1], anchor[2], wp[3]) for wp in waypoints]
        else:
            # For moving actors / walkers / egos: freeze stop-and-go + Gaussian smooth
            # (handles the residual annotation jitter that survives lane snapping)
            waypoints = smooth_track_waypoints(waypoints, frames, kind=kind)
            # Reject single-frame speed anomalies (snap-stuck frames where
            # the snap lagged raw and then teleported to catch up)
            waypoints = reject_speed_anomalies(waypoints, max_speed_mps=22.0)
            # Drop trailing teleports
            waypoints = trim_end_anomalies(waypoints, max_speed_mps=22.0)

        if kind == "ego":
            xml_name = f"{town.lower()}_custom_ego_vehicle_{ego_count}.xml"
            xml_path = out_dir / xml_name
            write_ego_route(
                xml_path,
                route_id=scenario_name,
                town=town,
                waypoints=waypoints,
            )
            ego_entries.append({
                "file": xml_name,
                "route_id": scenario_name,
                "town": town,
                "name": f"ego_{ego_count}",
                "kind": "ego",
                "model": actor_model(track, "ego"),
            })
            ego_count += 1
        else:
            xml_name = actor_filename(scenario_name, town, tid, kind)
            xml_path = actors_dir / kind / xml_name
            write_actor_route(
                xml_path,
                route_id=scenario_name,
                town=town,
                role=kind,
                model=actor_model(track, kind),
                waypoints=waypoints,
                control_mode="replay",
            )
            entry = {
                "file": str(xml_path.relative_to(out_dir)),
                "route_id": scenario_name,
                "town": town,
                "name": tid,
                "kind": kind,
                "model": actor_model(track, kind),
                "control_mode": "replay",
            }
            length = track.get("length")
            width = track.get("width")
            if length is not None:
                entry["length"] = str(length) if isinstance(length, (int, float)) else length
            if width is not None:
                entry["width"] = str(width) if isinstance(width, (int, float)) else width
            actors_by_kind[kind].append(entry)

    manifest: Dict[str, Any] = {}
    if ego_entries:
        manifest["ego"] = ego_entries
    for kind in sorted(actors_by_kind.keys()):
        if actors_by_kind[kind]:
            manifest[kind] = actors_by_kind[kind]
    (out_dir / "actors_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # carla_control_config.json
    config = {
        "ego_path_source": "auto",
        "actor_control_mode": "replay",
        "walker_control_mode": "replay",
        "encode_timing": True,
        "snap_to_road": False,
        "static_spawn_only": False,
        "town": town,
        "route_id": scenario_name,
        "manifest_path": "actors_manifest.json",
        "ego_count": len(ego_entries),
        "npc_count": len(actors_by_kind.get("npc", [])),
        "walker_count": len(actors_by_kind.get("walker", [])),
        "static_count": len(actors_by_kind.get("static", [])),
    }
    (out_dir / "carla_control_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    if n_dropped_off_road:
        print(f"  Dropped {n_dropped_off_road} off-road vehicle track(s)")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--html", nargs="+", required=True, help="HTML files produced by pipeline_runtime")
    p.add_argument("--out", required=True, help="Output root (e.g., scenarioset/v2xpnp)")
    p.add_argument("--map-offset-json", required=True, help="map_offset_carla.json")
    p.add_argument("--town", default="ucla_v2")
    p.add_argument("--route-id-from", choices=["scenario_name", "html_stem"], default="scenario_name",
                   help="How to derive the route_id (default: dataset.scenario_name)")
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    transform = load_pkl_to_carla_transform(Path(args.map_offset_json))
    print(f"Transform: tx={transform['tx']} ty={transform['ty']} flip_y={transform['flip_y']}")

    for hp in args.html:
        hp_path = Path(hp).resolve()
        ds = parse_html_dataset(hp_path)
        if args.route_id_from == "html_stem":
            scenario_name = hp_path.stem
        else:
            scenario_name = ds.get("scenario_name") or hp_path.stem
        print(f"\n=== {scenario_name} ===")
        m = convert_dataset(ds, scenario_name, out_root, transform, town=args.town)
        n_ego = len(m.get("ego", []))
        n_npc = len(m.get("npc", []))
        n_walker = len(m.get("walker", []))
        n_static = len(m.get("static", []))
        print(f"  ego={n_ego} npc={n_npc} walker={n_walker} static={n_static}")
        print(f"  out: {out_root / scenario_name}")


if __name__ == "__main__":
    main()

"""Trajectory ingest internals: visualization and CLI orchestration."""

from __future__ import annotations

from v2xpnp.pipeline import trajectory_ingest_stage_01_types_io as _ingest_s1
from v2xpnp.pipeline import trajectory_ingest_stage_02_matching as _ingest_s2
from v2xpnp.pipeline import trajectory_ingest_stage_03_spawn_logic as _ingest_s3
from v2xpnp.pipeline.trajectory_ingest_stage_01_types_io import *  # noqa: F401,F403
from v2xpnp.pipeline.trajectory_ingest_stage_02_matching import *  # noqa: F401,F403
from v2xpnp.pipeline.trajectory_ingest_stage_03_spawn_logic import *  # noqa: F401,F403


for _mod in (_ingest_s1, _ingest_s2, _ingest_s3):
    for _name, _value in vars(_mod).items():
        if _name.startswith("__"):
            continue
        globals().setdefault(_name, _value)



def _yaml_dir_sort_key(path: Path) -> Tuple[int, object]:
    name = path.name
    try:
        return (0, int(name))
    except Exception:
        return (1, name)


def _is_negative_subdir(path: Path) -> bool:
    try:
        return int(path.name) < 0
    except Exception:
        return False


def _parse_id_list(raw: str) -> List[int]:
    ids: List[int] = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        try:
            ids.append(int(token))
        except Exception:
            continue
    return ids


def _apply_spawn_preprocess_maximal_profile(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "spawn_preprocess_maximal", False)):
        return

    args.spawn_preprocess = True
    args.spawn_preprocess_require_carla = True
    args.use_carla_map = True
    args.encode_timing = True
    args.maximize_safe_early_spawn = True
    args.maximize_safe_late_despawn = True

    args.spawn_preprocess_align = True
    args.spawn_preprocess_align_ego = True
    args.spawn_preprocess_align_ego_piecewise = True
    args.spawn_preprocess_refine_piecewise = True
    args.spawn_preprocess_normalize_z = True
    args.snap_ego_to_lane = True
    args.parked_clearance = True

    args.spawn_preprocess_max_shift = max(float(args.spawn_preprocess_max_shift), 5.0)
    args.spawn_preprocess_random_samples = max(int(args.spawn_preprocess_random_samples), 160)
    args.spawn_preprocess_max_candidates = max(int(args.spawn_preprocess_max_candidates), 120)
    args.spawn_preprocess_sample_dt = min(float(args.spawn_preprocess_sample_dt), 0.35)
    args.spawn_preprocess_align_samples = max(int(args.spawn_preprocess_align_samples), 18)
    args.spawn_preprocess_align_windows = max(int(args.spawn_preprocess_align_windows), 5)
    args.spawn_preprocess_align_neighbor_radius = max(
        float(args.spawn_preprocess_align_neighbor_radius), 8.0
    )
    args.spawn_preprocess_align_neighbor_weight = max(
        float(args.spawn_preprocess_align_neighbor_weight), 0.22
    )
    args.spawn_preprocess_refine_max_local = max(float(args.spawn_preprocess_refine_max_local), 1.0)
    args.spawn_preprocess_refine_smooth_window = max(int(args.spawn_preprocess_refine_smooth_window), 11)
    args.spawn_preprocess_refine_max_step_delta = min(
        float(args.spawn_preprocess_refine_max_step_delta), 0.25
    )
    args.spawn_preprocess_align_ego_smooth_window = max(
        int(args.spawn_preprocess_align_ego_smooth_window), 13
    )
    args.spawn_preprocess_align_ego_max_step_delta = min(
        float(args.spawn_preprocess_align_ego_max_step_delta), 0.30
    )
    args.spawn_preprocess_bridge_max_gap_steps = max(int(args.spawn_preprocess_bridge_max_gap_steps), 8)
    args.spawn_preprocess_bridge_straight_thresh_deg = min(
        float(args.spawn_preprocess_bridge_straight_thresh_deg), 15.0
    )
    args.early_spawn_safety_margin = max(float(args.early_spawn_safety_margin), 0.30)
    args.late_despawn_safety_margin = max(float(args.late_despawn_safety_margin), 0.30)

    print("[INFO] Enabled maximal spawn preprocess profile (strict CARLA connectivity).")


def pick_yaml_dirs(scenario_dir: Path, chosen: str | None) -> List[Path]:
    subdirs = [d for d in scenario_dir.iterdir() if d.is_dir()]

    if chosen:
        if str(chosen).lower() == "all":
            yaml_subdirs = [d for d in subdirs if list_yaml_timesteps(d)]
            numeric_subdirs = [d for d in yaml_subdirs if re.fullmatch(r"-?\d+", d.name or "")]
            if numeric_subdirs:
                yaml_subdirs = numeric_subdirs
            if not yaml_subdirs:
                raise SystemExit(f"No YAML subfolders found under {scenario_dir}")
            return sorted(yaml_subdirs, key=_yaml_dir_sort_key)
        cand = scenario_dir / chosen
        if not cand.is_dir():
            raise SystemExit(f"--subdir {chosen} not found under {scenario_dir}")
        return [cand]

    if list_yaml_timesteps(scenario_dir):
        return [scenario_dir]

    yaml_subdirs = [d for d in subdirs if list_yaml_timesteps(d)]
    numeric_subdirs = [d for d in yaml_subdirs if re.fullmatch(r"-?\d+", d.name or "")]
    if numeric_subdirs:
        yaml_subdirs = numeric_subdirs
    if len(yaml_subdirs) == 1:
        return yaml_subdirs
    if len(yaml_subdirs) > 1:
        return sorted(yaml_subdirs, key=_yaml_dir_sort_key)

    raise SystemExit(f"No YAML files found under {scenario_dir}")


def _extract_map_lines(obj, depth=0, out: List[List[Tuple[float, float]]] | None = None):
    """Heuristic extractor for vector map polylines from arbitrary pickle structures."""
    if out is None:
        out = []
    if obj is None or depth > 10:
        return out

    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            try:
                out.append([(float(obj["x"]), float(obj["y"]))])
            except Exception:
                pass
        for v in obj.values():
            _extract_map_lines(v, depth + 1, out)
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(hasattr(it, "__len__") and len(it) >= 2 for it in obj if it is not None):
            try:
                pts = [(float(p[0]), float(p[1])) for p in obj if p is not None and len(p) >= 2]
                if len(pts) >= 2:
                    out.append(pts)
                    return out
            except Exception:
                pass
        for v in obj:
            _extract_map_lines(v, depth + 1, out)
        return out

    if hasattr(obj, "x") and hasattr(obj, "y"):
        try:
            out.append([(float(obj.x), float(obj.y))])
        except Exception:
            pass
        return out

    if hasattr(obj, "__dict__"):
        _extract_map_lines(obj.__dict__, depth + 1, out)
    return out


def _extract_map_line_records(obj) -> List[Dict[str, object]]:
    """
    Extract map line records with optional lane metadata.
    Returns records shaped as:
      {"points": [(x,y), ...], "road_id": int|None, "lane_id": int|None, "dir_sign": -1|1|None}
    """
    records: List[Dict[str, object]] = []
    if isinstance(obj, dict):
        raw = obj.get("line_records")
        if isinstance(raw, list):
            for rec in raw:
                if not isinstance(rec, dict):
                    continue
                pts_raw = rec.get("points")
                if not isinstance(pts_raw, (list, tuple)):
                    continue
                pts: List[Tuple[float, float]] = []
                for p in pts_raw:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    try:
                        pts.append((float(p[0]), float(p[1])))
                    except Exception:
                        continue
                if len(pts) < 2:
                    continue
                road_id = rec.get("road_id")
                lane_id = rec.get("lane_id")
                dir_sign = rec.get("dir_sign")
                try:
                    road_id = int(road_id) if road_id is not None else None
                except Exception:
                    road_id = None
                try:
                    lane_id = int(lane_id) if lane_id is not None else None
                except Exception:
                    lane_id = None
                try:
                    dir_sign = int(dir_sign) if dir_sign is not None else None
                except Exception:
                    dir_sign = None
                if dir_sign not in (-1, 1):
                    dir_sign = None
                point_yaws = rec.get("point_yaws")
                point_s = rec.get("point_s")
                if not isinstance(point_yaws, (list, tuple)):
                    point_yaws = None
                if not isinstance(point_s, (list, tuple)):
                    point_s = None
                try:
                    yaw_alignment_score = float(rec.get("yaw_alignment_score", float("nan")))
                except Exception:
                    yaw_alignment_score = float("nan")
                records.append(
                    {
                        "points": pts,
                        "road_id": road_id,
                        "lane_id": lane_id,
                        "dir_sign": dir_sign,
                        "point_yaws": [float(v) for v in point_yaws] if point_yaws is not None else None,
                        "point_s": [float(v) for v in point_s] if point_s is not None else None,
                        "direction_source": str(rec.get("direction_source", "")),
                        "travel_ordered": bool(rec.get("travel_ordered", False)),
                        "order_flipped_vs_s_asc": bool(rec.get("order_flipped_vs_s_asc", False)),
                        "yaw_alignment_score": float(yaw_alignment_score),
                    }
                )
    if records:
        return records

    # Fallback for old caches / generic pickle inputs.
    lines = _extract_map_lines(obj, out=[])
    for line in lines:
        if len(line) >= 2:
            records.append({"points": line, "road_id": None, "lane_id": None, "dir_sign": None})
    return records


def _integrate_geometry(
    x0: float,
    y0: float,
    hdg: float,
    length: float,
    curv_fn,
    step: float,
) -> List[Tuple[float, float]]:
    if length <= 0.0:
        return []
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    ds = length / n
    x = x0
    y = y0
    theta = hdg
    points = [(x, y)]
    for i in range(n):
        s_mid = (i + 0.5) * ds
        kappa = curv_fn(s_mid)
        theta_mid = theta + 0.5 * kappa * ds
        x += ds * math.cos(theta_mid)
        y += ds * math.sin(theta_mid)
        theta += kappa * ds
        points.append((x, y))
    return points


def _sample_geometry(geom: ET.Element, step: float) -> List[Tuple[float, float]]:
    x0 = float(geom.attrib.get("x", 0.0))
    y0 = float(geom.attrib.get("y", 0.0))
    hdg = float(geom.attrib.get("hdg", 0.0))
    length = float(geom.attrib.get("length", 0.0))

    child = next(iter(geom), None)
    if child is None:
        return [(x0, y0)]

    if child.tag == "line":
        curv_fn = lambda s: 0.0
    elif child.tag == "arc":
        curvature = float(child.attrib.get("curvature", 0.0))
        curv_fn = lambda s, k=curvature: k
    elif child.tag == "spiral":
        curv_start = float(child.attrib.get("curvStart", 0.0))
        curv_end = float(child.attrib.get("curvEnd", 0.0))

        def curv_fn(s: float, cs=curv_start, ce=curv_end, total=length) -> float:
            if total <= 0.0:
                return cs
            return cs + (ce - cs) * (s / total)
    else:
        curv_fn = lambda s: 0.0

    return _integrate_geometry(x0, y0, hdg, length, curv_fn, step)


def load_xodr_points(path: Path, step: float) -> List[Tuple[float, float]]:
    root = ET.parse(path).getroot()
    points: List[Tuple[float, float]] = []
    for geom in root.findall(".//planView/geometry"):
        points.extend(_sample_geometry(geom, step))
    return points


def _bounds_from_points(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), max(xs), min(ys), max(ys))


def _merge_bounds(bounds_list: Sequence[Tuple[float, float, float, float] | None]) -> Tuple[float, float, float, float] | None:
    mins = []
    maxs = []
    for b in bounds_list:
        if not b:
            continue
        mins.append((b[0], b[2]))
        maxs.append((b[1], b[3]))
    if not mins or not maxs:
        return None
    minx = min(m[0] for m in mins)
    miny = min(m[1] for m in mins)
    maxx = max(m[0] for m in maxs)
    maxy = max(m[1] for m in maxs)
    return (minx, maxx, miny, maxy)


def _plot_background_lines(ax, lines: List[List[Tuple[float, float]]], color: str, lw: float, alpha: float):
    for line in lines:
        if len(line) < 2:
            continue
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=1)


def _crop_lines_to_bounds(
    lines: List[List[Tuple[float, float]]],
    bounds: Tuple[float, float, float, float],
) -> List[List[Tuple[float, float]]]:
    minx, maxx, miny, maxy = bounds
    cropped: List[List[Tuple[float, float]]] = []
    for line in lines:
        if len(line) < 2:
            continue
        keep = False
        for x, y in line:
            if minx <= x <= maxx and miny <= y <= maxy:
                keep = True
                break
        if keep:
            cropped.append(line)
    return cropped


def _bbox_corners_2d(bbox, tf) -> List[Tuple[float, float]]:
    corners = []
    try:
        ext = bbox.extent
        center = bbox.location
        for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            loc = carla.Location(
                x=center.x + sx * ext.x,
                y=center.y + sy * ext.y,
                z=center.z,
            )
            world_loc = tf.transform(loc)
            corners.append((float(world_loc.x), float(world_loc.y)))
    except Exception:
        return []
    return corners


def _nearest_items(
    items: List[Dict[str, object]],
    center: carla.Location,
    max_dist: float,
    limit: int,
) -> List[Dict[str, object]]:
    out = []
    for item in items:
        loc = item.get("loc")
        if loc is None:
            continue
        try:
            dist = float(loc.distance(center))
        except Exception:
            continue
        if dist > max_dist:
            continue
        out.append((dist, item))
    out.sort(key=lambda x: x[0])
    results = []
    for dist, it in out[:limit]:
        payload = dict(it)
        payload["dist"] = float(dist)
        results.append(payload)
    return results


def _collect_spawn_debug(
    actor_id: int,
    base_wp: Waypoint,
    entry: Dict[str, object],
    world,
    world_map,
    blueprint,
    actor_items: List[Dict[str, object]],
    env_items: List[Dict[str, object]],
    max_dist: float,
    max_items: int,
    probe_yaw: bool,
) -> Dict[str, object]:
    debug: Dict[str, object] = {}
    loc = carla.Location(x=base_wp.x, y=base_wp.y, z=base_wp.z)
    ground_z = _resolve_ground_z(world, loc) if world is not None else None
    if ground_z is not None:
        debug["ground_z"] = float(ground_z)
        debug["z_delta"] = float(base_wp.z) - float(ground_z)
    try:
        wp_any = world_map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any) if world_map else None
        wp_drive = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving) if world_map else None
        wp_sidewalk = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk) if world_map else None
    except Exception:
        wp_any = wp_drive = wp_sidewalk = None
    if wp_any is not None:
        debug["wp_any"] = {
            "road_id": int(getattr(wp_any, "road_id", -1)),
            "lane_id": int(getattr(wp_any, "lane_id", -1)),
            "lane_type": str(getattr(wp_any, "lane_type", "")),
            "is_junction": bool(getattr(wp_any, "is_junction", False)),
        }
    if wp_drive is not None:
        debug["wp_drive"] = {
            "road_id": int(getattr(wp_drive, "road_id", -1)),
            "lane_id": int(getattr(wp_drive, "lane_id", -1)),
            "lane_type": str(getattr(wp_drive, "lane_type", "")),
            "is_junction": bool(getattr(wp_drive, "is_junction", False)),
        }
    if wp_sidewalk is not None:
        debug["wp_sidewalk"] = {
            "road_id": int(getattr(wp_sidewalk, "road_id", -1)),
            "lane_id": int(getattr(wp_sidewalk, "lane_id", -1)),
            "lane_type": str(getattr(wp_sidewalk, "lane_type", "")),
            "is_junction": bool(getattr(wp_sidewalk, "is_junction", False)),
        }

    near_actors = _nearest_items(actor_items, loc, max_dist, limit=max_items)
    near_env = _nearest_items(env_items, loc, max_dist, limit=max_items)
    debug["nearest_actors"] = [
        {
            "id": it.get("id"),
            "type": it.get("type"),
            "dist": it.get("dist"),
            "bbox": it.get("bbox"),
        }
        for it in near_actors
    ]
    debug["nearest_env_objects"] = [
        {
            "id": it.get("id"),
            "type": it.get("type"),
            "dist": it.get("dist"),
            "bbox": it.get("bbox"),
        }
        for it in near_env
    ]

    probe_results = []
    if blueprint is not None and world is not None:
        for dz in (0.0, 0.2, 0.5, 1.0, 2.0):
            spawn_loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + dz)
            spawn_tf = carla.Transform(
                spawn_loc,
                carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw, roll=base_wp.roll),
            )
            ok = False
            actor = None
            try:
                actor = world.try_spawn_actor(blueprint, spawn_tf)
                ok = actor is not None
            except Exception:
                ok = False
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
            probe_results.append({"dz": float(dz), "ok": bool(ok)})

        if probe_yaw:
            yaw_results = []
            for dyaw in (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0):
                spawn_tf = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z),
                    carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw + dyaw, roll=base_wp.roll),
                )
                ok = False
                actor = None
                try:
                    actor = world.try_spawn_actor(blueprint, spawn_tf)
                    ok = actor is not None
                except Exception:
                    ok = False
                if actor is not None:
                    try:
                        actor.destroy()
                    except Exception:
                        pass
                yaw_results.append({"dyaw": float(dyaw), "ok": bool(ok)})
            debug["probe_yaw"] = yaw_results

    debug["probe_z"] = probe_results
    return debug


def _plot_failed_spawn_visualizations(
    report: Dict[str, object],
    map_lines: List[List[Tuple[float, float]]],
    out_dir: Path,
    window_m: float,
    dpi: int,
) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping failed spawn visualization.")
        return

    actors = report.get("actors") or {}
    failed = []
    for actor_id, entry in actors.items():
        chosen = entry.get("chosen") or {}
        if chosen.get("status") != "no_valid_candidates":
            continue
        base = entry.get("spawn_base") or {}
        if not base:
            continue
        failed.append((actor_id, entry, base))

    if not failed:
        print("[INFO] No failed actors to visualize.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Overview plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    if map_lines:
        _plot_background_lines(ax, map_lines, color="#9e9e9e", lw=0.6, alpha=0.5)
    xs = []
    ys = []
    for actor_id, entry, base in failed:
        x = float(base.get("x", 0.0))
        y = float(base.get("y", 0.0))
        xs.append(x)
        ys.append(y)
        ax.scatter([x], [y], c="#d62728", s=30, marker="x", zorder=5)
        ax.text(x, y, str(actor_id), fontsize=6, color="#111111", zorder=6)
    if xs and ys:
        pad = max(10.0, 0.5 * window_m)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title("Failed Spawns Overview")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "failed_spawn_overview.png", dpi=dpi)
    plt.close(fig)

    # Per-actor zoomed plots
    half = max(10.0, 0.5 * float(window_m))
    for actor_id, entry, base in failed:
        cx = float(base.get("x", 0.0))
        cy = float(base.get("y", 0.0))
        bounds = (cx - half, cx + half, cy - half, cy + half)
        local_lines = _crop_lines_to_bounds(map_lines, bounds) if map_lines else []

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", adjustable="box")
        if local_lines:
            _plot_background_lines(ax, local_lines, color="#b0b0b0", lw=0.7, alpha=0.6)

        debug = entry.get("debug") or {}
        if patches is not None:
            for obj in debug.get("nearest_env_objects", []):
                poly = obj.get("bbox")
                if poly:
                    ax.add_patch(
                        patches.Polygon(
                            poly,
                            closed=True,
                            fill=False,
                            edgecolor="#ff9896",
                            linewidth=0.8,
                            alpha=0.8,
                            zorder=1,
                        )
                    )
            for obj in debug.get("nearest_actors", []):
                poly = obj.get("bbox")
                if poly:
                    ax.add_patch(
                        patches.Polygon(
                            poly,
                            closed=True,
                            fill=False,
                            edgecolor="#1f77b4",
                            linewidth=0.8,
                            alpha=0.8,
                            zorder=1,
                        )
                    )

        # Candidate points
        candidates = entry.get("candidates") or []
        invalid_x = []
        invalid_y = []
        valid_x = []
        valid_y = []
        for cand in candidates:
            loc = cand.get("spawn_loc")
            if not loc:
                continue
            if cand.get("valid"):
                valid_x.append(loc[0])
                valid_y.append(loc[1])
            else:
                invalid_x.append(loc[0])
                invalid_y.append(loc[1])
        if invalid_x:
            ax.scatter(invalid_x, invalid_y, s=8, c="#808080", alpha=0.45, label="invalid candidates", zorder=2)
        if valid_x:
            ax.scatter(valid_x, valid_y, s=12, c="#2ca02c", alpha=0.8, label="valid candidates", zorder=3)

        # Base spawn
        ax.scatter([cx], [cy], s=60, marker="x", c="#d62728", label="spawn base", zorder=5)

        # Best invalid
        best = entry.get("best_invalid_candidate") or {}
        if best:
            bl = best.get("spawn_loc")
            if bl:
                ax.scatter([bl[0]], [bl[1]], s=40, marker="o", c="#ff7f0e", label="best invalid", zorder=4)

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_title(f"Failed Spawn Actor {actor_id}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        meta = f"kind={entry.get('kind')} model={entry.get('model_used') or entry.get('model')}"
        stats = entry.get("candidate_stats") or {}
        detail = f"candidates={stats.get('total')} valid={stats.get('valid')} invalid={stats.get('invalid')}"
        ax.text(bounds[0], bounds[3], meta, fontsize=7, va="top")
        ax.text(bounds[0], bounds[3] - 0.05 * (bounds[3] - bounds[2]), detail, fontsize=7, va="top")
        if debug.get("probe_z"):
            z_ok = [str(r["dz"]) for r in debug.get("probe_z") if r.get("ok")]
            ax.text(
                bounds[0],
                bounds[3] - 0.10 * (bounds[3] - bounds[2]),
                f"probe_z_ok: {', '.join(z_ok) if z_ok else 'none'}",
                fontsize=7,
                va="top",
            )

        # Debug text: nearest actors/env objects
        lines = []
        near_actors = debug.get("nearest_actors") or []
        near_env = debug.get("nearest_env_objects") or []
        if near_actors:
            lines.append("nearest actors:")
            for item in near_actors[:5]:
                lines.append(
                    f"  id={item.get('id')} type={item.get('type')} d={item.get('dist'):.2f}"
                )
        if near_env:
            lines.append("nearest env:")
            for item in near_env[:5]:
                lines.append(
                    f"  id={item.get('id')} type={item.get('type')} d={item.get('dist'):.2f}"
                )
        if lines:
            ax.text(
                bounds[1],
                bounds[3],
                "\n".join(lines),
                fontsize=6,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
            )

        out_path = out_dir / f"failed_actor_{actor_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def _plot_offset_annotation(
    ax,
    aligned_pt: Tuple[float, float],
    spawn_pt: Tuple[float, float],
    label: str | None = None,
):
    dx = spawn_pt[0] - aligned_pt[0]
    dy = spawn_pt[1] - aligned_pt[1]
    if dx == 0.0 and dy == 0.0:
        return

    x_step = (aligned_pt[0] + dx, aligned_pt[1])

    # Highlight the pre-alignment (spawn) reference point.
    ax.scatter(
        [spawn_pt[0]],
        [spawn_pt[1]],
        s=130,
        marker="X",
        c="#ff7f0e",
        edgecolors="#111111",
        linewidths=0.6,
        label=label,
        zorder=8,
    )

    # Draw axis-aligned offset components.
    ax.plot(
        [aligned_pt[0], x_step[0]],
        [aligned_pt[1], x_step[1]],
        color="#ff7f0e",
        linewidth=1.8,
        zorder=7,
    )
    ax.plot(
        [x_step[0], spawn_pt[0]],
        [x_step[1], spawn_pt[1]],
        color="#1f77b4",
        linewidth=1.8,
        zorder=7,
    )

    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    ax.annotate(
        f"dx={dx:+.2f}m",
        xy=((aligned_pt[0] + x_step[0]) * 0.5, aligned_pt[1]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#ff7f0e",
        fontsize=9,
        bbox=bbox,
        zorder=9,
    )
    ax.annotate(
        f"dy={dy:+.2f}m",
        xy=(x_step[0], (x_step[1] + spawn_pt[1]) * 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color="#1f77b4",
        fontsize=9,
        bbox=bbox,
        zorder=9,
    )


def _plot_spawn_alignment(
    ax,
    aligned_points: Dict[int, Tuple[float, float]],
    spawn_points: Dict[int, Tuple[float, float]],
    actor_kind_by_id: Dict[int, str],
    ego_aligned: List[Tuple[float, float]],
    ego_spawn: List[Tuple[float, float]],
    title: str,
    show_offsets: bool = True,
    offset_pair: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    offset_label: str | None = None,
):
    kind_markers = {
        "npc": "o",
        "static": "s",
        "walker": "^",
        "walker_static": "^",
    }
    aligned_color = "#2ca02c"
    spawn_color = "#d62728"

    for kind, marker in kind_markers.items():
        ids = [vid for vid, k in actor_kind_by_id.items() if k == kind and vid in aligned_points]
        if not ids:
            continue
        a_pts = [aligned_points[vid] for vid in ids]
        s_pts = [spawn_points[vid] for vid in ids if vid in spawn_points]
        ax.scatter(
            [p[0] for p in a_pts],
            [p[1] for p in a_pts],
            s=20,
            marker=marker,
            c=aligned_color,
            alpha=0.7,
            label=f"{kind} aligned",
            zorder=3,
        )
        if s_pts:
            ax.scatter(
                [p[0] for p in s_pts],
                [p[1] for p in s_pts],
                s=40,
                marker=marker,
                facecolors="none",
                edgecolors=spawn_color,
                linewidths=1.0,
                label=f"{kind} spawn",
                zorder=4,
            )

    if ego_aligned:
        ax.scatter(
            [p[0] for p in ego_aligned],
            [p[1] for p in ego_aligned],
            s=80,
            marker="*",
            c="#111111",
            label="ego aligned",
            zorder=5,
        )
    if ego_spawn:
        ax.scatter(
            [p[0] for p in ego_spawn],
            [p[1] for p in ego_spawn],
            s=110,
            marker="*",
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidths=1.5,
            label="ego spawn",
            zorder=6,
        )

    if show_offsets:
        for vid, a_pt in aligned_points.items():
            s_pt = spawn_points.get(vid)
            if not s_pt:
                continue
            if a_pt == s_pt:
                continue
            ax.plot([a_pt[0], s_pt[0]], [a_pt[1], s_pt[1]], color="#555555", alpha=0.3, linewidth=0.8, zorder=2)

    if offset_pair is not None:
        _plot_offset_annotation(ax, offset_pair[0], offset_pair[1], label=offset_label)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)


def _pick_offset_reference(
    aligned_points: Dict[int, Tuple[float, float]],
    ref_points: Dict[int, Tuple[float, float]],
    ego_aligned: List[Tuple[float, float]],
    ego_ref: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float], str] | None:
    if ego_aligned and ego_ref:
        return ego_aligned[0], ego_ref[0], "ego"
    for vid in sorted(aligned_points.keys()):
        r_pt = ref_points.get(vid)
        if r_pt is None:
            continue
        return aligned_points[vid], r_pt, f"id {vid}"
    return None


def load_vector_map_from_pickle(path: Path) -> List[List[Tuple[float, float]]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return _extract_map_lines(obj, out=[])


def fetch_carla_map_lines(
    host: str,
    port: int,
    sample: float,
    cache_path: Path | None,
    expected_town: str | None = None,
    require_ground_truth_direction: bool = False,
) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float, float, float] | None, List[Dict[str, object]]]:
    """Connect to CARLA, ensure the desired map is loaded, sample waypoints, and optionally cache."""
    def _cache_has_ground_truth_direction(cached_obj: object) -> bool:
        if not isinstance(cached_obj, dict):
            return False
        if str(cached_obj.get("directionality_source", "")).strip().lower() == "carla_waypoint_yaw":
            return True
        recs = cached_obj.get("line_records")
        if not isinstance(recs, list) or not recs:
            return False
        checked = 0
        with_yaw = 0
        with_dir = 0
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            checked += 1
            if isinstance(rec.get("point_yaws"), (list, tuple)) and len(rec.get("point_yaws") or []) >= 2:
                with_yaw += 1
            if rec.get("dir_sign") in (-1, 1):
                with_dir += 1
            if checked >= 32:
                break
        if checked <= 0:
            return False
        return bool(with_yaw >= max(2, int(0.65 * checked)) and with_dir >= max(2, int(0.8 * checked)))

    def _build_travel_ordered_line(seq: List["carla.Waypoint"]) -> Tuple[List[Tuple[float, float]], List[float], List[float], bool, float]:
        pts: List[Tuple[float, float]] = []
        yaws: List[float] = []
        svals: List[float] = []
        for w in seq:
            x = float(w.transform.location.x)
            y = float(w.transform.location.y)
            pts.append((x, y))
            yaws.append(float(w.transform.rotation.yaw))
            svals.append(float(w.s))
        if len(pts) < 2:
            return pts, yaws, svals, False, 0.0

        align_sum = 0.0
        align_count = 0
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            dx = float(x1) - float(x0)
            dy = float(y1) - float(y0)
            seg = float(math.hypot(dx, dy))
            if seg < 1e-4:
                continue
            uy = math.sin(math.radians(float(yaws[i])))
            ux = math.cos(math.radians(float(yaws[i])))
            align_sum += float(dx / seg) * float(ux) + float(dy / seg) * float(uy)
            align_count += 1
        align = float(align_sum / max(1, align_count))
        flipped = False
        if align_count > 0 and align < 0.0:
            pts = list(reversed(pts))
            yaws = list(reversed(yaws))
            svals = list(reversed(svals))
            flipped = True
            align = float(-align)
        return pts, yaws, svals, flipped, float(align)

    # Cache reuse only if map name matches expectation (if provided).
    # Load cache before touching CARLA to avoid crashes in environments without a running server.
    if cache_path and cache_path.exists():
        try:
            cached = pickle.load(cache_path.open("rb"))
            if (
                isinstance(cached, dict)
                and "lines" in cached
                and (expected_town is None or expected_town in str(cached.get("map_name", "")))
            ):
                if bool(require_ground_truth_direction) and not _cache_has_ground_truth_direction(cached):
                    print(
                        f"[INFO] Cache at {cache_path} lacks ground-truth direction metadata; rebuilding from CARLA."
                    )
                else:
                    print(f"[INFO] Using cached map polylines from {cache_path} (map={cached.get('map_name')})")
                    records = _extract_map_line_records(cached)
                    return cached["lines"], cached.get("bounds"), records
        except Exception:
            pass

    if carla is None:
        raise SystemExit("carla Python module not available; install CARLA egg/wheel or omit --use-carla-map")

    _ingest_s1._assert_carla_endpoint_reachable(host, port, timeout_s=2.0)
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    available_maps = client.get_available_maps()
    world = client.get_world()
    cmap = world.get_map()
    print(f"[INFO] CARLA current map: {cmap.name}")
    print(f"[INFO] CARLA available maps: {', '.join(available_maps)}")

    if expected_town and expected_town not in (cmap.name or ""):
        candidates = [m for m in available_maps if expected_town in m]
        if not candidates:
            raise RuntimeError(f"CARLA map '{cmap.name}' does not match expected '{expected_town}', and no available map matches")
        target_map = candidates[0]
        print(f"[INFO] Loading map '{target_map}' to satisfy expected substring '{expected_town}'")
        world = client.load_world(target_map)
        cmap = world.get_map()
        print(f"[INFO] Loaded map: {cmap.name}")

    wps = cmap.generate_waypoints(distance=sample)
    buckets: Dict[Tuple[int, int], List[carla.Waypoint]] = {}
    for wp in wps:
        key = (wp.road_id, wp.lane_id)
        buckets.setdefault(key, []).append(wp)

    lines: List[List[Tuple[float, float]]] = []
    line_records: List[Dict[str, object]] = []
    bounds = [float("inf"), -float("inf"), float("inf"), -float("inf")]  # minx, maxx, miny, maxy
    for (road_id, lane_id), seq in buckets.items():
        seq.sort(key=lambda w: w.s)  # along-lane distance
        line, point_yaws, point_s, order_flipped, yaw_alignment = _build_travel_ordered_line(seq)
        for (x, y) in line:
            bounds[0] = min(bounds[0], x)
            bounds[1] = max(bounds[1], x)
            bounds[2] = min(bounds[2], y)
            bounds[3] = max(bounds[3], y)
        if len(line) >= 2:
            lines.append(line)
            line_records.append(
                {
                    "points": line,
                    "road_id": int(road_id),
                    "lane_id": int(lane_id),
                    # Lines are explicitly travel-ordered by waypoint yaw.
                    "dir_sign": 1,
                    "point_yaws": [float(v) for v in point_yaws],
                    "point_s": [float(v) for v in point_s],
                    "direction_source": "waypoint_yaw",
                    "travel_ordered": True,
                    "order_flipped_vs_s_asc": bool(order_flipped),
                    "yaw_alignment_score": float(yaw_alignment),
                }
            )

    btuple = None if bounds[0] == float("inf") else tuple(bounds)  # type: ignore

    if cache_path:
        try:
            pickle.dump(
                {
                    "lines": lines,
                    "line_records": line_records,
                    "bounds": btuple,
                    "map_name": cmap.name,
                    "cache_format_version": 2,
                    "directionality_source": "carla_waypoint_yaw",
                    "sample_distance_m": float(sample),
                },
                cache_path.open("wb"),
            )
            print(f"[INFO] Cached map polylines to {cache_path} (map={cmap.name})")
        except Exception:
            pass

    return lines, btuple, line_records


def main() -> None:
    args = parse_args()
    _apply_spawn_preprocess_maximal_profile(args)
    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    yaml_dirs = pick_yaml_dirs(scenario_dir, args.subdir)
    out_dir = Path(args.out_dir or (scenario_dir / "carla_log_export")).resolve()
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)

    # Optional transform overrides from JSON
    if args.coord_json:
        try:
            cfg = json.loads(Path(args.coord_json).read_text(encoding="utf-8"))
            json_tx = float(cfg.get("tx", 0.0))
            json_ty = float(cfg.get("ty", 0.0))
            json_tz = float(cfg.get("tz", 0.0)) if "tz" in cfg else 0.0
            json_theta_deg = (
                float(cfg.get("theta_deg", 0.0))
                if "theta_deg" in cfg
                else float(cfg.get("theta_rad", 0.0)) * 180.0 / math.pi if "theta_rad" in cfg else 0.0
            )
            json_flip = bool(cfg.get("flip_y", False) or cfg.get("y_flip", False))

            # Inverse transform: JSON describes CARLA->PKL; we need PKL->CARLA for XML
            if json_flip:
                args.tx += -json_tx
                args.ty += json_ty
                args.flip_y = True
            else:
                args.tx += -json_tx
                args.ty += -json_ty
            args.tz += -json_tz
            args.yaw_deg += -json_theta_deg

            # Allow XML-only offsets from the same file if present
            args.xml_tx += float(cfg.get("xml_tx", 0.0))
            args.xml_ty += float(cfg.get("xml_ty", 0.0))
        except Exception as exc:
            raise SystemExit(f"Failed to read coord_json {args.coord_json}: {exc}") from exc

    if len(yaml_dirs) > 1:
        print("[INFO] Using multiple YAML subfolders for actor locations:")
        for yd in yaml_dirs:
            print(f"  - {yd}")
        pos_subdirs = [yd for yd in yaml_dirs if not _is_negative_subdir(yd)]
        neg_subdirs = [yd for yd in yaml_dirs if _is_negative_subdir(yd)]
        if pos_subdirs:
            print("[INFO] Ego subfolders (non-negative):")
            for yd in pos_subdirs:
                print(f"  - {yd}")
        if neg_subdirs:
            print("[INFO] Non-ego subfolders (negative):")
            for yd in neg_subdirs:
                print(f"  - {yd}")

    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_trajs: List[List[Waypoint]] = []
    ego_times_list: List[List[float]] = []
    obj_info: Dict[int, Dict[str, object]] = {}

    for yd in yaml_dirs:
        is_negative_subdir = _is_negative_subdir(yd)
        v_map, v_times, ego_traj, ego_times, v_info = build_trajectories(
            yaml_dir=yd,
            dt=args.dt,
            tx=args.tx,
            ty=args.ty,
            tz=args.tz,
            yaw_deg=args.yaw_deg,
            flip_y=args.flip_y,
        )
        if ego_traj and not is_negative_subdir:
            ego_trajs.append(ego_traj)
            ego_times_list.append(ego_times)
        for vid, meta in v_info.items():
            existing = obj_info.get(vid, {})
            if not existing:
                obj_info[vid] = meta
                continue
            # Fill missing fields without overwriting existing obj_type/model
            if not existing.get("obj_type") and meta.get("obj_type"):
                existing["obj_type"] = meta.get("obj_type")
                if meta.get("model"):
                    existing["model"] = meta.get("model")
            if existing.get("length") is None and meta.get("length") is not None:
                existing["length"] = meta.get("length")
            if existing.get("width") is None and meta.get("width") is not None:
                existing["width"] = meta.get("width")
            obj_info[vid] = existing
        for vid, traj in v_map.items():
            if vid not in vehicles or len(traj) > len(vehicles[vid]):
                vehicles[vid] = traj
                vehicle_times[vid] = v_times.get(vid, [])

    if args.ego_only:
        ignored = len(vehicles)
        vehicles = {}
        vehicle_times = {}
        obj_info = {}
        print(f"[INFO] --ego-only enabled: ignoring {ignored} non-ego actors.")

    # Build actor metadata (used for preprocessing and export)
    actor_meta_by_id: Dict[int, Dict[str, object]] = {}
    skipped_non_vehicles = 0
    for vid, traj in vehicles.items():
        if not traj:
            continue
        info = obj_info.get(vid, {})
        obj_type_val = info.get("obj_type")
        if not obj_type_val:
            print(f"[WARN] Missing obj_type for actor id {vid}; defaulting to npc")
            obj_type_raw = "npc"
        else:
            obj_type_raw = str(obj_type_val)
        if not is_vehicle_type(obj_type_raw):
            skipped_non_vehicles += 1
            continue
        kind, is_ped, motion_stats = _classify_actor_kind(
            traj,
            obj_type_raw,
            times=vehicle_times.get(vid),
            default_dt=float(args.dt),
            static_path_threshold=float(getattr(args, "static_path_threshold", 1.2)),
            static_net_disp_threshold=float(getattr(args, "static_net_disp_threshold", 0.8)),
            static_bbox_extent_threshold=float(getattr(args, "static_bbox_extent_threshold", 0.9)),
            static_avg_speed_threshold=float(getattr(args, "static_avg_speed_threshold", 0.8)),
            static_heavy_path_threshold=float(getattr(args, "static_heavy_path_threshold", 8.0)),
            static_heavy_bbox_extent_threshold=float(getattr(args, "static_heavy_bbox_extent_threshold", 1.2)),
            static_heavy_avg_speed_threshold=float(getattr(args, "static_heavy_avg_speed_threshold", 0.8)),
        )
        model = info.get("model") or map_obj_type(obj_type_raw)
        actor_meta_by_id[vid] = {
            "kind": kind,
            "is_pedestrian": is_ped,
            "obj_type": obj_type_raw,
            "model": model,
            "length": info.get("length"),
            "width": info.get("width"),
            "motion_path_m": float(motion_stats.get("path_dist", 0.0)),
            "motion_net_m": float(motion_stats.get("net_disp_xy", 0.0)),
            "motion_bbox_m": float(motion_stats.get("bbox_extent_xy", 0.0)),
            "motion_avg_speed_mps": float(motion_stats.get("avg_speed_mps", 0.0)),
        }

    if skipped_non_vehicles > 0:
        print(f"[INFO] Skipped {skipped_non_vehicles} non-actor objects (props, static objects, etc.)")

    walker_diversity_stats = _diversify_nearby_walker_models(
        actor_meta_by_id=actor_meta_by_id,
        vehicles=vehicles,
        near_distance_m=WALKER_MODEL_NEAR_DISTANCE_M,
    )
    if walker_diversity_stats.get("walkers", 0) > 0:
        print(
            "[INFO] Walker model diversity: "
            f"walkers={walker_diversity_stats.get('walkers', 0)}, "
            f"near_pairs={walker_diversity_stats.get('near_pairs', 0)}, "
            f"same_model_pairs={walker_diversity_stats.get('same_pairs_before', 0)}"
            f"->{walker_diversity_stats.get('same_pairs_after', 0)}, "
            f"models_changed={walker_diversity_stats.get('models_changed', 0)}"
        )

    # Keep a copy for diagnostics/visualization (pre-spawn-preprocess state).
    ego_trajs_pre_align = [
        [
            Waypoint(
                x=float(wp.x),
                y=float(wp.y),
                z=float(wp.z),
                yaw=float(wp.yaw),
                pitch=float(wp.pitch),
                roll=float(wp.roll),
            )
            for wp in traj
        ]
        for traj in ego_trajs
    ]
    vehicles_pre_align: Dict[int, List[Waypoint]] = {
        int(vid): [
            Waypoint(
                x=float(wp.x),
                y=float(wp.y),
                z=float(wp.z),
                yaw=float(wp.yaw),
                pitch=float(wp.pitch),
                roll=float(wp.roll),
            )
            for wp in traj
        ]
        for vid, traj in vehicles.items()
    }

    if args.spawn_preprocess:
        report = _preprocess_spawn_positions(
            vehicles,
            vehicle_times,
            actor_meta_by_id,
            args,
            ego_trajs=ego_trajs,
            ego_times_list=ego_times_list,
        )
        spawn_report = report
        if args.spawn_preprocess_report:
            report_path = Path(args.spawn_preprocess_report)
            if not report_path.is_absolute():
                report_path = out_dir / report_path
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"[INFO] Spawn preprocess report written to {report_path}")
            except Exception as exc:
                print(f"[WARN] Failed to write spawn preprocess report: {exc}")

    # Export buffers (can diverge from raw trajectories when applying timing overrides).
    vehicles_export: Dict[int, List[Waypoint]] = {
        int(vid): [_copy_waypoint(wp) for wp in traj]
        for vid, traj in vehicles.items()
    }
    vehicle_times_export: Dict[int, List[float]] = {
        int(vid): _ensure_times(vehicles_export[int(vid)], vehicle_times.get(int(vid)), args.dt)
        for vid in vehicles_export.keys()
    }

    early_spawn_report: Dict[str, object] = {
        "enabled": bool(args.maximize_safe_early_spawn),
        "applied": False,
        "reason": None,
        "safety_margin": float(args.early_spawn_safety_margin),
        "adjusted_actor_ids": [],
        "adjusted_spawn_times": {},
    }
    if args.maximize_safe_early_spawn:
        if not args.encode_timing:
            early_spawn_report["reason"] = "encode_timing_disabled"
            print(
                "[INFO] Early-spawn optimization skipped because --encode-timing is disabled."
            )
        elif not actor_meta_by_id:
            early_spawn_report["reason"] = "no_actor_metadata"
            print("[INFO] Early-spawn optimization skipped because there are no actor trajectories.")
        else:
            selected_spawn_times, selection_report = _maximize_safe_early_spawn_actors(
                vehicles=vehicles_export,
                vehicle_times=vehicle_times_export,
                actor_meta=actor_meta_by_id,
                dt=float(args.dt),
                safety_margin=float(args.early_spawn_safety_margin),
            )
            (
                vehicles_export,
                vehicle_times_export,
                adjusted_ids,
                applied_spawn_times,
            ) = _apply_early_spawn_time_overrides(
                vehicles=vehicles_export,
                vehicle_times=vehicle_times_export,
                early_spawn_times=selected_spawn_times,
                dt=float(args.dt),
            )
            early_spawn_report.update(selection_report)
            early_spawn_report["applied"] = True
            early_spawn_report["adjusted_actor_ids"] = adjusted_ids
            early_spawn_report["adjusted_spawn_times"] = {
                str(int(vid)): float(t) for vid, t in sorted(applied_spawn_times.items())
            }
            print(
                "[EARLY_SPAWN] candidates={} adjusted={} at_t0={} dynamic_limited={} "
                "static_conflict_adjustments={} avg_advance={:.3f}s".format(
                    int(selection_report.get("candidates", 0)),
                    len(adjusted_ids),
                    int(selection_report.get("spawn_at_t0", 0)),
                    int(selection_report.get("dynamic_limited", 0)),
                    int(selection_report.get("static_conflict_adjustments", 0)),
                    float(selection_report.get("avg_advance_seconds", 0.0)),
                )
            )
    else:
        early_spawn_report["reason"] = "disabled_by_flag"

    if args.early_spawn_report:
        report_path = Path(args.early_spawn_report)
        if not report_path.is_absolute():
            report_path = out_dir / report_path
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(early_spawn_report, indent=2), encoding="utf-8")
            print(f"[INFO] Early-spawn report written to {report_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write early-spawn report: {exc}")

    late_despawn_report: Dict[str, object] = {
        "enabled": bool(args.maximize_safe_late_despawn),
        "applied": False,
        "reason": None,
        "safety_margin": float(args.late_despawn_safety_margin),
        "adjusted_actor_ids": [],
        "hold_until_time": None,
    }
    if args.maximize_safe_late_despawn:
        if not args.encode_timing:
            late_despawn_report["reason"] = "encode_timing_disabled"
            print(
                "[INFO] Late-despawn optimization skipped because --encode-timing is disabled."
            )
        elif not actor_meta_by_id:
            late_despawn_report["reason"] = "no_actor_metadata"
            print("[INFO] Late-despawn optimization skipped because there are no actor trajectories.")
        else:
            horizon_candidates: List[float] = []
            for times in vehicle_times_export.values():
                if times:
                    horizon_candidates.append(float(times[-1]))
            for ego_times in ego_times_list:
                if ego_times:
                    horizon_candidates.append(float(ego_times[-1]))
            hold_until_time = max(horizon_candidates) if horizon_candidates else 0.0
            late_despawn_report["hold_until_time"] = float(hold_until_time)
            if hold_until_time <= 0.0:
                late_despawn_report["reason"] = "non_positive_horizon"
            else:
                selected_ids, selection_report = _maximize_safe_late_despawn_actors(
                    vehicles=vehicles_export,
                    vehicle_times=vehicle_times_export,
                    actor_meta=actor_meta_by_id,
                    dt=float(args.dt),
                    safety_margin=float(args.late_despawn_safety_margin),
                    hold_until_time=float(hold_until_time),
                )
                vehicles_export, vehicle_times_export, adjusted_ids = _apply_late_despawn_time_overrides(
                    vehicles=vehicles_export,
                    vehicle_times=vehicle_times_export,
                    selected_late_hold_ids=selected_ids,
                    dt=float(args.dt),
                    hold_until_time=float(hold_until_time),
                )
                late_despawn_report.update(selection_report)
                late_despawn_report["applied"] = True
                late_despawn_report["adjusted_actor_ids"] = adjusted_ids
                print(
                    "[LATE_DESPAWN] candidates={} safe={} selected={} adjusted={} "
                    "pair_conflicts={} timeout_components={} horizon={:.3f}s".format(
                        int(selection_report.get("candidates", 0)),
                        int(selection_report.get("individually_safe", 0)),
                        int(selection_report.get("selected", 0)),
                        len(adjusted_ids),
                        int(selection_report.get("pair_conflicts", 0)),
                        int(selection_report.get("timed_out_components", 0)),
                        float(hold_until_time),
                    )
                )
    else:
        late_despawn_report["reason"] = "disabled_by_flag"

    if args.late_despawn_report:
        report_path = Path(args.late_despawn_report)
        if not report_path.is_absolute():
            report_path = out_dir / report_path
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(late_despawn_report, indent=2), encoding="utf-8")
            print(f"[INFO] Late-despawn report written to {report_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write late-despawn report: {exc}")

    parked_clearance_report: Dict[str, object] = {}
    vehicles_export, parked_clearance_report = _apply_parked_vehicle_path_clearance(
        vehicles_aligned=vehicles_export,
        vehicle_times_aligned=vehicle_times_export,
        vehicles_original=vehicles_pre_align,
        vehicle_times_original=vehicle_times,
        actor_meta=actor_meta_by_id,
        args=args,
    )

    # Write ego route (optional)
    ego_entries: List[dict] = []
    if not args.no_ego and ego_trajs:
        # Remove legacy ego_route.xml to avoid double-counting egos
        legacy_ego = out_dir / "ego_route.xml"
        if legacy_ego.exists():
            try:
                legacy_ego.unlink()
                print(f"[INFO] Removed legacy ego file {legacy_ego}")
            except Exception:
                pass
        for ego_idx, ego_traj in enumerate(ego_trajs):
            ego_times = ego_times_list[ego_idx] if ego_idx < len(ego_times_list) else []
            # Follow CustomRoutes naming: {town}_custom_ego_vehicle_{i}.xml
            ego_xml = out_dir / f"{args.town.lower()}_custom_ego_vehicle_{ego_idx}.xml"
            write_route_xml(
                ego_xml,
                route_id=args.route_id,
                role="ego",
                town=args.town,
                waypoints=ego_traj,
                times=ego_times if args.encode_timing else None,
                snap_to_road=False,
                xml_tx=args.xml_tx,
                xml_ty=args.xml_ty,
            )
            ego_entries.append({
                "file": ego_xml.name,
                "route_id": str(args.route_id),
                "town": args.town,
                "name": ego_xml.stem,
                "kind": "ego",
                "model": args.ego_model,
            })

    # Build actor entries after we know obj_type/model
    # Group by kind (npc, static, etc.) for manifest
    actors_by_kind: Dict[str, List[dict]] = {}
    actor_kind_by_id: Dict[int, str] = {}
    actor_xml_by_id: Dict[int, Path] = {}
    frozen_static_exports = 0

    if args.ego_only and actors_dir.exists():
        removed = 0
        for stale in actors_dir.rglob("*.xml"):
            try:
                stale.unlink()
                removed += 1
            except Exception:
                pass
        if removed > 0:
            print(f"[INFO] Removed {removed} stale actor XML files because --ego-only is enabled.")

    for vid, traj in vehicles_export.items():
        if not traj:
            continue
        meta = actor_meta_by_id.get(vid)
        if meta is None:
            continue
        obj_type_raw = str(meta.get("obj_type") or "npc")
        kind = str(meta.get("kind"))
        model = meta.get("model") or map_obj_type(obj_type_raw)
        length = meta.get("length")
        width = meta.get("width")
        
        # Use obj_type directly for actor type in filename
        # Clean it up to make it suitable for filenames
        actor_type = obj_type_raw.replace(" ", "_").replace("-", "_").title()
        if not actor_type or actor_type.lower() == "npc":
            actor_type = "Vehicle"
        
        # Follow CustomRoutes naming: {town}_custom_{ActorType}_{id}_{kind}.xml
        name = f"{args.town.lower()}_custom_{actor_type}_{vid}_{kind}"
        
        # Create subdirectory for actor kind
        kind_dir = actors_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        actor_xml = kind_dir / f"{name}.xml"

        traj_for_export = traj
        times_for_export = vehicle_times_export.get(vid) if args.encode_timing else None
        if kind in ("static", "walker_static") and len(traj) > 1:
            # Keep parked/static actors fixed over time to avoid replaying lidar jitter.
            anchor = _copy_waypoint(traj[0])
            traj_for_export = [_copy_waypoint(anchor) for _ in range(len(traj))]
            if times_for_export and len(times_for_export) != len(traj_for_export):
                times_for_export = _ensure_times(traj_for_export, times_for_export, args.dt)
            frozen_static_exports += 1

        write_route_xml(
            actor_xml,
            route_id=args.route_id,
            role=kind,
            town=args.town,
            waypoints=traj_for_export,
            times=times_for_export,
            snap_to_road=args.snap_to_road is True,
            xml_tx=args.xml_tx,
            xml_ty=args.xml_ty,
        )
        actor_xml_by_id[vid] = actor_xml
        speed = 0.0
        if len(traj_for_export) >= 2:
            dist = 0.0
            for a, b in zip(traj_for_export, traj_for_export[1:]):
                dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
            if args.encode_timing:
                if times_for_export and len(times_for_export) == len(traj_for_export):
                    total_time = times_for_export[-1] - times_for_export[0]
                    if total_time > 1e-6:
                        speed = dist / total_time
                    else:
                        speed = dist / max(args.dt * (len(traj_for_export) - 1), 1e-6)
                else:
                    speed = dist / max(args.dt * (len(traj_for_export) - 1), 1e-6)
            else:
                speed = dist / max(args.dt * (len(traj_for_export) - 1), 1e-6)
        
        entry = {
            "file": str(actor_xml.relative_to(out_dir)),
            "route_id": str(args.route_id),
            "town": args.town,
            "name": name,
            "kind": kind,
            "model": model,
        }
        
        # Add optional fields
        if speed > 0:
            entry["speed"] = speed
        if length is not None:
            entry["length"] = str(length) if isinstance(length, (int, float)) else length
        if width is not None:
            entry["width"] = str(width) if isinstance(width, (int, float)) else width
        
        if kind not in actors_by_kind:
            actors_by_kind[kind] = []
        actors_by_kind[kind].append(entry)
        actor_kind_by_id[vid] = kind

    if frozen_static_exports > 0:
        print(
            f"[INFO] Froze waypoint jitter for {frozen_static_exports} static actor trajectories "
            "(position held fixed across timestamps)."
        )

    save_manifest(out_dir / "actors_manifest.json", actors_by_kind, ego_entries)

    # Optional visualization
    if (
        args.gif
        or args.paths_png
        or args.spawn_viz
        or args.actor_yaw_viz_ids
        or args.actor_raw_yaml_viz_ids
        or args.ego_alignment_viz
        or args.ego_alignment_bev_viz
        or args.actor_alignment_bev_viz
        or args.spawn_preprocess_fail_viz
    ):
        if plt is None or (args.gif and imageio is None):
            raise SystemExit("matplotlib (and imageio for GIF) are required for visualization")
        map_lines: List[List[Tuple[float, float]]] = []
        map_line_records: List[Dict[str, object]] = []
        map_bounds = None
        # Priority: explicit map pickle -> CARLA live map (with cache) -> none
        if args.map_pkl:
            try:
                map_obj = pickle.load(Path(args.map_pkl).expanduser().open("rb"))
                map_lines = _extract_map_lines(map_obj, out=[])
                map_line_records = _extract_map_line_records(map_obj)
                if isinstance(map_obj, dict):
                    map_bounds = map_obj.get("bounds")
                print(f"[INFO] Loaded {len(map_lines)} polylines from {args.map_pkl}")
            except Exception as exc:
                print(f"[WARN] Failed to load map pickle {args.map_pkl}: {exc}")
        elif args.use_carla_map:
            cache_path = Path(args.carla_cache or (out_dir / "carla_map_cache.pkl"))
            try:
                sample = float(args.carla_sample)
                if args.spawn_preprocess_fail_viz:
                    sample = min(sample, float(args.spawn_preprocess_fail_viz_sample))
                map_lines, map_bounds, map_line_records = fetch_carla_map_lines(
                    host=args.carla_host,
                    port=args.carla_port,
                    sample=sample,
                    cache_path=cache_path,
                    expected_town=args.expected_town,
                )
                if map_lines:
                    print(f"[INFO] Loaded {len(map_lines)} map polylines from CARLA ({args.carla_host}:{args.carla_port})")
            except Exception as exc:
                print(f"[WARN] Failed to fetch map from CARLA: {exc}")
                map_bounds = None
        else:
            map_bounds = None

        map_image = None
        map_image_bounds = None
        if args.map_image:
            try:
                map_image = plt.imread(args.map_image)
                if args.map_image_bounds:
                    map_image_bounds = tuple(float(v) for v in args.map_image_bounds)  # type: ignore
                elif map_bounds:
                    map_image_bounds = map_bounds
            except Exception as exc:
                print(f"[WARN] Failed to load map image {args.map_image}: {exc}")
                map_image = None
                map_image_bounds = None

        if args.gif:
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            max_len = max((len(t) for t in vehicles.values()), default=0)
            for et in ego_trajs:
                max_len = max(max_len, len(et))
            axes_limits = None
            # Precompute global limits for stable camera
            xs: List[float] = []
            ys: List[float] = []
            for traj in vehicles.values():
                xs.extend([wp.x for wp in traj])
                ys.extend([wp.y for wp in traj])
            for et in ego_trajs:
                for wp in et:
                    xs.append(wp.x)
                    ys.append(wp.y)
            for line in map_lines:
                for x, y in line:
                    xs.append(x)
                    ys.append(y)
            if xs and ys:
                pad = max(0.0, float(args.axis_pad))
                axes_limits = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

            for i in range(max_len):
                plot_frame(
                    i,
                    vehicles,
                    ego_trajs,
                    frames_dir / f"frame_{i:06d}.png",
                    axes_limits,
                    map_lines=map_lines,
                    invert_plot_y=args.invert_plot_y,
                )
            gif_path = Path(args.gif_path or (out_dir / "replay.gif"))
            write_gif(frames_dir, gif_path)
            print(f"[OK] GIF written to {gif_path}")

        if args.paths_png:
            png_path = Path(args.paths_png).expanduser()
            write_paths_png(
                actors_by_id=vehicles,
                ego_trajs=ego_trajs,
                map_lines=map_lines,
                out_path=png_path,
                axis_pad=float(args.axis_pad),
                invert_plot_y=args.invert_plot_y,
            )
            print(f"[OK] Paths PNG written to {png_path}")

        if args.actor_yaw_viz_ids:
            actor_ids = _parse_id_list(args.actor_yaw_viz_ids)
            out_dir_yaw = Path(args.actor_yaw_viz_dir or (out_dir / "actor_yaw_viz")).expanduser()
            out_dir_yaw.mkdir(parents=True, exist_ok=True)
            for vid in actor_ids:
                gt_traj = vehicles.get(vid) or []
                xml_path = actor_xml_by_id.get(vid)
                if xml_path is None:
                    # best-effort fallback search
                    matches = list(actors_dir.rglob(f"*_{vid}_*.xml"))
                    xml_path = matches[0] if matches else None
                if not gt_traj:
                    print(f"[WARN] No GT trajectory found for actor id {vid}")
                    continue
                if xml_path is None or not xml_path.exists():
                    print(f"[WARN] No XML found for actor id {vid}")
                    continue
                xml_traj = parse_route_xml(xml_path)
                if not xml_traj:
                    print(f"[WARN] XML had no waypoints for actor id {vid}: {xml_path}")
                    continue
                out_path = out_dir_yaw / f"actor_{vid}_yaw_viz.png"
                write_actor_yaw_viz(
                    actor_id=vid,
                    gt_traj=gt_traj,
                    xml_traj=xml_traj,
                    map_lines=map_lines,
                    out_path=out_path,
                    arrow_step=max(1, int(args.actor_yaw_viz_step)),
                    arrow_len=float(args.actor_yaw_viz_arrow_len),
                    pad=float(args.actor_yaw_viz_pad),
                    invert_plot_y=args.invert_plot_y,
                )
                print(f"[OK] Actor yaw viz written to {out_path}")

        if args.actor_raw_yaml_viz_ids:
            actor_ids = _parse_id_list(args.actor_raw_yaml_viz_ids)
            out_dir_raw = Path(args.actor_raw_yaml_viz_dir or (out_dir / "actor_raw_yaml_viz")).expanduser()
            out_dir_raw.mkdir(parents=True, exist_ok=True)

            # Collect per-subdir points directly from YAML (with transform + XML offsets applied)
            points_by_actor: Dict[int, Dict[str, List[Tuple[float, float, float]]]] = {vid: {} for vid in actor_ids}
            for yd in yaml_dirs:
                sub_name = yd.name
                yaml_paths = list_yaml_timesteps(yd)
                for idx, path in enumerate(yaml_paths):
                    try:
                        frame_idx = int(path.stem)
                    except Exception:
                        frame_idx = idx
                    t = float(frame_idx) * float(args.dt)
                    data = load_yaml(path)
                    vehs = data.get("vehicles", {}) or {}
                    for vid in actor_ids:
                        payload = vehs.get(vid) if vid in vehs else vehs.get(str(vid))
                        if not payload:
                            continue
                        loc = payload.get("location") or [0, 0, 0]
                        x0 = float(loc[0]) if len(loc) > 0 else 0.0
                        y0 = float(loc[1]) if len(loc) > 1 else 0.0
                        x, y = apply_se2((x0, y0), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y)
                        x += float(args.xml_tx)
                        y += float(args.xml_ty)
                        points_by_actor.setdefault(vid, {}).setdefault(sub_name, []).append((x, y, t))

            for vid in actor_ids:
                points = points_by_actor.get(vid, {})
                if not points:
                    print(f"[WARN] No YAML points found for actor id {vid}")
                    continue
                out_path = out_dir_raw / f"actor_{vid}_raw_yaml_points.png"
                write_actor_raw_yaml_viz(
                    actor_id=vid,
                    points_by_subdir=points,
                    map_lines=map_lines,
                    out_path=out_path,
                    pad=float(args.actor_raw_yaml_viz_pad),
                    invert_plot_y=args.invert_plot_y,
                )
                print(f"[OK] Actor raw YAML viz written to {out_path}")

        if args.ego_alignment_viz:
            out_dir_ego = Path(args.ego_alignment_viz_dir or (out_dir / "ego_alignment_viz")).expanduser()
            out_dir_ego.mkdir(parents=True, exist_ok=True)
            n_pre = len(ego_trajs_pre_align)
            n_post = len(ego_trajs)
            if n_pre == 0 or n_post == 0:
                print("[WARN] Ego alignment viz requested but no ego trajectories are available.")
            else:
                if n_pre != n_post:
                    print(f"[WARN] Ego count mismatch for viz: pre={n_pre} post={n_post}; using min count.")
                for ego_idx in range(min(n_pre, n_post)):
                    out_path = out_dir_ego / f"ego_{ego_idx}_pre_vs_post_alignment.png"
                    write_ego_alignment_viz(
                        ego_idx=ego_idx,
                        pre_align_traj=ego_trajs_pre_align[ego_idx],
                        post_align_traj=ego_trajs[ego_idx],
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    print(f"[OK] Ego alignment viz written to {out_path}")

        if args.ego_alignment_bev_viz:
            out_dir_ego_bev = Path(args.ego_alignment_bev_viz_dir or (out_dir / "ego_alignment_bev_viz")).expanduser()
            out_dir_ego_bev.mkdir(parents=True, exist_ok=True)
            n_pre = len(ego_trajs_pre_align)
            n_post = len(ego_trajs)
            if n_pre == 0 or n_post == 0:
                print("[WARN] Ego BEV alignment viz requested but no ego trajectories are available.")
            else:
                if map_image is None and not bool(args.ego_alignment_bev_capture_from_carla):
                    print("[WARN] Ego BEV viz requested without --map-image; using vector map fallback instead of raster CARLA image.")
                if n_pre != n_post:
                    print(f"[WARN] Ego count mismatch for BEV viz: pre={n_pre} post={n_post}; using min count.")
                for ego_idx in range(min(n_pre, n_post)):
                    out_path = out_dir_ego_bev / f"ego_{ego_idx}_alignment_bev_nodes.png"
                    pre_xy = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for wp in ego_trajs_pre_align[ego_idx]
                    ]
                    post_xy = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for wp in ego_trajs[ego_idx]
                    ]
                    ego_bounds = _bounds_from_points(pre_xy + post_xy)
                    captured_bev = None
                    if (
                        bool(args.ego_alignment_bev_capture_from_carla)
                        and ego_bounds is not None
                        and carla is not None
                        and np is not None
                    ):
                        try:
                            captured_bev = _capture_carla_topdown_bev(
                                host=str(args.carla_host),
                                port=int(args.carla_port),
                                bounds=ego_bounds,
                                image_w=int(args.ego_alignment_bev_capture_width),
                                image_h=int(args.ego_alignment_bev_capture_height),
                                fov_deg=float(args.ego_alignment_bev_capture_fov),
                                margin_scale=float(args.ego_alignment_bev_capture_margin),
                                expected_town=str(args.expected_town),
                            )
                            if captured_bev is None:
                                print(f"[WARN] Ego {ego_idx} CARLA BEV capture returned no image; using fallback background.")
                        except Exception as exc:
                            print(f"[WARN] Ego {ego_idx} CARLA BEV capture failed: {exc}; using fallback background.")
                            captured_bev = None
                    write_ego_alignment_bev_viz(
                        ego_idx=ego_idx,
                        pre_align_traj=ego_trajs_pre_align[ego_idx],
                        post_align_traj=ego_trajs[ego_idx],
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        map_image=map_image,
                        map_image_bounds=map_image_bounds,
                        captured_bev=captured_bev,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        node_step=int(args.ego_alignment_bev_node_step),
                        match_radius=float(args.ego_alignment_bev_match_radius),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    print(f"[OK] Ego BEV alignment viz written to {out_path}")

        if args.actor_alignment_bev_viz:
            actor_ids_all = sorted(set(int(v) for v in vehicles_pre_align.keys()) | set(int(v) for v in vehicles.keys()))
            if not actor_ids_all:
                print("[WARN] Actor BEV alignment viz skipped: no actor trajectories available.")
            else:
                out_dir_actor_bev = Path(
                    args.actor_alignment_bev_viz_dir or (out_dir / "actor_alignment_bev_viz")
                ).expanduser()
                out_dir_actor_bev.mkdir(parents=True, exist_ok=True)
                out_dir_actor_indiv = out_dir_actor_bev / "individual"
                out_dir_actor_indiv.mkdir(parents=True, exist_ok=True)

                if map_image is None and not bool(args.ego_alignment_bev_capture_from_carla):
                    print("[WARN] Actor BEV viz requested without CARLA capture and without --map-image; using vector map fallback.")

                walker_ids = [
                    vid
                    for vid in actor_ids_all
                    if str((actor_meta_by_id.get(vid) or {}).get("kind", "")) in ("walker", "walker_static")
                ]
                npc_ids = [
                    vid
                    for vid in actor_ids_all
                    if str((actor_meta_by_id.get(vid) or {}).get("kind", "")) == "npc"
                ]
                subset_specs = [
                    ("all_actors", "Actors: all", actor_ids_all),
                    ("all_walkers", "Actors: all walkers", walker_ids),
                    ("all_npc", "Actors: all NPC vehicles", npc_ids),
                ]

                for key, title, ids in subset_specs:
                    if not ids:
                        continue
                    pre_subset = {vid: vehicles_pre_align.get(vid, []) for vid in ids}
                    post_subset = {vid: vehicles.get(vid, []) for vid in ids}
                    pre_pts = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for vid in ids
                        for wp in pre_subset.get(vid, [])
                    ]
                    post_pts = [
                        (wp.x + float(args.xml_tx), wp.y + float(args.xml_ty))
                        for vid in ids
                        for wp in post_subset.get(vid, [])
                    ]
                    subset_bounds = _bounds_from_points(pre_pts + post_pts)
                    captured_bev = None
                    if (
                        bool(args.ego_alignment_bev_capture_from_carla)
                        and subset_bounds is not None
                        and carla is not None
                        and np is not None
                    ):
                        try:
                            captured_bev = _capture_carla_topdown_bev(
                                host=str(args.carla_host),
                                port=int(args.carla_port),
                                bounds=subset_bounds,
                                image_w=int(args.ego_alignment_bev_capture_width),
                                image_h=int(args.ego_alignment_bev_capture_height),
                                fov_deg=float(args.ego_alignment_bev_capture_fov),
                                margin_scale=float(args.ego_alignment_bev_capture_margin),
                                expected_town=str(args.expected_town),
                            )
                        except Exception as exc:
                            print(f"[WARN] Actor subset '{key}' CARLA BEV capture failed: {exc}; using fallback background.")
                            captured_bev = None
                    out_path = out_dir_actor_bev / f"{key}_alignment_bev_nodes.png"
                    write_actor_alignment_bev_viz(
                        title=title,
                        pre_trajs=pre_subset,
                        post_trajs=post_subset,
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        map_image=map_image,
                        map_image_bounds=map_image_bounds,
                        captured_bev=captured_bev,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        node_step=int(args.ego_alignment_bev_node_step),
                        match_radius=float(args.ego_alignment_bev_match_radius),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    print(f"[OK] Actor BEV alignment viz written to {out_path}")

                for idx, vid in enumerate(actor_ids_all):
                    pre_single = {vid: vehicles_pre_align.get(vid, [])}
                    post_single = {vid: vehicles.get(vid, [])}
                    pre_pts = [(wp.x + float(args.xml_tx), wp.y + float(args.xml_ty)) for wp in pre_single.get(vid, [])]
                    post_pts = [(wp.x + float(args.xml_tx), wp.y + float(args.xml_ty)) for wp in post_single.get(vid, [])]
                    single_bounds = _bounds_from_points(pre_pts + post_pts)
                    captured_bev = None
                    if (
                        bool(args.ego_alignment_bev_capture_from_carla)
                        and single_bounds is not None
                        and carla is not None
                        and np is not None
                    ):
                        try:
                            captured_bev = _capture_carla_topdown_bev(
                                host=str(args.carla_host),
                                port=int(args.carla_port),
                                bounds=single_bounds,
                                image_w=int(args.ego_alignment_bev_capture_width),
                                image_h=int(args.ego_alignment_bev_capture_height),
                                fov_deg=float(args.ego_alignment_bev_capture_fov),
                                margin_scale=float(args.ego_alignment_bev_capture_margin),
                                expected_town=str(args.expected_town),
                            )
                        except Exception:
                            captured_bev = None

                    kind = str((actor_meta_by_id.get(vid) or {}).get("kind", "actor"))
                    out_path = out_dir_actor_indiv / f"actor_{vid}_{kind}_alignment_bev_nodes.png"
                    write_actor_alignment_bev_viz(
                        title=f"Actor {vid} ({kind}) alignment",
                        pre_trajs=pre_single,
                        post_trajs=post_single,
                        map_lines=map_lines,
                        map_line_records=map_line_records,
                        map_image=map_image,
                        map_image_bounds=map_image_bounds,
                        captured_bev=captured_bev,
                        out_path=out_path,
                        xml_tx=float(args.xml_tx),
                        xml_ty=float(args.xml_ty),
                        pad=float(args.ego_alignment_viz_pad),
                        node_step=int(args.ego_alignment_bev_node_step),
                        match_radius=float(args.ego_alignment_bev_match_radius),
                        invert_plot_y=bool(args.invert_plot_y),
                    )
                    if idx % 10 == 0 or idx == len(actor_ids_all) - 1:
                        print(f"[OK] Actor individual BEV viz progress: {idx + 1}/{len(actor_ids_all)}")

        if args.spawn_viz:
            spawn_viz_path = Path(args.spawn_viz_path or (out_dir / "spawn_alignment_viz.png")).expanduser()

            aligned_points: Dict[int, Tuple[float, float]] = {}
            spawn_points: Dict[int, Tuple[float, float]] = {}
            pre_align_points: Dict[int, Tuple[float, float]] = {}
            for vid, traj in vehicles.items():
                if not traj:
                    continue
                wp0 = traj[0]
                aligned_points[vid] = (wp0.x, wp0.y)
                spawn_points[vid] = (wp0.x + args.xml_tx, wp0.y + args.xml_ty)
                pre_align_points[vid] = invert_se2((wp0.x, wp0.y), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y)

            ego_aligned: List[Tuple[float, float]] = []
            ego_spawn: List[Tuple[float, float]] = []
            ego_pre_align: List[Tuple[float, float]] = []
            for ego_traj in ego_trajs:
                if not ego_traj:
                    continue
                wp0 = ego_traj[0]
                ego_aligned.append((wp0.x, wp0.y))
                ego_spawn.append((wp0.x + args.xml_tx, wp0.y + args.xml_ty))
                ego_pre_align.append(invert_se2((wp0.x, wp0.y), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y))

            offset_pair = None
            offset_label = None
            offset_ref = _pick_offset_reference(aligned_points, pre_align_points, ego_aligned, ego_pre_align)
            if offset_ref:
                offset_pair = (offset_ref[0], offset_ref[1])
                offset_label = f"pre-align ref ({offset_ref[2]})"

            xodr_points: List[Tuple[float, float]] = []
            xodr_path = Path(args.xodr).expanduser() if args.xodr else None
            if xodr_path and xodr_path.exists():
                try:
                    xodr_points = load_xodr_points(xodr_path, args.xodr_step)
                    print(f"[INFO] Loaded {len(xodr_points)} XODR points from {xodr_path}")
                except Exception as exc:
                    print(f"[WARN] Failed to load XODR {xodr_path}: {exc}")
                    xodr_points = []
            else:
                # Best-effort default XODR in v2xpnp/ (if present)
                default_xodr = Path(__file__).resolve().parents[1] / "map" / "ucla_v2.xodr"
                if default_xodr.exists():
                    try:
                        xodr_points = load_xodr_points(default_xodr, args.xodr_step)
                        print(f"[INFO] Loaded {len(xodr_points)} XODR points from {default_xodr}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load XODR {default_xodr}: {exc}")

            map_points: List[Tuple[float, float]] = []
            for line in map_lines:
                map_points.extend(line)

            kind_by_id = dict(actor_kind_by_id)
            for vid in aligned_points.keys():
                kind_by_id.setdefault(vid, "npc")

            bounds = _merge_bounds(
                [
                    map_bounds,
                    _bounds_from_points(map_points),
                    _bounds_from_points(xodr_points),
                    _bounds_from_points(
                        list(aligned_points.values())
                        + list(spawn_points.values())
                        + list(pre_align_points.values())
                        + ego_aligned
                        + ego_spawn
                        + ego_pre_align
                    ),
                    map_image_bounds,
                ]
            )
            if bounds:
                pad = max(0.0, float(args.axis_pad))
                minx, maxx, miny, maxy = bounds
                bounds = (minx - pad, maxx + pad, miny - pad, maxy + pad)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            ax_map, ax_xodr = axes

            # Left: CARLA map layer (image if provided, else polylines)
            if map_image is not None:
                if map_image_bounds:
                    minx, maxx, miny, maxy = map_image_bounds
                elif bounds:
                    minx, maxx, miny, maxy = bounds
                else:
                    minx, maxx, miny, maxy = 0.0, 1.0, 0.0, 1.0
                ax_map.imshow(map_image, extent=(minx, maxx, miny, maxy), origin="lower", alpha=0.8, zorder=0)
            elif map_lines:
                _plot_background_lines(ax_map, map_lines, color="#9e9e9e", lw=0.8, alpha=0.6)
            _plot_spawn_alignment(
                ax_map,
                aligned_points,
                spawn_points,
                kind_by_id,
                ego_aligned,
                ego_spawn,
                title="CARLA Map Layer",
                offset_pair=offset_pair,
                offset_label=offset_label,
            )

            # Right: XODR layer
            if xodr_points:
                ax_xodr.scatter(
                    [p[0] for p in xodr_points],
                    [p[1] for p in xodr_points],
                    s=1,
                    c="#1f77b4",
                    alpha=0.5,
                    label="XODR geometry",
                    zorder=1,
                )
            _plot_spawn_alignment(
                ax_xodr,
                aligned_points,
                spawn_points,
                kind_by_id,
                ego_aligned,
                ego_spawn,
                title="XODR Layer",
                offset_pair=offset_pair,
                offset_label=offset_label,
            )

            for ax in axes:
                if bounds:
                    minx, maxx, miny, maxy = bounds
                    ax.set_xlim(minx, maxx)
                    ax.set_ylim(miny, maxy)
                if args.invert_plot_y:
                    ax.invert_yaxis()

            # Global legend and info
            handles, labels = ax_map.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", frameon=True)

            info_lines = [
                f"Actors: {len(aligned_points)} (npc={sum(1 for k in kind_by_id.values() if k == 'npc')}, "
                f"static={sum(1 for k in kind_by_id.values() if k == 'static')}, "
                f"walker={sum(1 for k in kind_by_id.values() if k in ('walker', 'walker_static'))})",
                f"Egos: {len(ego_aligned)}",
                f"Alignment tx/ty/yaw: {args.tx:.2f}, {args.ty:.2f}, {args.yaw_deg:.2f}",
                f"XML offset xml_tx/xml_ty: {args.xml_tx:.2f}, {args.xml_ty:.2f}",
                f"flip_y: {args.flip_y}",
            ]
            fig.text(0.01, 0.01, "\n".join(info_lines), fontsize=9, ha="left", va="bottom")
            fig.suptitle("Spawn vs Aligned Positions (CARLA vs XODR)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(spawn_viz_path, dpi=180)
            plt.close(fig)
            print(f"[OK] Spawn alignment visualization written to {spawn_viz_path}")

        if args.spawn_preprocess_fail_viz:
            if spawn_report is None:
                print("[WARN] Spawn preprocess visualization requested but no report available.")
            else:
                fail_dir = args.spawn_preprocess_fail_viz_dir
                if not fail_dir:
                    fail_dir = str(out_dir / "spawn_preprocess_fail_viz")
                out_path = Path(fail_dir).expanduser()
                _plot_failed_spawn_visualizations(
                    report=spawn_report,
                    map_lines=map_lines,
                    out_dir=out_path,
                    window_m=float(args.spawn_preprocess_fail_viz_window),
                    dpi=int(args.spawn_preprocess_fail_viz_dpi),
                )
                print(f"[OK] Failed spawn visualization written to {out_path}")

    # Optional: run custom eval with generated routes
    if args.run_custom_eval:
        repo_root = Path(__file__).resolve().parents[2]
        python_bin = sys.executable
        cmd = [
            python_bin,
            str(repo_root / "tools" / "run_custom_eval.py"),
            "--routes-dir",
            str(out_dir),
            "--port",
            str(args.eval_port),
            "--overwrite",
        ]
        if args.eval_planner:
            cmd.extend(["--planner", args.eval_planner])
        print("[INFO] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] run_custom_eval failed with exit code {exc.returncode}")

    print(f"[OK] Export complete -> {out_dir}")
    print("Files:")
    if ego_entries:
        for entry in ego_entries:
            print(f"  - {entry['file']}")
    print(f"  - actors_manifest.json")
    total_actors = sum(len(entries) for entries in actors_by_kind.values())
    if args.ego_only:
        print("  - actors/*/*.xml (skipped by --ego-only)")
    else:
        print(f"  - actors/*/*.xml ({total_actors} actors across {len(actors_by_kind)} categories)")
        for kind, entries in sorted(actors_by_kind.items()):
            print(f"    - {kind}: {len(entries)} actors")
    if args.gif:
        print(f"  - replay.gif")


if __name__ == "__main__":
    main()

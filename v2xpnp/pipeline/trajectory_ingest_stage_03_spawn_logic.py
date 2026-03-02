"""Trajectory ingest internals: spawn logic and safety checks."""

from __future__ import annotations

from v2xpnp.pipeline.trajectory_ingest_stage_01_types_io import *  # noqa: F401,F403
from v2xpnp.pipeline.trajectory_ingest_stage_02_matching import *  # noqa: F401,F403


# ---------------------- Core conversion ---------------------- #

def build_trajectories(
    yaml_dir: Path,
    dt: float,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    flip_y: bool = False,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[Waypoint],
    List[float],
    Dict[int, Dict[str, object]],
]:
    """Parse YAML sequence into per-vehicle trajectories and ego path, plus per-waypoint times."""
    yaml_paths = list_yaml_timesteps(yaml_dir)
    if not yaml_paths:
        raise SystemExit(f"No YAML files found under {yaml_dir}")

    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_traj: List[Waypoint] = []
    ego_times: List[float] = []
    obj_info: Dict[int, Dict[str, object]] = {}
    spawn_report: Dict[str, object] | None = None

    for idx, path in enumerate(yaml_paths):
        try:
            frame_idx = int(path.stem)
        except Exception:
            frame_idx = idx
        frame_time = float(frame_idx) * float(dt)
        data = load_yaml(path)
        ego_pose = data.get("true_ego_pose") or data.get("lidar_pose")
        if ego_pose:
            ex, ey, ez = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
            ex, ey = apply_se2((ex, ey), yaw_deg, tx, ty, flip_y=flip_y)
            ego_yaw = yaw_from_pose(ego_pose)
            if flip_y:
                ego_yaw = -ego_yaw
            ego_traj.append(
                Waypoint(
                    x=ex,
                    y=ey,
                    z=ez + tz,
                    yaw=ego_yaw + yaw_deg,
                    pitch=float(ego_pose[3]) if len(ego_pose) > 3 else 0.0,
                    roll=float(ego_pose[5]) if len(ego_pose) > 5 else 0.0,
                )
            )
            ego_times.append(frame_time)

        vehs = data.get("vehicles", {}) or {}
        for vid_str, payload in vehs.items():
            try:
                vid = int(vid_str)
            except Exception:
                continue
            if isinstance(payload, dict):
                existing = obj_info.get(vid, {})
                obj_type = payload.get("obj_type")
                if obj_type and (not existing.get("obj_type")):
                    existing["obj_type"] = obj_type
                    existing["model"] = map_obj_type(obj_type)
                elif obj_type and existing.get("obj_type") and obj_type != existing.get("obj_type"):
                    # Keep first seen obj_type but note mismatch once
                    if not existing.get("_obj_type_conflict"):
                        print(f"[WARN] obj_type conflict for id {vid}: '{existing.get('obj_type')}' vs '{obj_type}' (keeping first)")
                        existing["_obj_type_conflict"] = True
                ext = payload.get("extent") or []
                if isinstance(ext, Sequence):
                    length = float(ext[0]) * 2 if len(ext) > 0 else None
                    width = float(ext[1]) * 2 if len(ext) > 1 else None
                    if length is not None and existing.get("length") is None:
                        existing["length"] = length
                    if width is not None and existing.get("width") is None:
                        existing["width"] = width
                if existing:
                    obj_info[vid] = existing
            loc = payload.get("location") or [0, 0, 0]
            ang = payload.get("angle") or [0, 0, 0]
            pitch = float(ang[0]) if len(ang) > 0 else 0.0
            yaw = yaw_from_angle(ang)
            if flip_y:
                yaw = -yaw
            yaw += yaw_deg
            roll = float(ang[2]) if len(ang) > 2 else 0.0
            x, y = apply_se2((float(loc[0]), float(loc[1])), yaw_deg, tx, ty, flip_y=flip_y)
            z = float(loc[2]) + tz if len(loc) > 2 else tz
            wp = Waypoint(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
            vehicles.setdefault(vid, []).append(wp)
            vehicle_times.setdefault(vid, []).append(frame_time)

    # Compute simple average speed (m/s) per vehicle from path length
    speeds: Dict[int, float] = {}
    for vid, traj in vehicles.items():
        dist = 0.0
        for a, b in zip(traj, traj[1:]):
            dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
        speeds[vid] = dist / max(dt * max(len(traj) - 1, 1), 1e-6)

    return vehicles, vehicle_times, ego_traj, ego_times, obj_info


def extract_obj_info(yaml_dir: Path) -> Dict[int, Dict[str, object]]:
    """Gather obj_type/model/size from the first timestep in a YAML directory."""
    obj_info: Dict[int, Dict[str, object]] = {}
    first_yaml = next(iter(list_yaml_timesteps(yaml_dir)), None)
    if not first_yaml:
        return obj_info
    data0 = load_yaml(first_yaml)
    vehs0 = data0.get("vehicles", {}) or {}
    for vid_str, payload in vehs0.items():
        try:
            vid = int(vid_str)
        except Exception:
            continue
        obj_type = payload.get("obj_type") or "npc"
        model = map_obj_type(obj_type)
        ext = payload.get("extent") or []
        length = float(ext[0]) * 2 if len(ext) > 0 else None
        width = float(ext[1]) * 2 if len(ext) > 1 else None
        obj_info[vid] = {
            "obj_type": obj_type,
            "model": model,
            "length": length,
            "width": width,
        }
    return obj_info


def write_route_xml(
    path: Path,
    route_id: str,
    role: str,
    town: str,
    waypoints: List[Waypoint],
    times: List[float] | None = None,
    snap_to_road: bool = False,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
) -> None:
    root = ET.Element("routes")
    route = ET.SubElement(
        root,
        "route",
        {
            "id": str(route_id),
            "town": town,
            "role": role,
            "snap_to_road": "true" if snap_to_road else "false",
        },
    )
    for idx, wp in enumerate(waypoints):
        attrs = {
            "x": f"{wp.x + xml_tx:.6f}",
            "y": f"{wp.y + xml_ty:.6f}",
            "z": f"{wp.z:.6f}",
            "yaw": f"{wp.yaw:.6f}",
            "pitch": "0.000000",
            "roll": "0.000000",
        }
        if times and idx < len(times):
            try:
                attrs["time"] = f"{float(times[idx]):.6f}"
            except (TypeError, ValueError):
                pass
        ET.SubElement(
            route,
            "waypoint",
            attrs,
        )
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def parse_route_xml(path: Path) -> List[Waypoint]:
    """Load waypoints (x,y,z,yaw) from a CARLA route XML."""
    tree = ET.parse(path)
    root = tree.getroot()
    wps: List[Waypoint] = []
    for node in root.findall(".//waypoint"):
        try:
            x = float(node.attrib.get("x", 0.0))
            y = float(node.attrib.get("y", 0.0))
            z = float(node.attrib.get("z", 0.0))
            yaw = float(node.attrib.get("yaw", 0.0))
            wps.append(Waypoint(x=x, y=y, z=z, yaw=yaw))
        except Exception:
            continue
    return wps


def save_manifest(
    manifest_path: Path,
    actors_by_kind: Dict[str, List[dict]],
    ego_entries: List[dict],
) -> None:
    """Save manifest with actors organized by kind (ego, npc, static, etc.)."""
    manifest: Dict[str, List[dict]] = {}
    if ego_entries:
        manifest["ego"] = ego_entries
    # Add all other actor kinds
    for kind, entries in sorted(actors_by_kind.items()):
        manifest[kind] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------- Visualization ---------------------- #

def plot_frame(
    timestep: int,
    actors_by_id: Dict[int, List[Waypoint]],
    ego_trajs: Sequence[List[Waypoint]],
    out_path: Path,
    axes_limits: Tuple[float, float, float, float] | None = None,
    map_lines: List[List[Tuple[float, float]]] | None = None,
    invert_plot_y: bool = False,
):
    if plt is None or patches is None or transforms is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib imageio")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Timestep {timestep:06d}")

    xs: List[float] = []
    ys: List[float] = []
    for vid, traj in actors_by_id.items():
        if timestep >= len(traj):
            continue
        wp = traj[timestep]
        width = 2.0
        height = 4.0
        rect = patches.Rectangle(
            (wp.x - width / 2, wp.y - height / 2),
            width,
            height,
            linewidth=1.0,
            edgecolor="C0",
            facecolor="C0",
            alpha=0.4,
        )
        rot = transforms.Affine2D().rotate_deg_around(wp.x, wp.y, wp.yaw) + ax.transData
        rect.set_transform(rot)
        ax.add_patch(rect)
        ax.text(wp.x, wp.y, f"{vid}", fontsize=7, ha="center", va="center")
        xs.append(wp.x)
        ys.append(wp.y)

    if ego_trajs:
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            idx = min(timestep, len(ego_traj) - 1)
            ego = ego_traj[idx]
            color = "orange" if ego_idx == 0 else f"C{(ego_idx + 1) % 10}"
            tri = patches.RegularPolygon(
                (ego.x, ego.y),
                numVertices=3,
                radius=2.5,
                orientation=math.radians(ego.yaw),
                color=color,
                alpha=0.6,
            )
            ax.add_patch(tri)
            ax.text(ego.x, ego.y, f"ego{ego_idx}", fontsize=7, ha="center", va="center")
            xs.append(ego.x)
            ys.append(ego.y)

    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            lx = [p[0] for p in line]
            ly = [p[1] for p in line]
            ax.plot(lx, ly, color="gray", linewidth=1.0, alpha=0.5)
            xs.extend(lx)
            ys.extend(ly)

    if axes_limits:
        minx, maxx, miny, maxy = axes_limits
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    elif xs and ys:
        pad = 10.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_gif(frames_dir: Path, gif_path: Path, fps: float = 10.0) -> None:
    if imageio is None:
        raise RuntimeError("imageio is required for GIF output; install imageio")
    imgs = []
    for png in sorted(frames_dir.glob("frame_*.png")):
        imgs.append(imageio.imread(png))
    if not imgs:
        raise RuntimeError("No frames produced for GIF")
    duration_ms = 1000.0 / float(fps)
    imageio.mimsave(gif_path, imgs, duration=duration_ms / 1000.0)


def write_paths_png(
    actors_by_id: Dict[int, List[Waypoint]],
    ego_trajs: Sequence[List[Waypoint]],
    map_lines: List[List[Tuple[float, float]]],
    out_path: Path,
    axis_pad: float = 10.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None or patches is None:
        raise RuntimeError("matplotlib is required for --paths-png; install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Actor Paths")

    xs: List[float] = []
    ys: List[float] = []

    # Map
    for line in map_lines:
        if len(line) < 2:
            continue
        lx = [p[0] for p in line]
        ly = [p[1] for p in line]
        ax.plot(lx, ly, color="gray", linewidth=0.8, alpha=0.5, zorder=0)
        xs.extend(lx)
        ys.extend(ly)

    # Actors
    for vid, traj in actors_by_id.items():
        if len(traj) < 2:
            continue
        lx = [wp.x for wp in traj]
        ly = [wp.y for wp in traj]
        ax.plot(lx, ly, linewidth=1.5, alpha=0.9, label=f"id {vid}")
        ax.scatter(lx[0], ly[0], s=15, marker="o")
        xs.extend(lx)
        ys.extend(ly)

    # Ego(s)
    if ego_trajs:
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            lx = [wp.x for wp in ego_traj]
            ly = [wp.y for wp in ego_traj]
            color = "black" if ego_idx == 0 else f"C{(ego_idx + 1) % 10}"
            label = "ego" if ego_idx == 0 else f"ego{ego_idx}"
            ax.plot(lx, ly, color=color, linewidth=2.0, alpha=0.8, label=label)
            ax.scatter(lx[0], ly[0], s=30, marker="*", color=color)
            xs.extend(lx)
            ys.extend(ly)

    if xs and ys:
        pad = max(0.0, axis_pad)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    if len(actors_by_id) <= 20:  # avoid huge legends
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_actor_yaw_viz(
    actor_id: int,
    gt_traj: List[Waypoint],
    xml_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    out_path: Path,
    arrow_step: int = 5,
    arrow_len: float = 0.8,
    pad: float = 5.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actor {actor_id} yaw: GT vs XML")

    # Map layer
    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.7, alpha=0.6, zorder=0)

    # Paths
    gt_x = [wp.x for wp in gt_traj]
    gt_y = [wp.y for wp in gt_traj]
    xml_x = [wp.x for wp in xml_traj]
    xml_y = [wp.y for wp in xml_traj]
    # Draw XML first, then GT on top with markers so overlap is visible
    ax.plot(xml_x, xml_y, color="#d95f0e", linewidth=2.0, alpha=0.9, label="XML path", zorder=2)
    ax.plot(
        gt_x,
        gt_y,
        color="#2c7fb8",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=2.5,
        markevery=max(1, int(len(gt_x) / 20)),
        label="GT path",
        zorder=3,
    )

    # If GT and XML are effectively identical, annotate it
    min_len = min(len(gt_traj), len(xml_traj))
    if min_len > 0:
        max_diff = 0.0
        for i in range(min_len):
            dx = gt_traj[i].x - xml_traj[i].x
            dy = gt_traj[i].y - xml_traj[i].y
            max_diff = max(max_diff, math.hypot(dx, dy))
        if max_diff < 1e-3:
            ax.text(
                0.02,
                0.98,
                "GT == XML (overlapping paths)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=10,
            )

    # Yaw arrows
    def _quiver(traj: List[Waypoint], color: str, label: str) -> None:
        if not traj:
            return
        step = max(1, int(arrow_step))
        xs = [wp.x for i, wp in enumerate(traj) if i % step == 0]
        ys = [wp.y for i, wp in enumerate(traj) if i % step == 0]
        us = [math.cos(math.radians(wp.yaw)) * arrow_len for i, wp in enumerate(traj) if i % step == 0]
        vs = [math.sin(math.radians(wp.yaw)) * arrow_len for i, wp in enumerate(traj) if i % step == 0]
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.7,
            width=0.002,
            label=label,
        )

    _quiver(gt_traj, "#2c7fb8", "GT yaw")
    _quiver(xml_traj, "#d95f0e", "XML yaw")

    # Bounds
    xs_all = gt_x + xml_x
    ys_all = gt_y + xml_y
    if xs_all and ys_all:
        pad_val = max(0.0, float(pad))
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sample_map_nodes(
    map_line_records: List[Dict[str, object]] | None,
    map_lines: List[List[Tuple[float, float]]] | None,
    step: int,
) -> List[Tuple[float, float]]:
    step = max(1, int(step))
    nodes: List[Tuple[float, float]] = []
    seen: set[Tuple[int, int]] = set()

    recs = map_line_records or []
    if recs:
        for rec in recs:
            pts = rec.get("points") if isinstance(rec, dict) else None
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            for i in range(0, len(pts), step):
                try:
                    x = float(pts[i][0])
                    y = float(pts[i][1])
                except Exception:
                    continue
                key = (int(round(x * 100.0)), int(round(y * 100.0)))
                if key in seen:
                    continue
                seen.add(key)
                nodes.append((x, y))
    elif map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            for i in range(0, len(line), step):
                x, y = line[i]
                key = (int(round(float(x) * 100.0)), int(round(float(y) * 100.0)))
                if key in seen:
                    continue
                seen.add(key)
                nodes.append((float(x), float(y)))
    return nodes


def _nearest_node_indices(
    query_points: Sequence[Tuple[float, float]],
    nodes: Sequence[Tuple[float, float]],
    max_radius: float,
) -> set[int]:
    if not query_points or not nodes:
        return set()
    max_r2 = float(max_radius) * float(max_radius)
    touched: set[int] = set()
    for qx, qy in query_points:
        best_idx = -1
        best_d2 = float("inf")
        for idx, (nx, ny) in enumerate(nodes):
            dx = float(nx) - float(qx)
            dy = float(ny) - float(qy)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        if best_idx >= 0 and best_d2 <= max_r2:
            touched.add(best_idx)
    return touched


def _capture_carla_topdown_bev(
    host: str,
    port: int,
    bounds: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    fov_deg: float,
    margin_scale: float,
    expected_town: str | None = None,
) -> Dict[str, object] | None:
    """
    Capture a top-down RGB image from CARLA centered around bounds.
    Returns dict with image and projection data for world->pixel overlays.
    """
    if carla is None:
        return None
    if np is None:
        return None
    _assert_carla_endpoint_reachable(host, port, timeout_s=2.0)
    image_w = max(256, int(image_w))
    image_h = max(256, int(image_h))
    fov_deg = min(120.0, max(15.0, float(fov_deg)))
    margin_scale = max(1.0, float(margin_scale))

    minx, maxx, miny, maxy = bounds
    cx = 0.5 * (float(minx) + float(maxx))
    cy = 0.5 * (float(miny) + float(maxy))
    span_x = max(5.0, float(maxx) - float(minx)) * margin_scale
    span_y = max(5.0, float(maxy) - float(miny)) * margin_scale

    hfov = math.radians(fov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (float(image_h) / float(image_w)))
    alt_x = span_x / (2.0 * math.tan(hfov / 2.0))
    alt_y = span_y / (2.0 * math.tan(vfov / 2.0))

    client = carla.Client(host, int(port))
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    if expected_town and expected_town not in str(cmap.name or ""):
        candidates = [m for m in client.get_available_maps() if expected_town in m]
        if candidates:
            world = client.load_world(candidates[0])
            cmap = world.get_map()

    ground_z = 0.0
    try:
        wp = cmap.get_waypoint(
            carla.Location(x=float(cx), y=float(cy), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if wp is not None:
            ground_z = float(wp.transform.location.z)
    except Exception:
        ground_z = 0.0

    cam_z = float(ground_z) + max(float(alt_x), float(alt_y)) + 2.0
    cam_tf = carla.Transform(
        carla.Location(x=float(cx), y=float(cy), z=float(cam_z)),
        carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
    )

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(image_w))
    cam_bp.set_attribute("image_size_y", str(image_h))
    cam_bp.set_attribute("fov", str(float(fov_deg)))
    cam_bp.set_attribute("sensor_tick", "0.0")
    cam_bp.set_attribute("motion_blur_intensity", "0.0")

    cam_actor = None
    img = None
    img_q: "queue.Queue[object]" = queue.Queue()
    try:
        cam_actor = world.spawn_actor(cam_bp, cam_tf)
        cam_actor.listen(lambda data: img_q.put(data))

        sync = bool(getattr(world.get_settings(), "synchronous_mode", False))
        for _ in range(20):
            if sync:
                world.tick()
            else:
                world.wait_for_tick(seconds=1.0)
            try:
                img = img_q.get(timeout=0.5)
                break
            except Exception:
                continue
        if img is None:
            return None

        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((int(img.height), int(img.width), 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

        focal = float(image_w) / (2.0 * math.tan(hfov / 2.0))
        K = np.array(
            [
                [focal, 0.0, float(image_w) / 2.0],
                [0.0, focal, float(image_h) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        w2c = np.array(cam_actor.get_transform().get_inverse_matrix(), dtype=np.float64)
        return {
            "image": rgb,
            "width": int(image_w),
            "height": int(image_h),
            "K": K,
            "world_to_camera": w2c,
            "ground_z": float(ground_z),
            "camera_z": float(cam_z),
        }
    finally:
        try:
            if cam_actor is not None:
                cam_actor.stop()
        except Exception:
            pass
        try:
            if cam_actor is not None:
                cam_actor.destroy()
        except Exception:
            pass


def _project_world_xy_to_image(
    points_xy: Sequence[Tuple[float, float]],
    world_to_camera,
    K,
    width: int,
    height: int,
    z_world: float,
) -> List[Tuple[float, float]]:
    if np is None:
        return []
    if not points_xy:
        return []
    out: List[Tuple[float, float]] = []
    w = float(max(1, int(width)))
    h = float(max(1, int(height)))
    for x, y in points_xy:
        world_pt = np.array([float(x), float(y), float(z_world), 1.0], dtype=np.float64)
        cam_pt = np.dot(world_to_camera, world_pt)
        # Convert CARLA camera coordinates to conventional pinhole coordinates.
        cam_cv = np.array([cam_pt[1], -cam_pt[2], cam_pt[0]], dtype=np.float64)
        depth = float(cam_cv[2])
        if depth <= 1e-4:
            continue
        u = float(K[0, 0] * cam_cv[0] / depth + K[0, 2])
        v = float(K[1, 1] * cam_cv[1] / depth + K[1, 2])
        if -2.0 <= u <= w + 2.0 and -2.0 <= v <= h + 2.0:
            out.append((u, v))
    return out


def write_ego_alignment_bev_viz(
    ego_idx: int,
    pre_align_traj: List[Waypoint],
    post_align_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    map_line_records: List[Dict[str, object]] | None,
    map_image,
    map_image_bounds: Tuple[float, float, float, float] | None,
    captured_bev: Dict[str, object] | None,
    out_path: Path,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
    pad: float = 24.0,
    node_step: int = 2,
    match_radius: float = 1.8,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    pre_xy = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in pre_align_traj]
    post_xy = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in post_align_traj]
    if not pre_xy and not post_xy:
        return

    all_nodes = _sample_map_nodes(map_line_records, map_lines, step=node_step)
    touched = _nearest_node_indices(post_xy, all_nodes, max_radius=max(0.1, float(match_radius)))
    touched_nodes = [all_nodes[i] for i in sorted(touched)]

    map_bounds = map_image_bounds
    if map_bounds is None and map_lines:
        pts: List[Tuple[float, float]] = []
        for ln in map_lines:
            pts.extend(ln)
        map_bounds = _bounds_from_points(pts)
    path_bounds = _bounds_from_points(pre_xy + post_xy)
    merged = _merge_bounds([map_bounds, path_bounds])

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_title(f"Ego {ego_idx}: Alignment on CARLA BEV")
    used_carla_capture = False

    if captured_bev is not None and np is not None:
        try:
            image = captured_bev.get("image")
            w2c = captured_bev.get("world_to_camera")
            K = captured_bev.get("K")
            width = int(captured_bev.get("width", 0))
            height = int(captured_bev.get("height", 0))
            z_world = float(captured_bev.get("ground_z", 0.0))
            if image is not None and w2c is not None and K is not None and width > 0 and height > 0:
                used_carla_capture = True
                ax.imshow(image, origin="upper", zorder=0)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal", adjustable="box")

                nodes_px = _project_world_xy_to_image(
                    all_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                touched_px = _project_world_xy_to_image(
                    touched_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                pre_px = _project_world_xy_to_image(
                    pre_xy, w2c, K, width=width, height=height, z_world=z_world
                )
                post_px = _project_world_xy_to_image(
                    post_xy, w2c, K, width=width, height=height, z_world=z_world
                )

                if nodes_px:
                    ax.scatter(
                        [p[0] for p in nodes_px],
                        [p[1] for p in nodes_px],
                        s=8,
                        marker="s",
                        c="#1f1f1f",
                        alpha=0.24,
                        linewidths=0.0,
                        label="all map nodes",
                        zorder=1,
                    )
                if touched_px:
                    ax.scatter(
                        [p[0] for p in touched_px],
                        [p[1] for p in touched_px],
                        s=15,
                        marker="s",
                        c="#00e5ff",
                        alpha=0.82,
                        linewidths=0.0,
                        label="nodes touched by aligned ego",
                        zorder=2,
                    )

                if pre_px:
                    ax.plot(
                        [p[0] for p in pre_px],
                        [p[1] for p in pre_px],
                        color="#1f77b4",
                        linestyle="--",
                        linewidth=2.2,
                        alpha=0.95,
                        label="GT from YAML + offsets (pre-align)",
                        zorder=4,
                    )
                    ax.scatter([pre_px[0][0]], [pre_px[0][1]], s=55, c="#1f77b4", marker="o", zorder=5, label="pre start")
                    ax.scatter([pre_px[-1][0]], [pre_px[-1][1]], s=55, c="#1f77b4", marker="x", zorder=5, label="pre end")
                if post_px:
                    ax.plot(
                        [p[0] for p in post_px],
                        [p[1] for p in post_px],
                        color="#e53935",
                        linewidth=2.4,
                        alpha=0.98,
                        label="post alignment/refinement",
                        zorder=6,
                    )
                    ax.scatter([post_px[0][0]], [post_px[0][1]], s=65, c="#e53935", marker="o", zorder=7, label="post start")
                    ax.scatter([post_px[-1][0]], [post_px[-1][1]], s=65, c="#e53935", marker="x", zorder=7, label="post end")

                # Zoom to ego ROI inside captured image
                roi = pre_px + post_px
                if roi:
                    rx = [p[0] for p in roi]
                    ry = [p[1] for p in roi]
                    span = max(20.0, max(max(rx) - min(rx), max(ry) - min(ry)))
                    pad_px = span * 0.22
                    x0 = max(0.0, min(rx) - pad_px)
                    x1 = min(float(width), max(rx) + pad_px)
                    y0 = max(0.0, min(ry) - pad_px)
                    y1 = min(float(height), max(ry) + pad_px)
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y1, y0)
        except Exception:
            used_carla_capture = False

    if not used_carla_capture:
        ax.set_aspect("equal", adjustable="box")
        if map_image is not None:
            if merged:
                ext = map_bounds or merged
                ax.imshow(map_image, extent=(ext[0], ext[1], ext[2], ext[3]), origin="lower", alpha=0.97, zorder=0)
            else:
                ax.imshow(map_image, origin="lower", alpha=0.97, zorder=0)
        elif map_lines:
            _plot_background_lines(ax, map_lines, color="#b8b8b8", lw=0.8, alpha=0.7)
            ax.text(
                0.02,
                0.98,
                "No captured CARLA BEV; using vector/raster fallback",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#666666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=9,
            )

        if all_nodes:
            ax.scatter(
                [p[0] for p in all_nodes],
                [p[1] for p in all_nodes],
                s=9,
                marker="s",
                c="#3b3b3b",
                alpha=0.22,
                linewidths=0.0,
                label="all map nodes",
                zorder=1,
            )
        if touched_nodes:
            ax.scatter(
                [p[0] for p in touched_nodes],
                [p[1] for p in touched_nodes],
                s=16,
                marker="s",
                c="#00bcd4",
                alpha=0.85,
                linewidths=0.0,
                label="nodes touched by aligned ego",
                zorder=2,
            )

        if pre_xy:
            ax.plot(
                [p[0] for p in pre_xy],
                [p[1] for p in pre_xy],
                color="#1f77b4",
                linestyle="--",
                linewidth=2.2,
                alpha=0.95,
                label="GT from YAML + offsets (pre-align)",
                zorder=4,
            )
            ax.scatter([pre_xy[0][0]], [pre_xy[0][1]], s=55, c="#1f77b4", marker="o", zorder=5, label="pre start")
            ax.scatter([pre_xy[-1][0]], [pre_xy[-1][1]], s=55, c="#1f77b4", marker="x", zorder=5, label="pre end")
        if post_xy:
            ax.plot(
                [p[0] for p in post_xy],
                [p[1] for p in post_xy],
                color="#e53935",
                linewidth=2.4,
                alpha=0.98,
                label="post alignment/refinement",
                zorder=6,
            )
            ax.scatter([post_xy[0][0]], [post_xy[0][1]], s=65, c="#e53935", marker="o", zorder=7, label="post start")
            ax.scatter([post_xy[-1][0]], [post_xy[-1][1]], s=65, c="#e53935", marker="x", zorder=7, label="post end")

        if merged:
            pad_val = max(0.0, float(pad)) * 1.2
            ax.set_xlim(merged[0] - pad_val, merged[1] + pad_val)
            ax.set_ylim(merged[2] - pad_val, merged[3] + pad_val)

        if invert_plot_y:
            ax.invert_yaxis()

    info = [
        f"background: {'CARLA captured BEV' if used_carla_capture else 'fallback'}",
        f"all nodes: {len(all_nodes)}",
        f"touched nodes: {len(touched_nodes)}",
        f"touch radius: {float(match_radius):.2f} m",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
        zorder=10,
    )

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_actor_alignment_bev_viz(
    title: str,
    pre_trajs: Dict[int, List[Waypoint]],
    post_trajs: Dict[int, List[Waypoint]],
    map_lines: List[List[Tuple[float, float]]] | None,
    map_line_records: List[Dict[str, object]] | None,
    map_image,
    map_image_bounds: Tuple[float, float, float, float] | None,
    captured_bev: Dict[str, object] | None,
    out_path: Path,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
    pad: float = 24.0,
    node_step: int = 2,
    match_radius: float = 1.8,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    actor_ids = sorted(set(int(k) for k in pre_trajs.keys()) | set(int(k) for k in post_trajs.keys()))
    if not actor_ids:
        return

    pre_xy_by_id: Dict[int, List[Tuple[float, float]]] = {}
    post_xy_by_id: Dict[int, List[Tuple[float, float]]] = {}
    for aid in actor_ids:
        pre = pre_trajs.get(aid) or []
        post = post_trajs.get(aid) or []
        p_pre = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in pre]
        p_post = [(wp.x + float(xml_tx), wp.y + float(xml_ty)) for wp in post]
        if p_pre:
            pre_xy_by_id[aid] = p_pre
        if p_post:
            post_xy_by_id[aid] = p_post
    if not pre_xy_by_id and not post_xy_by_id:
        return

    all_pre = [pt for path in pre_xy_by_id.values() for pt in path]
    all_post = [pt for path in post_xy_by_id.values() for pt in path]
    all_nodes = _sample_map_nodes(map_line_records, map_lines, step=node_step)
    touched = _nearest_node_indices(all_post, all_nodes, max_radius=max(0.1, float(match_radius)))
    touched_nodes = [all_nodes[i] for i in sorted(touched)]

    map_bounds = map_image_bounds
    if map_bounds is None and map_lines:
        pts: List[Tuple[float, float]] = []
        for ln in map_lines:
            pts.extend(ln)
        map_bounds = _bounds_from_points(pts)
    path_bounds = _bounds_from_points(all_pre + all_post)
    merged = _merge_bounds([map_bounds, path_bounds])

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_title(title)
    used_carla_capture = False
    individual_mode = len(actor_ids) == 1

    if captured_bev is not None and np is not None:
        try:
            image = captured_bev.get("image")
            w2c = captured_bev.get("world_to_camera")
            K = captured_bev.get("K")
            width = int(captured_bev.get("width", 0))
            height = int(captured_bev.get("height", 0))
            z_world = float(captured_bev.get("ground_z", 0.0))
            if image is not None and w2c is not None and K is not None and width > 0 and height > 0:
                used_carla_capture = True
                ax.imshow(image, origin="upper", zorder=0)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal", adjustable="box")

                nodes_px = _project_world_xy_to_image(
                    all_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                touched_px = _project_world_xy_to_image(
                    touched_nodes, w2c, K, width=width, height=height, z_world=z_world
                )
                if nodes_px:
                    ax.scatter(
                        [p[0] for p in nodes_px],
                        [p[1] for p in nodes_px],
                        s=7,
                        marker="s",
                        c="#1a1a1a",
                        alpha=0.22,
                        linewidths=0.0,
                        label="all map nodes",
                        zorder=1,
                    )
                if touched_px:
                    ax.scatter(
                        [p[0] for p in touched_px],
                        [p[1] for p in touched_px],
                        s=14,
                        marker="s",
                        c="#00e5ff",
                        alpha=0.82,
                        linewidths=0.0,
                        label="nodes touched by aligned paths",
                        zorder=2,
                    )

                pre_label_drawn = False
                post_label_drawn = False
                roi_pts: List[Tuple[float, float]] = []
                for aid in actor_ids:
                    pre_px = _project_world_xy_to_image(
                        pre_xy_by_id.get(aid, []), w2c, K, width=width, height=height, z_world=z_world
                    )
                    post_px = _project_world_xy_to_image(
                        post_xy_by_id.get(aid, []), w2c, K, width=width, height=height, z_world=z_world
                    )
                    roi_pts.extend(pre_px)
                    roi_pts.extend(post_px)
                    if pre_px:
                        ax.plot(
                            [p[0] for p in pre_px],
                            [p[1] for p in pre_px],
                            color="#1f77b4",
                            linestyle="--",
                            linewidth=2.1 if individual_mode else 1.1,
                            alpha=0.95 if individual_mode else 0.38,
                            label=None if pre_label_drawn else "GT from YAML + offsets (pre-align)",
                            zorder=4,
                        )
                        pre_label_drawn = True
                        if individual_mode:
                            ax.scatter([pre_px[0][0]], [pre_px[0][1]], s=50, c="#1f77b4", marker="o", zorder=5, label="pre start")
                            ax.scatter([pre_px[-1][0]], [pre_px[-1][1]], s=50, c="#1f77b4", marker="x", zorder=5, label="pre end")
                    if post_px:
                        ax.plot(
                            [p[0] for p in post_px],
                            [p[1] for p in post_px],
                            color="#e53935",
                            linewidth=2.3 if individual_mode else 1.3,
                            alpha=0.98 if individual_mode else 0.45,
                            label=None if post_label_drawn else "post alignment/refinement",
                            zorder=6,
                        )
                        post_label_drawn = True
                        if individual_mode:
                            ax.scatter([post_px[0][0]], [post_px[0][1]], s=58, c="#e53935", marker="o", zorder=7, label="post start")
                            ax.scatter([post_px[-1][0]], [post_px[-1][1]], s=58, c="#e53935", marker="x", zorder=7, label="post end")

                if roi_pts:
                    rx = [p[0] for p in roi_pts]
                    ry = [p[1] for p in roi_pts]
                    span = max(24.0, max(max(rx) - min(rx), max(ry) - min(ry)))
                    pad_px = span * 0.24
                    x0 = max(0.0, min(rx) - pad_px)
                    x1 = min(float(width), max(rx) + pad_px)
                    y0 = max(0.0, min(ry) - pad_px)
                    y1 = min(float(height), max(ry) + pad_px)
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y1, y0)
        except Exception:
            used_carla_capture = False

    if not used_carla_capture:
        ax.set_aspect("equal", adjustable="box")
        if map_image is not None:
            if merged:
                ext = map_bounds or merged
                ax.imshow(map_image, extent=(ext[0], ext[1], ext[2], ext[3]), origin="lower", alpha=0.97, zorder=0)
            else:
                ax.imshow(map_image, origin="lower", alpha=0.97, zorder=0)
        elif map_lines:
            _plot_background_lines(ax, map_lines, color="#b8b8b8", lw=0.8, alpha=0.7)

        if all_nodes:
            ax.scatter(
                [p[0] for p in all_nodes],
                [p[1] for p in all_nodes],
                s=9,
                marker="s",
                c="#3b3b3b",
                alpha=0.22,
                linewidths=0.0,
                label="all map nodes",
                zorder=1,
            )
        if touched_nodes:
            ax.scatter(
                [p[0] for p in touched_nodes],
                [p[1] for p in touched_nodes],
                s=16,
                marker="s",
                c="#00bcd4",
                alpha=0.85,
                linewidths=0.0,
                label="nodes touched by aligned paths",
                zorder=2,
            )

        pre_label_drawn = False
        post_label_drawn = False
        for aid in actor_ids:
            p_pre = pre_xy_by_id.get(aid, [])
            p_post = post_xy_by_id.get(aid, [])
            if p_pre:
                ax.plot(
                    [p[0] for p in p_pre],
                    [p[1] for p in p_pre],
                    color="#1f77b4",
                    linestyle="--",
                    linewidth=2.1 if individual_mode else 1.1,
                    alpha=0.95 if individual_mode else 0.38,
                    label=None if pre_label_drawn else "GT from YAML + offsets (pre-align)",
                    zorder=4,
                )
                pre_label_drawn = True
                if individual_mode:
                    ax.scatter([p_pre[0][0]], [p_pre[0][1]], s=50, c="#1f77b4", marker="o", zorder=5, label="pre start")
                    ax.scatter([p_pre[-1][0]], [p_pre[-1][1]], s=50, c="#1f77b4", marker="x", zorder=5, label="pre end")
            if p_post:
                ax.plot(
                    [p[0] for p in p_post],
                    [p[1] for p in p_post],
                    color="#e53935",
                    linewidth=2.3 if individual_mode else 1.3,
                    alpha=0.98 if individual_mode else 0.45,
                    label=None if post_label_drawn else "post alignment/refinement",
                    zorder=6,
                )
                post_label_drawn = True
                if individual_mode:
                    ax.scatter([p_post[0][0]], [p_post[0][1]], s=58, c="#e53935", marker="o", zorder=7, label="post start")
                    ax.scatter([p_post[-1][0]], [p_post[-1][1]], s=58, c="#e53935", marker="x", zorder=7, label="post end")

        if merged:
            pad_val = max(0.0, float(pad)) * 1.2
            ax.set_xlim(merged[0] - pad_val, merged[1] + pad_val)
            ax.set_ylim(merged[2] - pad_val, merged[3] + pad_val)
        if invert_plot_y:
            ax.invert_yaxis()

    info = [
        f"background: {'CARLA captured BEV' if used_carla_capture else 'fallback'}",
        f"actors: {len(actor_ids)}",
        f"all nodes: {len(all_nodes)}",
        f"touched nodes: {len(touched_nodes)}",
        f"touch radius: {float(match_radius):.2f} m",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
        zorder=10,
    )

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_ego_alignment_viz(
    ego_idx: int,
    pre_align_traj: List[Waypoint],
    post_align_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    map_line_records: List[Dict[str, object]] | None,
    out_path: Path,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
    pad: float = 24.0,
    invert_plot_y: bool = False,
) -> None:
    """
    Visualize ego trajectory before/after spawn preprocess alignment.
    pre_align_traj should be the original GT+global-transform path (before preprocess).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    if not pre_align_traj and not post_align_traj:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Ego {ego_idx}: GT+offset vs aligned/refined")

    pre_x = [wp.x + float(xml_tx) for wp in pre_align_traj]
    pre_y = [wp.y + float(xml_ty) for wp in pre_align_traj]
    post_x = [wp.x + float(xml_tx) for wp in post_align_traj]
    post_y = [wp.y + float(xml_ty) for wp in post_align_traj]
    xs_all = pre_x + post_x
    ys_all = pre_y + post_y

    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#c8c8c8", linewidth=0.7, alpha=0.6, zorder=0)

        # Add lane direction arrows near the ego path ROI.
        # Uses lane metadata when available (road_id/lane_id/dir_sign).
        qx: List[float] = []
        qy: List[float] = []
        qu: List[float] = []
        qv: List[float] = []
        if xs_all and ys_all:
            roi_pad = max(8.0, float(pad) * 1.1)
            minx = min(xs_all) - roi_pad
            maxx = max(xs_all) + roi_pad
            miny = min(ys_all) - roi_pad
            maxy = max(ys_all) + roi_pad
        else:
            minx = miny = -float("inf")
            maxx = maxy = float("inf")

        recs = map_line_records or []
        if not recs:
            recs = [{"points": ln, "dir_sign": None} for ln in map_lines]

        for rec in recs:
            line = rec.get("points") if isinstance(rec, dict) else None
            if not isinstance(line, list) or len(line) < 3:
                continue
            dir_sign = rec.get("dir_sign") if isinstance(rec, dict) else None
            try:
                dir_sign = int(dir_sign) if dir_sign is not None else None
            except Exception:
                dir_sign = None
            if dir_sign not in (-1, 1):
                # Old cache or unknown direction metadata -> skip to avoid misleading arrows.
                continue
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            if max(xs) < minx or min(xs) > maxx or max(ys) < miny or min(ys) > maxy:
                continue

            stride = max(6, len(line) // 20)
            start = stride // 2
            for i in range(start, len(line) - 1, stride):
                if dir_sign > 0:
                    x0, y0 = line[i]
                    x1, y1 = line[i + 1]
                else:
                    x0, y0 = line[i + 1]
                    x1, y1 = line[i]
                dx = x1 - x0
                dy = y1 - y0
                seg = math.hypot(dx, dy)
                if seg < 0.2:
                    continue
                # Keep arrows readable and consistent.
                arr_len = min(1.8, max(0.9, 0.55 * seg))
                qx.append(x0)
                qy.append(y0)
                qu.append((dx / seg) * arr_len)
                qv.append((dy / seg) * arr_len)

        if qx:
            ax.quiver(
                qx,
                qy,
                qu,
                qv,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#9a9a9a",
                alpha=0.65,
                width=0.0015,
                headwidth=3.8,
                headlength=4.5,
                headaxislength=4.0,
                zorder=1,
                label="lane direction",
            )
        elif map_lines:
            ax.text(
                0.02,
                0.985,
                "lane-direction arrows skipped (cache lacks lane metadata)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#666666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=9,
            )

    if pre_x and pre_y:
        ax.plot(
            pre_x,
            pre_y,
            color="#1f77b4",
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            label="GT from YAML + offsets (pre-align)",
            zorder=2,
        )
        ax.scatter([pre_x[0]], [pre_y[0]], s=55, color="#1f77b4", marker="o", zorder=3, label="pre start")
        ax.scatter([pre_x[-1]], [pre_y[-1]], s=55, color="#1f77b4", marker="x", zorder=3, label="pre end")

    if post_x and post_y:
        ax.plot(
            post_x,
            post_y,
            color="#d62728",
            linewidth=2.2,
            alpha=0.95,
            label="Post alignment/refinement",
            zorder=4,
        )
        ax.scatter([post_x[0]], [post_y[0]], s=65, color="#d62728", marker="o", zorder=5, label="post start")
        ax.scatter([post_x[-1]], [post_y[-1]], s=65, color="#d62728", marker="x", zorder=5, label="post end")

    min_len = min(len(pre_align_traj), len(post_align_traj))
    shift_vals: List[float] = []
    if min_len > 0:
        stride = max(1, min_len // 120)
        for i in range(0, min_len, stride):
            x0 = pre_align_traj[i].x + float(xml_tx)
            y0 = pre_align_traj[i].y + float(xml_ty)
            x1 = post_align_traj[i].x + float(xml_tx)
            y1 = post_align_traj[i].y + float(xml_ty)
            shift_vals.append(math.hypot(x1 - x0, y1 - y0))
            ax.plot([x0, x1], [y0, y1], color="#444444", alpha=0.25, linewidth=0.7, zorder=1)
        if not shift_vals:
            shift_vals.append(0.0)

    if xs_all and ys_all:
        pad_val = max(0.0, float(pad)) * 1.15
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if shift_vals:
        txt = (
            f"paired points: {min_len}\n"
            f"shift median: {_median(shift_vals):.3f} m\n"
            f"shift max: {max(shift_vals):.3f} m"
        )
        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
            zorder=10,
        )

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_actor_raw_yaml_viz(
    actor_id: int,
    points_by_subdir: Dict[str, List[Tuple[float, float, float]]],
    map_lines: List[List[Tuple[float, float]]] | None,
    out_path: Path,
    pad: float = 20.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actor {actor_id} raw YAML points by subfolder")

    # Map layer
    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.7, alpha=0.6, zorder=0)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

    xs_all: List[float] = []
    ys_all: List[float] = []
    for idx, (subdir, pts) in enumerate(sorted(points_by_subdir.items(), key=lambda kv: _yaml_dir_sort_key(Path(kv[0])))):
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[2])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        xs_all.extend(xs)
        ys_all.extend(ys)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, s=14, color=color, marker=marker, alpha=0.85, zorder=3, label=f"{subdir} (n={len(xs)})")
        # Start/end annotations
        ax.scatter([xs[0]], [ys[0]], s=40, color=color, marker="o", zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=40, color=color, marker="x", zorder=4)
        ax.annotate(f"t={pts_sorted[0][2]:.1f}", (xs[0], ys[0]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=color)
        ax.annotate(f"t={pts_sorted[-1][2]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=color)

    if xs_all and ys_all:
        pad_val = max(0.0, float(pad))
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------- CLI ---------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert V2XPnP YAML logs to CARLA route XML + manifest")
    p.add_argument("--scenario-dir", required=True, help="Path to the scenario folder containing subfolders with YAML frames")
    p.add_argument(
        "--subdir",
        default="all",
        help=(
            "Specific subfolder inside scenario-dir to use (e.g., -1). "
            "Use 'all' to process all subfolders for actor locations. If omitted and multiple "
            "subfolders exist, behavior is the same. Non-negative subfolders produce ego routes; "
            "negative subfolders contribute actors only."
        ),
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: <scenario-dir>/carla_log_export)")
    p.add_argument("--route-id", default="0", help="Route id to assign to ego and actors (default: 0)")
    p.add_argument("--town", default="ucla_v2", help="CARLA town/map name to embed in XML (default: ucla_v2)")
    p.add_argument("--ego-name", default="ego", help="Name for ego vehicle")
    p.add_argument("--ego-model", default="vehicle.lincoln.mkz2017", help="Blueprint for ego vehicle")
    p.add_argument("--dt", type=float, default=0.1, help="Timestep spacing in seconds (for speed estimation)")
    p.add_argument(
        "--static-path-threshold",
        type=float,
        default=1.2,
        help="Classify vehicle as static if total XY path length stays below this threshold and net displacement is small.",
    )
    p.add_argument(
        "--static-net-disp-threshold",
        type=float,
        default=0.8,
        help="Max XY start-to-end displacement (meters) for vehicle static classification.",
    )
    p.add_argument(
        "--static-bbox-extent-threshold",
        type=float,
        default=0.9,
        help="Max XY bounding-box extent (meters) for noisy-in-place static fallback classification.",
    )
    p.add_argument(
        "--static-avg-speed-threshold",
        type=float,
        default=0.8,
        help="Max average speed (m/s) for noisy-in-place static fallback classification.",
    )
    p.add_argument(
        "--static-heavy-path-threshold",
        type=float,
        default=8.0,
        help="For heavy vehicles (truck/bus/etc), max XY path length (meters) for parked/static fallback classification.",
    )
    p.add_argument(
        "--static-heavy-bbox-extent-threshold",
        type=float,
        default=1.2,
        help="For heavy vehicles (truck/bus/etc), max XY compact-extent (meters) for parked/static fallback classification.",
    )
    p.add_argument(
        "--static-heavy-avg-speed-threshold",
        type=float,
        default=0.8,
        help="For heavy vehicles (truck/bus/etc), max avg speed (m/s) for parked/static fallback classification.",
    )
    p.add_argument(
        "--encode-timing",
        action="store_true",
        help="Embed per-waypoint timing in XML using frame index * dt (enables log replay).",
    )
    p.add_argument(
        "--maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_true",
        help=(
            "For each late-detected actor, choose the earliest safe spawn time between "
            "scenario start and first detection, while enforcing strict non-interference "
            "against other actor trajectories."
        ),
    )
    p.add_argument(
        "--no-maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_false",
        help="Disable safe early-spawn optimization.",
    )
    p.set_defaults(maximize_safe_early_spawn=True)
    p.add_argument(
        "--early-spawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (meters) added to actor radii for early-spawn interference checks.",
    )
    p.add_argument(
        "--early-spawn-report",
        default=None,
        help="Optional JSON path for early-spawn optimization diagnostics.",
    )
    p.add_argument(
        "--maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_true",
        help=(
            "Keep as many actors as possible alive after their last timestamp by holding "
            "their final waypoint until scenario horizon, with strict non-interference checks."
        ),
    )
    p.add_argument(
        "--no-maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_false",
        help="Disable safe late-despawn hold optimization.",
    )
    p.set_defaults(maximize_safe_late_despawn=True)
    p.add_argument(
        "--late-despawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (meters) added to actor radii for late-despawn hold checks.",
    )
    p.add_argument(
        "--late-despawn-report",
        default=None,
        help="Optional JSON path for late-despawn optimization diagnostics.",
    )
    p.add_argument("--tx", type=float, default=0.0, help="Translation X to apply to all coordinates")
    p.add_argument("--ty", type=float, default=0.0, help="Translation Y to apply to all coordinates")
    p.add_argument("--tz", type=float, default=0.0, help="Translation Z to apply to all coordinates")
    p.add_argument("--xml-tx", type=float, default=0.0, help="Additional X offset applied only when writing XML outputs")
    p.add_argument("--xml-ty", type=float, default=0.0, help="Additional Y offset applied only when writing XML outputs")
    p.add_argument(
        "--coord-json",
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Optional JSON file containing transform keys like tx, ty, theta_deg/rad, flip_y; applied to all coordinates",
    )
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Global yaw rotation (degrees, applied before translation)")
    p.add_argument("--snap-to-road", action="store_true", default=True, help="Enable road snapping for actors (defaults to on)")
    p.add_argument("--no-ego", action="store_true", help="Skip writing ego_route.xml")
    p.add_argument(
        "--ego-only",
        action="store_true",
        help="Ignore all non-ego actors and export/process only ego routes.",
    )
    p.add_argument("--gif", action="store_true", help="Generate GIF visualization")
    p.add_argument("--gif-path", default=None, help="Path for GIF (default: <out-dir>/replay.gif)")
    p.add_argument("--paths-png", default=None, help="If set, render a single PNG with each actor's full path as a polyline")
    p.add_argument(
        "--actor-yaw-viz-ids",
        default="",
        help="Comma/space-separated actor ids to plot GT vs XML yaw over the CARLA map.",
    )
    p.add_argument(
        "--actor-yaw-viz-dir",
        default=None,
        help="Output directory for actor yaw visualizations (default: <out-dir>/actor_yaw_viz).",
    )
    p.add_argument(
        "--actor-yaw-viz-step",
        type=int,
        default=10,
        help="Stride for yaw arrows in actor visualizations (default: 10).",
    )
    p.add_argument(
        "--actor-yaw-viz-arrow-len",
        type=float,
        default=0.8,
        help="Arrow length (meters) for yaw visualizations (default: 0.8).",
    )
    p.add_argument(
        "--actor-yaw-viz-pad",
        type=float,
        default=5.0,
        help="Padding (meters) around GT/XML path extents for actor yaw visualizations.",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-ids",
        default="",
        help="Comma/space-separated actor ids to plot raw YAML points by subfolder.",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-dir",
        default=None,
        help="Output directory for raw YAML actor visualizations (default: <out-dir>/actor_raw_yaml_viz).",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-pad",
        type=float,
        default=20.0,
        help="Padding (meters) around raw YAML points for actor visualizations.",
    )
    p.add_argument(
        "--ego-alignment-viz",
        action="store_true",
        help="Plot ego GT path (YAML + offsets) vs post-alignment/refinement path.",
    )
    p.add_argument(
        "--ego-alignment-viz-dir",
        default=None,
        help="Output directory for ego alignment visualizations (default: <out-dir>/ego_alignment_viz).",
    )
    p.add_argument(
        "--ego-alignment-viz-pad",
        type=float,
        default=24.0,
        help="Padding (meters) around ego pre/post alignment paths (default: 24).",
    )
    p.add_argument(
        "--ego-alignment-bev-viz",
        action="store_true",
        help="Write ego alignment overlay on a CARLA BEV/raster image with map-node highlights.",
    )
    p.add_argument(
        "--ego-alignment-bev-viz-dir",
        default=None,
        help="Output directory for ego BEV alignment visualizations (default: <out-dir>/ego_alignment_bev_viz).",
    )
    p.add_argument(
        "--actor-alignment-bev-viz",
        action="store_true",
        help="Write actor alignment BEV visualizations (all actors, subsets, and per-actor).",
    )
    p.add_argument(
        "--actor-alignment-bev-viz-dir",
        default=None,
        help="Output directory for actor BEV alignment visualizations (default: <out-dir>/actor_alignment_bev_viz).",
    )
    p.add_argument(
        "--ego-alignment-bev-node-step",
        type=int,
        default=2,
        help="Stride when sampling map nodes for ego BEV viz boxes (default: 2).",
    )
    p.add_argument(
        "--ego-alignment-bev-match-radius",
        type=float,
        default=1.8,
        help="Radius (m) used to mark map nodes touched by aligned ego path (default: 1.8).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-from-carla",
        action="store_true",
        help="Capture a real top-down CARLA RGB image for ego BEV alignment viz (default: enabled).",
    )
    p.add_argument(
        "--no-ego-alignment-bev-capture-from-carla",
        dest="ego_alignment_bev_capture_from_carla",
        action="store_false",
        help="Disable CARLA RGB capture and use map image/vector fallback only.",
    )
    p.set_defaults(ego_alignment_bev_capture_from_carla=True)
    p.add_argument(
        "--ego-alignment-bev-capture-width",
        type=int,
        default=2048,
        help="Captured CARLA BEV width in pixels (default: 2048).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-height",
        type=int,
        default=2048,
        help="Captured CARLA BEV height in pixels (default: 2048).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-fov",
        type=float,
        default=70.0,
        help="Camera FOV (deg) for CARLA BEV capture (default: 70).",
    )
    p.add_argument(
        "--ego-alignment-bev-capture-margin",
        type=float,
        default=1.18,
        help="Margin scale around ego bounds when capturing CARLA BEV image (default: 1.18).",
    )
    p.add_argument("--map-pkl", default=None, help="Optional pickle containing vector map polylines to overlay")
    p.add_argument("--use-carla-map", default=True, action="store_true", help="Connect to CARLA to fetch map polylines for overlay")
    p.add_argument("--carla-host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    p.add_argument("--carla-port", type=int, default=2010, help="CARLA port (default: 2010)")
    p.add_argument("--carla-sample", type=float, default=2.0, help="Waypoint sampling distance in meters (default: 2.0)")
    p.add_argument("--carla-cache", default=None, help="Path to cache map polylines (default: <out-dir>/carla_map_cache.pkl)")
    p.add_argument("--expected-town", default="ucla_v2", help="Assert CARLA map name contains this string when using --use-carla-map")
    p.add_argument("--axis-pad", type=float, default=10.0, help="Padding (meters) around actor/ego extents for visualization axes")
    p.add_argument("--flip-y", action="store_true", help="Mirror dataset Y axis and negate yaw (useful if overlay appears upside-down)")
    p.add_argument("--invert-plot-y", action="store_true", help="Invert matplotlib Y axis for visualization only")
    p.add_argument(
        "--spawn-viz",
        action="store_true",
        help="Generate a spawn-vs-aligned visualization over CARLA map and XODR layers.",
    )
    p.add_argument(
        "--spawn-viz-path",
        default=None,
        help="Output path for spawn-vs-aligned visualization (default: <out-dir>/spawn_alignment_viz.png).",
    )
    p.add_argument(
        "--xodr",
        default=None,
        help="Path to the OpenDRIVE XODR file for spawn visualization overlay.",
    )
    p.add_argument(
        "--xodr-step",
        type=float,
        default=2.0,
        help="Sampling step size (meters) for XODR geometry (default: 2.0).",
    )
    p.add_argument(
        "--map-image",
        default=None,
        help="Optional raster map image to use as the CARLA background layer (PNG/JPG).",
    )
    p.add_argument(
        "--map-image-bounds",
        nargs=4,
        type=float,
        default=None,
        metavar=("MINX", "MAXX", "MINY", "MAXY"),
        help="World bounds for the map image (minx maxx miny maxy). If omitted, bounds are inferred.",
    )
    p.add_argument(
        "--spawn-preprocess",
        action="store_true",
        help="Run CARLA-in-the-loop spawn preprocessing to improve actor spawn success.",
    )
    p.add_argument(
        "--spawn-preprocess-require-carla",
        action="store_true",
        help="Fail export if spawn preprocessing cannot import/connect CARLA (instead of skipping).",
    )
    p.add_argument(
        "--spawn-preprocess-maximal",
        action="store_true",
        help=(
            "Enable an aggressive CARLA-in-the-loop preprocessing preset: "
            "strict CARLA connectivity, higher candidate/refinement budgets, "
            "and timing optimizations."
        ),
    )
    p.add_argument(
        "--spawn-preprocess-report",
        default=None,
        help="Optional JSON report path for spawn preprocessing results.",
    )
    p.add_argument(
        "--spawn-preprocess-max-shift",
        type=float,
        default=4.0,
        help="Maximum XY shift (meters) when generating spawn candidates (default: 4.0).",
    )
    p.add_argument(
        "--spawn-preprocess-random-samples",
        type=int,
        default=80,
        help="Number of random candidate offsets per actor (default: 80).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz",
        action="store_true",
        help="Generate visualization for actors that failed to spawn (over CARLA map).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-dir",
        default=None,
        help="Output directory for failed spawn visualizations (default: <out-dir>/spawn_preprocess_fail_viz).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-window",
        type=float,
        default=60.0,
        help="Window size (meters) for per-actor failed spawn plots (default: 60).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-dpi",
        type=int,
        default=220,
        help="DPI for failed spawn visualizations (default: 220).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-sample",
        type=float,
        default=1.0,
        help="CARLA map sampling distance for failed spawn visualizations (default: 1.0).",
    )
    p.add_argument(
        "--spawn-preprocess-debug-radius",
        type=float,
        default=30.0,
        help="Radius (meters) for collecting nearby actors/env objects in failed spawn debug (default: 30).",
    )
    p.add_argument(
        "--spawn-preprocess-debug-max-items",
        type=int,
        default=10,
        help="Max nearby actors/env objects to record per failed spawn (default: 10).",
    )
    p.add_argument(
        "--spawn-preprocess-grid",
        default="0.0,0.2,-0.2,0.4,-0.4,0.8,-0.8,1.2,-1.2",
        help="Comma/space-separated XY offsets (meters) for local candidate grid.",
    )
    p.add_argument(
        "--spawn-preprocess-sample-dt",
        type=float,
        default=0.5,
        help="Sampling timestep (seconds) for collision scoring (default: 0.5).",
    )
    p.add_argument(
        "--spawn-preprocess-grid-size",
        type=float,
        default=5.0,
        help="Spatial hash grid size (meters) for collision checks (default: 5.0).",
    )
    p.add_argument(
        "--spawn-preprocess-max-candidates",
        type=int,
        default=60,
        help="Maximum candidate offsets per actor (default: 60).",
    )
    p.add_argument(
        "--spawn-preprocess-collision-weight",
        type=float,
        default=50.0,
        help="Weight for collision penalty in candidate scoring (default: 50.0).",
    )
    p.add_argument(
        "--spawn-preprocess-verbose",
        action="store_true",
        help="Enable verbose spawn preprocessing logs.",
    )
    p.add_argument(
        "--spawn-preprocess-align",
        action="store_true",
        help="Enable multi-waypoint alignment candidates during spawn preprocess (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-align",
        dest="spawn_preprocess_align",
        action="store_false",
        help="Disable multi-waypoint alignment candidates during spawn preprocess.",
    )
    p.set_defaults(spawn_preprocess_align=True)
    p.add_argument(
        "--spawn-preprocess-align-samples",
        type=int,
        default=12,
        help="Number of trajectory samples for alignment candidate generation.",
    )
    p.add_argument(
        "--spawn-preprocess-align-windows",
        type=int,
        default=3,
        help="Number of trajectory windows for slice-based alignment candidates.",
    )
    p.add_argument(
        "--spawn-preprocess-align-intent-margin",
        type=float,
        default=0.8,
        help="Distance margin (m) used to infer sidewalk vs road intent for walkers.",
    )
    p.add_argument(
        "--spawn-preprocess-align-neighbor-radius",
        type=float,
        default=6.0,
        help="Radius (m) for neighbor-aware alignment bias among nearby vehicles.",
    )
    p.add_argument(
        "--spawn-preprocess-align-neighbor-weight",
        type=float,
        default=0.15,
        help="Weight for neighbor offset coherence during global candidate selection.",
    )
    p.add_argument(
        "--spawn-preprocess-refine-piecewise",
        action="store_true",
        help="After global actor shift selection, run bounded per-waypoint refinement (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-refine-piecewise",
        dest="spawn_preprocess_refine_piecewise",
        action="store_false",
        help="Disable second-stage per-waypoint actor refinement.",
    )
    p.set_defaults(spawn_preprocess_refine_piecewise=True)
    p.add_argument(
        "--spawn-preprocess-refine-max-local",
        type=float,
        default=0.8,
        help="Max local deviation (m) from global actor shift in piecewise refinement (default: 0.8).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-smooth-window",
        type=int,
        default=7,
        help="Smoothing window (waypoints) for actor piecewise refinement (default: 7).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-max-step-delta",
        type=float,
        default=0.35,
        help="Max change (m) between adjacent actor offsets in piecewise refinement (default: 0.35).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-early-lane-lock-seconds",
        type=float,
        default=1.0,
        help="For NPC/static piecewise refinement, keep initial lane for this many seconds unless alternative is clearly better (default: 1.0).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-early-lane-switch-override-margin",
        type=float,
        default=0.9,
        help="Minimum score improvement required to allow early lane switch during initial-lane lock (default: 0.9).",
    )
    p.add_argument(
        "--spawn-preprocess-refine-collision-slack",
        type=float,
        default=0.0,
        help="Allowable collision score increase when accepting actor refinement (default: 0.0).",
    )
    p.add_argument(
        "--spawn-preprocess-bridge-max-gap-steps",
        type=int,
        default=6,
        help="Max consecutive waypoints treated as a map-projection gap for lane-bridge interpolation (default: 6). Set 0 to disable.",
    )
    p.add_argument(
        "--spawn-preprocess-bridge-straight-thresh-deg",
        type=float,
        default=18.0,
        help="Max heading change (deg) across a bridged segment; larger turns will not be bridged (default: 18).",
    )
    p.add_argument(
        "--spawn-preprocess-align-ego",
        action="store_true",
        help="Align ego trajectories during spawn preprocess (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-align-ego",
        dest="spawn_preprocess_align_ego",
        action="store_false",
        help="Disable ego trajectory alignment during spawn preprocess.",
    )
    p.set_defaults(spawn_preprocess_align_ego=True)
    p.add_argument(
        "--spawn-preprocess-align-ego-piecewise",
        action="store_true",
        help="Use per-waypoint ego alignment (not a single global offset) during spawn preprocess (default: enabled).",
    )
    p.add_argument(
        "--no-spawn-preprocess-align-ego-piecewise",
        dest="spawn_preprocess_align_ego_piecewise",
        action="store_false",
        help="Use only a single global ego offset during spawn preprocess.",
    )
    p.set_defaults(spawn_preprocess_align_ego_piecewise=True)
    p.add_argument(
        "--spawn-preprocess-align-ego-smooth-window",
        type=int,
        default=9,
        help="Smoothing window (waypoints) for piecewise ego offset profile (default: 9).",
    )
    p.add_argument(
        "--spawn-preprocess-align-ego-max-step-delta",
        type=float,
        default=0.45,
        help="Max change (m) in ego offset between adjacent waypoints for piecewise mode (default: 0.45).",
    )
    p.add_argument(
        "--snap-ego-to-lane",
        dest="snap_ego_to_lane",
        action="store_true",
        help=(
            "When aligning ego trajectories, lock snapping to a continuous forward lane-center path "
            "(prevents lane changes while still allowing turns; default: enabled)."
        ),
    )
    p.add_argument(
        "--no-snap-ego-to-lane",
        dest="snap_ego_to_lane",
        action="store_false",
        help="Disable continuous lane-lock snapping for ego trajectories.",
    )
    p.set_defaults(snap_ego_to_lane=True)
    p.add_argument(
        "--parked-clearance",
        dest="parked_clearance",
        action="store_true",
        help=(
            "For static parked/off-side vehicles, apply minimal XY nudges when alignment introduces "
            "new bounding-box interference with moving vehicle paths (default: enabled)."
        ),
    )
    p.add_argument(
        "--no-parked-clearance",
        dest="parked_clearance",
        action="store_false",
        help="Disable parked/off-side clearance nudging.",
    )
    p.set_defaults(parked_clearance=True)
    p.add_argument(
        "--parked-clearance-sample-dt",
        type=float,
        default=0.2,
        help="Sampling timestep (seconds) for parked clearance overlap checks (default: 0.2).",
    )
    p.add_argument(
        "--parked-clearance-max-shift",
        type=float,
        default=1.0,
        help="Maximum XY nudge (meters) allowed for parked clearance adjustment (default: 1.0).",
    )
    p.add_argument(
        "--parked-clearance-shift-step",
        type=float,
        default=0.15,
        help="Search step (meters) for parked clearance shift candidates (default: 0.15).",
    )
    p.add_argument(
        "--parked-clearance-angle-count",
        type=int,
        default=16,
        help="Number of angular directions sampled per parked-clearance radius (default: 16).",
    )
    p.add_argument(
        "--spawn-preprocess-normalize-z",
        action="store_true",
        default=True,
        help="Use ground projection when validating spawn candidates (default: on).",
    )
    p.add_argument(
        "--no-spawn-preprocess-normalize-z",
        dest="spawn_preprocess_normalize_z",
        action="store_false",
        help="Disable ground projection during spawn candidate validation.",
    )
    p.add_argument("--run-custom-eval", action="store_true", help="After export, call tools/run_custom_eval.py with the generated routes dir")
    p.add_argument(
        "--eval-planner",
        default="",
        help="Planner for run_custom_eval (empty string means no planner flag; e.g., pass 'tcp' or 'log_replay')",
    )
    p.add_argument("--eval-port", type=int, default=2014, help="CARLA port for run_custom_eval (default: 2014)")
    return p.parse_args()

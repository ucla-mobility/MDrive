"""Shared open-loop instrumentation for any leaderboard agent that wants to
participate in the perception-swap evaluation pipeline (AP@IoU vs CARLA GT,
visible-actor filter against the v2x-real-pnp source dataset, GT-replay
loading for ADE/FDE, JSON output schema compatible with
``tools/openloop_post_metrics.py``).

Why this module exists
----------------------
The full pipeline was originally written into ``perception_swap_agent.py``.
Bringing colmdriver / colmdriver_rulebase / codriving into the same
benchmark requires the same wiring (AP collection per tick, visible-id
filter, replay-anchored ADE) without duplicating code. This module exposes
two stable building blocks any agent can opt into:

    OpenLoopAPHook
        A self-contained orchestrator. Constructed once in setup(), held by
        the agent, fed perception predictions per tick, dumps the
        ``perception_swap_ap_ego{N}.json`` files at destroy. When openloop
        mode isn't active (``--openloop`` flag off), the hook detects that
        and stays inert — every method returns immediately. So adding
        ``self._oloop = OpenLoopAPHook(...)`` to a closed-loop agent has
        zero behavioral cost.

    CarlaGTApCollector
        The per-ego AP accumulator. World-frame predictions are accepted
        natively (codriving / colmdriver path); the legacy ego-frame path
        used by perception_swap_agent.py also still works. GT comes from
        ``world.get_actors().filter('vehicle.*')`` and is restricted by the
        visible-id filter when active.

Behavioral guarantees
---------------------
- When ``OpenLoopAPHook.enabled`` is False (the env flag isn't set), every
  call is a fast no-op. There's no CARLA query, no file IO, no allocation.
- When enabled, every CARLA query is wrapped in try/except so a perception
  failure can't kill the drive. Errors are rate-limited prints, not raises.
- File output is always to per-scene paths under ``SAVE_PATH`` so concurrent
  scenarios in the harness pool don't clobber each other.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# REPLAY xml parsing + GT lookup (used by per-tick ADE display in BEV viz)
# ──────────────────────────────────────────────────────────────────────────
def parse_ego_replay_xml(xml_path: str):
    """Parse a ucla_v2_custom_ego_vehicle_<i>_REPLAY.xml into (times, xy).

    Returns (np.ndarray (T,), np.ndarray (T, 2)) sorted by time, or
    (None, None) if the file is missing/unparseable.
    """
    if not xml_path or not os.path.isfile(xml_path):
        return None, None
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, None
    ts, xs, ys = [], [], []
    for wp in root.iter("waypoint"):
        try:
            ts.append(float(wp.attrib["time"]))
            xs.append(float(wp.attrib["x"]))
            ys.append(float(wp.attrib["y"]))
        except (KeyError, ValueError):
            continue
    if not ts:
        return None, None
    order = np.argsort(ts)
    times = np.asarray(ts, dtype=np.float64)[order]
    xy = np.asarray(list(zip(xs, ys)), dtype=np.float64)[order]
    return times, xy


def gt_xy_at(times, xy, t):
    """Linear interpolation of (times, xy) at scalar ``t``. Clamps to ends.

    Returns ``np.ndarray (2,)`` or ``None`` if data is missing.
    """
    if times is None or xy is None or len(times) == 0:
        return None
    if t <= times[0]:
        return xy[0].copy()
    if t >= times[-1]:
        return xy[-1].copy()
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = max(0, min(i, len(times) - 2))
    span = times[i + 1] - times[i]
    if span <= 1e-9:
        return xy[i].copy()
    frac = (t - times[i]) / span
    return xy[i] + frac * (xy[i + 1] - xy[i])


# ──────────────────────────────────────────────────────────────────────────
# Visible-actor filter from the v2x-real-pnp source dataset
# ──────────────────────────────────────────────────────────────────────────
_V2X_REAL_PNP_SPLITS = ("train", "val", "test", "v2v_test")


def resolve_real_dataset_scene_dir(scene_basename: str,
                                    real_root: str) -> Optional[str]:
    """Return the v2x-real-pnp <split>/<scene> dir for ``scene_basename``,
    or None if the scene isn't present in any split."""
    if not scene_basename or not real_root or not os.path.isdir(real_root):
        return None
    for split in _V2X_REAL_PNP_SPLITS:
        cand = os.path.join(real_root, split, scene_basename)
        if os.path.isdir(cand):
            return cand
    return None


def load_visible_ids_per_frame(agent_dir: str):
    """Parse all <frame>.yaml files in a v2x-real-pnp agent dir into
    ``{frame_idx: set[str]}`` of vehicle IDs that agent observed at that frame.

    The IDs come from the YAML's top-level ``vehicles:`` map. Keys are cast to
    str so they can be compared against CARLA ``actor.attributes['role_name']``
    (which is always a string regardless of how the manifest names were typed).

    Returns ``({}, 0)`` if the dir is missing or no frames parse.

    Avoids the PyYAML dependency (not all conda envs have it) — the
    ``vehicles:`` block is shallow enough that a hand-rolled scan over only the
    integer-keyed entries at the right indent works fine.
    """
    out: Dict[int, set] = {}
    if not agent_dir or not os.path.isdir(agent_dir):
        return out, 0
    yaml_files = sorted(
        f for f in os.listdir(agent_dir)
        if f.endswith(".yaml") and f[:-5].isdigit()
    )
    parsed = 0
    for fn in yaml_files:
        try:
            frame_idx = int(fn[:-5])
        except ValueError:
            continue
        path = os.path.join(agent_dir, fn)
        ids = set()
        in_vehicles = False
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("vehicles:"):
                        in_vehicles = True
                        continue
                    if in_vehicles:
                        # End of the block when we hit another top-level key.
                        if line and not line.startswith((" ", "\t")) and line.strip():
                            break
                        # Keys are at exactly two leading spaces, integer + colon.
                        s = line.rstrip("\n")
                        if (len(s) > 3 and s.startswith("  ")
                                and not s.startswith("   ")
                                and s.endswith(":")):
                            key = s[2:-1].strip()
                            if key.isdigit():
                                ids.add(key)
        except Exception:
            continue
        out[frame_idx] = ids
        parsed += 1
    return out, parsed


# ──────────────────────────────────────────────────────────────────────────
# Lazy opencood imports (heavy; only pay the cost when openloop is enabled)
# ──────────────────────────────────────────────────────────────────────────
def _lazy_eval_imports():
    import torch
    from opencood.utils.eval_utils import caluclate_tp_fp, calculate_ap
    from opencood.utils.box_utils import boxes_to_corners_3d
    from opencood.utils.transformation_utils import x_to_world
    return torch, caluclate_tp_fp, calculate_ap, boxes_to_corners_3d, x_to_world


# ──────────────────────────────────────────────────────────────────────────
# AP eval range — single source of truth across agents
# ──────────────────────────────────────────────────────────────────────────
def _openloop_active_from_env() -> bool:
    """Return True if any known openloop signal is set.

    Two flags exist for historical reasons: ``PERCEPTION_SWAP_WRITE_PRED_FILES``
    (the canonical hook enable, set by ``run_custom_eval --openloop``) and
    ``CUSTOM_EGO_LOG_REPLAY`` (legacy ego-teleport mode, opt-in only). Either
    triggers the openloop AP crop.
    """
    return (
        os.environ.get("PERCEPTION_SWAP_WRITE_PRED_FILES", "").lower()
            in ("1", "true", "yes", "on")
        or os.environ.get("CUSTOM_EGO_LOG_REPLAY", "0") == "1"
    )


def resolve_openloop_ap_eval_range(model_lidar_range,
                                   *,
                                   openloop_active: Optional[bool] = None):
    """Single source of truth for the AABB used to crop AP scoring + BEV.

    In openloop runs, return an asymmetric crop matching the codriving /
    colmdriver detection-output envelope: 36 m forward, 12 m rear, ±12 m
    lateral (i.e. ``x ∈ [-12, 36], y ∈ [-12, 12]``, 48 × 24 m, front-heavy).
    These are the bounds beyond which the codriving / colmdriver heads do
    not produce predictions, so AABBs larger than this just contribute
    guaranteed false-negatives to AP and unfairly depress it.

    Tunable via four env vars (all metres, >= 0):
        OPENLOOP_AP_RANGE_FRONT   default 36
        OPENLOOP_AP_RANGE_REAR    default 12
        OPENLOOP_AP_RANGE_LEFT    default 12
        OPENLOOP_AP_RANGE_RIGHT   default 12

    The legacy symmetric env vars ``OPENLOOP_AP_RANGE_X / _Y`` (and aliases
    ``PERCEPTION_SWAP_AP_RANGE_X / _Y``) are still honored for back-compat
    and expand to ``front=rear=X, left=right=Y``. Per-direction vars
    override the legacy pair.

    Per-tick prediction NPZs are NOT range-filtered before being written,
    so re-running with a wider eval range will pick up any predictions
    the model emitted out there. To compare AP at different ranges, set
    ``OPENLOOP_AP_RANGE_FRONT/REAR/LEFT/RIGHT`` to the desired bounds
    and rerun (or post-process the cached preds).

    In closed-loop runs, return the model's full ``cav_lidar_range`` — the
    same thing the perception forward pass sees. Closed-loop AP is unchanged
    by this function (only DS/RC/CR/HUG are primary closed-loop metrics).

    Pass ``openloop_active=True/False`` to override env detection (used by
    tests and by perception_swap_agent which has its own openloop signal).
    """
    if openloop_active is None:
        openloop_active = _openloop_active_from_env()
    lr = (list(model_lidar_range)
          if model_lidar_range is not None and len(model_lidar_range) >= 6
          else [-70.4, -40.0, -3.0, 70.4, 40.0, 1.0])
    if not openloop_active:
        return lr
    # Legacy symmetric envs (if set) become the fallback for all four
    # asymmetric directions: front=rear=X, left=right=Y. Per-direction
    # envs override them.
    legacy_x = os.environ.get(
        "OPENLOOP_AP_RANGE_X",
        os.environ.get("PERCEPTION_SWAP_AP_RANGE_X"),
    )
    legacy_y = os.environ.get(
        "OPENLOOP_AP_RANGE_Y",
        os.environ.get("PERCEPTION_SWAP_AP_RANGE_Y"),
    )
    front = float(os.environ.get("OPENLOOP_AP_RANGE_FRONT",
                                 legacy_x if legacy_x is not None else "36"))
    rear  = float(os.environ.get("OPENLOOP_AP_RANGE_REAR",
                                 legacy_x if legacy_x is not None else "12"))
    left  = float(os.environ.get("OPENLOOP_AP_RANGE_LEFT",
                                 legacy_y if legacy_y is not None else "12"))
    right = float(os.environ.get("OPENLOOP_AP_RANGE_RIGHT",
                                 legacy_y if legacy_y is not None else "12"))
    # Convention: bbox is [x_min, y_min, z_min, x_max, y_max, z_max].
    # +x is forward, +y is left (right-handed OPV2V/HEAL frame).
    return [-rear, -right, lr[2], front, left, lr[5]]


# ──────────────────────────────────────────────────────────────────────────
# CarlaGTApCollector — per-ego AP accumulator
# ──────────────────────────────────────────────────────────────────────────
class CarlaGTApCollector:
    """Per-detector AP accumulator using live CARLA actors as the oracle GT.

    The collector is frame-agnostic: pass either ego-frame predictions
    (``pred_frame='ego'``, the perception_swap convention) or world-frame
    predictions (``pred_frame='world'``, the codriving / colmdriver
    convention). GT is built in the same frame so IoU is meaningful.

    Usage:
        coll = CarlaGTApCollector(iou_thresholds=(0.3, 0.5, 0.7),
                                  visible_ids_per_frame=...)
        # each tick:
        coll.update(
            primary_pose_world=[x, y, z, roll, yaw_deg, pitch],
            pred_corners=det.corners,         # (P, 8, 3)
            pred_scores=det.scores,
            lidar_range=cav_lidar_range,
            ego_actor_ids={hero_id_0, ...},
            pred_frame="ego",                 # or "world"
            frame_idx=...,
        )
        # at destroy:
        coll.save("/path/to/perception_swap_ap_ego0.json", extra_meta={...})
    """
    def __init__(self,
                 iou_thresholds=(0.1, 0.3, 0.5, 0.7),
                 self_filter_radius_m: float = 2.0,
                 visible_ids_per_frame: Optional[Dict[int, set]] = None):
        self.iou_thresholds = tuple(iou_thresholds)
        self.self_filter_radius_m = float(self_filter_radius_m)
        self.visible_ids_per_frame: Optional[Dict[int, set]] = (
            visible_ids_per_frame
        )
        self.result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
                            for iou in self.iou_thresholds}
        self.n_frames = 0
        self.cum_n_pred = 0
        self.cum_n_pred_after_self_filter = 0
        self.cum_n_gt = 0
        self.cum_n_self_dets = 0
        self.cum_frames_with_filter = 0
        self.cum_frames_filter_empty = 0

        # Last-frame snapshot, populated by update(). Used by the BEV
        # renderer (codriving / colmdriver) so the on-screen panel can show
        # GT boxes + per-frame TP/FP/FN without re-querying CARLA. The
        # primary IoU threshold defaults to the lowest threshold the
        # collector tracks (so per-frame counts are non-trivial); callers
        # can pass ``iou`` to ``last_frame_state`` to read at a specific IoU.
        self.last_frame: Optional[Dict[str, Any]] = None

    def update(self,
               primary_pose_world,
               pred_corners,
               pred_scores,
               lidar_range,
               ego_actor_ids,
               pred_frame: str = "ego",
               frame_idx: Optional[int] = None) -> None:
        """Append one frame's TP/FP stats.

        Parameters
        ----------
        primary_pose_world : list of 6 floats
            ``[x, y, z, roll, yaw_deg, pitch]``. Used to project GT into
            the prediction frame and to filter to ``lidar_range``.
        pred_corners : np.ndarray of shape (P, 8, 3)
            Predicted bounding boxes. Frame is ``pred_frame``.
        pred_scores : np.ndarray of shape (P,)
        lidar_range : sequence of 6 floats
            ``[xmin, ymin, zmin, xmax, ymax, zmax]`` in the prediction frame.
        ego_actor_ids : iterable of int
            CARLA actor IDs of ego vehicles — excluded from GT.
        pred_frame : "ego" | "world"
            "ego" matches perception_swap_agent's ``Detections.corners``;
            "world" matches codriving's ``pred_box_world`` ndarray.
        frame_idx : int or None
            Source-dataset frame index (0..N-1 in v2x-real-pnp's 10 Hz
            stream) for the visible-actor filter. Pass ``None`` to disable
            filtering for this tick.
        """
        torch, caluclate_tp_fp, _, _, _ = _lazy_eval_imports()

        if pred_frame not in ("ego", "world"):
            raise ValueError(f"pred_frame must be 'ego' or 'world', got {pred_frame!r}")

        visible_ids = self._visible_ids_for_frame(frame_idx)
        gt_corners, gt_corners_world = self._build_gt_corners(
            primary_pose_world=primary_pose_world,
            lidar_range=lidar_range,
            ego_actor_ids=ego_actor_ids,
            pred_frame=pred_frame,
            visible_ids=visible_ids,
        )

        # Self-filter: drop predictions within ``self_filter_radius_m`` of
        # the *ego* in the prediction frame. In world frame "ego" is at
        # ``primary_pose_world[:2]``; in ego frame it's at the origin.
        # Range filter: lidar_range is in EGO/LIDAR frame for both
        # pred_frame variants — for "world" we rotate world centroids back
        # into ego frame before checking against the AABB so the rotated
        # rectangle (not its world-axis-aligned bounding box) is used.
        ego_xy_w = np.asarray(primary_pose_world[:2], dtype=np.float64)
        ego_yaw_deg = float(primary_pose_world[4]) if len(primary_pose_world) > 4 else 0.0
        ego_yaw_rad = np.radians(ego_yaw_deg)
        cy, sy = float(np.cos(ego_yaw_rad)), float(np.sin(ego_yaw_rad))

        n_self_filtered = 0
        if len(pred_corners) > 0:
            centers = pred_corners.mean(axis=1)[:, :2]
            if pred_frame == "world":
                # Translate then rotate world centroids into ego frame.
                dx = centers[:, 0] - ego_xy_w[0]
                dy = centers[:, 1] - ego_xy_w[1]
                xe = cy * dx + sy * dy
                ye = -sy * dx + cy * dy
                centers_ego = np.stack([xe, ye], axis=1)
            else:
                centers_ego = centers
            r2 = self.self_filter_radius_m ** 2
            not_self = (
                centers_ego[:, 0] * centers_ego[:, 0]
                + centers_ego[:, 1] * centers_ego[:, 1]
            ) > r2
            in_range = (
                (centers_ego[:, 0] >= lidar_range[0])
                & (centers_ego[:, 0] <= lidar_range[3])
                & (centers_ego[:, 1] >= lidar_range[1])
                & (centers_ego[:, 1] <= lidar_range[4])
            )
            n_self_filtered = int((~not_self).sum())
            keep = in_range & not_self
            pred_corners = pred_corners[keep]
            pred_scores = pred_scores[keep]
        self.cum_n_self_dets += n_self_filtered
        self.cum_n_pred_after_self_filter += int(len(pred_corners))

        pc = torch.from_numpy(pred_corners.astype(np.float32)) if len(pred_corners) else None
        ps = torch.from_numpy(pred_scores.astype(np.float32)) if len(pred_scores) else None
        gc = torch.from_numpy(gt_corners.astype(np.float32))   # (M, 8, 3)

        # Snapshot pre-call list lengths so we can derive per-frame TP/FP/FN
        # for each IoU threshold from the deltas. ``caluclate_tp_fp`` appends
        # one element per *prediction* to fp/tp lists (after the in-loop
        # match), so ``len_after - len_before`` is exactly this frame's pred
        # count, and ``sum(tp[len_before:])`` is the per-frame TP count.
        per_frame: Dict[float, Dict[str, int]] = {}
        pre_lens = {iou: (len(self.result_stat[iou]["tp"]),
                          int(self.result_stat[iou]["gt"]))
                    for iou in self.iou_thresholds}

        for iou in self.iou_thresholds:
            caluclate_tp_fp(pc, ps, gc, self.result_stat, iou_thresh=iou)

        for iou in self.iou_thresholds:
            tp_pre, gt_pre = pre_lens[iou]
            tp_slice = self.result_stat[iou]["tp"][tp_pre:]
            fp_slice = self.result_stat[iou]["fp"][tp_pre:]
            n_tp_f = int(sum(tp_slice))
            n_fp_f = int(sum(fp_slice))
            n_gt_f = int(self.result_stat[iou]["gt"]) - gt_pre
            per_frame[float(iou)] = {
                "tp": n_tp_f,
                "fp": n_fp_f,
                "fn": max(0, n_gt_f - n_tp_f),
                "gt": n_gt_f,
            }

        self.n_frames += 1
        self.cum_n_pred += int(len(pred_corners))
        self.cum_n_gt += int(len(gt_corners))

        # Per-box matching for BEV viz: greedy score-descending, IoU >= 0.5.
        # Mirrors what ``caluclate_tp_fp`` does internally so the labels on
        # screen agree with the AP@0.5 numbers. Cost is one extra IoU pass
        # per frame (negligible vs. model inference). Pred status: 1=TP,
        # 0=FP. GT status: True=matched, False=missed (FN).
        viz_iou = 0.5
        n_pred = int(len(pred_corners))
        n_gt = int(len(gt_corners))
        pred_status_viz = np.zeros(n_pred, dtype=np.int8)
        pred_iou_viz = np.zeros(n_pred, dtype=np.float32)
        gt_matched_viz = np.zeros(n_gt, dtype=bool)
        if n_pred > 0 and n_gt > 0:
            try:
                from opencood.utils import common_utils
                pred_polys = list(common_utils.convert_format(pred_corners))
                gt_polys = list(common_utils.convert_format(gt_corners))
                score_order = np.argsort(-pred_scores)
                for k in score_order:
                    available = ~gt_matched_viz
                    if not available.any():
                        break
                    avail_idx = np.where(available)[0]
                    avail_polys = [gt_polys[g] for g in avail_idx]
                    ious = common_utils.compute_iou(pred_polys[int(k)], avail_polys)
                    best_local = int(np.argmax(ious))
                    best_iou = float(ious[best_local])
                    pred_iou_viz[int(k)] = best_iou
                    if best_iou >= viz_iou:
                        gt_matched_viz[int(avail_idx[best_local])] = True
                        pred_status_viz[int(k)] = 1
            except Exception:
                # Never let viz matching break AP collection. Worst case the
                # BEV gets uniform-colored boxes (the pre-matcher behavior).
                pred_status_viz = np.zeros(n_pred, dtype=np.int8)
                gt_matched_viz = np.zeros(n_gt, dtype=bool)
                pred_iou_viz = np.zeros(n_pred, dtype=np.float32)

        # Stash GT corners for the BEV renderer. ``gt_corners_world`` is
        # always CARLA world coords (built by ``_build_gt_corners``
        # alongside the pred-frame copy used for AP) so the renderer can
        # plot them directly regardless of which ``pred_frame`` the agent
        # uses. ``gt_corners`` is the AP-frame copy (legacy field name).
        # ``pred_status`` / ``gt_matched`` enable per-box TP/FP/FN coloring.
        self.last_frame = {
            "pred_frame": pred_frame,
            "gt_corners": gt_corners.copy() if len(gt_corners) else
                          np.zeros((0, 8, 3), dtype=np.float32),
            "gt_corners_world": (gt_corners_world.copy() if len(gt_corners_world)
                                  else np.zeros((0, 8, 3), dtype=np.float32)),
            "pred_status": pred_status_viz,   # (P,) int8: 1=TP, 0=FP
            "pred_iou":    pred_iou_viz,      # (P,) float32: best IoU per pred
            "gt_matched":  gt_matched_viz,    # (M,) bool: True=matched
            "n_pred_after_filter": int(len(pred_corners)),
            "per_frame": per_frame,
        }

    def _visible_ids_for_frame(self, frame_idx: Optional[int]):
        """Return the set of role_names visible at ``frame_idx`` per the
        v2x-real-pnp source dataset, or ``None`` to disable filtering.

        Falls back to the nearest recorded frame if the requested frame_idx
        is outside the recorded range — this absorbs minor tick→frame
        rounding without silently dropping all GT.
        """
        if self.visible_ids_per_frame is None or frame_idx is None:
            return None
        if frame_idx in self.visible_ids_per_frame:
            return self.visible_ids_per_frame[frame_idx]
        keys = self.visible_ids_per_frame.keys()
        if not keys:
            return set()
        nearest = min(keys, key=lambda k: abs(k - frame_idx))
        return self.visible_ids_per_frame[nearest]

    def _build_gt_corners(self, primary_pose_world, lidar_range,
                          ego_actor_ids, pred_frame: str,
                          visible_ids=None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Query the CARLA world for vehicles. Returns
        ``(gt_corners_pred_frame, gt_corners_world)`` — both arrays cover the
        SAME actors in the SAME order, so an index in either refers to the
        same GT box. Each array is ``(M, 8, 3)`` with corners[:4] = bottom
        face CCW (matches opencood ``boxes_to_corners_3d`` convention).

        The world-frame copy is used by the BEV renderer (which always plots
        in CARLA world coords); the pred-frame copy is what gets fed into
        ``caluclate_tp_fp`` so AP matches the same set the renderer shows.
        """
        # Imports gated to avoid eager carla_data_provider load.
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        _, _, _, _, x_to_world = _lazy_eval_imports()

        empty = np.zeros((0, 8, 3), dtype=np.float32)
        world = CarlaDataProvider.get_world()
        if world is None:
            return empty, empty

        all_vehicles = list(world.get_actors().filter("vehicle.*"))
        gt_actors = [a for a in all_vehicles if a.id not in ego_actor_ids]
        if visible_ids is not None:
            self.cum_frames_with_filter += 1
            if not visible_ids:
                self.cum_frames_filter_empty += 1
            gt_actors = [
                a for a in gt_actors
                if str(a.attributes.get("role_name", "")) in visible_ids
            ]
        if not gt_actors:
            return empty, empty

        # Build world→ego transform once. Only used for ``pred_frame='ego'``.
        T_ego_world = None
        ex_w, ey_w = float(primary_pose_world[0]), float(primary_pose_world[1])
        if pred_frame == "ego":
            T_world_ego = x_to_world(primary_pose_world)
            T_ego_world = np.linalg.inv(T_world_ego)

        ego_yaw_deg = float(primary_pose_world[4]) if len(primary_pose_world) > 4 else 0.0
        ego_yaw_rad = np.radians(ego_yaw_deg)
        cy_, sy_ = float(np.cos(ego_yaw_rad)), float(np.sin(ego_yaw_rad))

        all_corners_pred: List[np.ndarray] = []
        all_corners_world: List[np.ndarray] = []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            corners_w = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if corners_w.shape != (8, 3):
                continue

            # Reorder world corners so corners[:4] = bottom face CCW. This
            # ordering is the canonical one we expose to the renderer; the
            # pred-frame copy below uses the SAME index permutation so the
            # two views stay aligned per-corner.
            order_z_w = np.argsort(corners_w[:, 2])
            bot_w = corners_w[order_z_w[:4]]
            top_w = corners_w[order_z_w[4:]]
            bot_w_c = bot_w[:, :2].mean(axis=0)
            ang_w = np.arctan2(bot_w[:, 1] - bot_w_c[1],
                               bot_w[:, 0] - bot_w_c[0])
            bot_order = np.argsort(ang_w)
            bot_w = bot_w[bot_order]
            matched_top_w = np.zeros_like(top_w)
            top_match_idx = np.full(4, -1, dtype=int)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top_w[:, :2] - bot_w[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                matched_top_w[i] = top_w[j]
                top_match_idx[i] = j
            corn_w = np.concatenate([bot_w, matched_top_w], axis=0)

            # Build the pred-frame corners using the same 8 source indices so
            # the per-corner correspondence with corn_w is preserved.
            bot_w_idx = order_z_w[:4][bot_order]
            top_w_idx = order_z_w[4:][top_match_idx]
            src_idx = np.concatenate([bot_w_idx, top_w_idx])
            if pred_frame == "ego":
                ones = np.ones((8, 1), dtype=np.float64)
                hom_w = np.concatenate([corners_w, ones], axis=1)
                hom_e = (T_ego_world @ hom_w.T).T
                corn_p = hom_e[src_idx, :3]
            else:
                corn_p = corn_w  # already world

            # Range filter on centroid (in ego/lidar frame for both variants).
            c_w = corn_w.mean(axis=0)
            if pred_frame == "world":
                cx_w, cy_w = float(c_w[0]), float(c_w[1])
                dx_, dy_ = cx_w - ex_w, cy_w - ey_w
                xe_ = cy_ * dx_ + sy_ * dy_
                ye_ = -sy_ * dx_ + cy_ * dy_
                if not (lidar_range[0] <= xe_ <= lidar_range[3] and
                        lidar_range[1] <= ye_ <= lidar_range[4]):
                    continue
            else:
                c_p = corn_p.mean(axis=0)
                if not (lidar_range[0] <= c_p[0] <= lidar_range[3] and
                        lidar_range[1] <= c_p[1] <= lidar_range[4]):
                    continue
            all_corners_pred.append(corn_p.astype(np.float32))
            all_corners_world.append(corn_w.astype(np.float32))

        if not all_corners_pred:
            return empty, empty
        return (np.stack(all_corners_pred, axis=0),
                np.stack(all_corners_world, axis=0))

    def _build_gt_world_with_meta(self, ego_actor_ids, visible_actor_ids=None):
        """Snapshot every CARLA vehicle (excluding egos) in WORLD frame plus
        per-actor metadata, for per-tick GT cache dumps. No range filter — the
        downstream AP recompute applies its own range. Mirrors the schema
        used by ``perception_swap_agent.CarlaGTApCollector._build_gt_world_with_meta``
        so a single post-hoc analysis script can consume both pipelines.

        Returns ``(corners_world, actor_ids, role_names, type_ids)`` with
        ``corners_world`` shape ``(M, 8, 3)`` (bottom-CCW first, top second —
        same ordering as opencood ``boxes_to_corners_3d``). When
        ``visible_actor_ids`` is non-None it must be the set of actor IDs to
        keep (visibility-filter); ``None`` keeps every non-ego vehicle.
        """
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        world = CarlaDataProvider.get_world()
        if world is None:
            return (np.zeros((0, 8, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), [], [])
        all_vehicles = list(world.get_actors().filter("vehicle.*"))
        gt_actors = [a for a in all_vehicles if a.id not in ego_actor_ids]
        if visible_actor_ids is not None:
            gt_actors = [a for a in gt_actors if a.id in visible_actor_ids]
        if not gt_actors:
            return (np.zeros((0, 8, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), [], [])
        out_corners, out_ids, out_roles, out_types = [], [], [], []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            corners_w = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if corners_w.shape != (8, 3):
                continue
            order_z = np.argsort(corners_w[:, 2])
            bot = corners_w[order_z[:4]]
            top = corners_w[order_z[4:]]
            bot_c = bot[:, :2].mean(axis=0)
            ang = np.arctan2(bot[:, 1] - bot_c[1], bot[:, 0] - bot_c[0])
            bot = bot[np.argsort(ang)]
            matched_top = np.zeros_like(top)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                matched_top[i] = top[j]
            out_corners.append(np.concatenate([bot, matched_top], axis=0).astype(np.float32))
            out_ids.append(int(actor.id))
            out_roles.append(str(actor.attributes.get("role_name", "")))
            out_types.append(str(actor.type_id))
        if not out_corners:
            return (np.zeros((0, 8, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), [], [])
        return (np.stack(out_corners, axis=0),
                np.array(out_ids, dtype=np.int64),
                out_roles, out_types)

    def save(self, out_path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Compute final AP table and write JSON. Idempotent — call from
        destroy or at periodic snapshot times during the run."""
        _, _, calculate_ap, _, _ = _lazy_eval_imports()
        ap_table: Dict[str, Any] = {}
        for iou in self.iou_thresholds:
            n_tp = int(np.sum(self.result_stat[iou]["tp"]))
            n_fp = int(np.sum(self.result_stat[iou]["fp"]))
            n_gt = int(self.result_stat[iou]["gt"])
            if n_gt == 0 or (n_tp + n_fp) == 0:
                ap_value = 0.0
            else:
                try:
                    ap_value, _, _ = calculate_ap(self.result_stat, iou)
                    ap_value = float(ap_value)
                except ZeroDivisionError:
                    ap_value = 0.0
            ap_table[f"ap_{int(iou*100):02d}"] = {
                "ap":   ap_value,
                "iou":  float(iou),
                "n_tp": n_tp,
                "n_fp": n_fp,
                "n_gt": n_gt,
            }
        summary = {
            "n_frames":   self.n_frames,
            "cum_n_pred": self.cum_n_pred,
            "cum_n_pred_after_self_filter": self.cum_n_pred_after_self_filter,
            "cum_n_self_dets": self.cum_n_self_dets,
            "cum_n_gt":   self.cum_n_gt,
            "gt_filter": {
                "enabled": self.visible_ids_per_frame is not None,
                "cum_frames_with_filter": self.cum_frames_with_filter,
                "cum_frames_filter_empty": self.cum_frames_filter_empty,
                "n_frames_indexed": (
                    len(self.visible_ids_per_frame)
                    if self.visible_ids_per_frame is not None else 0
                ),
            },
            "ap":         ap_table,
        }
        if extra_meta:
            summary.update(extra_meta)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────
# OpenLoopAPHook — single object an agent holds to opt into the pipeline
# ──────────────────────────────────────────────────────────────────────────
def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "").lower() in (
        "1", "true", "yes", "on"
    )


class OpenLoopAPHook:
    """Per-agent open-loop instrumentation. Construct once in the agent's
    setup(); call ``update_world(...)`` or ``update_ego(...)`` per tick from
    run_step(); call ``save_all()`` from destroy().

    When ``--openloop`` isn't active, ``self.enabled`` is False and every
    method is a no-op — the hook costs nothing in closed-loop runs.

    Detection of openloop mode:
        ``PERCEPTION_SWAP_WRITE_PRED_FILES=1`` is the canonical openloop
        signal — set by ``run_custom_eval --openloop``. We treat that as
        the enable flag for AP collection too (any agent that's emitting
        per-tick pred files in openloop mode wants AP collection).
    """

    AP_FILE_FMT = "perception_swap_ap_ego{ego_idx}.json"

    def __init__(self,
                 ego_vehicles_num: int,
                 *,
                 detector_label: str = "external",
                 lidar_range: Optional[List[float]] = None,
                 iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
                 ap_save_root: Optional[str] = None,
                 enabled: Optional[bool] = None) -> None:
        self.ego_vehicles_num = int(ego_vehicles_num)
        self.detector_label = str(detector_label)
        self.lidar_range = list(lidar_range) if lidar_range else \
            [-102.4, -40.0, -3.0, 102.4, 40.0, 1.0]
        self.iou_thresholds = tuple(iou_thresholds)
        # Where to write the AP json. Defaults to <SAVE_PATH>/.
        if ap_save_root is None:
            ap_save_root = os.environ.get("SAVE_PATH", "")
        self.ap_save_root = ap_save_root or "/tmp"

        # Auto-detect openloop unless caller forced it.
        if enabled is None:
            enabled = _env_flag("PERCEPTION_SWAP_WRITE_PRED_FILES")
        self.enabled = bool(enabled)

        # Lazy: only do the rest if enabled.
        self.collectors: List[Optional[CarlaGTApCollector]] = (
            [None] * self.ego_vehicles_num
        )
        self.visible_ids_per_frame: List[Dict[int, set]] = [
            {} for _ in range(self.ego_vehicles_num)
        ]
        self.gt_replays: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
            None
        ] * self.ego_vehicles_num
        self.scenarioset_dir: Optional[str] = None
        self.real_scene_dir: Optional[str] = None
        self.real_dataset_fps: float = float(
            os.environ.get("PERCEPTION_SWAP_REAL_DATASET_FPS", "10.0")
        )
        self.gt_from_real_dataset = _env_flag(
            "PERCEPTION_SWAP_GT_FROM_REAL_DATASET", default=False,
        )
        # Hard-override that forces every visibility-filter source OFF.
        # Used by perception-quality correlation studies where AP must
        # count every CARLA vehicle in the eval range — not just those
        # the dataset annotated, and not just those the live semantic
        # LiDAR hit ≥N times. Same env var is honored by perception_swap
        # so codriving_* and perception_swap_* variants share methodology.
        if _env_flag("OPENLOOP_DISABLE_VISIBILITY_FILTER", default=False):
            self.gt_from_real_dataset = False
            print("[openloop-hook] OPENLOOP_DISABLE_VISIBILITY_FILTER=1 — "
                  "visibility filter disabled (all CARLA vehicles in range count as GT)")

        self._err_count = 0

        if not self.enabled:
            return

        self._resolve_paths()
        self._load_replays_and_visible_ids()
        self._build_collectors()
        # Write a debug snapshot of init state so we can verify the hook
        # actually fired with sane paths (setup-time prints go to stdout
        # before per-scenario log redirection so they're often invisible).
        try:
            import json as _json
            _root = self.ap_save_root or "/tmp"
            _dbg = os.path.join(_root, f"openloop_hook_init_ego{0}.json")
            with open(_dbg, "w") as _f:
                _json.dump({
                    "enabled": bool(self.enabled),
                    "ap_save_root": str(self.ap_save_root),
                    "scenarioset_dir": self.scenarioset_dir,
                    "real_scene_dir": self.real_scene_dir,
                    "gt_from_real_dataset": bool(self.gt_from_real_dataset),
                    "n_collectors": sum(1 for c in self.collectors if c is not None),
                    "n_replays_loaded": sum(1 for r in self.gt_replays if r is not None),
                    "visible_ids_n_frames_per_ego": [
                        len(v) for v in self.visible_ids_per_frame
                    ],
                    "ego_vehicles_num": int(self.ego_vehicles_num),
                    "detector_label": str(self.detector_label),
                }, _f, indent=2)
        except Exception:
            pass

    # ── Resolution helpers ────────────────────────────────────────────
    def _resolve_paths(self) -> None:
        """Walk SAVE_PATH parents to find scenarioset/v2xpnp/<scene>/, and
        pair that with /data/dataset/v2x-real-pnp/<split>/<scene>/ for the
        visible-id filter. Both are best-effort: if they don't resolve,
        the corresponding feature stays off but the rest of the pipeline
        still runs."""
        scenarioset_dir = os.environ.get("OPENLOOP_SCENARIOSET_DIR", "")
        save_path_root = self.ap_save_root
        if not scenarioset_dir and save_path_root:
            import re as _re
            _repo = os.environ.get(
                "OPENLOOP_REPO_ROOT",
                os.path.abspath(os.path.join(
                    os.path.dirname(__file__), "..", "..", ".."
                )),
            )
            _v2xpnp_root = os.path.join(_repo, "scenarioset", "v2xpnp")
            _path = os.path.abspath(save_path_root.rstrip("/"))
            # The colmdriver/codriving action module appends a per-scenario
            # timestamp suffix _MM_DD_HH_MM_SS to the run dir basename
            # (see colmdriver_action.py:884), so the literal basename never
            # matches the scenarioset/<scene> dir. Strip both _partial_<…>
            # AND the trailing _DD_DD_DD_DD_DD timestamp before matching.
            _ts_suffix = _re.compile(r"_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
            for _ in range(8):  # cap walk depth
                _basename = os.path.basename(_path).split("_partial_", 1)[0]
                _basename_stripped = _ts_suffix.sub("", _basename)
                # Try both — sometimes the run dir already lacks the suffix.
                for _try in (_basename, _basename_stripped):
                    _candidate = os.path.join(_v2xpnp_root, _try)
                    if _try and os.path.isdir(_candidate):
                        scenarioset_dir = _candidate
                        break
                if scenarioset_dir:
                    break
                _parent = os.path.dirname(_path)
                if _parent == _path:
                    break
                _path = _parent
        self.scenarioset_dir = scenarioset_dir or None
        if self.scenarioset_dir:
            print(f"[openloop-hook] scenarioset_dir resolved → {self.scenarioset_dir}")
        else:
            print(f"[openloop-hook] WARN: scenarioset_dir NOT resolved from save_path={save_path_root}; "
                  f"REPLAY/visible-id features disabled (set OPENLOOP_SCENARIOSET_DIR to override)")

        if self.scenarioset_dir:
            scene_basename = os.path.basename(self.scenarioset_dir)
            real_root = os.environ.get(
                "PERCEPTION_SWAP_V2X_REAL_PNP_ROOT",
                "/data/dataset/v2x-real-pnp",
            )
            self.real_scene_dir = resolve_real_dataset_scene_dir(
                scene_basename, real_root,
            )

    def _load_replays_and_visible_ids(self) -> None:
        if self.scenarioset_dir:
            for i in range(self.ego_vehicles_num):
                xml = os.path.join(
                    self.scenarioset_dir,
                    f"ucla_v2_custom_ego_vehicle_{i}_REPLAY.xml",
                )
                t_arr, xy_arr = parse_ego_replay_xml(xml)
                self.gt_replays[i] = (
                    (t_arr, xy_arr) if t_arr is not None else None
                )
            n_loaded = sum(1 for r in self.gt_replays if r is not None)
            print(
                f"[openloop-hook] BEV GT overlay: "
                f"{n_loaded}/{self.ego_vehicles_num} ego REPLAY loaded "
                f"from {self.scenarioset_dir}"
            )
        else:
            print(
                f"[openloop-hook] BEV GT overlay disabled — "
                f"no scenarioset/v2xpnp/<scene> matched (SAVE_PATH={self.ap_save_root})"
            )

        if self.gt_from_real_dataset:
            if self.real_scene_dir is None:
                print(
                    f"[openloop-hook] visible-actor GT filter requested but "
                    f"no v2x-real-pnp scene matched; falling back to all-CARLA-vehicle GT."
                )
            else:
                # ego_vehicle_<i> ↔ agent dir str(i+1) in source dataset.
                # Verified empirically against true_ego_pose alignment.
                for i in range(self.ego_vehicles_num):
                    agent_id = str(i + 1)
                    agent_dir = os.path.join(self.real_scene_dir, agent_id)
                    visible, n_parsed = load_visible_ids_per_frame(agent_dir)
                    self.visible_ids_per_frame[i] = visible
                    avg_visible = (
                        sum(len(v) for v in visible.values())
                        / max(1, len(visible))
                    )
                    print(
                        f"[openloop-hook] visible-actor filter ego{i}: "
                        f"agent dir {agent_dir!r} → {n_parsed} frames, "
                        f"avg visible={avg_visible:.1f}"
                    )

    def _build_collectors(self) -> None:
        for i in range(self.ego_vehicles_num):
            visible = self.visible_ids_per_frame[i]
            self.collectors[i] = CarlaGTApCollector(
                iou_thresholds=self.iou_thresholds,
                visible_ids_per_frame=(
                    visible if (self.gt_from_real_dataset and visible) else None
                ),
            )

    # ── Per-tick API ──────────────────────────────────────────────────
    def frame_idx_at(self, sim_t: float) -> int:
        """Convert sim time (s) to source-dataset frame index for visible-id lookup."""
        return int(round(float(sim_t) * self.real_dataset_fps))

    def update_world(self,
                     ego_idx: int,
                     primary_pose_world,
                     pred_corners_world,
                     pred_scores,
                     ego_actor_ids,
                     sim_t: float = 0.0) -> None:
        """Feed perception predictions in WORLD frame (codriving / colmdriver)."""
        self._update(
            ego_idx, primary_pose_world, pred_corners_world, pred_scores,
            ego_actor_ids, sim_t, pred_frame="world",
        )

    def update_ego(self,
                   ego_idx: int,
                   primary_pose_world,
                   pred_corners_ego,
                   pred_scores,
                   ego_actor_ids,
                   sim_t: float = 0.0) -> None:
        """Feed perception predictions in EGO frame (perception_swap)."""
        self._update(
            ego_idx, primary_pose_world, pred_corners_ego, pred_scores,
            ego_actor_ids, sim_t, pred_frame="ego",
        )

    def _update(self, ego_idx, primary_pose_world, pred_corners, pred_scores,
                ego_actor_ids, sim_t, pred_frame) -> None:
        if not self.enabled:
            return
        if ego_idx >= len(self.collectors) or self.collectors[ego_idx] is None:
            return
        try:
            self.collectors[ego_idx].update(
                primary_pose_world=primary_pose_world,
                pred_corners=np.asarray(pred_corners, dtype=np.float32),
                pred_scores=np.asarray(pred_scores, dtype=np.float32),
                lidar_range=self.lidar_range,
                ego_actor_ids=set(ego_actor_ids),
                pred_frame=pred_frame,
                frame_idx=self.frame_idx_at(sim_t),
            )
        except Exception as exc:
            self._err_count += 1
            if self._err_count <= 3 or self._err_count % 100 == 0:
                print(f"[openloop-hook] AP update fail #{self._err_count} "
                      f"(ego {ego_idx}): {exc}")

    def last_frame_state(self,
                         ego_idx: int,
                         iou: float = 0.5,
                         ) -> Optional[Dict[str, Any]]:
        """Return the BEV-renderer view of the most recent ``update_*`` call
        for ``ego_idx`` at ``iou`` threshold, or ``None`` if no frame has
        been collected yet.

        Returns dict::

            {
                "gt_corners":   ndarray (M, 8, 3) — frame is ``pred_frame``
                                ('world' for codriving/colmdriver), ready to
                                pass into ``BEVRenderer.render(gt_corners_world=...)``,
                "n_tp_frame":   int,
                "n_fp_frame":   int,
                "n_fn_frame":   int,
                "n_gt_frame":   int,
            }
        """
        if not (0 <= ego_idx < len(self.collectors)):
            return None
        coll = self.collectors[ego_idx]
        if coll is None or coll.last_frame is None:
            return None
        per_frame = coll.last_frame.get("per_frame", {})
        # Find the IoU bucket: prefer exact match, else the closest threshold
        # the collector tracks. The renderer's default is 0.5, which both
        # codriving and perception_swap include.
        if float(iou) in per_frame:
            stats = per_frame[float(iou)]
        else:
            keys = list(per_frame.keys())
            if not keys:
                return None
            stats = per_frame[min(keys, key=lambda k: abs(k - float(iou)))]
        return {
            "gt_corners": coll.last_frame.get("gt_corners"),
            "gt_corners_world": coll.last_frame.get("gt_corners_world"),
            "pred_frame": coll.last_frame.get("pred_frame"),
            "pred_status": coll.last_frame.get("pred_status"),
            "pred_iou":    coll.last_frame.get("pred_iou"),
            "gt_matched":  coll.last_frame.get("gt_matched"),
            "n_tp_frame": int(stats.get("tp", 0)),
            "n_fp_frame": int(stats.get("fp", 0)),
            "n_fn_frame": int(stats.get("fn", 0)),
            "n_gt_frame": int(stats.get("gt", 0)),
        }

    # ── BEV-rendering inputs ──────────────────────────────────────────
    def gt_future_for_render(self, ego_idx: int, sim_t: float,
                             traj: Optional[np.ndarray],
                             ) -> Tuple[Optional[np.ndarray], float, float]:
        """Build the anchor-shifted GT future trajectory (for the BEV
        green-dashed line) plus per-tick ADE / FDE.

        Returns ``(gt_xy_world, ade, fde)``. ``gt_xy_world`` is None when
        no replay is loaded for this ego or ``traj`` is empty/missing.
        ADE / FDE are NaN in those cases.
        """
        replay = (
            self.gt_replays[ego_idx]
            if 0 <= ego_idx < len(self.gt_replays) else None
        )
        if replay is None or traj is None or len(traj) == 0:
            return None, float("nan"), float("nan")
        t_arr, xy_arr = replay
        T = len(traj)
        dt = 3.0 / max(T, 1)  # mirrors planner horizon convention
        samples = []
        aligned = []
        for k in range(T):
            xy = gt_xy_at(t_arr, xy_arr, sim_t + (k + 1) * dt)
            if xy is not None:
                samples.append(xy)
                aligned.append(k)
        if not samples:
            return None, float("nan"), float("nan")
        gt = np.asarray(samples, dtype=np.float64)
        pred = np.asarray([traj[k] for k in aligned], dtype=np.float64)
        gt_shifted = gt - gt[0] + pred[0]   # anchor-shift to pred[0]
        diffs = np.linalg.norm(pred - gt_shifted, axis=1)
        return gt_shifted, float(diffs.mean()), float(diffs[-1])

    # ── Persist ───────────────────────────────────────────────────────
    def save_all(self,
                 ap_save_root: Optional[str] = None,
                 extra_meta_per_ego: Optional[Dict[int, Dict[str, Any]]] = None
                 ) -> None:
        """Write one ``perception_swap_ap_ego{N}.json`` per ego. Safe to
        call multiple times (final file is the last call's snapshot)."""
        if not self.enabled:
            return
        root = ap_save_root or self.ap_save_root
        if not root:
            return
        for i, coll in enumerate(self.collectors):
            if coll is None:
                continue
            extra = {
                "detector":     self.detector_label,
                "ego_index":    i,
            }
            if extra_meta_per_ego and i in extra_meta_per_ego:
                extra.update(extra_meta_per_ego[i])
            out_path = os.path.join(root, self.AP_FILE_FMT.format(ego_idx=i))
            try:
                coll.save(out_path, extra_meta=extra)
            except Exception as exc:
                self._err_count += 1
                if self._err_count <= 3 or self._err_count % 100 == 0:
                    print(f"[openloop-hook] AP save fail #{self._err_count} "
                          f"(ego {i}): {exc}")


# ──────────────────────────────────────────────────────────────────────────
# BEVRenderer — side-by-side front-cam + top-down BEV, faithful to the
# perception_swap_agent visualization (orange pred trajectory, green-dashed
# anchor-shifted GT future, ADE/FDE bracket, AP/ADE/CR stat panel, optional
# per-pred-box and per-GT-box overlays). Reused by codriving / colmdriver
# agents so all three planners produce visually consistent BEV output.
# ──────────────────────────────────────────────────────────────────────────
# CARLA top-down camera convention (matches perception_swap_agent.py):
#   width = height = 800 px
#   fov = 90°, mounted at z = 60 m, pitch = -90° looking straight down
#   image-up = ego forward (+x), image-right = ego right (+y)
_BEV_TOP_W = 800
_BEV_TOP_H = 800
_BEV_TOP_FOV_DEG = 90.0
_BEV_TOP_Z = 60.0


def _project_ego_xy_to_top_img(x_ego: float, y_ego: float,
                                W: int = _BEV_TOP_W, H: int = _BEV_TOP_H,
                                fov_deg: float = _BEV_TOP_FOV_DEG,
                                z_top: float = _BEV_TOP_Z):
    """Pin-hole projection of an ego-frame ground point to top-down image px.

    Identical to ``PerceptionSwapAgent._project_ego_xy_to_top_img`` so the
    output is interchangeable with perception_swap's BEV.
    """
    fx = W / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fy = fx
    u = W / 2.0 + (y_ego / z_top) * fx
    v = H / 2.0 - (x_ego / z_top) * fy
    return float(u), float(v)


class BEVRenderer:
    """Per-tick side-by-side BEV image emitter.

    The class is constructed once in the agent's setup() (alongside
    OpenLoopAPHook) and called per tick from run_step(). Emit cadence is
    every ``viz_every`` ticks. Output goes to
    ``<viz_dir>/tick_NNNNN_egoK.png``.

    Inputs are explicit ndarrays/scalars (no ``Detections`` namedtuple
    dependency) so the same renderer works for perception_swap (ego-frame
    detector outputs) and codriving / colmdriver (world-frame perception
    + planner waypoints).

    When ``viz_dir`` is empty or the openloop env signal is missing, every
    method is a no-op — the renderer is safe to instantiate in closed-loop
    runs.
    """

    TOP_W = _BEV_TOP_W
    TOP_H = _BEV_TOP_H
    TOP_FOV_DEG = _BEV_TOP_FOV_DEG
    TOP_Z = _BEV_TOP_Z

    def __init__(self,
                 *,
                 viz_dir: Optional[str] = None,
                 viz_every: int = 10,
                 detector_label: str = "external") -> None:
        self.viz_dir = (
            viz_dir or os.environ.get("PERCEPTION_SWAP_VIZ_DIR", "")
        )
        # If neither override nor explicit dir, default to <SAVE_PATH>/perception_swap_viz
        if not self.viz_dir:
            sp = os.environ.get("SAVE_PATH", "")
            if sp:
                self.viz_dir = os.path.join(sp, "perception_swap_viz")
        self.viz_every = int(viz_every if viz_every is not None
                             else os.environ.get("PERCEPTION_SWAP_VIZ_EVERY", "10"))
        self.detector_label = str(detector_label)
        self._err_count = 0
        if self.viz_dir:
            try:
                os.makedirs(self.viz_dir, exist_ok=True)
                print(f"[bev-render] viz_dir={self.viz_dir} (every {self.viz_every} ticks, label={self.detector_label})")
            except Exception as exc:
                print(f"[bev-render] viz dir setup failed: {exc}")
                self.viz_dir = ""

    def render(self,
               *,
               tick: int,
               ego_idx: int,
               ego_xy: Tuple[float, float],
               ego_yaw_rad: float,
               ego_speed_mps: float,
               traj_world: Optional[np.ndarray] = None,
               gt_future_world: Optional[np.ndarray] = None,
               pred_corners_world: Optional[np.ndarray] = None,
               pred_scores: Optional[np.ndarray] = None,
               gt_corners_world: Optional[np.ndarray] = None,
               cam_img: Optional[np.ndarray] = None,
               top_img: Optional[np.ndarray] = None,
               ade: float = float("nan"),
               fde: float = float("nan"),
               running_ap: Optional[float] = None,
               ap_iou: float = 0.5,
               n_tp_frame: int = 0,
               n_fp_frame: int = 0,
               n_fn_frame: int = 0,
               brake_now: bool = False,
               ttc_s: float = float("inf"),
               ap_eval_range: Optional[List[float]] = None,
               pred_status: Optional[np.ndarray] = None,
               gt_matched: Optional[np.ndarray] = None,
               ) -> None:
        """Emit one BEV frame. All inputs in WORLD coordinates.

        Skips silently if ``viz_dir`` is empty OR ``tick % viz_every != 0``.

        Per-box overlays:
          * ``pred_corners_world`` — (P, 8, 3) predicted boxes (red outline)
          * ``gt_corners_world``   — (M, 8, 3) ground-truth boxes (gray outline)
          * If both arrays are passed, set TPs / FPs / FNs by counting via
            ``n_tp_frame``/``n_fp_frame``/``n_fn_frame`` and color the boxes
            uniformly. (Per-box TP/FP coloring requires the matcher state
            and is out of scope for this minimal renderer.)

        ``ap_eval_range`` (``[xmin, ymin, zmin, xmax, ymax, zmax]`` in
        ego/lidar frame): if provided, draws a green dashed rectangle
        showing the AP scoring crop on the BEV pane. Helps the viewer see
        which preds / GT actually contributed to AP.
        """
        if not self.viz_dir:
            return
        if tick % self.viz_every != 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.lines import Line2D
        except Exception:
            return

        try:
            ex, ey = float(ego_xy[0]), float(ego_xy[1])
            cy_, sy_ = float(np.cos(ego_yaw_rad)), float(np.sin(ego_yaw_rad))

            def world_to_uv(xw: float, yw: float):
                dx, dy = xw - ex, yw - ey
                xe = cy_ * dx + sy_ * dy
                ye = -sy_ * dx + cy_ * dy
                return _project_ego_xy_to_top_img(xe, ye, self.TOP_W, self.TOP_H,
                                                   self.TOP_FOV_DEG, self.TOP_Z)

            fig, (ax_cam, ax) = plt.subplots(
                1, 2, figsize=(16, 8),
                gridspec_kw={"width_ratios": [1, 1]},
            )

            # ── Camera pane (left) ────────────────────────────────────
            if cam_img is not None and hasattr(cam_img, "shape") and cam_img.size > 0:
                if cam_img.ndim == 3 and cam_img.shape[-1] == 4:
                    cam_rgb = cam_img[..., [2, 1, 0]]
                else:
                    cam_rgb = cam_img
                ax_cam.imshow(cam_rgb)
            ax_cam.set_title(f"front cam · ego_{ego_idx}")
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])

            # ── BEV pane (right) ──────────────────────────────────────
            use_topdown_bg = (
                top_img is not None and hasattr(top_img, "shape") and top_img.size > 0
            )
            if use_topdown_bg:
                if top_img.ndim == 3 and top_img.shape[-1] == 4:
                    top_rgb = top_img[..., [2, 1, 0]]
                else:
                    top_rgb = top_img
                ax.imshow(top_rgb, origin="upper",
                          extent=(0, self.TOP_W, self.TOP_H, 0))
                ax.set_xlim(0, self.TOP_W)
                ax.set_ylim(self.TOP_H, 0)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Synthetic fallback: ±60 m around ego in world coordinates.
                ax.set_xlim(ex - 60.0, ex + 60.0)
                ax.set_ylim(ey - 60.0, ey + 60.0)
                ax.set_aspect("equal")
                ax.grid(alpha=0.3)

            # Both pred and GT lines start at the ego anchor so they share
            # a visual origin even when the first sampled step is offset.
            ego_uv = world_to_uv(ex, ey) if use_topdown_bg else (ex, ey)

            def _to_uv(p):
                return world_to_uv(float(p[0]), float(p[1])) if use_topdown_bg \
                    else (float(p[0]), float(p[1]))

            pred_uv_end = None
            if traj_world is not None and len(traj_world) > 0:
                uvs = [ego_uv] + [_to_uv(p) for p in traj_world]
                ax.plot([u for u, v in uvs], [v for u, v in uvs],
                        color="orange", linewidth=2.4,
                        label="pred (3 s)", zorder=8)
                u_e, v_e = uvs[-1]
                ax.plot([u_e], [v_e], "o", color="orange",
                        markersize=7, alpha=0.95, zorder=9)
                pred_uv_end = (u_e, v_e)

            gt_uv_end = None
            if gt_future_world is not None and len(gt_future_world) > 0:
                uvs = [ego_uv] + [_to_uv(p) for p in gt_future_world]
                ax.plot([u for u, v in uvs], [v for u, v in uvs],
                        color="#1a8b2a", linestyle="--", linewidth=2.2,
                        alpha=0.95, label="GT future", zorder=8)
                step_idx = list(range(0, len(uvs), 5))
                if (len(uvs) - 1) not in step_idx:
                    step_idx.append(len(uvs) - 1)
                ax.plot(
                    [uvs[i][0] for i in step_idx],
                    [uvs[i][1] for i in step_idx],
                    "o", color="#1a8b2a", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.6,
                    alpha=0.95, zorder=9,
                )
                u_e, v_e = uvs[-1]
                ax.plot([u_e], [v_e], "*", color="#1a8b2a",
                        markersize=12, markeredgecolor="white",
                        markeredgewidth=0.8, alpha=0.98, zorder=10)
                gt_uv_end = (u_e, v_e)

            # Shared anchor dot
            if traj_world is not None or gt_future_world is not None:
                ax.plot([ego_uv[0]], [ego_uv[1]], "o", color="white",
                        markeredgecolor="black", markersize=4.5,
                        markeredgewidth=0.8, zorder=11)

            # FDE bracket between endpoints
            if (pred_uv_end is not None and gt_uv_end is not None
                    and fde == fde):
                ax.plot([pred_uv_end[0], gt_uv_end[0]],
                        [pred_uv_end[1], gt_uv_end[1]],
                        ":", color="#7d2bff", linewidth=1.6, alpha=0.95, zorder=8)
                midu = 0.5 * (pred_uv_end[0] + gt_uv_end[0])
                midv = 0.5 * (pred_uv_end[1] + gt_uv_end[1])
                ade_part = (
                    f"\nADE={ade:.1f}m" if ade == ade else ""
                )
                ax.text(midu, midv, f"FDE={fde:.1f}m{ade_part}",
                        fontsize=7, ha="center", va="center", color="#7d2bff",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="#7d2bff",
                                  alpha=0.9, linewidth=0.7),
                        zorder=11)

            # Pred boxes & GT boxes — bottom face only. Per-box coloring
            # uses the matcher state stashed in CarlaGTApCollector.last_frame
            # (greedy IoU≥0.5 match in score order; same algorithm as
            # opencood's caluclate_tp_fp). Falls back to uniform red/gray
            # when matcher state is missing.
            def _draw_one_box(box, color, linewidth, zorder, fill):
                pts3 = np.asarray(box, dtype=np.float64)
                order_z = np.argsort(pts3[:, 2])
                bot = pts3[order_z[:4]]
                bot_c = bot[:, :2].mean(axis=0)
                ang = np.arctan2(bot[:, 1] - bot_c[1], bot[:, 0] - bot_c[0])
                bot = bot[np.argsort(ang)]
                pts2 = [_to_uv(p) for p in bot[:, :2]]
                facecolor = (color + "33"
                             if (fill and len(color) == 7)
                             else "none")
                ax.add_patch(MplPolygon(
                    pts2, closed=True, edgecolor=color,
                    facecolor=facecolor, linewidth=linewidth, zorder=zorder,
                ))

            # GT boxes: gray if matched (TP partner), orange if FN.
            _gt_matched_arr = (np.asarray(gt_matched, dtype=bool)
                                if gt_matched is not None else None)
            if gt_corners_world is not None and len(gt_corners_world) > 0:
                for gi, box in enumerate(gt_corners_world):
                    if (_gt_matched_arr is not None
                            and gi < len(_gt_matched_arr)
                            and not _gt_matched_arr[gi]):
                        _draw_one_box(box, "#f08020", linewidth=1.6,
                                      zorder=4, fill=True)  # FN — orange
                    else:
                        _draw_one_box(box, "#7f7f7f", linewidth=1.2,
                                      zorder=3, fill=True)  # matched — gray

            # Pred boxes: green if TP, red if FP.
            _pred_status_arr = (np.asarray(pred_status, dtype=np.int8)
                                 if pred_status is not None else None)
            if pred_corners_world is not None and len(pred_corners_world) > 0:
                for pi, box in enumerate(pred_corners_world):
                    if (_pred_status_arr is not None
                            and pi < len(_pred_status_arr)
                            and _pred_status_arr[pi] == 1):
                        _draw_one_box(box, "#1a8b2a", linewidth=1.6,
                                      zorder=5, fill=False)  # TP — green
                    else:
                        _draw_one_box(box, "#d62828", linewidth=1.6,
                                      zorder=5, fill=False)  # FP — red

            # AP eval-range rectangle (ego-frame AABB → 4 world corners,
            # rotated by ego yaw). Drawn last so it sits on top but with low
            # alpha so it doesn't drown out the boxes.
            if ap_eval_range is not None and len(ap_eval_range) >= 6:
                xmin, ymin, _, xmax, ymax, _ = (
                    float(ap_eval_range[0]), float(ap_eval_range[1]),
                    float(ap_eval_range[2]), float(ap_eval_range[3]),
                    float(ap_eval_range[4]), float(ap_eval_range[5]),
                )
                rect_local = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                if use_topdown_bg:
                    rect_pts = [
                        _project_ego_xy_to_top_img(lx, ly,
                                                    self.TOP_W, self.TOP_H,
                                                    self.TOP_FOV_DEG, self.TOP_Z)
                        for (lx, ly) in rect_local
                    ]
                else:
                    rect_pts = [
                        (ex + cy_ * lx - sy_ * ly,
                         ey + sy_ * lx + cy_ * ly)
                        for (lx, ly) in rect_local
                    ]
                ax.add_patch(MplPolygon(
                    rect_pts, closed=True, edgecolor="#b8860b",
                    facecolor="none", linewidth=1.4, linestyle="--",
                    alpha=0.85, zorder=2,
                ))

            # Ego marker (yaw-rotated arrowhead). Build in ego-local coords
            # then project — same shape as perception_swap.
            ego_local = [
                ( 4.5 * 0.6,  0.0),
                (-4.5 * 0.4,  2.0),
                (-4.5 * 0.2,  0.0),
                (-4.5 * 0.4, -2.0),
            ]
            if use_topdown_bg:
                ego_pts = [
                    _project_ego_xy_to_top_img(lx, ly,
                                                self.TOP_W, self.TOP_H,
                                                self.TOP_FOV_DEG, self.TOP_Z)
                    for (lx, ly) in ego_local
                ]
            else:
                # World-frame: rotate local → world by ego yaw.
                ego_pts = [
                    (ex + cy_ * lx - sy_ * ly,
                     ey + sy_ * lx + cy_ * ly)
                    for (lx, ly) in ego_local
                ]
            ax.add_patch(MplPolygon(ego_pts, closed=True, facecolor="blue",
                                     edgecolor="black", linewidth=1.0,
                                     alpha=0.95, zorder=7))

            # Stat panel (upper-left, monospace, white background).
            ap_iou_int = int(round(ap_iou * 100))
            ap_str = (f"{running_ap:.3f}" if running_ap is not None
                      and running_ap == running_ap else "n/a")
            denom_p = max(1, n_tp_frame + n_fp_frame)
            denom_r = max(1, n_tp_frame + n_fn_frame)
            ade_text = f"{ade:.1f} m" if ade == ade else "—"
            cr_flag = 1 if (ttc_s < 2.5 and ttc_s != float("inf")) else 0
            stat_lines = [
                f"AP@{ap_iou_int} (run): {ap_str}",
                f"this frame: TP={n_tp_frame}  FP={n_fp_frame}  FN={n_fn_frame}  "
                f"P={n_tp_frame/denom_p:.2f}  R={n_tp_frame/denom_r:.2f}",
                f"ADE: {ade_text}    CR: {cr_flag}"
                + (f" (ttc={ttc_s:.1f}s)" if cr_flag else ""),
            ]
            if use_topdown_bg:
                tx, ty = 8, 8
            else:
                tx = ex - 60.0 + 2.0
                ty = ey - 60.0 + 4.0  # synthetic frame origin top-left in world coords
            ax.text(tx, ty, "\n".join(stat_lines),
                    fontsize=10, ha="left", va="top", color="black",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="black", alpha=0.85, linewidth=0.8),
                    zorder=12)

            # Title bar with concise per-tick summary.
            title = (
                f"tick {tick:04d} ego_{ego_idx}  "
                f"spd={ego_speed_mps:.1f} m/s  "
                f"P={(0 if pred_corners_world is None else len(pred_corners_world))}  "
                f"GT={(0 if gt_corners_world is None else len(gt_corners_world))}  "
                f"brake={'Y' if brake_now else 'N'}  "
                f"ADE={ade_text}  AP@{ap_iou_int}={ap_str}  "
                f"[{self.detector_label}]"
            )
            ax.set_title(title, fontsize=10)

            # Legend
            handles = [
                Line2D([0], [0], color="orange", lw=2.2, label="pred (3 s)"),
                Line2D([0], [0], color="#1a8b2a", lw=2.2, linestyle="--",
                       label="GT future"),
            ]
            # Per-box coloring legend: TP / FP / matched-GT / FN. Counts come
            # from the matcher state (n_tp_frame / n_fn_frame) so they agree
            # with the box colors above. Falls back to uniform labels when
            # matcher state is missing.
            n_pred_total = (len(pred_corners_world)
                            if pred_corners_world is not None else 0)
            n_gt_total = (len(gt_corners_world)
                          if gt_corners_world is not None else 0)
            if n_pred_total > 0:
                if pred_status is not None:
                    n_tp_box = int(np.sum(np.asarray(pred_status) == 1))
                    n_fp_box = int(n_pred_total - n_tp_box)
                    if n_tp_box > 0:
                        handles.append(Line2D([0], [0], color="#1a8b2a", lw=2,
                                              label=f"pred TP ({n_tp_box})"))
                    if n_fp_box > 0:
                        handles.append(Line2D([0], [0], color="#d62828", lw=2,
                                              label=f"pred FP ({n_fp_box})"))
                else:
                    handles.append(Line2D([0], [0], color="#d62828", lw=2,
                                          label=f"pred boxes ({n_pred_total})"))
            if n_gt_total > 0:
                if gt_matched is not None:
                    n_gt_match = int(np.sum(np.asarray(gt_matched, dtype=bool)))
                    n_gt_fn = int(n_gt_total - n_gt_match)
                    if n_gt_match > 0:
                        handles.append(Line2D([0], [0], color="#7f7f7f", lw=2,
                                              label=f"GT matched ({n_gt_match})"))
                    if n_gt_fn > 0:
                        handles.append(Line2D([0], [0], color="#f08020", lw=2,
                                              label=f"GT FN ({n_gt_fn})"))
                else:
                    handles.append(Line2D([0], [0], color="#7f7f7f", lw=2,
                                          label=f"GT boxes ({n_gt_total})"))
            if ap_eval_range is not None and len(ap_eval_range) >= 6:
                _xs = float(ap_eval_range[3]) - float(ap_eval_range[0])
                _ys = float(ap_eval_range[4]) - float(ap_eval_range[1])
                handles.append(Line2D(
                    [0], [0], color="#b8860b", lw=1.4, linestyle="--",
                    label=f"AP eval ({_xs:.0f}×{_ys:.0f} m)",
                ))
            ax.legend(handles=handles, loc="upper right", fontsize=8)

            out_path = os.path.join(
                self.viz_dir, f"tick_{tick:05d}_ego{ego_idx}.png"
            )
            fig.savefig(out_path, dpi=80, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            self._err_count += 1
            if self._err_count <= 3 or self._err_count % 100 == 0:
                print(f"[bev-render] viz frame failed #{self._err_count}: {exc}")

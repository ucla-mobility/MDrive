import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .assets import get_asset_bbox
from .constants import LANE_WIDTH_M, LATERAL_TO_M, LATERAL_ALIASES
from .geometry import (
    cumulative_dist,
    heading_deg_from_vec,
    point_and_tangent_at_s,
    wrap180,
    _project_point_to_path_s_m,
)
from .guardrails import _infer_vehicle_turn_exit_indices
from .path_extension import _compute_path_length, extend_path_if_needed
from .spawn import compute_spawn_from_anchor
from .utils import _parse_vehicle_num


DIST_BUCKET_TO_M = {
    "touching": 1.0,
    "close": 4.0,
    "medium": 10.0,
    "far": 20.0,
    "unknown": None,
}
ALLOWED_PHASES = {"on_approach", "in_intersection", "after_turn", "after_exit", "after_merge", "any", "unknown"}
ALLOWED_REL_TYPES = {"ahead_of", "behind_of", "left_of", "right_of", "near"}
ALLOWED_DIST_BUCKETS = set(DIST_BUCKET_TO_M.keys())
ALLOWED_GROUP_PATTERNS = {"across_lane", "along_lane", "scatter", "unknown"}

AFTER_MERGE_CLEARANCE_M = 2.0
LANE_DROP_AFTER_MERGE_OFFSET_M = 1.0
LANE_DROP_AFTER_MERGE_WINDOW_M = 4.0
LANE_DROP_CANDIDATE_STEP_M = 1.0
MERGE_POINT_MAX_DIST_M = 6.0

MIN_ACTOR_SEPARATION_M = 0.5


def normalize_lateral(lat: str) -> str:
    """Normalize lateral position string, handling aliases like 'side_of_road' -> 'sidewalk_right'."""
    lat = str(lat or "unknown").strip().lower()
    # Check aliases first (e.g., "side of the road" -> "sidewalk_right")
    if lat in LATERAL_ALIASES:
        return LATERAL_ALIASES[lat]
    # Check if it's a known lateral position
    if lat in LATERAL_TO_M:
        return lat
    # Handle natural language variations
    lat_words = lat.replace("_", " ").replace("-", " ")
    if lat_words in LATERAL_ALIASES:
        return LATERAL_ALIASES[lat_words]
    return lat  # return as-is, caller will handle unknown


@dataclass
class CandidatePlacement:
    vehicle_num: int
    segment_index: int   # 1-based within vehicle path
    seg_id: int
    s_along: float
    lateral_relation: str
    x: float
    y: float
    yaw_deg: float
    path_s_m: float      # cumulative distance along vehicle path
    base_score: float


def _compute_merge_min_s_by_vehicle(
    picked_payload: Dict[str, Any],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    max_dist_m: float = MERGE_POINT_MAX_DIST_M,
) -> Dict[int, float]:
    """
    Map target vehicle -> path_s_m of merge point (from path refiner), when available.
    Uses the closest merge point along the target's path and ignores far/off-path points.
    """
    refinement = picked_payload.get("refinement", {})
    if not isinstance(refinement, dict):
        return {}
    lane_change_debug = refinement.get("lane_change_debug", [])
    if not isinstance(lane_change_debug, list):
        return {}

    points_by_target: Dict[int, List[Tuple[float, float]]] = {}
    for item in lane_change_debug:
        if not isinstance(item, dict):
            continue
        dbg = item.get("debug") or {}
        if not isinstance(dbg, dict) or not dbg.get("applied"):
            continue
        mp = dbg.get("merge_point") or {}
        if not isinstance(mp, dict):
            continue
        try:
            x = float(mp.get("x"))
            y = float(mp.get("y"))
        except Exception:
            continue
        lane_change = item.get("lane_change") or {}
        target = lane_change.get("target") or lane_change.get("target_vehicle")
        target_num = _parse_vehicle_num(target)
        if target_num is None:
            continue
        points_by_target.setdefault(target_num, []).append((x, y))

    if not points_by_target:
        return {}

    picked_by_vehicle = {}
    for p in picked_list:
        vnum = _parse_vehicle_num(p.get("vehicle"))
        if vnum is not None:
            picked_by_vehicle[vnum] = p

    out: Dict[int, float] = {}
    for veh_num, pts in points_by_target.items():
        picked_entry = picked_by_vehicle.get(veh_num)
        if not picked_entry:
            continue
        best_s = None
        for x, y in pts:
            proj = _project_point_to_path_s_m(picked_entry, seg_by_id, np.array([x, y], dtype=float))
            if not proj:
                continue
            dist, path_s = proj
            if dist > max_dist_m:
                continue
            if best_s is None or path_s < best_s:
                best_s = path_s
        if best_s is not None:
            out[veh_num] = best_s
    return out


def _inside_crop_xy(x: float, y: float, crop_region: Any, margin_m: float = 0.0) -> bool:
    if not isinstance(crop_region, dict):
        return True
    try:
        xmin = float(crop_region.get("xmin"))
        xmax = float(crop_region.get("xmax"))
        ymin = float(crop_region.get("ymin"))
        ymax = float(crop_region.get("ymax"))
    except Exception:
        return True
    return (xmin + margin_m <= x <= xmax - margin_m) and (ymin + margin_m <= y <= ymax - margin_m)


def _actor_radius_m(category: str, actor_kind: str, asset_id: Optional[str] = None) -> float:
    """
    Return the collision radius for an actor. This is used to prevent
    actors from spawning on top of each other.
    
    If asset_id is provided and has a bounding box, use that for accurate sizing.
    Otherwise fall back to category-based defaults.
    """
    # Try to use bbox if available
    if asset_id:
        bbox = get_asset_bbox(asset_id)
        if bbox:
            # Use the larger of length/width divided by 2 as radius
            # (conservative: use max dimension for circular collision)
            return max(bbox.length, bbox.width) / 2.0
    
    # Fallback to category-based defaults
    c = str(category).lower()
    k = str(actor_kind).lower()
    if c == "vehicle" or k in ("parked_vehicle", "npc_vehicle"):
        return 2.5  # Vehicles need more space
    if c in ("walker", "cyclist") or k in ("walker", "cyclist"):
        return 0.8  # Walkers/cyclists
    # Static props like cones, barriers - still need some separation
    return 0.6  # Increased from 0.5 to avoid overlaps


def get_actor_bbox_dims(asset_id: Optional[str], category: str, actor_kind: str) -> Tuple[float, float]:
    """
    Get length and width for an actor (in meters).
    Returns (length, width) tuple. Uses bbox if available, else defaults.
    """
    if asset_id:
        bbox = get_asset_bbox(asset_id)
        if bbox:
            return (bbox.length, bbox.width)
    
    # Fallback to category-based defaults
    c = str(category).lower()
    k = str(actor_kind).lower()
    if c == "vehicle" or k in ("parked_vehicle", "npc_vehicle"):
        return (4.5, 1.8)  # Typical car
    if c == "walker" or k == "walker":
        return (0.6, 0.6)  # Pedestrian
    if c == "cyclist" or k == "cyclist":
        return (1.8, 0.6)  # Bicycle
    return (0.6, 0.6)  # Small static prop


def _find_opposite_lane_segments(
    ref_vehicle_num: int,
    picked_list: List[Dict[str, Any]],
    all_segments: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    heading_diff_threshold: float = 135.0,
) -> List[Dict[str, Any]]:
    """
    Find segments that go in the opposite direction to a reference vehicle's path.
    Returns list of segment dicts with seg_id, points, and heading.
    """
    # Get reference vehicle's heading from first segment
    ref_entry = next((p for p in picked_list if _parse_vehicle_num(p.get("vehicle")) == ref_vehicle_num), None)
    if not ref_entry:
        return []
    
    sig = ref_entry.get("signature", {})
    entry = sig.get("entry", {})
    ref_heading = entry.get("heading_deg")
    if ref_heading is None:
        return []
    ref_heading = float(ref_heading)
    
    # Find segments with approximately opposite heading
    opposite_segs = []
    for seg in all_segments:
        seg_id = seg.get("seg_id")
        pts = seg_by_id.get(seg_id)
        if pts is None or len(pts) < 2:
            continue
        
        # Compute segment heading from first/last points
        seg_heading = heading_deg_from_vec(pts[-1] - pts[0])
        heading_diff = abs(wrap180(seg_heading - ref_heading))
        
        if heading_diff >= heading_diff_threshold:
            # Check if segment is in crop region
            mid_pt = pts[len(pts)//2]
            if _inside_crop_xy(float(mid_pt[0]), float(mid_pt[1]), crop_region, margin_m=2.0):
                opposite_segs.append({
                    "seg_id": seg_id,
                    "points": pts,
                    "heading_deg": seg_heading,
                    "length_m": float(cumulative_dist(pts)[-1]),
                })
    
    return opposite_segs


def build_stage2_constraints_prompt(
    description: str,
    vehicle_segments: Dict[str, Any],
    stage1_entities: List[Dict[str, Any]],
    per_entity_options: Dict[str, List[Dict[str, Any]]],
) -> str:
    payload = {
        "task": "Emit actor_specs with symbolic preferences/relations only. Solver picks concrete anchors later.",
        "description": description,
        "ego_paths": vehicle_segments,
        "stage1_entities": stage1_entities,
        "asset_candidates_by_entity_id": per_entity_options,
        "allowed_enums": {
            "phase": sorted(list(ALLOWED_PHASES)),
            "lateral_relation": sorted(list(LATERAL_TO_M.keys())),
            "relation_type": sorted(list(ALLOWED_REL_TYPES)),
            "distance_bucket": sorted(list(ALLOWED_DIST_BUCKETS)),
            "group_pattern": sorted(list(ALLOWED_GROUP_PATTERNS)),
            "category": ["static", "vehicle", "walker", "cyclist"],
        },
        "motion_type_definitions": {
            "static": "Object does not move (parked vehicles, cones, barriers, debris).",
            "follow_lane": "Moves ALONG the road in the lane direction (same direction as traffic). Use when description says 'walks in the direction of the road', 'walks along the road', 'travels in the lane direction', etc.",
            "cross_perpendicular": "Moves ACROSS the road (perpendicular to traffic). Use when description says 'crosses the road', 'crosses the street', 'jaywalking', etc.",
            "straight_line": "Moves in a straight line (not necessarily along or across road).",
            "zigzag_lane": "Weaves between lanes.",
            "unknown": "Motion not specified or unclear."
        },
        "motion_hint_to_type_mapping": {
            "static": "static",
            "crossing": "cross_perpendicular",
            "follow_lane": "follow_lane",
            "unknown": "unknown"
        },
        "output_schema": {
            "actor_specs": [
                {
                    "id": "entity_1",
                    "asset_id": "MUST be one of the asset_id strings provided for that entity_id",
                    "category": "static|vehicle|walker|cyclist",
                    "actor_kind": "copy from Stage1 actor_kind",
                    "quantity": "IGNORED (quantity is taken from Stage1). You may omit or set to 1.",
                    "anchor": {
                        "target_vehicle": "Vehicle N | none | unknown",
                        "phase": "on_approach|in_intersection|after_turn|after_exit|after_merge|any|unknown",
                        "lateral_preference": "one of lateral_relation enums or unknown"
                    },
                    "relations": [
                        {
                            "type": "ahead_of|behind_of|left_of|right_of|near",
                            "other_id": "entity_*",
                            "distance": "touching|close|medium|far|unknown",
                            "evidence": "EXACT quote (<=20 words) from description"
                        }
                    ],
                    "group_pattern": {
                        "pattern": "across_lane|along_lane|scatter|unknown",
                        "start_lateral": "optional lateral enum",
                        "end_lateral": "optional lateral enum",
                        "spacing_bucket": "tight|normal|sparse|auto"
                    },
                    "motion": {
                        "type": "DERIVE from stage1 motion_hint using motion_hint_to_type_mapping. 'follow_lane' means along road, 'crossing' means cross_perpendicular.",
                        "speed_profile": "slow|normal|fast|erratic|unknown",
                        "cross_direction": "left|right|unknown (ONLY for cross_perpendicular: direction the actor moves. 'from right to left' = 'left', 'from left to right' = 'right')"
                    },
                    "confidence": "float in [0,1]"
                }
            ]
        },
        "critical_rules": [
            "Return JSON ONLY. No prose, no markdown.",
            "CLOSED WORLD: use ONLY enums provided.",
            "Do NOT output segment_index, seg_id, s_along, x/y, yaw.",
            "Do NOT invent constraints. If unsure, use 'unknown' or omit.",
            "If a Stage1 entity seems like map geometry (e.g., intersection/road/lane/sidewalk) or lacks a clear actor noun, OMIT it from actor_specs (do not force a placement).",
            
            "ANCHOR TARGET_VEHICLE IS CRITICAL - COPY FROM stage1_entities 'affects_vehicle' FIELD:",
            "  - If Stage1 has affects_vehicle='Vehicle X', set anchor.target_vehicle='Vehicle X'",
            "  - DO NOT set anchor.target_vehicle to 'unknown' if Stage1 provides 'affects_vehicle'!",
            "  - This field determines which ego vehicle the actor is intended to interact with.",
            
            "PHASE IS CRITICAL - COPY FROM stage1_entities 'when' FIELD:",
            "  - If Stage1 has when='after_exit', set anchor.phase='after_exit'",
            "  - 'On Vehicle X's exit road' = phase='after_exit' (NOT on_approach!)",
            "  - 'after the intersection' = phase='after_exit'",
            "  - 'on the approach road' = phase='on_approach'",
            "  - DO NOT default to 'unknown' if Stage1 provides 'when' value!",
            
            "SEQUENTIAL/DISTANCE RELATIONS ARE CRITICAL:",
            "  - If description says 'X meters later', 'X meters after', 'X meters down', use relation type='ahead_of' with appropriate distance bucket.",
            "  - Distance mapping: 1-3m='touching', 4-8m='close', 9-15m='medium', 16-30m='far'.",
            "  - If description says 'farther down', 'further along', 'past the X', use type='ahead_of' with distance='far'.",
            "  - The entity mentioned FIRST in the description is typically BEHIND the one mentioned later (so later entity is 'ahead_of' the earlier one).",
            
            "Every relation MUST have non-empty evidence quote. Otherwise omit.",
            "Keep constraints minimal (0-2 relations per actor), but DO extract explicitly stated distance/sequence relationships.",
            
            "GROUP PATTERN CRITICAL: For entities with quantity > 1 in stage1_entities:",
            "  - You MUST provide group_pattern with pattern, start_lateral, end_lateral.",
            "  - 'diagonal' pattern from Stage1 should use pattern='along_lane' with start_lateral and end_lateral set.",
            "  - Example: if Stage1 says 'diagonal' with start_lateral='right_edge', end_lateral='left_edge', output:",
            "    group_pattern: {pattern: 'along_lane', start_lateral: 'right_edge', end_lateral: 'left_edge', spacing_bucket: 'normal'}",
            
            "MOTION TYPE CRITICAL: Use motion_hint from stage1_entities to set motion.type:",
            "  - If motion_hint='follow_lane' -> motion.type='follow_lane' (moves ALONG the road)",
            "  - If motion_hint='crossing' -> motion.type='cross_perpendicular' (moves ACROSS the road)",
            "  - If motion_hint='static' -> motion.type='static'",
            "  - 'walks in the direction of the road' means ALONG the road = follow_lane, NOT crossing!",
            "SPEED PROFILE: Use speed_hint from stage1_entities. If speed_hint='erratic', use speed_profile='erratic'."
        ]
    }
    return (
        "You are a constraint extractor for a driving-scene CSP.\n"
        "Your output is used by a solver; invented/over-specific constraints hurt feasibility.\n"
        "Follow payload rules EXACTLY.\n\n"
        "PAYLOAD (JSON):\n" + json.dumps(payload, indent=2)
    )


def _normalize_actor_spec_id(spec: Dict[str, Any]) -> Optional[str]:
    if isinstance(spec, dict):
        if spec.get("id"):
            return str(spec.get("id"))
        if spec.get("entity_id"):
            return str(spec.get("entity_id"))
    return None


def _get_vehicle_lane_id(picked: List[Dict[str, Any]], vehicle_name: str, phase: str) -> Optional[int]:
    """
    Get the lane_id for a vehicle at the relevant phase.
    For 'after_exit', 'after_merge', 'after_turn' -> use last segment.
    For 'on_approach' -> use first segment.
    For others -> use last segment (where conflicts typically happen).
    """
    for p in picked:
        if str(p.get("vehicle", "")) == vehicle_name:
            sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
            segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed"), list) else []
            if not segs:
                return None
            # For approach phases, use first segment; otherwise use last
            if phase in ("on_approach",):
                return segs[0].get("lane_id")
            else:
                return segs[-1].get("lane_id")
    return None


def validate_actor_specs(
    actor_specs: Any,
    stage1_entities: List[Dict[str, Any]],
    per_entity_options: Dict[str, List[Dict[str, Any]]],
    picked: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    # Debug: show what we're validating
    print(f"[DEBUG] validate_actor_specs: received {len(actor_specs) if isinstance(actor_specs, list) else 'non-list'} specs", flush=True)
    print(f"[DEBUG] validate_actor_specs: stage1_entities={[e.get('entity_id') for e in stage1_entities]}", flush=True)
    print(f"[DEBUG] validate_actor_specs: per_entity_options keys={list(per_entity_options.keys())}", flush=True)
    if isinstance(actor_specs, list):
        for i, spec in enumerate(actor_specs[:5]):
            print(f"[DEBUG] validate_actor_specs: raw spec[{i}] id={spec.get('id') or spec.get('entity_id')}", flush=True)
    if not isinstance(actor_specs, list):
        return [], ["actor_specs must be a list"]
    if picked is None:
        picked = []
    stage1_by_id = {str(e.get("entity_id")): e for e in stage1_entities if e.get("entity_id")}
    clean: List[Dict[str, Any]] = []
    for raw in actor_specs:
        if not isinstance(raw, dict):
            continue
        sid = _normalize_actor_spec_id(raw)
        if not sid or sid not in stage1_by_id:
            warnings.append(f"Dropped actor_spec with unknown id: {sid}")
            continue
        e = stage1_by_id[sid]
        kind = str(raw.get("actor_kind") or e.get("actor_kind") or "static_prop")
        cat = str(raw.get("category") or "").lower()
        if cat not in ("static", "vehicle", "walker", "cyclist"):
            if kind in ("walker",):
                cat = "walker"
            elif kind in ("cyclist",):
                cat = "cyclist"
            elif kind in ("parked_vehicle", "npc_vehicle"):
                cat = "vehicle"
            else:
                cat = "static"

        # asset_id strict: must be from candidate list
        asset_id = str(raw.get("asset_id") or "")
        allowed = [str(o.get("asset_id")) for o in per_entity_options.get(sid, []) if isinstance(o, dict) and o.get("asset_id")]
        if asset_id not in set(allowed):
            if allowed:
                asset_id = allowed[0]
                warnings.append(f"{sid}: asset_id not in candidates; fell back to {asset_id}")
            else:
                warnings.append(f"{sid}: no asset candidates; dropped spec")
                continue

        qty = e.get("quantity", e.get("count", 1))  # Fix B: quantity is authoritative from Stage 1
        try:
            qty = int(qty)
        except Exception:
            qty = 1
        qty = max(1, qty)

        anchor = raw.get("anchor", {}) if isinstance(raw.get("anchor", {}), dict) else {}
        # Priority: Stage2 anchor.target_vehicle -> Stage1 affects_vehicle -> "unknown"
        stage2_tv = anchor.get("target_vehicle", "")
        stage1_tv = e.get("affects_vehicle", "unknown")
        
        if stage2_tv and str(stage2_tv).lower() not in ("", "unknown", "none"):
            tv = str(stage2_tv)
        elif stage1_tv and str(stage1_tv).lower() not in ("", "unknown", "none"):
            tv = str(stage1_tv)
        else:
            tv = "unknown"
        
        # DEBUG: Log when there's a mismatch
        if str(stage2_tv).lower() not in ("", "unknown", "none") and str(stage2_tv) != str(stage1_tv):
            print(f"[DEBUG] Anchor target_vehicle mismatch for {e.get('entity_id')}: stage2={stage2_tv} vs stage1={stage1_tv}, using={tv}", flush=True)
        elif str(tv) != "unknown" and str(tv) != str(stage1_tv):
            print(f"[DEBUG] Using Stage1 affects_vehicle for {e.get('entity_id')}: {tv}", flush=True)
        
        # Phase fallback: Stage2 anchor.phase -> Stage1 when -> "unknown"
        # If Stage2 outputs "unknown" or empty, prefer Stage1's value
        stage2_phase = anchor.get("phase", "")
        stage1_phase = e.get("when", "unknown")
        if stage2_phase and stage2_phase != "unknown":
            phase = str(stage2_phase)
        elif stage1_phase and stage1_phase != "unknown":
            phase = str(stage1_phase)
        else:
            phase = "unknown"
        # For lateral preference, try: anchor.lateral_preference -> e.lateral_relation -> e.start_lateral
        lat = normalize_lateral(anchor.get("lateral_preference", "unknown"))
        if lat == "unknown":
            lat = normalize_lateral(e.get("lateral_relation", "unknown"))
        if lat == "unknown":
            # Fall back to start_lateral (e.g., "pedestrian appears to the left" -> start_lateral=left_edge)
            lat = normalize_lateral(e.get("start_lateral") or "unknown")

        tvs = str(tv)
        if tvs not in ("none", "unknown") and _parse_vehicle_num(tvs) is None:
            tvs = "unknown"

        if phase not in ALLOWED_PHASES:
            phase = "unknown"
        if lat not in LATERAL_TO_M and lat != "unknown":
            lat = "unknown"

        # relations: drop invalid/missing evidence
        rels: List[Dict[str, Any]] = []
        rin = raw.get("relations", [])
        if isinstance(rin, list):
            for r in rin:
                if not isinstance(r, dict):
                    continue
                rtype = str(r.get("type", ""))
                oid = str(r.get("other_id", ""))
                dist = str(r.get("distance", "unknown"))
                ev = str(r.get("evidence", "")).strip()
                if rtype not in ALLOWED_REL_TYPES:
                    continue
                if oid not in stage1_by_id:
                    continue
                if dist not in ALLOWED_DIST_BUCKETS:
                    dist = "unknown"
                if not ev:
                    continue
                rels.append({"type": rtype, "other_id": oid, "distance": dist, "evidence": ev})

        # group pattern optional (only meaningful for qty>1)
        gp = raw.get("group_pattern")
        group_pattern = None
        if qty > 1:
            # First try Stage 2's group_pattern, then fall back to Stage 1 fields
            if isinstance(gp, dict):
                patt = str(gp.get("pattern", "unknown"))
                sl = gp.get("start_lateral")
                el = gp.get("end_lateral")
                spacing = str(gp.get("spacing_bucket", "auto"))
            else:
                # Use Stage 1 fields
                patt = str(e.get("group_pattern", "unknown"))
                sl = e.get("start_lateral")
                el = e.get("end_lateral")
                spacing = "auto"

            # Map "diagonal" to "along_lane" with start/end laterals
            if patt == "diagonal":
                patt = "along_lane"

            if patt not in ALLOWED_GROUP_PATTERNS:
                patt = "along_lane"
            if sl is not None and str(sl) not in LATERAL_TO_M:
                sl = None
            if el is not None and str(el) not in LATERAL_TO_M:
                el = None
            if spacing not in ("tight", "normal", "sparse", "auto"):
                spacing = "auto"
            group_pattern = {"pattern": patt, "start_lateral": sl, "end_lateral": el, "spacing_bucket": spacing}

        # Motion type: prioritize Stage 1 motion_hint over Stage 2 output
        motion = raw.get("motion", {}) if isinstance(raw.get("motion", {}), dict) else {}
        stage1_motion_hint = str(e.get("motion_hint", "unknown"))
        motion_hint_to_type = {
            "static": "static",
            "crossing": "cross_perpendicular",
            "follow_lane": "follow_lane",
            "unknown": "unknown",
        }

        if stage1_motion_hint in motion_hint_to_type and stage1_motion_hint != "unknown":
            mtype = motion_hint_to_type[stage1_motion_hint]
        else:
            mtype = str(motion.get("type", "static"))
            if mtype not in ("static", "follow_lane", "cross_perpendicular", "straight_line", "zigzag_lane", "unknown"):
                mtype = "unknown"

        # Default motion for NPC vehicles and cyclists
        if mtype in ("unknown", "static") and kind in ("npc_vehicle", "cyclist"):
            mtype = "follow_lane"

        sp = str(motion.get("speed_profile", e.get("speed_hint", "unknown")))
        if sp not in ("slow", "normal", "fast", "erratic", "stopped", "unknown"):
            sp = "unknown"
        if sp == "unknown" and kind == "npc_vehicle":
            sp = "normal"

        # NPC vehicles with distance triggers should keep a natural speed; spawn placement will
        # be biased to keep them close enough for the trigger to fire without crawling.
        trigger = e.get("trigger")
        action = e.get("action")
        if kind == "npc_vehicle" and mtype == "follow_lane":
            trigger_type = trigger.get("type", "") if isinstance(trigger, dict) else ""
            action_type = action.get("type", "") if isinstance(action, dict) else ""
            # If waiting for ego to catch up and will perform an action
            if trigger_type == "distance_to_vehicle" and action_type in ("hard_brake", "lane_change", "sudden_stop"):
                sp = "normal"
                print(f"[INFO] {sid}: NPC vehicle with distance trigger + {action_type} action -> using normal speed and trigger-distance spawn bias")

        conf = raw.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        conf = max(0.0, min(1.0, conf))

        direction_relative_to = e.get("direction_relative_to")
        if direction_relative_to is not None and not isinstance(direction_relative_to, dict):
            direction_relative_to = None

        # trigger and action already extracted above (for speed profile adjustment)
        if trigger is not None and not isinstance(trigger, dict):
            trigger = None

        if action is not None and not isinstance(action, dict):
            action = None

        lat_pref = lat

        # Cross direction: FIRST check explicit stage1 crossing_direction, then infer from lane
        cross_dir = "unknown"
        
        # Priority 1: Use explicit crossing_direction from stage1 (e.g., "cross from right to left")
        stage1_cross_dir = e.get("crossing_direction")
        print(f"[DEBUG CSP] Entity {sid}: stage1_cross_dir = {stage1_cross_dir}")  # DEBUG
        if stage1_cross_dir in ("left", "right"):
            cross_dir = stage1_cross_dir
        
        # Priority 2: Use motion.cross_direction if set
        if cross_dir == "unknown":
            cross_dir = str(motion.get("cross_direction", "unknown"))
        
        # Priority 3: Infer from start_lateral (if starting on right, cross left; if starting on left, cross right)
        if cross_dir == "unknown":
            stage1_start_lat = normalize_lateral(e.get("start_lateral") or "unknown")
            if "right" in stage1_start_lat:
                cross_dir = "left"
            elif "left" in stage1_start_lat:
                cross_dir = "right"
        
        # Priority 4: Infer from lane_id (last resort)
        if cross_dir == "unknown" and mtype == "cross_perpendicular":
            lane_id = _get_vehicle_lane_id(picked, tvs, phase)
            if lane_id is not None:
                if lane_id < 0:
                    cross_dir = "left"
                elif lane_id > 0:
                    cross_dir = "right"

        # For crossing motion, use start_lateral from stage1 if available
        if mtype == "cross_perpendicular":
            stage1_start_lat = normalize_lateral(e.get("start_lateral") or "unknown")
            print(f"[DEBUG CSP] Entity {sid}: stage1_start_lat = {stage1_start_lat}, mtype = {mtype}, cross_dir = {cross_dir}")  # DEBUG
            if stage1_start_lat != "unknown" and stage1_start_lat in LATERAL_TO_M:
                lat_pref = stage1_start_lat
            elif lat_pref == "unknown":
                # Infer from cross_dir as fallback
                if cross_dir == "left":
                    lat_pref = "sidewalk_right"
                elif cross_dir == "right":
                    lat_pref = "sidewalk_left"
            print(f"[DEBUG CSP] Entity {sid}: lat_pref = {lat_pref}")  # DEBUG

        clean.append({
            "id": sid,
            "semantic": str(raw.get("semantic") or e.get("mention") or sid),
            "category": cat,
            "actor_kind": kind,
            "asset_id": asset_id,
            "quantity": qty,
            "anchor": {"target_vehicle": tvs, "phase": phase, "lateral_preference": lat_pref},
            "relations": rels,
            "group_pattern": group_pattern,
            "motion": {
                "type": mtype,
                "speed_profile": sp,
                "cross_direction": cross_dir,
                "start_lateral": lat_pref,  # Pass lateral position to spawn.py for crossing motion
            },
            "confidence": conf,
            "direction_relative_to": direction_relative_to,
            "trigger": trigger,
            "action": action,
        })

    # Fallback: Create default specs for entities that the LLM didn't output specs for
    processed_ids = {str(spec["id"]) for spec in clean}
    for e in stage1_entities:
        eid = str(e.get("entity_id", ""))
        if not eid or eid in processed_ids:
            continue
        # Check if we have asset candidates for this entity
        allowed = [str(o.get("asset_id")) for o in per_entity_options.get(eid, []) if isinstance(o, dict) and o.get("asset_id")]
        if not allowed:
            warnings.append(f"{eid}: LLM skipped and no asset candidates; cannot create fallback")
            continue
        
        # Create a fallback spec using Stage 1 information
        kind = str(e.get("actor_kind") or "static_prop")
        if kind in ("walker",):
            cat = "walker"
        elif kind in ("cyclist",):
            cat = "cyclist"
        elif kind in ("parked_vehicle", "npc_vehicle"):
            cat = "vehicle"
        else:
            cat = "static"
        
        # Get phase and target vehicle from Stage 1
        phase = str(e.get("when", "unknown"))
        if phase not in ALLOWED_PHASES:
            phase = "on_approach"
        tv = str(e.get("affects_vehicle", "unknown"))
        
        # Get lateral preference
        lat = normalize_lateral(e.get("lateral_relation", "unknown"))
        if lat == "unknown":
            lat = normalize_lateral(e.get("start_lateral") or "unknown")
        
        # Determine motion type from stage1
        motion_hint = str(e.get("motion_hint", "unknown"))
        if motion_hint == "crossing":
            mtype = "cross_perpendicular"
        elif motion_hint == "follow_lane":
            mtype = "follow_lane"
        elif motion_hint == "static":
            mtype = "static"
        else:
            mtype = "static" if kind in ("parked_vehicle", "static_prop") else "unknown"
        
        # Speed
        sp = str(e.get("speed_hint", "unknown"))
        if sp not in ("slow", "normal", "fast", "erratic", "stopped", "unknown"):
            sp = "stopped" if mtype == "static" else "normal"
        
        # Cross direction for crossing motion
        cross_dir = "unknown"
        if mtype == "cross_perpendicular":
            stage1_cross_dir = e.get("crossing_direction")
            if stage1_cross_dir in ("left", "right"):
                cross_dir = stage1_cross_dir
            elif "right" in lat:
                cross_dir = "left"
            elif "left" in lat:
                cross_dir = "right"
        
        qty = int(e.get("quantity", 1))
        
        fallback_spec = {
            "id": eid,
            "semantic": str(e.get("mention") or eid),
            "category": cat,
            "actor_kind": kind,
            "asset_id": allowed[0],  # Use first available asset
            "quantity": qty,
            "anchor": {"target_vehicle": tv, "phase": phase, "lateral_preference": lat},
            "relations": [],
            "group_pattern": None,
            "motion": {
                "type": mtype,
                "speed_profile": sp,
                "cross_direction": cross_dir,
                "start_lateral": lat,
            },
            "confidence": 0.5,  # Lower confidence for fallback
            "direction_relative_to": e.get("direction_relative_to"),
            "trigger": e.get("trigger"),
            "action": e.get("action"),
        }
        clean.append(fallback_spec)
        warnings.append(f"{eid}: LLM skipped; created fallback spec with asset={allowed[0]}")

    return clean, warnings


def _phase_score(segment_index: int, veh_info: Dict[str, Any], phase: str) -> float:
    """Soft-but-strong preference for placing an actor at the requested route phase.

    The earlier version used small +/- scores, which made it easy for the solver to
    place 'after_exit' objects near the approach if other terms dominated. Here we
    keep the solver soft (to avoid UNSAT), but make phase mismatches much more costly.
    """
    if phase in ("any", "unknown", None, ""):
        return 0.0

    turn_indices = veh_info.get("turn_indices", set()) if isinstance(veh_info, dict) else set()
    exit_idx = veh_info.get("exit_index", None) if isinstance(veh_info, dict) else None
    nsegs = len(veh_info.get("seg_ids", [])) if isinstance(veh_info, dict) else 0
    nsegs = max(1, int(nsegs)) if nsegs else 1

    # progress ratio along segments (1..n)
    try:
        ratio = float(segment_index) / float(nsegs)
    except Exception:
        ratio = 0.0

    # Helper windows (fallback if geometry inference is noisy)
    early_cut = max(1, int(math.ceil(0.35 * nsegs)))
    late_cut = max(1, int(math.floor(0.65 * nsegs)))

    if phase == "in_intersection":
        if turn_indices:
            return 5.0 if segment_index in turn_indices else -5.0
        # If no inferred turn indices, prefer the middle segment(s).
        mid = (nsegs + 1) / 2.0
        dist = abs(segment_index - mid)
        return 3.0 - 3.0 * dist

    if phase == "after_turn":
        if exit_idx is not None:
            ex = int(exit_idx)
            if segment_index == ex:
                return 5.0
            if segment_index > ex:
                return 2.5
            return -5.0
        return 2.5 if ratio >= 0.55 else -2.5

    if phase == "on_approach":
        if turn_indices:
            ft = min(turn_indices)
            if segment_index < ft:
                return 5.0
            if segment_index == ft:
                return 1.0
            return -5.0
        return 3.0 if segment_index <= early_cut else -3.0

    if phase == "after_exit":
        if exit_idx is not None:
            ex = int(exit_idx)
            if segment_index == ex:
                return 5.0
            if segment_index > ex:
                return 2.0
            return -5.0
        return 3.0 if segment_index >= late_cut else -3.0

    if phase == "after_merge":
        # "After merge" means on the exit road, similar to after_exit
        if exit_idx is not None:
            ex = int(exit_idx)
            if segment_index == ex:
                return 5.0
            if segment_index > ex:
                return 2.0
            return -5.0
        return 3.0 if segment_index >= late_cut else -3.0

    return 0.0


def _s_along_centering_score(s_along: float, phase: str) -> float:
    """Bonus for placing actors near the center of a segment for 'in_intersection' phase.
    
    Returns a score from 0.0 (at edges) to 2.0 (at center s_along=0.5).
    This ensures that 'in_intersection' placements land in the middle of the junction,
    not at the very start or end.
    """
    if phase not in ("in_intersection",):
        return 0.0
    # s_along ranges from 0.0 to 1.0; we want to prefer ~0.5
    # Use a simple parabola: 2.0 * (1 - (2*s - 1)^2) = 2.0 - 2*(2s-1)^2
    # At s=0.5: 2.0, at s=0 or s=1: 0.0
    deviation = 2.0 * s_along - 1.0  # ranges from -1 to 1
    return 2.0 * (1.0 - deviation * deviation)


def _lateral_score(lat: str, pref: str) -> float:
    if pref in ("unknown", None, ""):
        return 0.0
    return 1.0 if lat == pref else -0.2


def _trigger_distance_bias(spec: Dict[str, Any], vehicle_num: int, path_s_m: float) -> float:
    """
    Soft bias to keep distance-trigger NPCs close to the referenced vehicle's start,
    so triggers can fire without forcing unnaturally slow speeds.
    """
    if spec.get("actor_kind") != "npc_vehicle":
        return 0.0
    trigger = spec.get("trigger")
    action = spec.get("action")
    if not isinstance(trigger, dict) or trigger.get("type") != "distance_to_vehicle":
        return 0.0
    action_type = action.get("type") if isinstance(action, dict) else ""
    if action_type not in ("hard_brake", "lane_change", "sudden_stop"):
        return 0.0
    trigger_vehicle = _parse_vehicle_num(trigger.get("vehicle"))
    anchor_vehicle = _parse_vehicle_num((spec.get("anchor") or {}).get("target_vehicle"))
    if trigger_vehicle is None or anchor_vehicle is None:
        return 0.0
    if trigger_vehicle != anchor_vehicle or trigger_vehicle != vehicle_num:
        return 0.0
    try:
        dist = float(trigger.get("distance_m", 8.0))
    except Exception:
        dist = 8.0
    dist = max(1.0, min(50.0, dist))
    buffer_m = min(6.0, max(2.0, 0.3 * dist))
    desired = dist + buffer_m
    tol = max(4.0, 0.5 * buffer_m)
    diff = abs(path_s_m - desired)
    if diff <= tol:
        bonus = 3.0
    else:
        bonus = max(-3.0, 3.0 - ((diff - tol) / max(tol, 1e-6)) * 3.0)
    if path_s_m < dist:
        bonus -= 1.0
    return bonus


def _relation_score(a: CandidatePlacement, b: CandidatePlacement, rel: Dict[str, Any]) -> float:
    # Positive if relation satisfied (ahead/behind/left/right/near)
    typ = rel.get("type", "")
    dist = rel.get("distance", "unknown")
    dtarget = DIST_BUCKET_TO_M.get(dist, None)

    # Compute along-path difference when possible
    # Use path_s_m for along-lane ordering.
    if typ in ("ahead_of", "behind_of"):
        if a.vehicle_num != b.vehicle_num:
            # Cross-vehicle ordering: approximate by along values anyway
            pass
        delta = a.path_s_m - b.path_s_m
        if typ == "behind_of":
            delta = -delta
        if dtarget is None:
            return 1.0 if delta > 0 else -1.0
        # Prefer being ahead by at least dtarget (softly)
        if delta >= dtarget:
            return 2.0
        if delta >= 0:
            return 1.0 + (delta / max(1e-6, dtarget))
        return -2.0

    if typ in ("left_of", "right_of"):
        # Use lateral relationship based on x,y difference, projecting onto right-normal at A.
        # Approx: compare lateral offset between A and B around A's heading.
        # We'll use their yaw_deg in world frame to build right-normal.
        ang = math.radians(float(a.yaw_deg))
        fwd = (math.cos(ang), math.sin(ang))
        # right normal
        rn = (-fwd[1], fwd[0])
        dx, dy = (a.x - b.x), (a.y - b.y)
        lat = dx * rn[0] + dy * rn[1]
        if typ == "left_of":
            lat = -lat
        # positive lat means a is on the requested side
        if lat > 0.0:
            return 1.5
        return -1.5

    if typ == "near":
        dx = a.x - b.x
        dy = a.y - b.y
        d = math.hypot(dx, dy)
        if dtarget is None:
            return 0.5
        # Prefer being within target distance
        if d <= dtarget:
            return 1.2
        return -0.5

    return 0.0


def generate_candidates_for_actor(
    spec: Dict[str, Any],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    veh_info: Dict[int, Dict[str, Any]],
    merge_min_s_by_vehicle: Optional[Dict[int, float]] = None,
    all_segments: Optional[List[Dict[str, Any]]] = None,
    max_candidates: int = 120,
    ds_m: float = 6.0,
    crop_margin_m: float = 1.0,
    ego_spawns: Optional[List[Dict[str, Any]]] = None,
    candidate_debug: Optional[Dict[str, Any]] = None,
) -> List[CandidatePlacement]:
    anchor = spec.get("anchor", {})
    tv = str(anchor.get("target_vehicle", "unknown"))
    pref_phase = str(anchor.get("phase", "unknown"))
    pref_lat = str(anchor.get("lateral_preference", "unknown"))
    motion = spec.get("motion", {}) if isinstance(spec.get("motion", {}), dict) else {}
    mtype = str(motion.get("type", "unknown")).lower()
    cross_dir = str(motion.get("cross_direction", "unknown")).lower()
    preferred = _parse_vehicle_num(tv)
    
    # Check if this is an NPC vehicle with "opposite" direction constraint
    direction_rel = spec.get("direction_relative_to")
    is_opposite_npc = False
    opposite_ref_vehicle = None
    if direction_rel and isinstance(direction_rel, dict):
        if direction_rel.get("direction") == "opposite":
            is_opposite_npc = True
            opposite_ref_vehicle = _parse_vehicle_num(direction_rel.get("vehicle"))

    veh_domain: List[int] = []
    for p in picked_list:
        n = _parse_vehicle_num(p.get("vehicle"))
        if n is not None:
            veh_domain.append(n)
    veh_domain = sorted(list(set(veh_domain)))
    if preferred is not None:
        veh_domain = [preferred]
        print(f"[DEBUG] Actor {spec.get('id')} constrained to vehicle_domain={veh_domain} (preferred={preferred})", flush=True)
    else:
        print(f"[DEBUG] Actor {spec.get('id')} using all vehicles: vehicle_domain={veh_domain} (target_vehicle was '{tv}')", flush=True)

    cat = str(spec.get("category","static")).lower()
    if cat == "vehicle":
        laterals = ["center", "half_right", "half_left"]
    elif cat in ("walker","cyclist"):
        laterals = ["center", "half_right", "half_left", "right_edge", "left_edge"]
        if mtype == "cross_perpendicular" and pref_lat in ("unknown", None, ""):
            if cross_dir == "right":
                laterals = ["left_edge", "half_left", "right_edge", "half_right", "center"]
            elif cross_dir == "left":
                laterals = ["right_edge", "half_right", "left_edge", "half_left", "center"]
            else:
                laterals = ["right_edge", "left_edge", "half_right", "half_left", "center"]
    else:
        laterals = ["right_edge", "left_edge", "half_right", "half_left", "center", "offroad_right", "offroad_left"]
    if pref_lat in laterals:
        laterals = [pref_lat] + [x for x in laterals if x != pref_lat]

    sid = str(spec.get("id", "unknown"))
    debug_entry = {
        "actor_id": sid,
        "vehicle_domain": list(veh_domain),
        "attempted_samples": 0,
        "outside_crop": 0,
        "before_merge_window": 0,
        "after_merge_window": 0,
        "added_primary": 0,
        "added_fallback": 0,
        "added_far": 0,
        "ego_buffer_filtered": 0,
        "is_opposite_direction_mode": is_opposite_npc,
        "segment_ids": [],
        "segment_lengths_m": [],
    }

    cands: List[CandidatePlacement] = []
    fallback_cands: List[CandidatePlacement] = []
    far_cands: List[CandidatePlacement] = []

    gp = spec.get("group_pattern", {}) if isinstance(spec.get("group_pattern", {}), dict) else {}
    group_pattern_name = str(gp.get("pattern", "unknown") if isinstance(gp, dict) else gp).lower()
    pref_lat_norm = str(pref_lat).lower()
    is_lane_drop_marker = (
        pref_phase == "after_merge"
        and group_pattern_name == "along_lane"
        and pref_lat_norm in ("left_edge", "right_edge", "half_left", "half_right")
    )

    merge_min_s_m = None
    merge_max_s_m = None
    if pref_phase == "after_merge" and preferred is not None and merge_min_s_by_vehicle:
        base_merge_s = merge_min_s_by_vehicle.get(preferred)
        if base_merge_s is not None:
            # Lane Drop markers should appear just after the merge point.
            clearance = LANE_DROP_AFTER_MERGE_OFFSET_M if is_lane_drop_marker else AFTER_MERGE_CLEARANCE_M
            merge_min_s_m = float(base_merge_s) + clearance + _group_back_buffer_m(spec)
            if is_lane_drop_marker:
                merge_max_s_m = merge_min_s_m + LANE_DROP_AFTER_MERGE_WINDOW_M
    
    # Special handling for NPC vehicles that travel in the OPPOSITE direction
    if is_opposite_npc and opposite_ref_vehicle is not None and all_segments:
        opposite_segs = _find_opposite_lane_segments(
            opposite_ref_vehicle, picked_list, all_segments, seg_by_id, crop_region
        )
        if opposite_segs:
            for seg_info in opposite_segs:
                seg_id = seg_info["seg_id"]
                pts = seg_info["points"]
                seg_len = seg_info["length_m"]
                if seg_len < 2.0:
                    continue
                debug_entry["segment_ids"].append(int(seg_id))
                debug_entry["segment_lengths_m"].append(float(seg_len))
                
                step = max(2.0, float(ds_m))
                s_values_m = list(np.arange(min(crop_margin_m, 0.2*seg_len), max(seg_len - crop_margin_m, 0.8*seg_len), step))
                if len(s_values_m) < 3:
                    s_values_m = [0.2*seg_len, 0.5*seg_len, 0.8*seg_len]
                
                for s_m in s_values_m:
                    s_along = float(min(1.0, max(0.0, s_m / seg_len)))
                    for lat in laterals:
                        debug_entry["attempted_samples"] += 1
                        spawn = compute_spawn_from_anchor(pts, s_along, lat, None)
                        if not _inside_crop_xy(spawn["x"], spawn["y"], crop_region, margin_m=crop_margin_m):
                            debug_entry["outside_crop"] += 1
                            continue
                        base = 3.0 + _lateral_score(lat, pref_lat)  # High bonus for matching opposite direction
                        cands.append(CandidatePlacement(
                            vehicle_num=0,  # 0 indicates "not on any ego path"
                            segment_index=1,  # Single segment
                            seg_id=int(seg_id),
                            s_along=float(s_along),
                            lateral_relation=str(lat),
                            x=float(spawn["x"]),
                            y=float(spawn["y"]),
                            yaw_deg=float(spawn["yaw_deg"]),
                            path_s_m=float(s_m),
                            base_score=float(base),
                        ))
                        debug_entry["added_primary"] += 1
            # If we found opposite-lane candidates, return them (don't mix with ego paths)
            if cands:
                cands.sort(key=lambda c: c.base_score, reverse=True)
                cands = cands[:max_candidates]
                debug_entry["final_candidates"] = len(cands)
                if candidate_debug is not None:
                    candidate_debug[sid] = debug_entry
                print(
                    f"[DEBUG] Actor {sid} (opposite direction): kept {len(cands)} candidates "
                    f"(attempted={debug_entry['attempted_samples']}, "
                    f"outside_crop={debug_entry['outside_crop']})",
                    flush=True,
                )
                return cands
    
    # Standard candidate generation for ego vehicle paths
    for veh_num in veh_domain:
        picked_entry = next((p for p in picked_list if _parse_vehicle_num(p.get("vehicle")) == veh_num), None)
        if not picked_entry:
            continue
        seg_ids = picked_entry.get("signature", {}).get("segment_ids", [])
        if not isinstance(seg_ids, list) or not seg_ids:
            continue

        seg_lengths: List[float] = []
        seg_pts_cache: List[Optional[np.ndarray]] = []
        for seg_id_raw in seg_ids:
            try:
                seg_id = int(seg_id_raw)
            except Exception:
                seg_lengths.append(0.0)
                seg_pts_cache.append(None)
                continue
            pts = seg_by_id.get(seg_id)
            if pts is None or len(pts) < 2:
                seg_lengths.append(0.0)
                seg_pts_cache.append(None)
                continue
            cum = cumulative_dist(pts)
            seg_lengths.append(float(cum[-1]))
            seg_pts_cache.append(pts)
        try:
            debug_entry["segment_ids"].extend([int(sid) for sid in seg_ids if sid is not None])
        except Exception:
            pass
        debug_entry["segment_lengths_m"].extend([float(l) for l in seg_lengths if l is not None])

        total_path_len = float(sum(seg_lengths))
        if merge_min_s_m is not None and total_path_len > 0.0 and merge_min_s_m >= total_path_len:
            merge_min_s_m = None
        debug_entry["merge_window_m"] = {
            "min": float(merge_min_s_m) if merge_min_s_m is not None else None,
            "max": float(merge_max_s_m) if merge_max_s_m is not None else None,
        }

        for idx0, seg_id_raw in enumerate(seg_ids):
            segment_index = idx0 + 1
            pts = seg_pts_cache[idx0]
            seg_len = seg_lengths[idx0]
            if pts is None or seg_len < 2.0:
                continue
            if is_lane_drop_marker:
                step = max(0.5, float(LANE_DROP_CANDIDATE_STEP_M))
            else:
                step = max(2.0, float(ds_m))
            s_values_m = list(np.arange(min(crop_margin_m, 0.2*seg_len), max(seg_len - crop_margin_m, 0.8*seg_len), step))
            if len(s_values_m) < 3:
                s_values_m = [0.2*seg_len, 0.5*seg_len, 0.8*seg_len]

            info = veh_info.get(veh_num, {})
            phase_bonus = _phase_score(segment_index, info, pref_phase)

            for s_m in s_values_m:
                s_along = float(min(1.0, max(0.0, s_m / seg_len)))
                centering_bonus = _s_along_centering_score(s_along, pref_phase)
                for lat in laterals:
                    debug_entry["attempted_samples"] += 1
                    spawn = compute_spawn_from_anchor(pts, s_along, lat, None)
                    if not _inside_crop_xy(spawn["x"], spawn["y"], crop_region, margin_m=crop_margin_m):
                        debug_entry["outside_crop"] += 1
                        continue
                    path_s_m = float(sum(seg_lengths[:idx0]) + float(s_m))
                    if merge_min_s_m is not None and path_s_m < merge_min_s_m:
                        trigger_bias = _trigger_distance_bias(spec, int(veh_num), path_s_m)
                        fallback_cands.append(CandidatePlacement(
                            vehicle_num=int(veh_num),
                            segment_index=int(segment_index),
                            seg_id=int(seg_id_raw),
                            s_along=float(s_along),
                            lateral_relation=str(lat),
                            x=float(spawn["x"]),
                            y=float(spawn["y"]),
                            yaw_deg=float(spawn["yaw_deg"]),
                            path_s_m=float(path_s_m),
                            base_score=float(phase_bonus + centering_bonus + _lateral_score(lat, pref_lat) + trigger_bias + (1.0 if preferred is not None else 0.0)),
                        ))
                        debug_entry["before_merge_window"] += 1
                        debug_entry["added_fallback"] += 1
                        continue
                    if merge_max_s_m is not None and path_s_m > merge_max_s_m:
                        base = phase_bonus + centering_bonus + _lateral_score(lat, pref_lat)
                        base += _trigger_distance_bias(spec, int(veh_num), path_s_m)
                        if preferred is not None:
                            base += 1.0
                        far_cands.append(CandidatePlacement(
                            vehicle_num=int(veh_num),
                            segment_index=int(segment_index),
                            seg_id=int(seg_id_raw),
                            s_along=float(s_along),
                            lateral_relation=str(lat),
                            x=float(spawn["x"]),
                            y=float(spawn["y"]),
                            yaw_deg=float(spawn["yaw_deg"]),
                            path_s_m=float(path_s_m),
                            base_score=float(base),
                        ))
                        debug_entry["after_merge_window"] += 1
                        debug_entry["added_far"] += 1
                        continue
                    base = phase_bonus + centering_bonus + _lateral_score(lat, pref_lat)
                    base += _trigger_distance_bias(spec, int(veh_num), path_s_m)
                    if preferred is not None:
                        base += 1.0
                    cands.append(CandidatePlacement(
                        vehicle_num=int(veh_num),
                        segment_index=int(segment_index),
                        seg_id=int(seg_id_raw),
                        s_along=float(s_along),
                        lateral_relation=str(lat),
                        x=float(spawn["x"]),
                        y=float(spawn["y"]),
                        yaw_deg=float(spawn["yaw_deg"]),
                        path_s_m=float(path_s_m),
                        base_score=float(base),
                    ))
                    debug_entry["added_primary"] += 1
    if merge_max_s_m is not None and not cands:
        if far_cands:
            cands = far_cands
        elif fallback_cands:
            cands = fallback_cands
    elif merge_min_s_m is not None and not cands and fallback_cands:
        cands = fallback_cands
    cands.sort(key=lambda c: c.base_score, reverse=True)
    
    # Filter candidates to maintain minimum buffer from ego vehicle spawns
    MIN_BUFFER_TO_EGO_M = 15.0
    if ego_spawns:
        pre_buffer_len = len(cands)
        filtered_cands = []
        for cand in cands:
            violates_buffer = False
            for ego_spawn_info in ego_spawns:
                ego_spawn = ego_spawn_info.get("spawn", {})
                if not isinstance(ego_spawn, dict):
                    continue
                ego_x = ego_spawn.get("x")
                ego_y = ego_spawn.get("y")
                if ego_x is None or ego_y is None:
                    continue
                
                # Check distance from this candidate to ego spawn
                dist = math.hypot(float(cand.x) - float(ego_x), float(cand.y) - float(ego_y))
                if dist < MIN_BUFFER_TO_EGO_M:
                    violates_buffer = True
                    break
            
            if not violates_buffer:
                filtered_cands.append(cand)
        
        cands = filtered_cands
        debug_entry["ego_buffer_filtered"] = max(0, pre_buffer_len - len(cands))

    if len(cands) > max_candidates:
        cands = cands[:max_candidates]

    debug_entry["final_candidates"] = len(cands)
    if candidate_debug is not None:
        candidate_debug[sid] = debug_entry
    print(
        f"[DEBUG] Actor {sid}: kept {len(cands)} candidates "
        f"(attempted={debug_entry['attempted_samples']}, "
        f"outside_crop={debug_entry['outside_crop']}, "
        f"before_merge_window={debug_entry['before_merge_window']}, "
        f"after_merge_window={debug_entry['after_merge_window']}, "
        f"ego_buffer_filtered={debug_entry['ego_buffer_filtered']}, "
        f"primary={debug_entry['added_primary']}, "
        f"fallback={debug_entry['added_fallback']}, far={debug_entry['added_far']}, "
        f"vehicle_domain={debug_entry['vehicle_domain']}, "
        f"seg_lens_m={[round(x,2) for x in debug_entry.get('segment_lengths_m', [])]})",
        flush=True,
    )
    
    return cands


def solve_weighted_csp(
    specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    all_segments: Optional[List[Dict[str, Any]]] = None,
    merge_min_s_by_vehicle: Optional[Dict[int, float]] = None,
    min_sep_scale: float = 1.0,
    max_backtrack: int = 30000,
    ego_spawns: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, CandidatePlacement], Dict[str, Any]]:
    veh_info = _infer_vehicle_turn_exit_indices(picked_list, seg_by_id)
    domains: Dict[str, List[CandidatePlacement]] = {}
    candidate_debug: Dict[str, Any] = {}
    for s in specs:
        sid = str(s["id"])
        domains[sid] = generate_candidates_for_actor(
            s,
            picked_list,
            seg_by_id,
            crop_region,
            veh_info,
            merge_min_s_by_vehicle=merge_min_s_by_vehicle,
            all_segments=all_segments,
            ego_spawns=ego_spawns,
            candidate_debug=candidate_debug,
        )
    empty_domains = [sid for sid, dom in domains.items() if not dom]
    if empty_domains:
        for sid in empty_domains:
            info = candidate_debug.get(sid, {})
            print(
                f"[WARNING] CSP: no placement candidates for {sid} "
                f"(attempted={info.get('attempted_samples')}, "
                f"outside_crop={info.get('outside_crop')}, "
                f"vehicle_domain={info.get('vehicle_domain')})",
                flush=True,
            )

    # Build dependency graph: if A has relation to B, B should be placed before A
    # Topological sort to respect dependencies, then sort by domain size within each level
    spec_by_id = {str(s["id"]): s for s in specs}
    all_ids = [str(s["id"]) for s in specs]
    
    # Build adjacency: depends_on[A] = set of entities A depends on (referenced in A's relations)
    depends_on: Dict[str, set] = {sid: set() for sid in all_ids}
    for s in specs:
        sid = str(s["id"])
        for rel in s.get("relations", []):
            other = str(rel.get("other_id", ""))
            if other in spec_by_id:
                depends_on[sid].add(other)
    
    # Topological sort using Kahn's algorithm
    in_degree = {sid: len(depends_on[sid]) for sid in all_ids}
    # Start with entities that have no dependencies
    queue = [sid for sid in all_ids if in_degree[sid] == 0]
    queue.sort(key=lambda k: len(domains.get(k, [])))  # Sort by domain size within level
    
    order = []
    while queue:
        sid = queue.pop(0)
        order.append(sid)
        # Decrease in_degree for entities that depend on this one
        for other_sid in all_ids:
            if sid in depends_on[other_sid]:
                in_degree[other_sid] -= 1
                if in_degree[other_sid] == 0:
                    queue.append(other_sid)
        queue.sort(key=lambda k: len(domains.get(k, [])))
    
    # If there's a cycle, just add remaining in domain-size order
    remaining = [sid for sid in all_ids if sid not in order]
    remaining.sort(key=lambda k: len(domains.get(k, [])))
    order.extend(remaining)
    
    # Use asset_id for more accurate radius calculation when bbox is available
    radii = {sid: _actor_radius_m(
        spec_by_id[sid]["category"], 
        spec_by_id[sid]["actor_kind"],
        spec_by_id[sid].get("asset_id")
    ) for sid in order}

    best: Dict[str, CandidatePlacement] = {}
    best_score = -1e18
    nodes = 0

    best_base = {sid: (domains[sid][0].base_score if domains.get(sid) else 0.0) for sid in order}
    suffix = [0.0]*(len(order)+1)
    for i in range(len(order)-1, -1, -1):
        suffix[i] = suffix[i+1] + float(best_base[order[i]])

    def ok_hard(c: CandidatePlacement, partial: Dict[str, CandidatePlacement], sid: str) -> bool:
        """
        Check if placing actor `sid` at candidate `c` would collide with any already-placed actors.
        Uses actor radii (derived from bboxes when available) plus a minimum separation margin.
        """
        rr = float(radii[sid]) * float(min_sep_scale)
        for oid, oc in partial.items():
            r = (rr + float(radii[oid]) * float(min_sep_scale)) + MIN_ACTOR_SEPARATION_M
            dx = c.x - oc.x
            dy = c.y - oc.y
            dist_sq = dx*dx + dy*dy
            if dist_sq < r*r:
                return False
        return True

    def _find_related_placement(oid: str, partial: Dict[str, CandidatePlacement]) -> Optional[CandidatePlacement]:
        """
        Find the placement for an entity referenced by other_id.
        Handles group expansion: if oid is "entity_2" but we have "entity_2_1", "entity_2_2", etc.,
        return the first matching expanded entity (they share the same base anchor).
        """
        if oid in partial:
            return partial[oid]
        # Check for expanded group members (entity_X_1, entity_X_2, etc.)
        for pid, placement in partial.items():
            if pid.startswith(oid + "_") and pid[len(oid)+1:].isdigit():
                return placement
        return None

    def soft_delta(c: CandidatePlacement, partial: Dict[str, CandidatePlacement], sid: str) -> float:
        s = spec_by_id[sid]
        sc = float(c.base_score)
        for rel in s.get("relations", []):
            oid = rel["other_id"]
            other_placement = _find_related_placement(oid, partial)
            if other_placement is not None:
                sc += float(_relation_score(c, other_placement, rel))
        return sc

    def bt(i: int, partial: Dict[str, CandidatePlacement], score: float) -> None:
        nonlocal best, best_score, nodes
        nodes += 1
        if nodes > max_backtrack:
            return
        if score + suffix[i] < best_score:
            return
        if i >= len(order):
            if score > best_score:
                best_score = score
                best = dict(partial)
            return
        sid = order[i]
        dom = domains.get(sid, [])
        if not dom:
            return
        for cand in dom:
            if not ok_hard(cand, partial, sid):
                continue
            d = soft_delta(cand, partial, sid)
            partial[sid] = cand
            bt(i+1, partial, score+d)
            partial.pop(sid, None)

    bt(0, {}, 0.0)

    dbg = {
        "nodes_searched": nodes,
        "best_score": best_score,
        "domain_sizes": {k: len(v) for k, v in domains.items()},
        "candidate_debug": candidate_debug,
    }
    if len(best) != len(order):
        # greedy fallback
        fallback: Dict[str, CandidatePlacement] = {}
        for sid in order:
            dom = domains.get(sid, [])
            chosen = None
            for cand in dom:
                if ok_hard(cand, fallback, sid):
                    chosen = cand
                    break
            if chosen is None and dom:
                chosen = dom[0]
            if chosen is not None:
                fallback[sid] = chosen
        dbg["fallback_used"] = True
        missing = [sid for sid in order if sid not in fallback]
        if missing:
            dbg["missing_entities"] = missing
            print(f"[WARNING] CSP fallback could not place: {missing}", flush=True)
        return fallback, dbg
    dbg["fallback_used"] = False
    dbg["missing_entities"] = []
    return best, dbg


# Distance mapping for checking constraint satisfaction
_DISTANCE_TO_M_TARGET = {
    "touching": 2.0,
    "close": 6.0,
    "medium": 12.0,
    "far": 20.0,  # Target for "Twenty meters later"
}


def _check_distance_constraints(
    chosen: Dict[str, CandidatePlacement],
    specs: List[Dict[str, Any]],
    threshold_ratio: float = 0.7,  # Allow 70% of target as acceptable
) -> List[Tuple[str, str, float, float]]:
    """
    Check if distance constraints between entities are satisfied.
    
    Returns list of (entity_id, other_id, actual_distance, target_distance) for unsatisfied constraints.
    """
    violations = []
    
    def _find_placement(entity_id: str) -> Optional[CandidatePlacement]:
        """Find placement, handling group expansion."""
        if entity_id in chosen:
            return chosen[entity_id]
        # Check for expanded group members (entity_X_1, entity_X_2, etc.)
        for pid, p in chosen.items():
            if pid.startswith(entity_id + "_") and pid[len(entity_id)+1:].split("_")[0].isdigit():
                return p
        return None
    
    for spec in specs:
        eid = str(spec.get("id", ""))
        placement = _find_placement(eid)
        if placement is None:
            continue
        
        for rel in spec.get("relations", []):
            rel_type = rel.get("type", "")
            if rel_type not in ("ahead_of", "behind_of"):
                continue
            
            other_id = str(rel.get("other_id", ""))
            other_placement = _find_placement(other_id)
            
            if other_placement is None:
                continue
            
            # Check distance
            distance_bucket = rel.get("distance", "medium")
            target = _DISTANCE_TO_M_TARGET.get(distance_bucket, 12.0)
            
            if rel_type == "ahead_of":
                actual = placement.path_s_m - other_placement.path_s_m
            else:  # behind_of
                actual = other_placement.path_s_m - placement.path_s_m
            
            # Check if constraint is satisfied (within threshold)
            if actual < target * threshold_ratio:
                violations.append((eid, other_id, actual, target))
    
    return violations


def _compute_extension_needed(
    violations: List[Tuple[str, str, float, float]],
    chosen: Dict[str, CandidatePlacement],
    specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Compute how much additional path length is needed for each vehicle.
    
    Returns dict mapping vehicle_num -> additional_meters_needed
    """
    spec_by_id = {str(s["id"]): s for s in specs}
    extension_needed: Dict[int, float] = {}
    
    for eid, other_id, actual, target in violations:
        spec = spec_by_id.get(eid)
        if spec is None:
            # Try with group expansion (entity_2 -> entity_2_1)
            for sid, s in spec_by_id.items():
                if sid.startswith(eid + "_"):
                    spec = s
                    break
        if spec is None:
            continue
        
        # Get vehicle number from anchor.target_vehicle
        anchor = spec.get("anchor", {})
        target_vehicle = anchor.get("target_vehicle", "")
        veh_num = None
        if target_vehicle:
            # Extract number from "Vehicle 1", "Vehicle 2", etc.
            import re
            m = re.search(r'(\d+)', target_vehicle)
            if m:
                veh_num = int(m.group(1))
        
        if veh_num is None:
            # Fallback: get from placement
            placement = chosen.get(eid)
            if placement is None:
                for pid, p in chosen.items():
                    if pid.startswith(eid + "_"):
                        placement = p
                        break
            if placement is not None:
                veh_num = placement.vehicle_num
        
        if veh_num is None:
            continue
        
        # How much more do we need?
        shortfall = target - actual
        if shortfall > 0:
            current = extension_needed.get(veh_num, 0.0)
            extension_needed[veh_num] = max(current, shortfall + 10.0)  # Add margin
    
    return extension_needed


def solve_weighted_csp_with_extension(
    specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    all_segments: List[Dict[str, Any]],
    nodes: Optional[Dict[str, Any]] = None,
    merge_min_s_by_vehicle: Optional[Dict[int, float]] = None,
    min_sep_scale: float = 1.0,
    max_backtrack: int = 30000,
    max_extension_iterations: int = 3,
    ego_spawns: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, CandidatePlacement], Dict[str, Any], Any]:
    """
    Solve CSP with iterative path extension.
    
    After each CSP solve, checks if distance constraints are satisfied.
    If not, extends paths and re-solves.
    
    Returns:
        - chosen: dict mapping entity_id -> CandidatePlacement
        - dbg: debug info dict
        - crop_region: the (possibly expanded) crop region
    """
    # Make a mutable copy of crop_region so we can expand it
    if isinstance(crop_region, dict):
        crop_region = dict(crop_region)
    
    iteration = 0
    total_extensions: Dict[int, List[int]] = {}
    
    while iteration < max_extension_iterations:
        iteration += 1
        
        # Solve CSP
        chosen, dbg = solve_weighted_csp(
            specs, picked_list, seg_by_id, crop_region,
            all_segments=all_segments,
            merge_min_s_by_vehicle=merge_min_s_by_vehicle,
            min_sep_scale=min_sep_scale,
            max_backtrack=max_backtrack,
            ego_spawns=ego_spawns,
        )
        
        # Check if distance constraints are satisfied
        violations = _check_distance_constraints(chosen, specs)
        
        if not violations:
            dbg["extension_iterations"] = iteration
            dbg["extensions_made"] = total_extensions
            return chosen, dbg, crop_region
        
        # Log violations
        print(f"[INFO] CSP iteration {iteration}: {len(violations)} distance constraint violation(s)")
        for eid, other_id, actual, target in violations:
            print(f"  - {eid} should be {target:.0f}m from {other_id}, but is only {actual:.1f}m")
        
        # Compute needed extensions
        extension_needed = _compute_extension_needed(violations, chosen, specs, picked_list, seg_by_id)
        
        if not extension_needed:
            print(f"[INFO] No extensions needed (couldn't determine vehicle)")
            break
        
        # Extend paths
        any_extended = False
        for veh_num, extra_m in extension_needed.items():
            # Find picked entry for this vehicle
            picked_entry = None
            for p in picked_list:
                if _parse_vehicle_num(p.get("vehicle")) == veh_num:
                    picked_entry = p
                    break
            
            if picked_entry is None:
                continue
            
            current_len = _compute_path_length(picked_entry, seg_by_id)
            target_len = current_len + extra_m
            
            print(f"[INFO] Vehicle {veh_num}: extending from {current_len:.1f}m to {target_len:.1f}m")
            
            was_extended, new_len, parallel_extended = extend_path_if_needed(
                picked_entry, seg_by_id, all_segments, target_len, nodes=nodes, picked_list=picked_list
            )
            
            if was_extended:
                any_extended = True
                if veh_num not in total_extensions:
                    total_extensions[veh_num] = []
                total_extensions[veh_num].append(new_len)
                print(f"[INFO] Vehicle {veh_num}: extended to {new_len:.1f}m")
                
                # Track parallel vehicle extensions
                for pv in parallel_extended:
                    if pv not in total_extensions:
                        total_extensions[pv] = []
                    total_extensions[pv].append(new_len)  # Approximate length
                
                # Expand crop_region to include the extended path points (primary + parallel vehicles)
                if isinstance(crop_region, dict):
                    # Collect all segment IDs from primary and parallel vehicles
                    all_seg_ids_to_check = []
                    
                    # Primary vehicle
                    sig = picked_entry.get("signature", {})
                    all_seg_ids_to_check.extend(sig.get("segment_ids", []))
                    
                    # Parallel vehicles
                    for pv in parallel_extended:
                        for p in picked_list:
                            if _parse_vehicle_num(p.get("vehicle")) == pv:
                                pv_sig = p.get("signature", {})
                                all_seg_ids_to_check.extend(pv_sig.get("segment_ids", []))
                                break
                    
                    for seg_id in all_seg_ids_to_check:
                        pts = seg_by_id.get(int(seg_id))
                        if pts is not None and len(pts) > 0:
                            pts = np.asarray(pts)
                            xmin = float(np.min(pts[:, 0]))
                            xmax = float(np.max(pts[:, 0]))
                            ymin = float(np.min(pts[:, 1]))
                            ymax = float(np.max(pts[:, 1]))
                            if xmin < crop_region.get("xmin", float('inf')):
                                crop_region["xmin"] = xmin - 5.0  # Add margin
                            if xmax > crop_region.get("xmax", float('-inf')):
                                crop_region["xmax"] = xmax + 5.0
                            if ymin < crop_region.get("ymin", float('inf')):
                                crop_region["ymin"] = ymin - 5.0
                            if ymax > crop_region.get("ymax", float('-inf')):
                                crop_region["ymax"] = ymax + 5.0
                    print(f"[INFO] Expanded crop region to include extended paths")
        
        if not any_extended:
            print(f"[INFO] Could not extend any paths further")
            break
    
    dbg["extension_iterations"] = iteration
    dbg["extensions_made"] = total_extensions
    dbg["unresolved_violations"] = [(e, o, f"{a:.1f}m", f"{t:.0f}m") for e, o, a, t in violations] if violations else []
    return chosen, dbg, crop_region


def _spacing_bucket_to_m(bucket: str, category: str = "static", actor_kind: str = "static_prop", asset_id: Optional[str] = None) -> float:
    """
    Convert spacing bucket to meters, ensuring minimum separation for collision avoidance.
    The spacing must be at least 2 * actor_radius + MIN_ACTOR_SEPARATION_M to prevent overlaps.
    Uses bbox-based radius if asset_id is provided and has a bounding box.
    """
    b = str(bucket)
    # Get minimum spacing based on actor radius (uses bbox if available)
    actor_radius = _actor_radius_m(category, actor_kind, asset_id)
    min_spacing = 2 * actor_radius + MIN_ACTOR_SEPARATION_M
    
    if b == "tight":
        return max(min_spacing, 1.0)
    if b == "sparse":
        return max(min_spacing, 3.0)
    if b == "normal":
        return max(min_spacing, 2.0)
    return max(min_spacing, 2.0)


def _group_back_buffer_m(spec: Dict[str, Any]) -> float:
    """
    Conservative backward offset to keep expanded groups after a phase boundary.
    """
    try:
        qty = int(spec.get("quantity", 1))
    except Exception:
        qty = 1
    if qty <= 1:
        return 0.0
    gp = spec.get("group_pattern") or {}
    spacing_m = _spacing_bucket_to_m(
        gp.get("spacing_bucket", "auto"),
        category=str(spec.get("category", "static")),
        actor_kind=str(spec.get("actor_kind", "static_prop")),
        asset_id=spec.get("asset_id"),
    )
    return 0.5 * float(qty - 1) * float(spacing_m)


def expand_group_to_actors(
    base_actor: Dict[str, Any],
    spec: Dict[str, Any],
    chosen: CandidatePlacement,
    seg_by_id: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Expand quantity>1 into multiple actor dicts with custom lateral_offset_m and/or s_along offsets.
    Works without touching world conversion: we add placement.lateral_offset_m.
    """
    qty = int(spec.get("quantity", 1))
    if qty <= 1:
        return [base_actor]
    # Log expansion intent for debugging
    print(f"[INFO] expand_group_to_actors: expanding {base_actor.get('id')} to {qty} actors")

    gp = spec.get("group_pattern") or {}
    pattern = str(gp.get("pattern", "along_lane"))
    if pattern not in ALLOWED_GROUP_PATTERNS:
        pattern = "along_lane"
    
    # Get category and actor_kind for collision radius calculation
    category = spec.get("category", "static")
    actor_kind = spec.get("actor_kind", "static_prop")
    asset_id = spec.get("asset_id")
    
    # Lane Drop scenarios need tight spacing for visual lane markers
    bucket = gp.get("spacing_bucket", "auto")
    if pattern == "along_lane" and actor_kind == "static_prop":
        # Traffic cones along lane edges should be tightly spaced for visual effect
        bucket = "tight"
    
    spacing_m = _spacing_bucket_to_m(bucket, category=category, actor_kind=actor_kind, asset_id=asset_id)

    pts = seg_by_id.get(int(chosen.seg_id))
    if pts is None or len(pts) < 2:
        print(f"[WARNING] expand_group_to_actors: segment {chosen.seg_id} not found or insufficient points; "
              f"expected {qty} actors for {base_actor.get('id')}, returning only 1")
        return [base_actor]

    seg_len = float(cumulative_dist(pts)[-1])
    p, t = point_and_tangent_at_s(pts, chosen.s_along)

    start_lat = gp.get("start_lateral")
    end_lat = gp.get("end_lateral")
    a = float(LATERAL_TO_M.get(start_lat or "right_edge", +0.45*LANE_WIDTH_M))
    b = float(LATERAL_TO_M.get(end_lat or "left_edge", -0.45*LANE_WIDTH_M))
    
    # Check if we have diagonal placement (start_lateral != end_lateral)
    has_lateral_transition = (start_lat is not None and end_lat is not None and start_lat != end_lat)
    lateral_span = abs(b - a)
    lateral_step = lateral_span / float(qty - 1) if qty > 1 else lateral_span
    needs_s_offset = lateral_step < spacing_m

    out: List[Dict[str, Any]] = []
    for i in range(qty):
        child = json.loads(json.dumps(base_actor))  # deep-ish copy
        child["id"] = f"{base_actor.get('id','entity')}_{i+1}"
        
        # Compute interpolation factor
        u = 0.0 if qty == 1 else i / float(qty - 1)

        if pattern == "across_lane":
            # Side-by-side across lane width
            lat_m = a*(1-u) + b*u
            child["placement"]["lateral_offset_m"] = float(lat_m)
            if needs_s_offset:
                offset = (i - (qty-1)/2.0) * spacing_m
                child_s = float(chosen.s_along) + float(offset / max(1e-6, seg_len))
                child["placement"]["s_along"] = float(min(1.0, max(0.0, child_s)))
        elif pattern == "along_lane":
            # Offset along segment
            offset = (i - (qty-1)/2.0) * spacing_m
            child_s = float(chosen.s_along) + float(offset / max(1e-6, seg_len))
            child["placement"]["s_along"] = float(min(1.0, max(0.0, child_s)))
            # If we have lateral transition (diagonal), also interpolate lateral position
            if has_lateral_transition:
                lat_m = a*(1-u) + b*u
                child["placement"]["lateral_offset_m"] = float(lat_m)
        else:  # scatter
            # small lateral jitter
            jitter = random.uniform(-1.0, 1.0)
            child["placement"]["lateral_offset_m"] = float(LATERAL_TO_M.get(chosen.lateral_relation, 0.0) + jitter)
            offset = (i - (qty-1)/2.0) * spacing_m
            child_s = float(chosen.s_along) + float(offset / max(1e-6, seg_len))
            child["placement"]["s_along"] = float(min(1.0, max(0.0, child_s)))

        out.append(child)
    return out


__all__ = [
    "AFTER_MERGE_CLEARANCE_M",
    "LANE_DROP_AFTER_MERGE_OFFSET_M",
    "LANE_DROP_AFTER_MERGE_WINDOW_M",
    "LANE_DROP_CANDIDATE_STEP_M",
    "ALLOWED_DIST_BUCKETS",
    "ALLOWED_GROUP_PATTERNS",
    "ALLOWED_PHASES",
    "ALLOWED_REL_TYPES",
    "CandidatePlacement",
    "DIST_BUCKET_TO_M",
    "MERGE_POINT_MAX_DIST_M",
    "MIN_ACTOR_SEPARATION_M",
    "_actor_radius_m",
    "_check_distance_constraints",
    "_compute_extension_needed",
    "_compute_merge_min_s_by_vehicle",
    "_find_opposite_lane_segments",
    "_get_vehicle_lane_id",
    "_group_back_buffer_m",
    "_inside_crop_xy",
    "_lateral_score",
    "_normalize_actor_spec_id",
    "_phase_score",
    "_relation_score",
    "_spacing_bucket_to_m",
    "build_stage2_constraints_prompt",
    "expand_group_to_actors",
    "generate_candidates_for_actor",
    "get_actor_bbox_dims",
    "solve_weighted_csp",
    "solve_weighted_csp_with_extension",
    "validate_actor_specs",
]

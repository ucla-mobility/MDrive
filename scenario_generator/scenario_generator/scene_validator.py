"""
Scene Validator

Deep validation of generated scenes against natural language descriptions.
Compares scene_objects.json content to what was requested in the scenario text.

Enhanced validation includes:
1. Basic structure validation (vehicle count, maneuvers, actors)
2. Constraint satisfaction (from picked_paths_detailed.json)
3. Path intersection analysis (critical for multi-agent scenarios)
4. Entry/exit direction validation
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .capabilities import CATEGORY_DEFINITIONS, TopologyType, RequiredRelation, EgoManeuver

class ValidationIssue(Enum):
    """Types of validation issues that can occur."""
    # Vehicle count issues
    VEHICLE_COUNT_MISMATCH = "vehicle_count_mismatch"
    MISSING_VEHICLE = "missing_vehicle"
    
    # Maneuver issues
    WRONG_MANEUVER = "wrong_maneuver"
    MISSING_MANEUVER = "missing_maneuver"
    
    # Constraint issues
    MISSING_RELATIONSHIP = "missing_relationship"
    WRONG_APPROACH_DIRECTION = "wrong_approach_direction"
    CONSTRAINT_NOT_SATISFIED = "constraint_not_satisfied"
    
    # Path issues (multi-agent specific)
    NO_PATH_INTERSECTION = "no_path_intersection"
    PATHS_NOT_CONFLICTING = "paths_not_conflicting"
    NO_INTERACTION = "no_interaction"

    # Topology/geometry issues
    TOPOLOGY_MISMATCH = "topology_mismatch"
    MISSING_MERGE_FEATURE = "missing_merge_feature"
    MISSING_MULTI_LANE = "missing_multi_lane"
    MISSING_LANE_CHANGE = "missing_lane_change"
    MISSING_LANE_DROP = "missing_lane_drop"
    MISSING_REQUIRED_RELATION = "missing_required_relation"
    
    # Actor issues
    MISSING_ACTOR = "missing_actor"
    WRONG_ACTOR_TYPE = "wrong_actor_type"
    WRONG_ACTOR_MOTION = "wrong_actor_motion"
    WRONG_ACTOR_POSITION = "wrong_actor_position"
    
    # General issues
    SCENE_EMPTY = "scene_empty"
    PARSE_ERROR = "parse_error"


@dataclass
class ValidationIssueDetail:
    """Detailed information about a validation issue."""
    issue_type: ValidationIssue
    severity: str  # "error", "warning", "info"
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    suggestion: Optional[str] = None
    pipeline_stage: Optional[str] = None  # Which stage likely caused this


@dataclass
class SceneValidationResult:
    """Result of validating a scene against its description."""
    is_valid: bool
    score: float  # 0.0 to 1.0, how well scene matches description
    issues: List[ValidationIssueDetail] = field(default_factory=list)
    
    # Extracted info from description
    expected_vehicles: int = 0
    expected_maneuvers: Dict[str, str] = field(default_factory=dict)
    expected_actors: List[Dict[str, Any]] = field(default_factory=list)
    expected_relationships: List[str] = field(default_factory=list)
    
    # Extracted info from scene
    actual_vehicles: int = 0
    actual_maneuvers: Dict[str, str] = field(default_factory=dict)
    actual_actors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Path intersection info (critical for multi-agent scenarios)
    paths_intersect: bool = False
    intersection_region: Optional[Dict[str, float]] = None
    
    def add_issue(self, issue_type: ValidationIssue, severity: str, message: str, 
                  expected: str = None, actual: str = None, 
                  suggestion: str = None, pipeline_stage: str = None):
        self.issues.append(ValidationIssueDetail(
            issue_type=issue_type,
            severity=severity,
            message=message,
            expected=expected,
            actual=actual,
            suggestion=suggestion,
            pipeline_stage=pipeline_stage,
        ))
    
    def get_errors(self) -> List[ValidationIssueDetail]:
        return [i for i in self.issues if i.severity == "error"]
    
    def get_warnings(self) -> List[ValidationIssueDetail]:
        return [i for i in self.issues if i.severity == "warning"]
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Validation: {'PASS' if self.is_valid else 'FAIL'} (score: {self.score:.2f})",
            f"Vehicles: expected {self.expected_vehicles}, got {self.actual_vehicles}",
        ]
        
        # Show path intersection status for multi-agent scenarios
        if self.expected_vehicles >= 2:
            if self.paths_intersect:
                region = self.intersection_region
                if region:
                    lines.append(f"Paths intersect: YES (near x={region['x']:.1f}, y={region['y']:.1f})")
                else:
                    lines.append("Paths intersect: YES")
            else:
                lines.append("Paths intersect: NO ⚠️")
        
        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            for issue in self.issues:
                prefix = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
                lines.append(f"  {prefix} {issue.message}")
                if issue.suggestion:
                    lines.append(f"      → {issue.suggestion}")
        return "\n".join(lines)


class SceneValidator:
    """
    Validates generated scenes against their natural language descriptions.
    
    Enhanced features:
    - Path intersection detection (critical for multi-agent scenarios)
    - Constraint satisfaction verification from picked_paths
    - Multi-agent coordination validation
    """
    
    # Relationship keywords that imply paths MUST intersect
    INTERSECTION_REQUIRED_PATTERNS = [
        r'intersect', r'cross(?:ing)?\s+paths?', r'paths?\s+cross', r'conflict(?:ing)?',
        r'negotiate', r'yield', r'right[\s-]?of[\s-]?way',
        r'deadlock', r'standoff', r'competition',
        r'unprotected\s+(?:left|right)', r'perpendicular',
        r'intersection', r'junction',
    ]
    
    def __init__(self):
        # Patterns for extracting info from natural language
        self.vehicle_pattern = re.compile(r'Vehicle\s+(\d+)', re.IGNORECASE)
        self.maneuver_patterns = {
            'left': re.compile(r'Vehicle\s+(\d+)[^.]*(?:turns?\s+left|left\s+turn|turning\s+left)', re.IGNORECASE),
            'right': re.compile(r'Vehicle\s+(\d+)[^.]*(?:turns?\s+right|right\s+turn|turning\s+right)', re.IGNORECASE),
            'straight': re.compile(r'Vehicle\s+(\d+)[^.]*(?:goes?\s+straight|straight\s+through|continues?\s+straight|travels?\s+straight)', re.IGNORECASE),
            'lane_change': re.compile(r'Vehicle\s+(\d+)[^.]*(?:changes?\s+lane|lane\s+change|merges?|merging)', re.IGNORECASE),
        }
        
        # Relationship patterns
        self.relationship_patterns = {
            'opposite_approach': re.compile(r'(?:opposite|oncoming|opposing)[\s_]+(?:direction|approach|traffic)', re.IGNORECASE),
            'perpendicular': re.compile(r'(?:perpendicular|from\s+the\s+(?:left|right)|cross(?:ing)?\s+paths?|paths?\s+cross)', re.IGNORECASE),
            'same_approach': re.compile(r'(?:same\s+(?:direction|approach)|behind|following|follows?)', re.IGNORECASE),
            'adjacent_lane': re.compile(r'(?:adjacent\s+lane|left\s+lane|right\s+lane|next\s+lane)', re.IGNORECASE),
            'merging': re.compile(r'(?:merg(?:es?|ing)|lane\s+drop|zipper)', re.IGNORECASE),
        }
        
        # Compiled intersection requirement pattern
        self.intersection_required_pattern = re.compile(
            '|'.join(self.INTERSECTION_REQUIRED_PATTERNS), re.IGNORECASE
        )
        # Actor patterns
        self.actor_patterns = {
            'walker': re.compile(r'(?:pedestrian|walker|person|crossing)\s*(?:cross(?:es|ing)?)?', re.IGNORECASE),
            'parked_vehicle': re.compile(r'(?:parked\s+(?:vehicle|car|truck)|stopped\s+vehicle|blocking|obstruction)', re.IGNORECASE),
            'cyclist': re.compile(r'(?:cyclist|bicycle|bike|cycling)', re.IGNORECASE),
            'static_prop': re.compile(r'(?:cone|barrier|traffic\s+cone|construction|debris)', re.IGNORECASE),
        }
        
        # Motion patterns for actors
        self.motion_patterns = {
            'cross_perpendicular': re.compile(r'(?:cross(?:es|ing)?(?:\s+the\s+road)?|perpendicular)', re.IGNORECASE),
            'static': re.compile(r'(?:parked|stopped|stationary|blocking|static)', re.IGNORECASE),
            'follow_lane': re.compile(r'(?:follow(?:s|ing)?\s+(?:the\s+)?lane|along\s+(?:the\s+)?lane)', re.IGNORECASE),
        }
        
        # Lateral position patterns
        self.lateral_patterns = {
            'center': re.compile(r'(?:center|middle|in\s+the\s+lane)', re.IGNORECASE),
            'right_edge': re.compile(r'(?:right\s+(?:edge|side)|at\s+the\s+right)', re.IGNORECASE),
            'left_edge': re.compile(r'(?:left\s+(?:edge|side)|at\s+the\s+left)', re.IGNORECASE),
            'half_right': re.compile(r'(?:half\s+right|partially\s+right)', re.IGNORECASE),
            'half_left': re.compile(r'(?:half\s+left|partially\s+left)', re.IGNORECASE),
        }
    
    def _vehicle_only_text(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        vehicle_sentences = [
            sentence for sentence in sentences
            if re.search(r'\bvehicle', sentence, re.IGNORECASE)
        ]
        return " ".join(vehicle_sentences)

    def _requires_path_intersection(
        self,
        text: str,
        category: Optional[str] = None,
        expected_flags: Optional[Dict[str, Any]] = None,
        expected_relationships: Optional[List[str]] = None,
        expected_maneuvers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Determine if the scenario description implies paths must intersect.
        This is critical for multi-agent coordination scenarios.
        """
        if expected_flags:
            topo = expected_flags.get("required_topology")
            if topo in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY}:
                return False

        if expected_relationships:
            if any(rel in ("perpendicular", "opposite_approach") for rel in expected_relationships):
                return True

        if expected_maneuvers:
            turners = [m for m in expected_maneuvers.values() if str(m).lower() in ("left", "right", "uturn")]
            if len(turners) >= 2:
                return True

        if category in {"Intersection Deadlock Resolution"}:
            return True

        return bool(self.intersection_required_pattern.search(self._vehicle_only_text(text)))
    
    def _extract_polylines_from_scene(self, scene_data: Dict[str, Any]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract polylines for each vehicle's path from scene_objects.json.
        Returns dict mapping vehicle name to list of (x, y) points.
        """
        polylines = {}
        
        for ego in scene_data.get('ego_picked', []):
            vehicle_name = ego.get('vehicle', 'unknown')
            points = []
            
            # Get polylines from segments_detailed
            for segment in ego.get('signature', {}).get('segments_detailed', []):
                for pt in segment.get('polyline_sample', []):
                    points.append((pt.get('x', 0), pt.get('y', 0)))
            
            if points:
                polylines[vehicle_name] = points
        
        return polylines

    def _extract_polyline_samples_with_meta(
        self,
        scene_data: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract polyline samples with lane/road metadata for temporal validation.
        Returns dict mapping vehicle name to list of sample dicts with x/y/s/road/lane.
        """
        polylines = {}
        for ego in scene_data.get('ego_picked', []):
            vehicle_name = ego.get('vehicle', 'unknown')
            samples = []
            cumulative_s = 0.0
            last_pt = None
            for segment in ego.get('signature', {}).get('segments_detailed', []):
                road_id = segment.get('road_id')
                section_id = segment.get('section_id')
                lane_id = segment.get('lane_id')
                for pt in segment.get('polyline_sample', []):
                    x = pt.get('x', 0.0)
                    y = pt.get('y', 0.0)
                    if last_pt is not None:
                        cumulative_s += math.hypot(x - last_pt[0], y - last_pt[1])
                    samples.append({
                        'x': x,
                        'y': y,
                        's': cumulative_s,
                        'road_id': road_id,
                        'section_id': section_id,
                        'lane_id': lane_id,
                    })
                    last_pt = (x, y)
            if samples:
                polylines[vehicle_name] = samples
        return polylines

    def _is_static_actor(self, actor: Dict[str, Any]) -> bool:
        motion = actor.get('motion') if isinstance(actor.get('motion'), dict) else {}
        motion_type = (motion.get('type') or "").lower()
        category = (actor.get('category') or "").lower()
        kind = (actor.get('kind') or "").lower()
        return motion_type == "static" or category == "static" or kind == "static_prop"

    def _sample_actor_path_points(self, actor: Dict[str, Any]) -> List[Tuple[float, float]]:
        waypoints = actor.get("world_waypoints", [])
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            return []

        points: List[Tuple[float, float]] = []
        for idx in range(len(waypoints) - 1):
            p0 = waypoints[idx]
            p1 = waypoints[idx + 1]
            if not isinstance(p0, dict) or not isinstance(p1, dict):
                continue
            x0, y0 = p0.get("x"), p0.get("y")
            x1, y1 = p1.get("x"), p1.get("y")
            if x0 is None or y0 is None or x1 is None or y1 is None:
                continue
            for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        return points

    def _extract_actor_spawns(self, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        actors = []
        for actor in scene_data.get('actors', []):
            spawn = actor.get('spawn') or {}
            if 'x' not in spawn or 'y' not in spawn:
                continue
            path_points = self._sample_actor_path_points(actor)
            if not path_points:
                path_points = [(spawn.get('x', 0.0), spawn.get('y', 0.0))]
            actors.append({
                'x': spawn.get('x', 0.0),
                'y': spawn.get('y', 0.0),
                'yaw_deg': spawn.get('yaw_deg'),
                'semantic': actor.get('semantic', ''),
                'category': actor.get('category', ''),
                'kind': actor.get('kind', ''),
                'placement': actor.get('placement', {}) if isinstance(actor.get('placement'), dict) else {},
                'motion': actor.get('motion', {}) if isinstance(actor.get('motion'), dict) else {},
                'is_static': self._is_static_actor(actor),
                'path_points': path_points,
            })
        return actors

    def _closest_sample_to_point(
        self,
        samples: List[Dict[str, Any]],
        x: float,
        y: float,
    ) -> Tuple[Optional[int], float]:
        min_dist = float('inf')
        min_idx = None
        for idx, pt in enumerate(samples):
            dist = math.hypot(pt['x'] - x, pt['y'] - y)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        return min_idx, min_dist

    def _heading_angle(
        self,
        samples: List[Dict[str, Any]],
        idx: int,
    ) -> Optional[float]:
        if len(samples) < 2:
            return None
        if idx <= 0:
            p0, p1 = samples[0], samples[1]
        elif idx >= len(samples) - 1:
            p0, p1 = samples[-2], samples[-1]
        else:
            p0, p1 = samples[idx - 1], samples[idx + 1]
        dx = p1['x'] - p0['x']
        dy = p1['y'] - p0['y']
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return math.atan2(dy, dx)

    def _angle_diff_deg(self, a: float, b: float) -> float:
        diff = abs(a - b) % (2.0 * math.pi)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff
        return math.degrees(diff)
    
    # =========================================================================
    # INTERACTION REGION DETECTION (Fix 3)
    # Detects path convergence points geometrically and validates all vehicles
    # approach the derived interaction region.
    # =========================================================================
    
    def _find_interaction_region(
        self,
        scene_data: Dict[str, Any],
        convergence_threshold_m: float = 15.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the interaction region by detecting path convergence points.
        
        Algorithm:
        1. For each pair of vehicle paths, find the point of minimum distance
        2. If multiple convergence points exist, compute their centroid
        3. Return the centroid as the interaction region center
        
        This is fully geometric - no category-specific logic.
        
        Args:
            scene_data: Loaded scene_objects.json data
            convergence_threshold_m: Maximum distance to consider paths "converging"
            
        Returns:
            Dict with 'x', 'y', 'radius_m', 'contributing_pairs' or None if no convergence found
        """
        polylines = self._extract_polylines_from_scene(scene_data)
        
        if len(polylines) < 2:
            return None
        
        convergence_points = []
        vehicle_names = list(polylines.keys())
        
        for i, v1_name in enumerate(vehicle_names):
            for v2_name in vehicle_names[i + 1:]:
                poly1 = polylines[v1_name]
                poly2 = polylines[v2_name]
                
                if len(poly1) < 2 or len(poly2) < 2:
                    continue
                
                # Find the point of minimum distance between the two polylines
                min_dist = float('inf')
                min_point = None
                
                for k in range(len(poly1) - 1):
                    a0, a1 = poly1[k], poly1[k + 1]
                    for j in range(len(poly2) - 1):
                        b0, b1 = poly2[j], poly2[j + 1]
                        dist, mid = self._segment_segment_distance(a0, a1, b0, b1)
                        if dist < min_dist:
                            min_dist = dist
                            min_point = mid
                
                if min_dist <= convergence_threshold_m and min_point is not None:
                    convergence_points.append({
                        'x': min_point[0],
                        'y': min_point[1],
                        'distance_m': min_dist,
                        'pair': (v1_name, v2_name),
                    })
        
        if not convergence_points:
            return None
        
        # Compute centroid of all convergence points
        total_x = sum(cp['x'] for cp in convergence_points)
        total_y = sum(cp['y'] for cp in convergence_points)
        centroid_x = total_x / len(convergence_points)
        centroid_y = total_y / len(convergence_points)
        
        # Compute radius as max distance from centroid to any convergence point
        max_radius = max(
            math.hypot(cp['x'] - centroid_x, cp['y'] - centroid_y)
            for cp in convergence_points
        )
        # Add buffer to radius
        radius_m = max(max_radius + 10.0, 20.0)
        
        return {
            'x': centroid_x,
            'y': centroid_y,
            'radius_m': radius_m,
            'contributing_pairs': [(cp['pair'], cp['distance_m']) for cp in convergence_points],
        }
    
    def _validate_all_vehicles_near_interaction_region(
        self,
        scene_data: Dict[str, Any],
        interaction_region: Dict[str, Any],
        approach_threshold_m: float = 30.0,
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that all vehicles approach the interaction region.
        
        Args:
            scene_data: Loaded scene_objects.json data
            interaction_region: Result from _find_interaction_region
            approach_threshold_m: Maximum distance from region center to be considered "approaching"
            
        Returns:
            Tuple of (approaching_vehicles, isolated_vehicles)
        """
        polylines = self._extract_polylines_from_scene(scene_data)
        region_x = interaction_region['x']
        region_y = interaction_region['y']
        
        approaching = []
        isolated = []
        
        for vehicle_name, path in polylines.items():
            if not path:
                isolated.append(vehicle_name)
                continue
            
            # Find minimum distance from vehicle path to interaction region center
            min_dist = float('inf')
            for point in path:
                dist = math.hypot(point[0] - region_x, point[1] - region_y)
                if dist < min_dist:
                    min_dist = dist
            
            if min_dist <= approach_threshold_m:
                approaching.append(vehicle_name)
            else:
                isolated.append(vehicle_name)
        
        return approaching, isolated
    
    def _point_segment_distance(
        self,
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> Tuple[float, Tuple[float, float]]:
        ax, ay = a
        bx, by = b
        px, py = p
        vx, vy = (bx - ax), (by - ay)
        denom = vx * vx + vy * vy
        if denom < 1e-9:
            return math.hypot(px - ax, py - ay), a
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        t = max(0.0, min(1.0, t))
        cx = ax + t * vx
        cy = ay + t * vy
        return math.hypot(px - cx, py - cy), (cx, cy)

    def _segment_segment_distance(
        self,
        a0: Tuple[float, float],
        a1: Tuple[float, float],
        b0: Tuple[float, float],
        b1: Tuple[float, float],
    ) -> Tuple[float, Tuple[float, float]]:
        candidates = []
        d, cp = self._point_segment_distance(a0, b0, b1)
        candidates.append((d, a0, cp))
        d, cp = self._point_segment_distance(a1, b0, b1)
        candidates.append((d, a1, cp))
        d, cp = self._point_segment_distance(b0, a0, a1)
        candidates.append((d, cp, b0))
        d, cp = self._point_segment_distance(b1, a0, a1)
        candidates.append((d, cp, b1))
        dmin, pa, pb = min(candidates, key=lambda x: x[0])
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        return dmin, mid

    def _check_polylines_intersect(
        self, 
        poly1: List[Tuple[float, float]], 
        poly2: List[Tuple[float, float]],
        threshold_m: float = 5.0,  # How close paths need to be to "intersect"
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Check if two polylines come within threshold distance of each other.
        Returns (intersects, intersection_point or None).
        
        This is a simplified check - finds minimum distance between polylines.
        """
        if len(poly1) < 2 or len(poly2) < 2:
            return (False, None)

        min_dist = float('inf')
        closest_point = None

        for i in range(len(poly1) - 1):
            a0, a1 = poly1[i], poly1[i + 1]
            for j in range(len(poly2) - 1):
                b0, b1 = poly2[j], poly2[j + 1]
                dist, mid = self._segment_segment_distance(a0, a1, b0, b1)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = mid
                    if min_dist <= threshold_m:
                        return (True, closest_point)

        return (min_dist <= threshold_m, closest_point if min_dist <= threshold_m else None)

    def _polyline_min_distance(
        self,
        poly1: List[Tuple[float, float]],
        poly2: List[Tuple[float, float]],
    ) -> float:
        if len(poly1) < 2 or len(poly2) < 2:
            return float("inf")
        min_dist = float("inf")
        for i in range(len(poly1) - 1):
            a0, a1 = poly1[i], poly1[i + 1]
            for j in range(len(poly2) - 1):
                b0, b1 = poly2[j], poly2[j + 1]
                dist, _ = self._segment_segment_distance(a0, a1, b0, b1)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _roundabout_overlap_ok(
        self,
        scene_data: Dict[str, Any],
        proximity_threshold_m: float = 6.0,
    ) -> Tuple[bool, str]:
        """Roundabout scenarios require non-disjoint ego interactions."""
        egos = scene_data.get("ego_picked", [])
        if not isinstance(egos, list) or len(egos) < 2:
            return False, "Roundabout requires at least two ego vehicles."

        payload = []
        for ego in egos:
            if not isinstance(ego, dict):
                continue
            vehicle = str(ego.get("vehicle", "")).strip() or "unknown"
            sig = ego.get("signature", {}) if isinstance(ego.get("signature"), dict) else {}
            entry = sig.get("entry", {}) if isinstance(sig.get("entry"), dict) else {}
            exit_info = sig.get("exit", {}) if isinstance(sig.get("exit"), dict) else {}
            poly = self._extract_polylines_from_scene({"ego_picked": [ego]}).get(vehicle, [])
            payload.append(
                {
                    "vehicle": vehicle,
                    "entry": str(entry.get("cardinal4", "unknown")).lower(),
                    "exit": str(exit_info.get("cardinal4", "unknown")).lower(),
                    "poly": poly,
                }
            )

        if len(payload) < 2:
            return False, "Roundabout ego paths missing geometry."

        for i in range(len(payload)):
            for j in range(i + 1, len(payload)):
                a = payload[i]
                b = payload[j]
                if a["exit"] != "unknown" and a["exit"] == b["exit"]:
                    return True, f"{a['vehicle']} and {b['vehicle']} share the same exit."
                if (
                    a["exit"] != "unknown"
                    and b["entry"] != "unknown"
                    and a["exit"] == b["entry"]
                ) or (
                    b["exit"] != "unknown"
                    and a["entry"] != "unknown"
                    and b["exit"] == a["entry"]
                ):
                    return True, f"{a['vehicle']} exit aligns with {b['vehicle']} entry (swap/merge)."
                d = self._polyline_min_distance(a["poly"], b["poly"])
                if d <= proximity_threshold_m:
                    return True, (
                        f"{a['vehicle']} and {b['vehicle']} approach within {d:.1f}m "
                        f"(<= {proximity_threshold_m:.1f}m)."
                    )

        min_pair = float("inf")
        pair_name = "unknown"
        for i in range(len(payload)):
            for j in range(i + 1, len(payload)):
                d = self._polyline_min_distance(payload[i]["poly"], payload[j]["poly"])
                if d < min_pair:
                    min_pair = d
                    pair_name = f"{payload[i]['vehicle']} vs {payload[j]['vehicle']}"
        return False, f"Closest ego-path pair ({pair_name}) is {min_pair:.1f}m apart (disjoint roundabout traversal)."

    def _validate_pedestrian_crosswalk_semantics(
        self,
        scene_data: Dict[str, Any],
        result: "SceneValidationResult",
    ) -> bool:
        """
        Ensure walkers start from road side (not middle of vehicle path) and
        then traverse into/through the vehicle corridor.
        Returns True when all hard checks pass.
        """
        actors = scene_data.get("actors", [])
        if not isinstance(actors, list):
            actors = []
        walkers = []
        for actor in actors:
            if not isinstance(actor, dict):
                continue
            cat = str(actor.get("category", "")).lower()
            kind = str(actor.get("kind", "")).lower()
            semantic = str(actor.get("semantic", "")).lower()
            if cat == "walker" or kind == "walker" or "walker" in semantic or "pedestrian" in semantic:
                walkers.append(actor)

        if not walkers:
            result.add_issue(
                ValidationIssue.MISSING_ACTOR,
                "error",
                "Pedestrian Crosswalk requires at least one walker actor.",
                pipeline_stage="step_05_object_placer",
            )
            return False

        samples_by_vehicle = self._extract_polyline_samples_with_meta(scene_data)
        ego_points: List[Tuple[float, float]] = []
        for samples in samples_by_vehicle.values():
            for pt in samples:
                ego_points.append((float(pt["x"]), float(pt["y"])))
        if not ego_points:
            # Other checks will already fail this scene; do not duplicate hard-fail here.
            return True

        def _min_dist_to_ego(x: float, y: float) -> float:
            return min(math.hypot(x - ex, y - ey) for ex, ey in ego_points)

        def _min_dist_to_vehicle(vehicle: str, x: float, y: float) -> Optional[float]:
            if not vehicle:
                return None
            samples = samples_by_vehicle.get(vehicle, [])
            if not samples:
                return None
            return min(
                math.hypot(x - float(pt["x"]), y - float(pt["y"]))
                for pt in samples
            )

        ok = True
        for walker in walkers:
            motion = walker.get("motion", {}) if isinstance(walker.get("motion"), dict) else {}
            motion_type = str(motion.get("type", "")).lower()
            if motion_type != "cross_perpendicular":
                result.add_issue(
                    ValidationIssue.WRONG_ACTOR_MOTION,
                    "error",
                    f"Walker '{walker.get('id', 'walker')}' is '{motion_type or 'unknown'}' instead of cross_perpendicular.",
                    pipeline_stage="step_05_object_placer",
                )
                ok = False
                continue

            spawn = walker.get("spawn", {}) if isinstance(walker.get("spawn"), dict) else {}
            if "x" not in spawn or "y" not in spawn:
                result.add_issue(
                    ValidationIssue.WRONG_ACTOR_POSITION,
                    "error",
                    f"Walker '{walker.get('id', 'walker')}' is missing spawn coordinates.",
                    pipeline_stage="step_05_object_placer",
                )
                ok = False
                continue

            sx = float(spawn["x"])
            sy = float(spawn["y"])
            trigger = walker.get("trigger", {}) if isinstance(walker.get("trigger"), dict) else {}
            placement = walker.get("placement", {}) if isinstance(walker.get("placement"), dict) else {}
            target_vehicle = str(trigger.get("vehicle", "")).strip() or str(placement.get("target_vehicle", "")).strip()

            # Hard floor against obvious lane-centerline starts.
            d_start_any = _min_dist_to_ego(sx, sy)
            if d_start_any < 0.8:
                result.add_issue(
                    ValidationIssue.WRONG_ACTOR_POSITION,
                    "error",
                    (
                        f"Walker '{walker.get('id', 'walker')}' starts nearly on an ego lane "
                        f"centerline ({d_start_any:.1f}m)."
                    ),
                    pipeline_stage="step_05_object_placer",
                )
                ok = False
            # For crosswalk semantics, evaluate side-of-road relative to intended vehicle.
            d_start_target = _min_dist_to_vehicle(target_vehicle, sx, sy)
            if d_start_target is not None and d_start_target < 2.2:
                result.add_issue(
                    ValidationIssue.WRONG_ACTOR_POSITION,
                    "error",
                    (
                        f"Walker '{walker.get('id', 'walker')}' starts too close to target "
                        f"vehicle lane ({target_vehicle}, {d_start_target:.1f}m)."
                    ),
                    pipeline_stage="step_05_object_placer",
                )
                ok = False

            path_points = self._sample_actor_path_points(walker)
            if not path_points:
                result.add_issue(
                    ValidationIssue.WRONG_ACTOR_MOTION,
                    "error",
                    f"Walker '{walker.get('id', 'walker')}' has no traversable waypoint path.",
                    pipeline_stage="step_05_object_placer",
                )
                ok = False
                continue

            if d_start_target is not None:
                d_min = min(_min_dist_to_vehicle(target_vehicle, px, py) or float("inf") for px, py in path_points)
                miss_msg = (
                    f"Walker '{walker.get('id', 'walker')}' never approaches target vehicle "
                    f"corridor ({target_vehicle}, closest {d_min:.1f}m)."
                )
            else:
                d_min = min(_min_dist_to_ego(px, py) for px, py in path_points)
                miss_msg = (
                    f"Walker '{walker.get('id', 'walker')}' never approaches vehicle corridor "
                    f"(closest {d_min:.1f}m)."
                )
            if d_min > 4.5:
                result.add_issue(
                    ValidationIssue.WRONG_ACTOR_POSITION,
                    "error",
                    miss_msg,
                    pipeline_stage="step_05_object_placer",
                )
                ok = False

        return ok
    
    def _validate_path_intersections(
        self, 
        scene_data: Dict[str, Any],
        scenario_text: str,
        result: 'SceneValidationResult',
        expected_flags: Optional[Dict[str, Any]] = None,
        expected_relationships: Optional[List[str]] = None,
        expected_maneuvers: Optional[Dict[str, str]] = None,
        category: Optional[str] = None,
    ) -> float:
        """
        Validate that paths intersect if the scenario requires it.
        
        ENHANCED: Also validates temporal-spatial interaction potential.
        Not enough for paths to be nearby - vehicles must be able to 
        interact meaningfully (temporal co-occurrence + conflict).
        
        Returns a score component (0.0 to 1.0).
        """
        requires_intersection = self._requires_path_intersection(
            scenario_text,
            category=category,
            expected_flags=expected_flags,
            expected_relationships=expected_relationships,
            expected_maneuvers=expected_maneuvers,
        )
        if expected_flags and expected_flags.get("required_topology") in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY}:
            requires_intersection = False
        
        polylines = self._extract_polylines_from_scene(scene_data)
        
        if len(polylines) < 2:
            # Single vehicle scenarios don't need interaction validation
            if requires_intersection:
                result.add_issue(
                    ValidationIssue.PATHS_NOT_CONFLICTING, "warning",
                    "Cannot verify path intersection with fewer than 2 vehicles",
                    pipeline_stage="step_03_path_picker"
                )
                return 0.7
            return 1.0
        
        # For multi-vehicle scenarios, ALWAYS validate temporal interaction
        # even if the category doesn't explicitly require intersection
        has_temporal_interaction = self._validate_temporal_interaction(scene_data, polylines, result)
        
        # If scenario doesn't require intersection, temporal interaction is the only check
        if not requires_intersection:
            return 1.0 if has_temporal_interaction else 0.6
        
        # Check all pairs of vehicles for geometric intersection
        vehicle_names = list(polylines.keys())
        any_intersection = False
        intersection_points = []
        
        for i in range(len(vehicle_names)):
            for j in range(i + 1, len(vehicle_names)):
                v1, v2 = vehicle_names[i], vehicle_names[j]
                intersects, point = self._check_polylines_intersect(
                    polylines[v1], polylines[v2], threshold_m=6.0
                )
                if intersects:
                    any_intersection = True
                    if point:
                        intersection_points.append(point)
        
        if any_intersection:
            result.paths_intersect = True
            if intersection_points:
                # Store approximate intersection region
                avg_x = sum(p[0] for p in intersection_points) / len(intersection_points)
                avg_y = sum(p[1] for p in intersection_points) / len(intersection_points)
                result.intersection_region = {'x': avg_x, 'y': avg_y}
            
            # If paths intersect but no temporal interaction, reduce score
            if not has_temporal_interaction:
                return 0.6  # Geometric intersection but no actual interaction potential
            return 1.0
        else:
            result.paths_intersect = False
            result.add_issue(
                ValidationIssue.NO_PATH_INTERSECTION, "error",
                "Scenario requires path intersection but paths do not cross",
                expected="paths within 6m of each other",
                actual="paths do not intersect",
                suggestion="Re-run path picker with intersection-causing constraints (opposite_approach, perpendicular)",
                pipeline_stage="step_03_path_picker"
            )
            return 0.3
    
    def _validate_temporal_interaction(
        self,
        scene_data: Dict[str, Any],
        polylines: Dict[str, List[Tuple[float, float]]],
        result: 'SceneValidationResult',
    ) -> bool:
        """
        Validate that vehicles have temporal-spatial interaction potential.
        
        This checks if vehicles are at the same spatial region at similar times,
        not just if their paths are geometrically close.
        
        Returns True if meaningful interaction potential exists.
        """
        ego_picked = scene_data.get('ego_picked', [])
        if len(ego_picked) < 2:
            return True  # Can't validate with <2 vehicles

        debug = os.getenv("SCENE_VALIDATOR_DEBUG") == "1"
        # Reduced from 10.0m to 8.0m for stricter interaction requirements
        spatial_window_m = 8.0
        temporal_window_s = 5.0
        merge_distance_m = 2.0
        non_parallel_angle_deg = 25.0

        # Extract refined spawn indices and speeds
        vehicle_data = {}
        samples_by_vehicle = self._extract_polyline_samples_with_meta(scene_data)
        for ego in ego_picked:
            veh_name = ego.get('vehicle', '')
            refined = ego.get('refined', {})
            if not refined:
                continue

            start_idx = refined.get('start_idx_global', 0)
            speed = refined.get('speed_mps', 8.0)  # default 8 m/s
            if not speed or speed <= 0:
                speed = 8.0

            samples = samples_by_vehicle.get(veh_name, [])
            if len(samples) < 2:
                continue

            spawn_idx = max(0, min(int(start_idx), len(samples) - 1))
            spawn_s = samples[spawn_idx]['s']
            headings = [self._heading_angle(samples, idx) for idx in range(len(samples))]

            vehicle_data[veh_name] = {
                'speed': speed,
                'spawn_s': spawn_s,
                'samples': samples,
                'headings': headings,
            }

        if len(vehicle_data) < 2:
            return True  # Can't validate

        if debug:
            vehicles_dbg = ", ".join(
                f"{name}(n={len(v['samples'])},speed={v['speed']:.2f})"
                for name, v in vehicle_data.items()
            )
            print(f"[TEMPORAL VALIDATION] vehicles: {vehicles_dbg}")

        # Check all pairs for temporal-spatial co-occurrence with conflict potential
        vehicle_names = list(vehicle_data.keys())
        any_interaction = False

        for i in range(len(vehicle_names)):
            for j in range(i + 1, len(vehicle_names)):
                v1_name, v2_name = vehicle_names[i], vehicle_names[j]
                v1, v2 = vehicle_data[v1_name], vehicle_data[v2_name]
                pair_interaction = False

                min_dist = float('inf')
                min_time_diff = float('inf')

                for idx1, pt1 in enumerate(v1['samples']):
                    t1 = (pt1['s'] - v1['spawn_s']) / v1['speed']
                    if t1 < 0:
                        continue
                    for idx2, pt2 in enumerate(v2['samples']):
                        t2 = (pt2['s'] - v2['spawn_s']) / v2['speed']
                        if t2 < 0:
                            continue
                        time_diff = abs(t1 - t2)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff

                        dx = pt1['x'] - pt2['x']
                        dy = pt1['y'] - pt2['y']
                        dist = math.hypot(dx, dy)
                        if dist < min_dist:
                            min_dist = dist

                        if dist > spatial_window_m:
                            continue
                        if time_diff > temporal_window_s:
                            continue

                        same_lane = (
                            pt1.get('road_id') == pt2.get('road_id')
                            and pt1.get('section_id') == pt2.get('section_id')
                            and pt1.get('lane_id') == pt2.get('lane_id')
                        )
                        if same_lane:
                            pair_interaction = True
                            break

                        adjacent_lane = (
                            pt1.get('road_id') == pt2.get('road_id')
                            and pt1.get('section_id') == pt2.get('section_id')
                            and pt1.get('lane_id') is not None
                            and pt2.get('lane_id') is not None
                            and abs(pt1.get('lane_id') - pt2.get('lane_id')) == 1
                        )
                        if adjacent_lane:
                            pair_interaction = True
                            break

                        if dist <= merge_distance_m:
                            pair_interaction = True
                            break

                        heading1 = v1['headings'][idx1]
                        heading2 = v2['headings'][idx2]
                        if heading1 is None or heading2 is None:
                            continue
                        angle_diff = self._angle_diff_deg(heading1, heading2)
                        if angle_diff >= non_parallel_angle_deg:
                            pair_interaction = True
                            break

                    if pair_interaction:
                        break

                if debug:
                    dist_dbg = f"{min_dist:.2f}m" if min_dist != float('inf') else "n/a"
                    time_dbg = f"{min_time_diff:.2f}s" if min_time_diff != float('inf') else "n/a"
                    print(
                        f"[TEMPORAL VALIDATION] pair {v1_name} vs {v2_name}: "
                        f"min_dist={dist_dbg} min_time_diff={time_dbg} interaction={pair_interaction}"
                    )

                if pair_interaction:
                    any_interaction = True
                    break
        
        if not any_interaction:
            result.add_issue(
                ValidationIssue.PATHS_NOT_CONFLICTING, "warning",
                "Vehicles have no temporal-spatial interaction potential",
                expected="vehicles within 8m at similar times (within 5s)",
                actual=f"vehicles do not co-occur spatially and temporally",
                suggestion="Ensure vehicles spawn positions and speeds allow them to interact. "
                          "Consider: opposite directions, perpendicular approaches, or following relationships.",
                pipeline_stage="step_03_path_picker"
            )
        
        return any_interaction

    def _validate_interaction_coverage(
        self,
        scene_data: Dict[str, Any],
        result: 'SceneValidationResult',
        scenario_spec: Optional[Any] = None,
    ) -> float:
        """
        Ensure each ego vehicle has some interaction potential.

        Interaction can be with another ego (temporal/spatial conflict) or with
        any actor/prop near its path. Adjacent-lane vehicles can be considered
        interacting when one vehicle has a static obstacle in its lane.
        
        Enhanced with constraint effectiveness analysis: for each constraint in
        the spec, compute minimum distance between the constrained vehicles and
        report which constraints created proximity (EFFECTIVE) vs which didn't
        (INEFFECTIVE). This feedback helps the LLM understand why constraints
        failed to produce interactions.
        """
        ego_picked = scene_data.get('ego_picked', [])
        if not ego_picked:
            return 1.0

        debug = os.getenv("SCENE_VALIDATOR_DEBUG") == "1"
        static_proximity_m = 4.0
        blocked_distance_m = 2.0
        adjacent_distance_m = 4.0
        temporal_window_s = 5.0
        parallel_angle_deg = 15.0

        samples_by_vehicle = self._extract_polyline_samples_with_meta(scene_data)
        vehicle_data: Dict[str, Dict[str, Any]] = {}
        for ego in ego_picked:
            veh_name = ego.get('vehicle', '')
            refined = ego.get('refined', {})
            if not refined:
                continue

            start_idx = refined.get('start_idx_global', 0)
            speed = refined.get('speed_mps', 8.0)
            if not speed or speed <= 0:
                speed = 8.0

            samples = samples_by_vehicle.get(veh_name, [])
            if len(samples) < 2:
                continue

            spawn_idx = max(0, min(int(start_idx), len(samples) - 1))
            spawn_s = samples[spawn_idx]['s']
            headings = [self._heading_angle(samples, idx) for idx in range(len(samples))]

            vehicle_data[veh_name] = {
                'speed': speed,
                'spawn_s': spawn_s,
                'samples': samples,
                'headings': headings,
            }

        if len(vehicle_data) < 1:
            return 1.0

        interactions = {name: False for name in vehicle_data}
        blocked_events: Dict[str, List[Dict[str, Any]]] = {name: [] for name in vehicle_data}

        actors = self._extract_actor_spawns(scene_data)
        for actor in actors:
            placement = actor.get('placement', {}) or {}
            target_vehicle = placement.get('target_vehicle')
            path_points = actor.get('path_points') or [(actor['x'], actor['y'])]

            for veh_name, vdata in vehicle_data.items():
                closest_idx = None
                closest_dist = float('inf')
                closest_point = None

                for px, py in path_points:
                    idx, dist = self._closest_sample_to_point(vdata['samples'], px, py)
                    if idx is None:
                        continue
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = idx
                        closest_point = (px, py)

                if closest_idx is None or closest_dist > static_proximity_m:
                    continue

                interactions[veh_name] = True

                if actor.get('is_static') and (closest_dist <= blocked_distance_m or target_vehicle == veh_name):
                    sample = vdata['samples'][closest_idx]
                    t = (sample['s'] - vdata['spawn_s']) / vdata['speed']
                    if t < 0:
                        continue
                    blocked_events[veh_name].append({
                        'x': closest_point[0] if closest_point else actor['x'],
                        'y': closest_point[1] if closest_point else actor['y'],
                        'time_s': t,
                        'road_id': sample.get('road_id'),
                        'section_id': sample.get('section_id'),
                        'lane_id': sample.get('lane_id'),
                        'heading': vdata['headings'][closest_idx],
                    })

        # Direct vehicle-vehicle interactions (same criteria as temporal validation)
        # Reduced from 10.0m to 8.0m for stricter interaction requirements
        spatial_window_m = 8.0
        merge_distance_m = 2.0
        non_parallel_angle_deg = 25.0

        vehicle_names = list(vehicle_data.keys())
        for i in range(len(vehicle_names)):
            for j in range(i + 1, len(vehicle_names)):
                v1_name, v2_name = vehicle_names[i], vehicle_names[j]
                v1, v2 = vehicle_data[v1_name], vehicle_data[v2_name]
                pair_interaction = False

                for idx1, pt1 in enumerate(v1['samples']):
                    t1 = (pt1['s'] - v1['spawn_s']) / v1['speed']
                    if t1 < 0:
                        continue
                    for idx2, pt2 in enumerate(v2['samples']):
                        t2 = (pt2['s'] - v2['spawn_s']) / v2['speed']
                        if t2 < 0:
                            continue

                        time_diff = abs(t1 - t2)
                        if time_diff > temporal_window_s:
                            continue

                        dist = math.hypot(pt1['x'] - pt2['x'], pt1['y'] - pt2['y'])
                        if dist > spatial_window_m:
                            continue

                        same_lane = (
                            pt1.get('road_id') == pt2.get('road_id')
                            and pt1.get('section_id') == pt2.get('section_id')
                            and pt1.get('lane_id') == pt2.get('lane_id')
                        )
                        if same_lane:
                            pair_interaction = True
                            break

                        adjacent_lane = (
                            pt1.get('road_id') == pt2.get('road_id')
                            and pt1.get('section_id') == pt2.get('section_id')
                            and pt1.get('lane_id') is not None
                            and pt2.get('lane_id') is not None
                            and abs(pt1.get('lane_id') - pt2.get('lane_id')) == 1
                        )
                        if adjacent_lane:
                            pair_interaction = True
                            break

                        if dist <= merge_distance_m:
                            pair_interaction = True
                            break

                        heading1 = v1['headings'][idx1]
                        heading2 = v2['headings'][idx2]
                        if heading1 is None or heading2 is None:
                            continue
                        angle_diff = self._angle_diff_deg(heading1, heading2)
                        if angle_diff >= non_parallel_angle_deg:
                            pair_interaction = True
                            break

                    if pair_interaction:
                        break

                if pair_interaction:
                    interactions[v1_name] = True
                    interactions[v2_name] = True
                    if debug:
                        print(f"[INTERACTION COVERAGE] {v1_name} <-> {v2_name}: INTERACTING")

        # Special check: follow_route_of constraints should have vehicles in same/adjacent lanes
        # Otherwise they're "phantom followers" with no actual interaction
        constraints = scene_data.get('constraints', [])
        for constraint in constraints:
            if isinstance(constraint, dict):
                ctype = constraint.get('type', '')
                if ctype == 'follow_route_of':
                    a_name = constraint.get('a')
                    b_name = constraint.get('b')
                    if a_name in vehicle_data and b_name in vehicle_data:
                        # Check if they ever get close enough to actually follow
                        a_samples = vehicle_data[a_name]['samples']
                        b_samples = vehicle_data[b_name]['samples']
                        
                        min_following_dist = float('inf')
                        for sample_a in a_samples:
                            for sample_b in b_samples:
                                dist = math.hypot(sample_a['x'] - sample_b['x'], sample_a['y'] - sample_b['y'])
                                if dist < min_following_dist:
                                    min_following_dist = dist
                        
                        # follow_route_of requires vehicles to be within 5m at some point
                        if min_following_dist > 5.0:
                            if debug:
                                print(f"[INTERACTION COVERAGE] {a_name} follow_route_of {b_name}: PHANTOM FOLLOWER (min_dist={min_following_dist:.1f}m)")
                            # Don't mark as interacting - they're too far apart despite the constraint
                            interactions[a_name] = False
                            interactions[b_name] = interactions.get(b_name, False)  # Don't unmark b if it has other interactions

        # Implied interaction: adjacent-lane vehicle near a static obstacle in another lane
        for blocked_vehicle, events in blocked_events.items():
            if not events:
                continue
            for other_name, other_data in vehicle_data.items():
                if other_name == blocked_vehicle:
                    continue
                if interactions.get(other_name):
                    continue

                for event in events:
                    idx, dist = self._closest_sample_to_point(
                        other_data['samples'], event['x'], event['y']
                    )
                    if idx is None or dist > adjacent_distance_m:
                        continue

                    pt = other_data['samples'][idx]
                    if (
                        pt.get('road_id') != event.get('road_id')
                        or pt.get('section_id') != event.get('section_id')
                    ):
                        continue

                    lane_id = pt.get('lane_id')
                    event_lane = event.get('lane_id')
                    if lane_id is None or event_lane is None:
                        continue
                    if abs(lane_id - event_lane) != 1:
                        continue

                    t_other = (pt['s'] - other_data['spawn_s']) / other_data['speed']
                    if t_other < 0:
                        continue
                    if abs(t_other - event.get('time_s', 0.0)) > temporal_window_s:
                        continue

                    heading_other = other_data['headings'][idx]
                    heading_blocked = event.get('heading')
                    if heading_other is None or heading_blocked is None:
                        continue
                    angle_diff = self._angle_diff_deg(heading_other, heading_blocked)
                    if angle_diff > parallel_angle_deg:
                        continue

                    interactions[other_name] = True
                    break

        missing = [name for name, ok in interactions.items() if not ok]
        if missing:
            stage = "step_05_object_placer" if actors else "step_03_path_picker"
            
            # Build detailed message about why each vehicle is isolated
            isolation_details = []
            for veh_name in missing:
                vdata = vehicle_data[veh_name]
                isolation_details.append(f"{veh_name}: no vehicle within 8m and 5s, no actors within 4m")
            
            # CONSTRAINT EFFECTIVENESS ANALYSIS
            # For each constraint in the spec, compute minimum distance between 
            # constrained vehicles and label as EFFECTIVE or INEFFECTIVE
            constraint_analysis = []
            pairwise_min_dist: Dict[Tuple[str, str], float] = {}
            
            # Compute pairwise minimum distances for all vehicle pairs
            for i, v1_name in enumerate(vehicle_names):
                for v2_name in vehicle_names[i + 1:]:
                    v1, v2 = vehicle_data[v1_name], vehicle_data[v2_name]
                    min_dist = float('inf')
                    for pt1 in v1['samples']:
                        for pt2 in v2['samples']:
                            dist = math.hypot(pt1['x'] - pt2['x'], pt1['y'] - pt2['y'])
                            if dist < min_dist:
                                min_dist = dist
                    # Store in both directions for easy lookup
                    pairwise_min_dist[(v1_name, v2_name)] = min_dist
                    pairwise_min_dist[(v2_name, v1_name)] = min_dist
            
            # Analyze constraints from scenario_spec if available
            if scenario_spec:
                constraints = self._get_spec_value(scenario_spec, "vehicle_constraints", []) or []
                for constraint in constraints:
                    if isinstance(constraint, dict):
                        ctype = constraint.get("type", "unknown")
                        a_name = constraint.get("a", "")
                        b_name = constraint.get("b", "")
                    else:
                        ctype = getattr(constraint, "constraint_type", None)
                        if hasattr(ctype, "value"):
                            ctype = ctype.value
                        a_name = getattr(constraint, "vehicle_a", "")
                        b_name = getattr(constraint, "vehicle_b", "")
                    
                    if not a_name or not b_name:
                        continue
                    
                    # Get min distance for this constrained pair
                    pair_key = (a_name, b_name)
                    min_d = pairwise_min_dist.get(pair_key)
                    if min_d is None:
                        # Try with vehicle_data names
                        min_d = pairwise_min_dist.get((a_name, b_name), float('inf'))
                    
                    if min_d < float('inf'):
                        status = "EFFECTIVE" if min_d < spatial_window_m else "INEFFECTIVE"
                        constraint_analysis.append(
                            f"{ctype}({a_name} -> {b_name}): min_dist={min_d:.1f}m [{status}]"
                        )
                    else:
                        constraint_analysis.append(
                            f"{ctype}({a_name} -> {b_name}): NO OVERLAP [INEFFECTIVE]"
                        )
            
            # Build interaction graph summary
            interaction_graph = []
            for (v1_name, v2_name), min_d in pairwise_min_dist.items():
                # Only report each pair once
                if v1_name < v2_name:
                    status = "CLOSE" if min_d < spatial_window_m else "FAR"
                    interaction_graph.append(f"{v1_name} <-> {v2_name}: {min_d:.1f}m [{status}]")
            
            # Build enhanced suggestion with constraint analysis
            suggestion_parts = [
                "Remove isolated vehicles or adjust their paths/timing to create interaction."
            ]
            
            if constraint_analysis:
                ineffective = [c for c in constraint_analysis if "INEFFECTIVE" in c]
                if ineffective:
                    suggestion_parts.append(
                        f"CONSTRAINT ANALYSIS: {len(ineffective)} constraint(s) failed to create proximity: " +
                        "; ".join(ineffective)
                    )
                    suggestion_parts.append(
                        "Try using different constraint types: same_lane_as (queues in same lane), "
                        "left_lane_of, right_lane_of (adjacent lanes), or merges_into_lane_of (merge conflict)."
                    )
            
            if interaction_graph:
                suggestion_parts.append(
                    f"INTERACTION GRAPH: {'; '.join(interaction_graph)}"
                )
            
            result.add_issue(
                ValidationIssue.NO_INTERACTION, "error",
                f"Ego vehicles with no interaction potential: {', '.join(missing)}",
                expected="each ego interacts with another ego or actor/obstacle",
                actual=f"{len(missing)} isolated vehicle(s): {'; '.join(isolation_details)}",
                suggestion=" | ".join(suggestion_parts),
                pipeline_stage=stage,
            )
            if debug:
                print(f"[INTERACTION COVERAGE] missing interactions: {missing}")
                for detail in isolation_details:
                    print(f"  {detail}")
                if constraint_analysis:
                    print(f"[INTERACTION COVERAGE] Constraint effectiveness:")
                    for ca in constraint_analysis:
                        print(f"  {ca}")
            return 0.0

        return 1.0
    
    def _validate_constraint_satisfaction(
        self,
        scene_dir: Path,
        expected: Dict[str, Any],
        result: 'SceneValidationResult',
    ) -> float:
        """
        Validate constraint satisfaction by reading picked_paths_detailed.json.
        This checks if the paths actually satisfy the spatial constraints.
        """
        picked_paths_path = scene_dir / "picked_paths_detailed.json"
        refined_paths_path = scene_dir / "picked_paths_refined.json"
        
        # Try refined first, then picked
        paths_file = refined_paths_path if refined_paths_path.exists() else picked_paths_path
        
        if not paths_file.exists():
            # Can't validate without picked paths
            return 0.8  # Give benefit of the doubt
        
        try:
            with open(paths_file, 'r') as f:
                paths_data = json.load(f)
        except Exception:
            return 0.8
        
        picked = paths_data.get('picked', paths_data.get('ego_picked', []))
        if len(picked) < 2:
            return 0.9  # Single vehicle, no constraints to validate
        
        # Extract entry/exit info for constraint checking
        entries = {}
        exits = {}
        roads = {}
        lanes = {}
        
        for p in picked:
            name = p.get('vehicle', p.get('name', 'unknown'))
            sig = p.get('signature', {})
            
            entry = sig.get('entry', {})
            exit_info = sig.get('exit', {})
            
            entries[name] = {
                'cardinal': entry.get('cardinal4', 'unknown'),
                'road_id': entry.get('road_id'),
                'lane_id': entry.get('lane_id'),
            }
            exits[name] = {
                'cardinal': exit_info.get('cardinal4', 'unknown'),
                'road_id': exit_info.get('road_id'),
                'lane_id': exit_info.get('lane_id'),
            }
            roads[name] = sig.get('roads', [])
            lanes[name] = sig.get('lanes', [])
        
        score = 1.0
        vehicle_names = list(entries.keys())
        
        # Check expected relationships
        for rel in expected.get('relationships', []):
            if rel == 'opposite_approach':
                # At least two vehicles should have opposite entry cardinals
                cardinals = [
                    entries[v]['cardinal'] for v in vehicle_names
                    if entries[v]['cardinal'] and entries[v]['cardinal'] != 'unknown'
                ]
                if len(cardinals) < 2:
                    continue
                opposite_map = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
                has_opposite = any(
                    opposite_map.get(cardinals[i]) == cardinals[j]
                    for i in range(len(cardinals))
                    for j in range(i+1, len(cardinals))
                )
                if not has_opposite:
                    score -= 0.2
                    result.add_issue(
                        ValidationIssue.CONSTRAINT_NOT_SATISFIED, "warning",
                        "opposite_approach constraint not satisfied",
                        expected="vehicles from opposite directions (N/S or E/W)",
                        actual=f"entry cardinals: {cardinals}",
                        pipeline_stage="step_03_path_picker"
                    )
            
            elif rel == 'perpendicular':
                # At least two vehicles should have perpendicular entry cardinals
                cardinals = [
                    entries[v]['cardinal'] for v in vehicle_names
                    if entries[v]['cardinal'] and entries[v]['cardinal'] != 'unknown'
                ]
                if len(cardinals) < 2:
                    continue
                perp_pairs = {('N', 'E'), ('N', 'W'), ('S', 'E'), ('S', 'W'),
                              ('E', 'N'), ('E', 'S'), ('W', 'N'), ('W', 'S')}
                has_perp = any(
                    (cardinals[i], cardinals[j]) in perp_pairs
                    for i in range(len(cardinals))
                    for j in range(i+1, len(cardinals))
                )
                if not has_perp:
                    score -= 0.2
                    result.add_issue(
                        ValidationIssue.CONSTRAINT_NOT_SATISFIED, "warning",
                        "perpendicular constraint not satisfied",
                        expected="vehicles from perpendicular directions",
                        actual=f"entry cardinals: {cardinals}",
                        pipeline_stage="step_03_path_picker"
                    )
            
            elif rel in ('same_approach', 'adjacent_lane'):
                # Vehicles should share some road segments
                all_roads = [set(roads[v]) for v in vehicle_names]
                if len(all_roads) >= 2:
                    shared = all_roads[0].intersection(*all_roads[1:])
                    if not shared:
                        score -= 0.15
                        result.add_issue(
                            ValidationIssue.CONSTRAINT_NOT_SATISFIED, "info",
                            f"{rel} constraint: no shared road segments found",
                            pipeline_stage="step_03_path_picker"
                        )
            
            elif rel == 'merging':
                # Check if vehicles converge (entries different, exits share road)
                entry_roads = [entries[v]['road_id'] for v in vehicle_names if entries[v]['road_id'] is not None]
                exit_roads = [exits[v]['road_id'] for v in vehicle_names if exits[v]['road_id'] is not None]
                if len(entry_roads) < 2 or len(exit_roads) < 1:
                    continue
                
                # Different entry roads, same exit road suggests merge
                if len(set(entry_roads)) > 1 and len(set(exit_roads)) == 1:
                    pass  # Good!
                else:
                    score -= 0.1
                    result.add_issue(
                        ValidationIssue.CONSTRAINT_NOT_SATISFIED, "info",
                        "merging constraint: paths don't converge as expected",
                        actual=f"entry roads: {entry_roads}, exit roads: {exit_roads}",
                        pipeline_stage="step_03_path_picker"
                    )
        
        return max(0, score)

    def _extract_vehicle_cardinals(self, text: str) -> Dict[str, str]:
        text_l = text.lower()
        patterns = [
            re.compile(r"vehicle\s*(\d+)[^.]*?\bapproach(?:es|ing)?\s+from\s+(north|south|east|west)\b"),
            re.compile(r"vehicle\s*(\d+)[^.]*?\bcoming\s+from\s+(north|south|east|west)\b"),
            re.compile(r"vehicle\s*(\d+)[^.]*?\bfrom\s+(north|south|east|west)\b"),
        ]
        found: Dict[str, str] = {}
        for pattern in patterns:
            for match in pattern.finditer(text_l):
                vehicle = f"Vehicle {match.group(1)}"
                if vehicle not in found:
                    found[vehicle] = match.group(2)
        return found

    def _infer_expected_features(
        self,
        scenario_text: str,
        category: Optional[str],
        scenario_spec: Optional[Any] = None,
    ) -> Dict[str, Any]:
        text_l = scenario_text.lower()
        if scenario_spec:
            topo_raw = self._get_spec_value(scenario_spec, "topology")
            if isinstance(topo_raw, TopologyType):
                topology = topo_raw
            else:
                try:
                    topology = TopologyType(str(topo_raw)) if topo_raw else None
                except ValueError:
                    topology = None
            flags = {
                "required_topology": topology,
                "needs_oncoming": bool(self._get_spec_value(scenario_spec, "needs_oncoming", False)),
                "needs_on_ramp": bool(self._get_spec_value(scenario_spec, "needs_on_ramp", False)),
                "needs_merge": bool(self._get_spec_value(scenario_spec, "needs_merge", False)),
                "needs_multi_lane": bool(self._get_spec_value(scenario_spec, "needs_multi_lane", False)),
                "needs_lane_change": False,
                "needs_lane_drop": False,
            }
        else:
            flags = {
                "required_topology": None,
                "needs_oncoming": False,
                "needs_on_ramp": bool(re.search(r"\bon[-\s]?ramp\b", text_l)),
                "needs_merge": bool(re.search(r"\bmerge\b|\bmerges\b|\bmerging\b", text_l)),
                "needs_multi_lane": bool(re.search(
                    r"(left\s+lane|right\s+lane|adjacent\s+lane|multi[-\s]?lane|two[-\s]?lane)",
                    text_l,
                )),
                "needs_lane_change": bool(re.search(r"(lane\s+change|changes?\s+lane|weav)", text_l)),
                "needs_lane_drop": bool(re.search(r"(lane\s+drop|zipper)", text_l)),
            }
        if category:
            cat = CATEGORY_DEFINITIONS.get(category)
            if cat:
                # Category definition is authoritative - override LLM spec flags
                flags["required_topology"] = cat.map.topology
                flags["needs_oncoming"] = cat.map.needs_oncoming
                flags["needs_on_ramp"] = cat.map.needs_on_ramp
                flags["needs_merge"] = cat.map.needs_merge
                flags["needs_multi_lane"] = cat.map.needs_multi_lane
                if "weaving" in cat.name.lower():
                    flags["needs_lane_change"] = True
                if "lane drop" in cat.name.lower():
                    flags["needs_lane_drop"] = True
        if flags["needs_on_ramp"]:
            flags["needs_merge"] = True
        return flags

    def _analyze_scene_geometry(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        entry_roads: List[Optional[int]] = []
        exit_roads: List[Optional[int]] = []
        entry_lanes: List[Optional[int]] = []
        exit_lanes: List[Optional[int]] = []
        entry_dirs: List[str] = []
        exit_dirs: List[str] = []
        turns: List[str] = []
        lane_changes: List[bool] = []

        for ego in scene_data.get("ego_picked", []):
            sig = ego.get("signature", {})
            entry = sig.get("entry", {})
            exit_info = sig.get("exit", {})

            entry_roads.append(entry.get("road_id"))
            exit_roads.append(exit_info.get("road_id"))
            entry_lanes.append(entry.get("lane_id"))
            exit_lanes.append(exit_info.get("lane_id"))
            entry_dirs.append(str(entry.get("cardinal4", "unknown")))
            exit_dirs.append(str(exit_info.get("cardinal4", "unknown")))
            turns.append(str(sig.get("entry_to_exit_turn", "unknown")).lower())

            lanes = sig.get("lanes", [])
            lane_changes.append(len(set(lanes)) > 1 if lanes else False)

        unique_entry_roads = {r for r in entry_roads if r is not None}
        unique_exit_roads = {r for r in exit_roads if r is not None}
        unique_entry_lanes = {l for l in entry_lanes if l is not None}
        unique_exit_lanes = {l for l in exit_lanes if l is not None}

        has_merge_onto_same_road = len(unique_entry_roads) > 1 and len(unique_exit_roads) == 1
        has_lane_change = any(lane_changes)
        has_multi_lane = len(unique_entry_lanes) > 1 or len(unique_exit_lanes) > 1 or has_lane_change
        has_turns = any(t in ("left", "right", "uturn") for t in turns)
        lane_drop_like = (
            len(unique_entry_roads) == 1
            and len(unique_exit_roads) == 1
            and len(unique_entry_lanes) >= 2
            and len(unique_exit_lanes) < len(unique_entry_lanes)
        )
        
        # Check for on-ramp usage: vehicles from different entry roads merging to same exit
        has_on_ramp_usage = False
        if len(unique_entry_roads) >= 2 and len(unique_exit_roads) == 1:
            has_on_ramp_usage = True

        return {
            "entry_roads": entry_roads,
            "exit_roads": exit_roads,
            "entry_lanes": entry_lanes,
            "exit_lanes": exit_lanes,
            "entry_dirs": entry_dirs,
            "exit_dirs": exit_dirs,
            "turns": turns,
            "distinct_entry_roads": len(unique_entry_roads),
            "distinct_exit_roads": len(unique_exit_roads),
            "distinct_entry_lanes": len(unique_entry_lanes),
            "distinct_exit_lanes": len(unique_exit_lanes),
            "has_merge_onto_same_road": has_merge_onto_same_road,
            "has_multi_lane": has_multi_lane,
            "has_lane_change": has_lane_change,
            "has_turns": has_turns,
            "lane_drop_like": lane_drop_like,
            "has_on_ramp_usage": has_on_ramp_usage,
        }

    # ------------------------------------------------------------------ #
    # Required relation checks (pair-agnostic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cardinal_relation(entry_a: str, entry_b: str) -> str:
        a = (entry_a or "unknown").lower()
        b = (entry_b or "unknown").lower()
        if a == b and a != "unknown":
            return "same"
        opposite_pairs = {("north", "south"), ("south", "north"), ("east", "west"), ("west", "east")}
        if (a, b) in opposite_pairs:
            return "opposite"
        perp_pairs = {
            ("north", "east"), ("north", "west"),
            ("south", "east"), ("south", "west"),
            ("east", "north"), ("west", "north"),
            ("east", "south"), ("west", "south"),
        }
        if (a, b) in perp_pairs:
            return "perpendicular"
        return "any"

    @staticmethod
    def _lane_relation(lane_a: Optional[int], lane_b: Optional[int]) -> str:
        if lane_a is None or lane_b is None:
            return "any"
        if lane_a == lane_b:
            return "same_lane"
        if abs(lane_a - lane_b) == 1:
            return "adjacent_lane"
        return "any"

    def _relation_satisfied_oriented(self, rel: RequiredRelation, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # Maneuvers
        if rel.first_maneuver != EgoManeuver.UNKNOWN and str(a["maneuver"]).lower() != rel.first_maneuver.value:
            return False
        if rel.second_maneuver != EgoManeuver.UNKNOWN and str(b["maneuver"]).lower() != rel.second_maneuver.value:
            return False

        # Entry relation
        entry_rel = self._cardinal_relation(a["entry_dir"], b["entry_dir"])
        if rel.entry_relation != "any" and entry_rel != rel.entry_relation:
            return False

        # Exit relation
        if rel.exit_relation == "same_exit":
            if a["exit_road"] is None or b["exit_road"] is None or a["exit_road"] != b["exit_road"]:
                return False
        elif rel.exit_relation == "different_exit":
            if a["exit_road"] is None or b["exit_road"] is None or a["exit_road"] == b["exit_road"]:
                return False

        # Entry lane relation
        entry_lane_rel = self._lane_relation(a["entry_lane"], b["entry_lane"])
        if rel.entry_lane_relation != "any" and entry_lane_rel != rel.entry_lane_relation:
            return False

        # Exit lane relation
        exit_lane_rel = self._lane_relation(a["exit_lane"], b["exit_lane"])
        if rel.exit_lane_relation == "merge_into":
            if a["exit_lane"] is None or b["exit_lane"] is None or a["exit_lane"] != b["exit_lane"]:
                return False
        elif rel.exit_lane_relation != "any" and exit_lane_rel != rel.exit_lane_relation:
            return False

        return True

    def _check_required_relations(self, scene_data: Dict[str, Any], relations: List[RequiredRelation]) -> Tuple[bool, List[str]]:
        """Return (passed, missing_descriptions)."""
        egos = []
        for ego in scene_data.get("ego_picked", []):
            sig = ego.get("signature", {})
            egos.append({
                "name": ego.get("vehicle", ""),
                "maneuver": sig.get("entry_to_exit_turn", "unknown"),
                "entry_dir": sig.get("entry", {}).get("cardinal4", "unknown"),
                "exit_dir": sig.get("exit", {}).get("cardinal4", "unknown"),
                "entry_road": sig.get("entry", {}).get("road_id"),
                "exit_road": sig.get("exit", {}).get("road_id"),
                "entry_lane": sig.get("entry", {}).get("lane_id"),
                "exit_lane": sig.get("exit", {}).get("lane_id"),
            })

        if not relations or len(egos) < 2:
            return True, []

        missing: List[str] = []
        for rel in relations:
            satisfied = False
            n = len(egos)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = egos[i], egos[j]
                    if self._relation_satisfied_oriented(rel, a, b) or self._relation_satisfied_oriented(rel, b, a):
                        satisfied = True
                        break
                if satisfied:
                    break
            if not satisfied:
                missing.append(
                    f"entry={rel.entry_relation}, maneuvers=({rel.first_maneuver.value},{rel.second_maneuver.value}), "
                    f"exit={rel.exit_relation}, entry_lane={rel.entry_lane_relation}, exit_lane={rel.exit_lane_relation}"
                )
        return len(missing) == 0, missing

    def _get_spec_value(self, spec: Any, key: str, default: Any = None) -> Any:
        if isinstance(spec, dict):
            return spec.get(key, default)
        return getattr(spec, key, default)

    def extract_expected_from_spec(self, scenario_spec: Any) -> Dict[str, Any]:
        """Extract expected scene elements directly from a scenario spec."""
        result = {
            'vehicles': [],
            'maneuvers': {},
            'relationships': [],
            'actors': [],
        }

        if not scenario_spec:
            return result

        vehicles = self._get_spec_value(scenario_spec, "ego_vehicles", []) or []
        for vehicle in vehicles:
            if isinstance(vehicle, dict):
                vehicle_id = vehicle.get("vehicle_id")
                maneuver = vehicle.get("maneuver")
            else:
                vehicle_id = getattr(vehicle, "vehicle_id", None)
                maneuver = getattr(vehicle, "maneuver", None)
                if hasattr(maneuver, "value"):
                    maneuver = maneuver.value
            if not vehicle_id:
                continue
            result["vehicles"].append(vehicle_id)
            if maneuver:
                result["maneuvers"][vehicle_id] = str(maneuver).lower()

        constraints = self._get_spec_value(scenario_spec, "vehicle_constraints", []) or []
        rels = set()
        relationship_map = {
            "opposite_approach_of": "opposite_approach",
            "perpendicular_left_of": "perpendicular",
            "perpendicular_right_of": "perpendicular",
            "same_approach_as": "same_approach",
            "same_road_as": "same_approach",
            "follow_route_of": "same_approach",
            "left_lane_of": "adjacent_lane",
            "right_lane_of": "adjacent_lane",
            "merges_into_lane_of": "merging",
        }
        for constraint in constraints:
            if isinstance(constraint, dict):
                ctype = constraint.get("type")
            else:
                ctype = getattr(constraint, "constraint_type", None)
                if hasattr(ctype, "value"):
                    ctype = ctype.value
            if not ctype:
                continue
            rel = relationship_map.get(str(ctype).lower())
            if rel:
                rels.add(rel)
        result["relationships"] = sorted(rels)

        allow_static_props = bool(self._get_spec_value(scenario_spec, "allow_static_props", True))
        actors = self._get_spec_value(scenario_spec, "actors", []) or []
        static_kind_exact = {
            "parked_vehicle",
            "static_prop",
            "construction_cone",
            "construction_barrier",
            "traffic_cone",
        }
        static_kind_tokens = ("static", "parked", "barrier", "cone", "debris", "obstacle", "prop")
        for actor in actors:
            if isinstance(actor, dict):
                kind = actor.get("kind")
                motion = actor.get("motion")
            else:
                kind = getattr(actor, "kind", None)
                motion = getattr(actor, "motion", None)
                if hasattr(kind, "value"):
                    kind = kind.value
                if hasattr(motion, "value"):
                    motion = motion.value
            if not kind:
                continue
            kind_l = str(kind).strip().lower()
            motion_l = str(motion).strip().lower() if motion is not None else ""
            if not allow_static_props:
                is_static_kind = kind_l in static_kind_exact or any(tok in kind_l for tok in static_kind_tokens)
                is_static_motion = motion_l in {"static", "stopped"}
                if is_static_kind or is_static_motion:
                    continue
            actor_entry = {"kind": str(kind)}
            if motion:
                actor_entry["motion"] = str(motion)
            result["actors"].append(actor_entry)

        return result

    def extract_expected_from_text(self, text: str) -> Dict[str, Any]:
        """Extract expected scene elements from natural language description."""
        result = {
            'vehicles': [],
            'maneuvers': {},
            'relationships': [],
            'actors': [],
        }

        vehicle_text = self._vehicle_only_text(text)
        
        # Extract vehicles
        vehicles = set(self.vehicle_pattern.findall(vehicle_text))
        result['vehicles'] = sorted([int(v) for v in vehicles])
        
        # Extract maneuvers
        for maneuver, pattern in self.maneuver_patterns.items():
            matches = pattern.findall(vehicle_text)
            for vehicle_num in matches:
                result['maneuvers'][f"Vehicle {vehicle_num}"] = maneuver
        
        # Extract relationships
        for rel_type, pattern in self.relationship_patterns.items():
            if pattern.search(vehicle_text):
                result['relationships'].append(rel_type)

        cardinals = self._extract_vehicle_cardinals(vehicle_text)
        if cardinals:
            card_set = set(cardinals.values())
            has_opposite = ("north" in card_set and "south" in card_set) or ("east" in card_set and "west" in card_set)
            has_perp = any(
                a in card_set and b in card_set
                for a, b in [("north", "east"), ("north", "west"), ("south", "east"), ("south", "west")]
            )
            if has_opposite and "opposite_approach" not in result["relationships"]:
                result["relationships"].append("opposite_approach")
            if has_perp and "perpendicular" not in result["relationships"]:
                result["relationships"].append("perpendicular")
        
        # Extract actors
        for actor_type, pattern in self.actor_patterns.items():
            if pattern.search(text):
                actor = {'kind': actor_type}
                
                # Try to determine motion
                for motion, motion_pattern in self.motion_patterns.items():
                    if motion_pattern.search(text):
                        actor['motion'] = motion
                        break
                
                # Try to determine lateral position
                for lateral, lateral_pattern in self.lateral_patterns.items():
                    if lateral_pattern.search(text):
                        actor['lateral'] = lateral
                        break
                
                result['actors'].append(actor)
        
        return result
    
    def extract_actual_from_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actual scene elements from scene_objects.json."""
        result = {
            'vehicles': [],
            'maneuvers': {},
            'actors': [],
            'entry_directions': {},
            'exit_directions': {},
            'lane_changes': {},
        }
        
        # Extract ego vehicles
        for ego in scene_data.get('ego_picked', []):
            vehicle_name = ego.get('vehicle', '')
            result['vehicles'].append(vehicle_name)
            
            sig = ego.get('signature', {})
            maneuver = sig.get('entry_to_exit_turn', 'unknown')
            result['maneuvers'][vehicle_name] = maneuver
            lanes = sig.get('lanes')
            if isinstance(lanes, list) and lanes:
                result['lane_changes'][vehicle_name] = len(set(lanes)) > 1
            else:
                result['lane_changes'][vehicle_name] = None
            
            entry = sig.get('entry', {})
            exit_info = sig.get('exit', {})
            result['entry_directions'][vehicle_name] = entry.get('cardinal4', 'unknown')
            result['exit_directions'][vehicle_name] = exit_info.get('cardinal4', 'unknown')
        
        # Extract actors
        for actor in scene_data.get('actors', []):
            result['actors'].append({
                'kind': actor.get('category', actor.get('kind', 'unknown')),
                'motion': actor.get('motion', {}).get('type', 'unknown') if isinstance(actor.get('motion'), dict) else 'unknown',
                'lateral': actor.get('placement', {}).get('lateral_relation', 'unknown') if isinstance(actor.get('placement'), dict) else 'unknown',
                'semantic': actor.get('semantic', ''),
            })
        
        return result
    
    def validate_scene(
        self,
        scene_path: str,
        scenario_text: str,
        category: Optional[str] = None,
        scenario_spec: Optional[Any] = None,
    ) -> SceneValidationResult:
        """
        Simplified validation: only hard-fail parse errors, empty scenes, missing required geometry/flags,
        or fully isolated vehicles. All other issues are treated as warnings with a lower score threshold.
        """
        result = SceneValidationResult(is_valid=True, score=1.0)

        # Load scene
        try:
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
        except Exception as e:
            result.is_valid = False
            result.score = 0.0
            result.add_issue(
                ValidationIssue.PARSE_ERROR,
                "error",
                f"Failed to load scene_objects.json: {e}",
                pipeline_stage="step_05_object_placer",
            )
            return result

        if not category and scenario_spec:
            category = self._get_spec_value(scenario_spec, "category")

        # Extract expected elements for reference
        if scenario_spec:
            expected = self.extract_expected_from_spec(scenario_spec)
        else:
            expected = self.extract_expected_from_text(scenario_text)
        expected_flags = self._infer_expected_features(
            scenario_text,
            category,
            scenario_spec=scenario_spec,
        )
        result.expected_vehicles = len(expected['vehicles'])
        result.expected_maneuvers = expected['maneuvers']
        result.expected_actors = expected['actors']
        result.expected_relationships = expected['relationships']

        # Extract actual elements
        actual = self.extract_actual_from_scene(scene_data)
        result.actual_vehicles = len(actual['vehicles'])
        result.actual_maneuvers = actual['maneuvers']
        result.actual_actors = actual['actors']

        if result.actual_vehicles == 0:
            result.is_valid = False
            result.score = 0.0
            result.add_issue(
                ValidationIssue.SCENE_EMPTY,
                "error",
                "Scene has no ego vehicles",
                expected=f"{result.expected_vehicles} vehicles" if result.expected_vehicles else "1+ vehicles",
                actual="0 vehicles",
                pipeline_stage="step_03_path_picker",
            )
            return result

        # Hard rule: Highway On-Ramp Merge must not place extra props/actors
        if category == "Highway On-Ramp Merge" and result.actual_actors:
            result.is_valid = False
            result.score = 0.0
            result.add_issue(
                ValidationIssue.WRONG_ACTOR_TYPE,
                "error",
                "Actors/props are not allowed for Highway On-Ramp Merge scenarios",
                actual=str(result.actual_actors),
                pipeline_stage="step_05_object_placer",
            )
            return result

        geometry = self._analyze_scene_geometry(scene_data)
        hard_error = False

        def add_error(issue_type, message, **kwargs):
            nonlocal hard_error
            hard_error = True
            result.add_issue(
                issue_type,
                "error",
                message,
                **kwargs,
            )

        # Required geometry/flag checks
        if expected_flags.get("needs_multi_lane") and not geometry["has_multi_lane"]:
            add_error(
                ValidationIssue.MISSING_MULTI_LANE,
                "Multi-lane geometry not present in picked paths",
                expected="at least 2 lanes",
                actual=f"entry lanes: {geometry['distinct_entry_lanes']}, exit lanes: {geometry['distinct_exit_lanes']}",
                pipeline_stage="step_01_crop",
            )
        merge_ok = geometry["has_merge_onto_same_road"] or geometry["lane_drop_like"] or geometry["has_lane_change"]
        if expected_flags.get("needs_merge") and not merge_ok:
            add_error(
                ValidationIssue.MISSING_MERGE_FEATURE,
                "Merge geometry not found",
                expected="multiple entry roads converging or a lane drop",
                actual=f"entry roads: {geometry['distinct_entry_roads']}, exit roads: {geometry['distinct_exit_roads']}",
                pipeline_stage="step_01_crop",
            )
        if expected_flags.get("needs_on_ramp"):
            if geometry["distinct_entry_roads"] < 2 or not geometry.get("has_on_ramp_usage", False):
                add_error(
                    ValidationIssue.MISSING_MERGE_FEATURE,
                    "On-ramp geometry or usage missing",
                    expected="at least one ramp entry plus mainline with a merge",
                    actual=f"entry roads: {geometry['distinct_entry_roads']}",
                    pipeline_stage="step_03_path_picker",
                )

        if category == "Pedestrian Crosswalk":
            if not self._validate_pedestrian_crosswalk_semantics(scene_data, result):
                hard_error = True

        if category == "Roundabout Navigation":
            overlap_ok, overlap_msg = self._roundabout_overlap_ok(scene_data)
            if not overlap_ok:
                add_error(
                    ValidationIssue.NO_INTERACTION,
                    f"Roundabout interaction requirement failed: {overlap_msg}",
                    pipeline_stage="step_03_path_picker",
                )

        # Interaction/coverage check
        interaction_region = self._find_interaction_region(scene_data)
        if result.actual_vehicles > 1 and not interaction_region:
            add_error(
                ValidationIssue.NO_INTERACTION,
                "Vehicles do not converge; all paths appear isolated",
                suggestion="Add active constraints or adjust path picker to force an interaction region",
                pipeline_stage="step_03_path_picker",
            )
        elif interaction_region:
            result.intersection_region = {
                'x': interaction_region['x'],
                'y': interaction_region['y'],
            }
            result.paths_intersect = True

        # Soft checks as warnings
        if result.expected_vehicles and result.actual_vehicles != result.expected_vehicles:
            result.add_issue(
                ValidationIssue.VEHICLE_COUNT_MISMATCH,
                "warning",
                f"Vehicle count mismatch: expected {result.expected_vehicles}, got {result.actual_vehicles}",
                expected=str(result.expected_vehicles),
                actual=str(result.actual_vehicles),
            )

        expected_actor_types = {a['kind'] for a in expected['actors']}
        actual_actor_types = {a.get('kind') for a in actual['actors']}
        for actor_type in expected_actor_types:
            if actor_type not in actual_actor_types:
                result.add_issue(
                    ValidationIssue.MISSING_ACTOR,
                    "warning",
                    f"Expected {actor_type} actor not found in scene",
                    expected=actor_type,
                    actual=str(actual_actor_types) if actual_actor_types else "none",
                    pipeline_stage="step_05_object_placer",
                )

        warning_count = len(result.get_warnings())

        # Score: penalize warnings lightly; hard errors mark invalid
        if hard_error:
            result.score = max(0.0, 0.2 - 0.05 * warning_count)
            result.is_valid = False
        else:
            result.score = max(0.0, 1.0 - 0.05 * warning_count)
            result.is_valid = result.score >= 0.3

        return result


    def get_failing_stage(self, result: SceneValidationResult) -> Optional[str]:
        """
        Determine which pipeline stage likely caused the validation failure.
        Returns the stage that should be re-run.
        """
        if not result.issues:
            return None
        
        # Priority order for re-running stages
        stage_priority = [
            "step_03_path_picker",
            "step_05_object_placer",
            "step_04_path_refiner",
            "step_01_crop",
        ]
        
        # Count issues per stage
        stage_counts = {}
        for issue in result.issues:
            if issue.pipeline_stage:
                stage_counts[issue.pipeline_stage] = stage_counts.get(issue.pipeline_stage, 0) + 1
        
        # Return highest priority stage with issues
        for stage in stage_priority:
            if stage in stage_counts:
                return stage
        
        return None
    
    def generate_validation_report(
        self,
        validation: SceneValidationResult,
        scene_path: str,
        scenario_text: str,
        category: str = "",
        repair_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate comprehensive validation report with spatial analysis.
        
        This report includes:
        - All validation issues (not just top 5)
        - Actor coordinates and paths
        - Spatial conflict analysis
        - Repair attempt history
        - Pipeline metadata
        
        Args:
            validation: The validation result
            scene_path: Path to scene_objects.json
            scenario_text: Natural language scenario description
            category: Scenario category
            repair_history: List of repair attempts and results
            
        Returns:
            Comprehensive report dictionary
        """
        from datetime import datetime
        
        # Load scene data
        scene_data = self._load_scene_data(scene_path)
        
        # Extract actor information with coordinates
        actors = self._extract_actor_coordinates(scene_data)
        
        # Analyze spatial conflicts
        spatial_analysis = self._analyze_spatial_conflicts(actors)
        
        # Build summary
        error_count = len(validation.get_errors())
        warning_count = len(validation.get_warnings())
        failing_stage = self.get_failing_stage(validation)
        
        # Determine primary failure reason
        failure_reason = "Unknown"
        if validation.issues:
            # Prioritize spatial conflicts
            spatial_issues = [i for i in validation.issues if 'spawn' in i.message.lower() or 'path' in i.message.lower() or 'distance' in i.message.lower()]
            if spatial_issues:
                failure_reason = spatial_issues[0].message
            else:
                failure_reason = validation.issues[0].message
        
        # Convert issues to dicts
        issues_list = []
        for issue in validation.issues:
            issue_dict = {
                "severity": issue.severity,
                "category": issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type),
                "message": issue.message,
            }
            if issue.expected:
                issue_dict["expected"] = issue.expected
            if issue.actual:
                issue_dict["actual"] = issue.actual
            if issue.suggestion:
                issue_dict["suggestion"] = issue.suggestion
            if issue.pipeline_stage:
                issue_dict["pipeline_stage"] = issue.pipeline_stage
            issues_list.append(issue_dict)
        
        # Build comprehensive report
        report = {
            "validation_score": round(validation.score, 3),
            "is_valid": validation.is_valid,
            "min_required_score": 0.3,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            
            "scenario": {
                "id": os.path.basename(os.path.dirname(scene_path)),
                "category": category,
                "description": scenario_text[:500] + ("..." if len(scenario_text) > 500 else ""),
            },
            
            "summary": {
                "total_issues": len(validation.issues),
                "error_count": error_count,
                "warning_count": warning_count,
                "failing_stage": failing_stage,
                "failure_reason": failure_reason,
            },
            
            "issues": issues_list,
            "actors": actors,
            "spatial_analysis": spatial_analysis,
            "repair_history": repair_history or [],
            
            "pipeline_metadata": {
                "scene_path": scene_path,
                "town": scene_data.get("town", "Unknown"),
                "crop_region": scene_data.get("crop_region", {}),
                "stages_completed": self._infer_completed_stages(scene_path),
            }
        }
        
        return report
    
    def _load_scene_data(self, scene_path: str) -> Dict:
        """Load scene_objects.json file."""
        try:
            with open(scene_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load scene: {e}"}
    
    def _extract_actor_coordinates(self, scene_data: Dict) -> Dict:
        """Extract spawn locations and paths for all actors."""
        vehicles = []
        pedestrians = []
        static_objects = []
        cyclists = []
        
        # Extract vehicles
        for vehicle in scene_data.get("ego_picked", []):
            path_len = 0.0
            waypoints = []
            
            # Get path from segments
            for seg in vehicle.get("signature", {}).get("segments_detailed", []):
                path_len += seg.get("length_m", 0.0)
                polyline = seg.get("polyline_sample", [])
                if polyline:
                    waypoints.extend([{"x": p["x"], "y": p["y"]} for p in polyline[:5]])  # Sample first 5 points
            
            entry = vehicle.get("signature", {}).get("entry", {}).get("point", {})
            exit_pt = vehicle.get("signature", {}).get("exit", {}).get("point", {})
            
            vehicles.append({
                "id": vehicle.get("vehicle", "Unknown"),
                "spawn": {
                    "x": round(entry.get("x", 0.0), 3),
                    "y": round(entry.get("y", 0.0), 3),
                    "z": 0.0,
                    "yaw_deg": round(vehicle.get("signature", {}).get("entry", {}).get("heading_deg", 0.0), 2),
                },
                "exit": {
                    "x": round(exit_pt.get("x", 0.0), 3),
                    "y": round(exit_pt.get("y", 0.0), 3),
                },
                "path_length_m": round(path_len, 2),
                "num_segments": len(vehicle.get("signature", {}).get("segment_ids", [])),
                "maneuver": vehicle.get("signature", {}).get("entry_to_exit_turn", "unknown"),
                "path_sample": waypoints[:10],  # First 10 waypoints
            })
        
        # Extract non-ego actors
        for actor in scene_data.get("actors_world", []):
            actor_id = actor.get("id", "unknown")
            category = actor.get("category", "unknown").lower()
            
            spawn = actor.get("spawn", {})
            waypoints = actor.get("world_waypoints", [])
            motion = actor.get("motion", {})
            
            actor_info = {
                "id": actor_id,
                "category": category,
                "spawn": {
                    "x": round(spawn.get("x", 0.0), 3),
                    "y": round(spawn.get("y", 0.0), 3),
                    "z": round(spawn.get("z", 0.0), 3),
                    "yaw_deg": round(spawn.get("yaw_deg", 0.0), 2),
                },
                "motion_type": motion.get("type", "unknown"),
                "waypoints": [
                    {"x": round(wp.get("x", 0.0), 3), "y": round(wp.get("y", 0.0), 3)}
                    for wp in waypoints[:10]  # First 10 waypoints
                ],
            }
            
            # Add motion-specific details
            if motion.get("type") == "cross_perpendicular":
                actor_info["cross_direction"] = motion.get("cross_direction", "unknown")
                actor_info["start_lateral"] = actor.get("placement", {}).get("lateral_relation", "unknown")
                
                # Calculate crossing distance
                if len(waypoints) >= 2:
                    x1, y1 = waypoints[0].get("x", 0), waypoints[0].get("y", 0)
                    x2, y2 = waypoints[-1].get("x", 0), waypoints[-1].get("y", 0)
                    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    actor_info["cross_distance_m"] = round(dist, 2)
            
            # Add trigger info if present
            trigger = actor.get("trigger", {})
            if trigger:
                actor_info["trigger"] = {
                    "type": trigger.get("type", "unknown"),
                    "vehicle": trigger.get("vehicle", ""),
                    "distance_m": trigger.get("distance_m", 0.0),
                }
            
            # Categorize actor
            if category == "walker":
                pedestrians.append(actor_info)
            elif category == "cyclist":
                cyclists.append(actor_info)
            else:
                static_objects.append(actor_info)
        
        return {
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "cyclists": cyclists,
            "static_objects": static_objects,
        }
    
    def _analyze_spatial_conflicts(self, actors: Dict) -> Dict:
        """
        Calculate distances between all actor pairs and identify conflicts.
        
        Detects:
        - Pedestrians spawning on vehicle paths
        - Vehicle-vehicle collision risks
        - Actors too close to spawn
        """
        conflicts = []
        distances = []
        
        vehicles = actors.get("vehicles", [])
        pedestrians = actors.get("pedestrians", [])
        
        # Check pedestrian-vehicle distances
        for ped in pedestrians:
            ped_spawn = ped.get("spawn", {})
            ped_pos = (ped_spawn.get("x", 0), ped_spawn.get("y", 0))
            
            for veh in vehicles:
                # Get vehicle path
                veh_path = veh.get("path_sample", [])
                if not veh_path:
                    continue
                
                # Calculate minimum distance from pedestrian to vehicle path
                min_dist = float('inf')
                closest_point = None
                
                for wp in veh_path:
                    vx, vy = wp.get("x", 0), wp.get("y", 0)
                    dist = math.sqrt((ped_pos[0] - vx)**2 + (ped_pos[1] - vy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = {"x": vx, "y": vy}
                
                # Record distance
                distance_entry = {
                    "actor1": ped["id"],
                    "actor1_type": "pedestrian",
                    "actor2": veh["id"],
                    "actor2_type": "vehicle",
                    "min_distance_m": round(min_dist, 2),
                    "pedestrian_spawn": ped_spawn,
                    "closest_point_on_vehicle_path": closest_point,
                }
                distances.append(distance_entry)
                
                # Check for conflict (pedestrian too close to vehicle path)
                SIDEWALK_OFFSET_M = 5.25  # From constants.py
                if min_dist < SIDEWALK_OFFSET_M:
                    conflicts.append({
                        "type": "pedestrian_on_vehicle_path",
                        "actors": [ped["id"], veh["id"]],
                        "severity": "critical",
                        "description": f"Pedestrian {ped['id']} spawned {min_dist:.2f}m from {veh['id']}'s path (should be ≥{SIDEWALK_OFFSET_M}m)",
                        "distance_m": round(min_dist, 2),
                        "threshold_m": SIDEWALK_OFFSET_M,
                        "details": distance_entry,
                    })
        
        # Check vehicle-vehicle spawn distances
        for i, veh1 in enumerate(vehicles):
            veh1_spawn = veh1.get("spawn", {})
            veh1_pos = (veh1_spawn.get("x", 0), veh1_spawn.get("y", 0))
            
            for veh2 in vehicles[i+1:]:
                veh2_spawn = veh2.get("spawn", {})
                veh2_pos = (veh2_spawn.get("x", 0), veh2_spawn.get("y", 0))
                
                spawn_dist = math.sqrt((veh1_pos[0] - veh2_pos[0])**2 + (veh1_pos[1] - veh2_pos[1])**2)
                
                distances.append({
                    "actor1": veh1["id"],
                    "actor1_type": "vehicle",
                    "actor2": veh2["id"],
                    "actor2_type": "vehicle",
                    "spawn_distance_m": round(spawn_dist, 2),
                })
                
                # Check for spawn collision
                MIN_SPAWN_DISTANCE = 3.0  # meters
                if spawn_dist < MIN_SPAWN_DISTANCE:
                    conflicts.append({
                        "type": "vehicle_spawn_collision",
                        "actors": [veh1["id"], veh2["id"]],
                        "severity": "critical",
                        "description": f"Vehicles {veh1['id']} and {veh2['id']} spawn too close ({spawn_dist:.2f}m < {MIN_SPAWN_DISTANCE}m)",
                        "distance_m": round(spawn_dist, 2),
                        "threshold_m": MIN_SPAWN_DISTANCE,
                    })
        
        return {
            "actor_pair_distances": distances,
            "conflicts": conflicts,
            "summary": {
                "total_pairs_analyzed": len(distances),
                "conflicts_detected": len(conflicts),
                "critical_conflicts": len([c for c in conflicts if c.get("severity") == "critical"]),
            }
        }
    
    def _infer_completed_stages(self, scene_path: str) -> List[str]:
        """Infer which pipeline stages completed based on output files."""
        scene_dir = Path(scene_path).parent
        stages = []
        
        if (scene_dir / "scenario_spec.json").exists():
            stages.append("step_00_schema_generation")
        if (scene_dir / "legal_paths_detailed.json").exists():
            stages.append("step_02_legal_paths")
        if (scene_dir / "picked_paths_detailed.json").exists():
            stages.append("step_03_path_picker")
        if (scene_dir / "picked_paths_refined.json").exists():
            stages.append("step_04_path_refiner")
        if (scene_dir / "scene_objects.json").exists():
            stages.append("step_05_object_placer")
        
        return stages
    
    def save_validation_report(
        self,
        report: Dict,
        output_path: str,
    ) -> None:
        """Save validation report to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"[Validation Report] Saved to {output_path}")
        except Exception as e:
            print(f"[Validation Report] ERROR: Failed to save report: {e}")


def validate_scene_file(
    scene_path: str,
    scenario_text: str,
    scenario_spec: Optional[Any] = None,
) -> SceneValidationResult:
    """Convenience function to validate a scene file."""
    validator = SceneValidator()
    return validator.validate_scene(scene_path, scenario_text, scenario_spec=scenario_spec)

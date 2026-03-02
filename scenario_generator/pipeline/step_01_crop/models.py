from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CropKey:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_str(self) -> str:
        return f"crop_{self.xmin:.1f}_{self.xmax:.1f}_{self.ymin:.1f}_{self.ymax:.1f}"


@dataclass
class CropFeatures:
    town: str
    crop: CropKey
    center_xy: Tuple[float, float]

    turns: List[str]
    entry_dirs: List[str]
    exit_dirs: List[str]
    dirs: List[str]
    has_oncoming_pair: bool
    is_t_junction: bool
    is_four_way: bool
    is_highway: bool
    has_merge_onto_same_road: bool
    has_on_ramp: bool
    lane_count_est: int
    has_multi_lane: bool

    maneuver_stats: Dict[str, Dict[str, float]]  # man -> {"count":..., "max_entry_dist":..., "max_exit_dist":...}

    n_paths: int
    junction_count: int
    area: float

    ramp_main_lanes: int = 0
    ramp_entry_min_lanes: int = 0
    ramp_entry_max_lanes: int = 0
    ramp_candidate_paths: int = 0

    _segments_full: Optional[List[Any]] = None
    _junction_centers: Optional[List[np.ndarray]] = None


@dataclass
class Scenario:
    sid: str
    text: str
    source: str = ""


@dataclass
class GeometrySpec:
    topology: str  # "intersection"|"t_junction"|"corridor"|"unknown"
    degree: int    # 3 or 4, 0 if unknown

    required_maneuvers: Dict[str, int]
    needs_oncoming: bool
    needs_merge_onto_same_road: bool
    needs_on_ramp: bool
    needs_multi_lane: bool
    min_lane_count: int

    min_entry_runup_m: float
    min_exit_runout_m: float

    preferred_entry_cardinals: List[str]
    avoid_extra_intersections: bool

    confidence: float
    notes: str


@dataclass
class AssignmentResult:
    mapping: Dict[str, Dict[str, List[str]]]
    detailed: Dict[str, Any]

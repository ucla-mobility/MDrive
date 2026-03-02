from .models import CropFeatures, GeometrySpec


def _maneuver_needed_count(spec: GeometrySpec, man: str) -> int:
    v = spec.required_maneuvers.get(man, 0)
    try:
        return int(v)
    except Exception:
        return 0


def crop_satisfies_spec(spec: GeometrySpec, crop: CropFeatures) -> bool:
    is_roundabout = spec.topology == "roundabout"

    if spec.topology == "highway":
        if not crop.is_highway:
            return False
        if crop.lane_count_est < 3:  # Highways must have 3+ lanes
            return False
    elif is_roundabout:
        # Roundabout uses a fixed Town03 crop and maneuver labels in this step can
        # classify long roundabout traversals as "uturn" instead of "left".
        # Keep topology/flag checks, but defer maneuver feasibility to legal_paths.
        pass
    elif spec.topology == "two_lane_corridor":
        # Enforce corridor-like geometry: no intersections/highways and at most 2 lanes.
        if crop.is_four_way or crop.is_t_junction or crop.is_highway:
            return False
        if crop.lane_count_est > 2:
            return False
    elif spec.topology == "t_junction":
        if not crop.is_t_junction:
            return False
        # Enforce true T-junction semantics: reject cross/four-way layouts.
        if crop.is_four_way or len(crop.dirs) > 3:
            return False
        if spec.degree == 3 and len(crop.dirs) < 3:
            return False
    elif spec.topology == "intersection":
        if len(crop.dirs) < 3:
            return False
        if spec.degree == 4 and not crop.is_four_way:
            return False
        if spec.degree == 3 and not crop.is_t_junction:
            return False

    for man in ["straight", "left", "right"]:
        need = _maneuver_needed_count(spec, man)
        if need > 0:
            if is_roundabout:
                continue
            if spec.needs_on_ramp and man == "straight":
                continue
            if crop.maneuver_stats.get(man, {}).get("count", 0.0) < 1.0:
                return False

    if spec.needs_oncoming and not crop.has_oncoming_pair:
        return False

    if spec.needs_merge_onto_same_road and not crop.has_merge_onto_same_road:
        return False

    if spec.needs_on_ramp and not crop.has_on_ramp:
        return False
    if spec.needs_on_ramp and spec.topology == "highway":
        if crop.ramp_entry_max_lanes != 1:
            return False

    if spec.needs_multi_lane:
        if crop.lane_count_est < max(2, spec.min_lane_count):
            return False

    if spec.preferred_entry_cardinals:
        if not any(d in crop.entry_dirs for d in spec.preferred_entry_cardinals):
            return False

    for man in ["straight", "left", "right"]:
        need = _maneuver_needed_count(spec, man)
        if need > 0:
            if is_roundabout:
                continue
            if spec.needs_on_ramp and man == "straight":
                continue
            st = crop.maneuver_stats.get(man, {})
            if float(st.get("max_entry_dist", 0.0)) < float(spec.min_entry_runup_m):
                return False
            if float(st.get("max_exit_dist", 0.0)) < float(spec.min_exit_runout_m):
                return False

    return True


def crop_base_cost(spec: GeometrySpec, crop: CropFeatures, junction_penalty: float) -> float:
    cost = crop.area
    if spec.avoid_extra_intersections:
        cost += junction_penalty * max(0, crop.junction_count - 1)

    if spec.topology == "highway" and crop.is_highway:
        cost *= 0.95  # Prefer highways when explicitly requested
    if spec.topology == "t_junction" and crop.is_t_junction:
        cost *= 0.97
    if spec.needs_multi_lane and crop.has_multi_lane:
        cost *= 0.98
    if spec.needs_merge_onto_same_road and crop.has_merge_onto_same_road:
        cost *= 0.98
    if spec.needs_on_ramp and crop.has_on_ramp:
        cost *= 0.98
        if crop.ramp_entry_max_lanes == 1 and crop.ramp_main_lanes >= 3:
            cost *= 0.92
        elif crop.ramp_entry_min_lanes == 1:
            cost *= 0.96
        elif crop.ramp_entry_min_lanes == 2:
            cost *= 0.98
    if spec.topology == "intersection" and spec.degree == 0 and crop.is_four_way:
        cost *= 0.98
    return float(cost)

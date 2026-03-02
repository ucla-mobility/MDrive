"""Stage 03: CARLA correspondence and projection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from v2xpnp.pipeline import pipeline_runtime


def build_projection_context(
    *,
    v2_map: pipeline_runtime.VectorMapData,
    lines_xy: Sequence[object],
    bbox: object,
    lane_corr_top_k: int,
    lane_corr_cache_dir: Optional[Path],
    lane_corr_driving_types: Sequence[str],
    verbose: bool = False,
) -> Dict[str, object]:
    return pipeline_runtime.build_carla_projection_context(
        chosen_map=v2_map,
        carla_lines_xy=lines_xy,
        carla_bbox=bbox,
        carla_source_path="",
        carla_name="carla_map_cache",
        lane_corr_top_k=int(lane_corr_top_k),
        lane_corr_cache_dir=lane_corr_cache_dir,
        lane_corr_driving_types=lane_corr_driving_types,
        carla_line_records=None,
        verbose=bool(verbose),
    )


def apply_projection(
    tracks: Dict[str, object],
    context: Dict[str, object],
    *,
    verbose: bool = False,
) -> Dict[str, object]:
    return pipeline_runtime.apply_carla_projection_to_tracks(
        tracks=tracks,
        carla_context=context,
        verbose=bool(verbose),
    )

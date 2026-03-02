"""Stage 02: map selection and V2 trajectory alignment."""

from __future__ import annotations

from typing import Dict, List

from v2xpnp.pipeline import pipeline_runtime

LaneMatcher = pipeline_runtime.LaneMatcher
TrajectoryAligner = pipeline_runtime.TrajectoryAligner
VectorMapData = pipeline_runtime.VectorMapData


def select_best_vector_map(
    ego_trajs: List[List[object]],
    map_data_list: List[VectorMapData],
    tx: float,
    ty: float,
    yaw_deg: float,
) -> VectorMapData:
    return pipeline_runtime.select_best_map(ego_trajs, map_data_list, tx, ty, yaw_deg)


def align_tracks(
    trajectories: Dict[str, object],
    chosen_map: VectorMapData,
    *,
    verbose: bool = False,
) -> Dict[str, object]:
    aligner = TrajectoryAligner(chosen_map, verbose=bool(verbose))
    return aligner.align_all_tracks(trajectories)


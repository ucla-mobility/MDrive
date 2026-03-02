"""Stage 01: input loading and trajectory ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from v2xpnp.pipeline import pipeline_runtime

VectorMapData = pipeline_runtime.VectorMapData
Waypoint = pipeline_runtime.Waypoint


def load_vector_maps(map_paths: List[Path]) -> List[VectorMapData]:
    return [pipeline_runtime.load_vector_map(Path(p).expanduser().resolve()) for p in map_paths]


def load_scenario_trajectories(
    scenario_dir: Path,
    *,
    subdir: Optional[str],
    dt: float,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    flip_y: bool,
) -> Dict[str, object]:
    return pipeline_runtime.load_trajectories(
        scenario_dir=Path(scenario_dir).expanduser().resolve(),
        subdir=subdir,
        dt=float(dt),
        tx=float(tx),
        ty=float(ty),
        tz=float(tz),
        yaw_deg=float(yaw_deg),
        flip_y=bool(flip_y),
    )


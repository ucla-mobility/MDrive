"""Central configuration for infrastructure LiDAR collaboration on v2xpnp scenarios.

The two poses below were derived from the v2x-real-pnp dataset's infra LiDAR
agents (-1 and -2) by applying the v2xpnp -> CARLA alignment transform from
`v2xpnp/map/ucla_map_offset_carla.json`.  See the visualizations under
`/tmp/infra_lidar_*.png` for validation against real-world LiDAR returns.

Activation is controlled by the env var COLMDRIVER_INFRA_COLLAB, which is set
by `tools/run_custom_eval.py` only when its `--infra-collab` flag is passed AND
the scenario belongs to the v2xpnp bucket.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InfraLidarPose:
    name: str
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


# CARLA-world poses, validated against real-world LiDAR returns at the UCLA
# Westwood intersection. Default is the real 4.081 m mount height from the
# v2x→CARLA calibration.
#
# Escape hatch: HEAL OPV2V checkpoints' voxelizer hard-clips points to
# sensor-local z ∈ [-3, 1] m, so at 4 m mount height the ground (z = -4.08)
# and lower ~1 m of every car body get discarded — the encoder sees only
# roof slices and produces OOD features that the fusion module amplifies
# into wrong predictions (cooperative AP regresses on every planner). For
# HEAL HeterBaseline cooperative-fusion runs, set INFRA_LIDAR_LOW_HEIGHT=1
# to lower infra_-1 to 2.5 m so the sensor sweep matches the ~2 m-mount
# distribution the detector was trained on. Default OFF.
_INFRA1_Z = 2.5 if os.environ.get("INFRA_LIDAR_LOW_HEIGHT") == "1" else 4.081

_ALL_INFRA_LIDAR_POSES: list[InfraLidarPose] = [
    InfraLidarPose(name="infra_-1",
                   x=-440.516, y=-24.001, z=_INFRA1_Z,
                   roll=-2.9435, pitch=-1.6348, yaw=-138.5399),
    InfraLidarPose(name="infra_-2",
                   x=-473.248, y=-43.009, z=2.504,
                   roll=1.6127, pitch=3.0153, yaw=2.6192),
]


def _filter_infra_poses() -> list[InfraLidarPose]:
    """Optional INFRA_LIDAR_NAMES env (comma-separated) keeps only the listed
    sensors. Empty/unset = keep all. Used for ablation: e.g. setting
    INFRA_LIDAR_NAMES=infra_-2 isolates the lower-mount sensor to test
    whether the high-mount infra_-1 is what's degrading fusion AP."""
    names = os.environ.get("INFRA_LIDAR_NAMES", "").strip()
    if not names:
        return list(_ALL_INFRA_LIDAR_POSES)
    keep = {n.strip() for n in names.split(",") if n.strip()}
    return [p for p in _ALL_INFRA_LIDAR_POSES if p.name in keep]


INFRA_LIDAR_POSES: list[InfraLidarPose] = _filter_infra_poses()

ENV_VAR = "COLMDRIVER_INFRA_COLLAB"


def is_infra_collab_enabled() -> bool:
    return os.environ.get(ENV_VAR, "0") == "1"


# JSONL filename for the per-tick fusion topology log (written next to other
# per-scenario artifacts under self.infer.log_path or SAVE_PATH).
DIAG_LOG_FILENAME = "infra_fusion_diag.jsonl"


def write_diag_line(log_dir: Path | str | None, **fields: Any) -> None:
    """Append one JSONL record to <log_dir>/infra_fusion_diag.jsonl.

    Best-effort: any IO error is swallowed so a logging failure can never
    break the agent's main loop. Always includes wall-clock timestamp and
    the current value of `is_infra_collab_enabled()` so on/off ablation
    runs can be distinguished even after the fact.

    Caller fields typically include:
        step, sim_t, planner_label,
        n_egos_total, n_egos_present,
        n_regular_rsus, n_infra_lidars,
        n_cooperators_in_fusion,
        ...
    """
    if log_dir is None:
        return
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "wall_t": time.time(),
            "infra_collab_env": is_infra_collab_enabled(),
            **fields,
        }
        with (log_dir / DIAG_LOG_FILENAME).open("a") as f:
            f.write(json.dumps(record))
            f.write("\n")
    except Exception:
        pass

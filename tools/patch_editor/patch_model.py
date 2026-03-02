"""
patch_model.py  —  Declarative override model with undo/redo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import QUndoCommand, QUndoStack

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Override data structures
# ---------------------------------------------------------------------------

@dataclass
class LaneSegmentOverride:
    lane_id: str                  # CARLA line index as string, e.g. "451"
    start_t: Optional[float] = None   # None = entire trajectory
    end_t: Optional[float] = None
    blend_in_s: Optional[float] = None  # Optional smooth transition after start_t

    def to_dict(self) -> dict:
        d: dict = {"lane_id": self.lane_id}
        if self.start_t is not None:
            d["start_t"] = round(self.start_t, 4)
        if self.end_t is not None:
            d["end_t"] = round(self.end_t, 4)
        if self.blend_in_s is not None:
            d["blend_in_s"] = round(self.blend_in_s, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LaneSegmentOverride:
        return cls(
            lane_id=str(d["lane_id"]),
            start_t=float(d["start_t"]) if "start_t" in d else None,
            end_t=float(d["end_t"]) if "end_t" in d else None,
            blend_in_s=float(d["blend_in_s"]) if "blend_in_s" in d else None,
        )


@dataclass
class WaypointOverride:
    frame_idx: int
    dx: float
    dy: float

    def to_dict(self) -> dict:
        return {"frame_idx": self.frame_idx, "dx": round(self.dx, 4), "dy": round(self.dy, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> WaypointOverride:
        return cls(frame_idx=int(d["frame_idx"]), dx=float(d["dx"]), dy=float(d["dy"]))


@dataclass
class ActorOverride:
    actor_id: str
    delete: bool = False
    snap_to_outermost: bool = False
    phase_override: Optional[str] = None          # "approach" | "turn" | "exit"
    lane_segment_overrides: List[LaneSegmentOverride] = field(default_factory=list)
    waypoint_overrides: List[WaypointOverride] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.delete
            and not self.snap_to_outermost
            and self.phase_override is None
            and not self.lane_segment_overrides
            and not self.waypoint_overrides
        )

    def to_dict(self) -> dict:
        d: dict = {"actor_id": self.actor_id}
        if self.delete:
            d["delete"] = True
        if self.snap_to_outermost:
            d["snap_to_outermost"] = True
        if self.phase_override is not None:
            d["phase_override"] = self.phase_override
        if self.lane_segment_overrides:
            d["lane_segment_overrides"] = [o.to_dict() for o in self.lane_segment_overrides]
        if self.waypoint_overrides:
            d["waypoint_overrides"] = [o.to_dict() for o in self.waypoint_overrides]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ActorOverride:
        return cls(
            actor_id=str(d["actor_id"]),
            delete=bool(d.get("delete", False)),
            snap_to_outermost=bool(d.get("snap_to_outermost", False)),
            phase_override=d.get("phase_override"),
            lane_segment_overrides=[
                LaneSegmentOverride.from_dict(x)
                for x in (d.get("lane_segment_overrides") or [])
            ],
            waypoint_overrides=[
                WaypointOverride.from_dict(x)
                for x in (d.get("waypoint_overrides") or [])
            ],
        )


# ---------------------------------------------------------------------------
# Undo commands
# ---------------------------------------------------------------------------

class CmdLaneSnap(QUndoCommand):
    def __init__(self, model: "PatchModel", actor_id: str, lane_id: str,
                 start_t: Optional[float], end_t: Optional[float]):
        super().__init__(f"Lane snap {actor_id} → {lane_id}")
        self._model = model
        self._actor_id = actor_id
        self._new = LaneSegmentOverride(lane_id=lane_id, start_t=start_t, end_t=end_t)
        # snapshot old state
        ov = model.overrides.get(actor_id)
        self._old_lso = list(ov.lane_segment_overrides) if ov else []

    def redo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.lane_segment_overrides = [self._new]
        self._model.changed.emit(self._actor_id)

    def undo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.lane_segment_overrides = list(self._old_lso)
        if ov.is_empty():
            self._model.overrides.pop(self._actor_id, None)
        self._model.changed.emit(self._actor_id)


class CmdPhaseOverride(QUndoCommand):
    def __init__(self, model: "PatchModel", actor_id: str, phase: Optional[str]):
        label = f"Phase {actor_id} → {phase or 'clear'}"
        super().__init__(label)
        self._model = model
        self._actor_id = actor_id
        self._new = phase
        ov = model.overrides.get(actor_id)
        self._old = ov.phase_override if ov else None

    def redo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.phase_override = self._new
        if ov.is_empty():
            self._model.overrides.pop(self._actor_id, None)
        self._model.changed.emit(self._actor_id)

    def undo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.phase_override = self._old
        if ov.is_empty():
            self._model.overrides.pop(self._actor_id, None)
        self._model.changed.emit(self._actor_id)


class CmdDelete(QUndoCommand):
    def __init__(self, model: "PatchModel", actor_id: str):
        super().__init__(f"Delete {actor_id}")
        self._model = model
        self._actor_id = actor_id
        ov = model.overrides.get(actor_id)
        self._was_deleted = ov.delete if ov else False

    def redo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.delete = True
        self._model.changed.emit(self._actor_id)

    def undo(self):
        ov = self._model.overrides.get(self._actor_id)
        if ov:
            ov.delete = self._was_deleted
            if ov.is_empty():
                self._model.overrides.pop(self._actor_id, None)
        self._model.changed.emit(self._actor_id)


class CmdOutermostSnap(QUndoCommand):
    def __init__(self, model: "PatchModel", actor_id: str, was: bool):
        super().__init__(f"Outermost snap {actor_id}")
        self._model = model
        self._actor_id = actor_id
        self._old = was

    def redo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.snap_to_outermost = True
        self._model.changed.emit(self._actor_id)

    def undo(self):
        ov = self._model.overrides.get(self._actor_id)
        if ov:
            ov.snap_to_outermost = self._old
            if ov.is_empty():
                self._model.overrides.pop(self._actor_id, None)
        self._model.changed.emit(self._actor_id)


class CmdWaypointAdjust(QUndoCommand):
    def __init__(self, model: "PatchModel", actor_id: str, frame_idx: int,
                 dx: float, dy: float, old_dx: float, old_dy: float):
        super().__init__(f"Waypoint {actor_id}[{frame_idx}]")
        self._model = model
        self._actor_id = actor_id
        self._frame_idx = frame_idx
        self._new = WaypointOverride(frame_idx=frame_idx, dx=dx, dy=dy)
        self._old = WaypointOverride(frame_idx=frame_idx, dx=old_dx, dy=old_dy) if (old_dx or old_dy) else None

    def redo(self):
        ov = self._model.get_or_create(self._actor_id)
        # Replace or add for this frame_idx
        ov.waypoint_overrides = [
            w for w in ov.waypoint_overrides if w.frame_idx != self._frame_idx
        ]
        if self._new.dx != 0 or self._new.dy != 0:
            ov.waypoint_overrides.append(self._new)
        self._model.changed.emit(self._actor_id)

    def undo(self):
        ov = self._model.get_or_create(self._actor_id)
        ov.waypoint_overrides = [
            w for w in ov.waypoint_overrides if w.frame_idx != self._frame_idx
        ]
        if self._old:
            ov.waypoint_overrides.append(self._old)
        if ov.is_empty():
            self._model.overrides.pop(self._actor_id, None)
        self._model.changed.emit(self._actor_id)


# ---------------------------------------------------------------------------
# PatchModel
# ---------------------------------------------------------------------------

from PyQt5.QtCore import QObject, pyqtSignal


class PatchModel(QObject):
    changed = pyqtSignal(str)   # actor_id that was modified

    def __init__(self, scenario_id: str, parent=None):
        super().__init__(parent)
        self.scenario_id = scenario_id
        self.overrides: Dict[str, ActorOverride] = {}
        self.undo_stack = QUndoStack(self)

    def get_or_create(self, actor_id: str) -> ActorOverride:
        if actor_id not in self.overrides:
            self.overrides[actor_id] = ActorOverride(actor_id=actor_id)
        return self.overrides[actor_id]

    def is_modified(self, actor_id: str) -> bool:
        ov = self.overrides.get(actor_id)
        return ov is not None and not ov.is_empty()

    def is_deleted(self, actor_id: str) -> bool:
        ov = self.overrides.get(actor_id)
        return ov is not None and ov.delete

    # --- commands ---

    def cmd_lane_snap(self, actor_id: str, lane_id: str,
                      start_t: Optional[float], end_t: Optional[float]):
        self.undo_stack.push(CmdLaneSnap(self, actor_id, lane_id, start_t, end_t))

    def cmd_phase_override(self, actor_id: str, phase: Optional[str]):
        self.undo_stack.push(CmdPhaseOverride(self, actor_id, phase))

    def cmd_delete(self, actor_id: str):
        self.undo_stack.push(CmdDelete(self, actor_id))

    def cmd_outermost_snap(self, actor_id: str):
        ov = self.overrides.get(actor_id)
        was = ov.snap_to_outermost if ov else False
        self.undo_stack.push(CmdOutermostSnap(self, actor_id, was))

    def cmd_waypoint_adjust(self, actor_id: str, frame_idx: int,
                            dx: float, dy: float):
        ov = self.overrides.get(actor_id)
        existing = next(
            (w for w in (ov.waypoint_overrides if ov else []) if w.frame_idx == frame_idx),
            None,
        )
        old_dx = existing.dx if existing else 0.0
        old_dy = existing.dy if existing else 0.0
        self.undo_stack.push(
            CmdWaypointAdjust(self, actor_id, frame_idx, dx, dy, old_dx, old_dy)
        )

    # --- serialization ---

    def to_dict(self) -> dict:
        overrides = [ov.to_dict() for ov in self.overrides.values() if not ov.is_empty()]
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "overrides": overrides,
        }

    def load_dict(self, d: dict):
        self.overrides.clear()
        self.undo_stack.clear()
        for raw in d.get("overrides") or []:
            ov = ActorOverride.from_dict(raw)
            if not ov.is_empty():
                self.overrides[ov.actor_id] = ov

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load(self, path: Path):
        if path.exists():
            self.load_dict(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def patch_path_for(html_path: Path) -> Path:
        return html_path.with_suffix(".patch.json")

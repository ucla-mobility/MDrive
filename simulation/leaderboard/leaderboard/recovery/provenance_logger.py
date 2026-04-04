#!/usr/bin/env python

"""
Recovery journal + frame provenance logging.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class RecoveryJournal:
    def __init__(self, recovery_dir: str | Path) -> None:
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.recovery_dir / "recovery_journal.jsonl"

    def append(self, event_type: str, **fields: Any) -> None:
        payload = {
            "timestamp_unix": time.time(),
            "event_type": str(event_type),
        }
        payload.update(fields)
        line = json.dumps(payload, sort_keys=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class FrameProvenanceLogger:
    HEADER = [
        "logical_frame_id",
        "source_segment_id",
        "checkpoint_id",
        "crash_generation",
        "carla_episode_frame",
        "frame_origin",
    ]

    def __init__(self, recovery_dir: str | Path) -> None:
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.recovery_dir / "frame_provenance.csv"
        self.summary_path = self.recovery_dir / "recovery_summary.txt"
        self._segment_id = 0
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.HEADER)
            writer.writeheader()

    def start_segment(
        self,
        *,
        checkpoint_id: int | None,
        crash_generation: int,
        planner_mode: str = "",
        notes: str = "",
    ) -> int:
        self._segment_id += 1
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "segment={} crash_generation={} checkpoint_id={} planner_mode={} notes={}\n".format(
                    self._segment_id,
                    int(crash_generation),
                    "" if checkpoint_id is None else int(checkpoint_id),
                    str(planner_mode or ""),
                    str(notes or ""),
                )
            )
        return int(self._segment_id)

    def log_frame(
        self,
        *,
        logical_frame_id: int,
        checkpoint_id: int | None,
        crash_generation: int,
        carla_episode_frame: int | None,
        frame_origin: str,
    ) -> None:
        row = {
            "logical_frame_id": int(logical_frame_id),
            "source_segment_id": int(self._segment_id),
            "checkpoint_id": "" if checkpoint_id is None else int(checkpoint_id),
            "crash_generation": int(crash_generation),
            "carla_episode_frame": "" if carla_episode_frame is None else int(carla_episode_frame),
            "frame_origin": str(frame_origin),
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.HEADER)
            writer.writerow(row)


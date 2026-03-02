"""
cam_view.py  —  Camera-frame synchroniser (headless-server edition).

Discovers *_cam1.jpeg images in the scenario folder, builds a
frame-number → path index, and on each show_t() call pushes the
nearest frame to cam_server so the browser viewer updates instantly.

The Qt widget itself only shows metadata text (URL, timestamp, filename).
The actual image is served at  http://localhost:<port>/  via cam_server.py.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from . import cam_server


class CamView(QWidget):
    """
    Tracks the scenario's cam1 images and keeps cam_server in sync.

    No image is rendered inside this widget — open the URL shown in the
    header on your local browser (SSH port-forward required on headless
    servers).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # ── Header row ──────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)

        self._hdr_label = QLabel("CAM1")
        self._hdr_label.setFont(QFont("Monospace", 8, QFont.Bold))
        self._hdr_label.setStyleSheet("color: #777;")
        hdr.addWidget(self._hdr_label)
        hdr.addStretch()

        self._subdir_combo = QComboBox()
        self._subdir_combo.setFont(QFont("Monospace", 8))
        self._subdir_combo.setStyleSheet(
            "QComboBox { background: #222; color: #bbb; border: 1px solid #444; }"
        )
        self._subdir_combo.setMaximumWidth(60)
        self._subdir_combo.currentIndexChanged.connect(self._on_subdir_changed)
        hdr.addWidget(self._subdir_combo)

        layout.addLayout(hdr)

        # ── Browser URL label (clickable-style, for copy-paste) ─────────
        self._url_label = QLabel()
        self._url_label.setFont(QFont("Monospace", 8))
        self._url_label.setStyleSheet(
            "color: #5af; background: #161a20; border: 1px solid #2a3a4a;"
            " padding: 3px 6px; border-radius: 3px;"
        )
        self._url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._url_label.setWordWrap(False)
        self._url_label.setText("(server not started)")
        layout.addWidget(self._url_label)

        # ── Per-frame info ───────────────────────────────────────────────
        self._info_label = QLabel("")
        self._info_label.setFont(QFont("Monospace", 7))
        self._info_label.setStyleSheet("color: #555; padding: 1px 2px;")
        layout.addWidget(self._info_label)

        # ── Internal state ───────────────────────────────────────────────
        self._subdirs: List[Path] = []
        self._active_subdir: Optional[Path] = None
        self._frame_nums: List[int] = []
        self._frame_paths: Dict[int, Path] = {}
        self._dt: float = 0.1
        self._t_min: float = 0.0
        self._current_t: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_server_url(self, url: str) -> None:
        """Call after cam_server.start() to display the URL."""
        label = f"Browser:  {url}"
        ssh_hint = f"ssh -L {cam_server._port}:localhost:{cam_server._port} <server>"
        self._url_label.setText(f"{label}\nSSH:      {ssh_hint}")

    def load_scenario(self, html_path: Path, scene_data=None):
        """
        Discover cam1 images in the scenario directory adjacent to html_path.
        """
        scenario_dir = html_path.parent
        self._subdirs = []
        self._active_subdir = None

        candidate_subdirs: List[Tuple[int, Path]] = []
        try:
            for child in sorted(scenario_dir.iterdir()):
                if not child.is_dir():
                    continue
                cam1_files = sorted(child.glob("*_cam1.jpeg"))
                if not cam1_files:
                    cam1_files = sorted(child.glob("*_cam1.jpg"))
                if cam1_files:
                    candidate_subdirs.append((len(cam1_files), child))
        except Exception:
            pass

        if not candidate_subdirs:
            self._set_no_images()
            return

        # Most frames first (primary ego subdir)
        candidate_subdirs.sort(key=lambda x: -x[0])
        self._subdirs = [sd for _, sd in candidate_subdirs]

        self._subdir_combo.blockSignals(True)
        self._subdir_combo.clear()
        for sd in self._subdirs:
            self._subdir_combo.addItem(sd.name)
        self._subdir_combo.blockSignals(False)

        # Infer dt / t_min from ego track
        if scene_data is not None:
            ego_tracks = [t for t in scene_data.tracks if t.role == "ego"]
            if ego_tracks and len(ego_tracks[0].frames) >= 2:
                f0 = ego_tracks[0].frames[0]
                f1 = ego_tracks[0].frames[1]
                dt = f1.t - f0.t
                if 0.01 < dt < 10.0:
                    self._dt = dt
                self._t_min = f0.t

        self._activate_subdir(self._subdirs[0])

    def show_t(self, t: float):
        """Push the cam1 frame nearest to timestamp t to cam_server."""
        self._current_t = t
        if not self._frame_nums:
            cam_server.push_frame(t, None)
            return

        frame_num = self._t_to_frame_num(t)
        nearest = self._nearest_frame_num(frame_num)
        if nearest is None or abs(nearest - frame_num) > 3:
            cam_server.push_frame(t, None)
            self._info_label.setText(f"gap: frame {frame_num} not found")
            return

        path = self._frame_paths[nearest]
        cam_server.push_frame(t, path)
        self._info_label.setText(
            f"frame {nearest:06d}  t={t:.3f}s  {path.parent.name}/{path.name}"
        )

    def clear(self):
        self._frame_nums.clear()
        self._frame_paths.clear()
        cam_server.push_frame(0.0, None)
        self._set_no_images()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_no_images(self):
        self._info_label.setText("no cam1 images found")
        self._hdr_label.setText("CAM1  (none)")

    def _activate_subdir(self, subdir: Path):
        self._active_subdir = subdir
        self._frame_nums = []
        self._frame_paths = {}

        for img in sorted(subdir.glob("*_cam1.jpeg")):
            try:
                num = int(img.stem.split("_")[0])
                self._frame_paths[num] = img
            except ValueError:
                continue
        if not self._frame_paths:
            for img in sorted(subdir.glob("*_cam1.jpg")):
                try:
                    num = int(img.stem.split("_")[0])
                    self._frame_paths[num] = img
                except ValueError:
                    continue

        self._frame_nums = sorted(self._frame_paths.keys())
        n = len(self._frame_nums)
        self._hdr_label.setText(f"CAM1  [{subdir.name}]  {n} frames")

        if self._frame_nums:
            self.show_t(self._current_t)

    def _on_subdir_changed(self, idx: int):
        if 0 <= idx < len(self._subdirs):
            self._activate_subdir(self._subdirs[idx])

    def _t_to_frame_num(self, t: float) -> int:
        return int(round((t - self._t_min) / self._dt))

    def _nearest_frame_num(self, target: int) -> Optional[int]:
        if not self._frame_nums:
            return None
        pos = bisect.bisect_left(self._frame_nums, target)
        candidates = []
        if pos < len(self._frame_nums):
            candidates.append(self._frame_nums[pos])
        if pos > 0:
            candidates.append(self._frame_nums[pos - 1])
        return min(candidates, key=lambda n: abs(n - target))

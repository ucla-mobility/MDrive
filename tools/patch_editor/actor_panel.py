"""
actor_panel.py  —  Actor list + per-actor patch summary.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from .loader import SceneData
from .patch_model import PatchModel

ROLE_ICON = {
    "ego":     "★",
    "vehicle": "▶",
    "walker":  "♟",
    "cyclist": "⚙",
}


class ActorPanel(QWidget):
    """
    Shows list of all actors; highlights modified/deleted ones.

    Signals
    -------
    actor_selected(track_id)   — user clicked an actor row
    """

    actor_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(160)
        self.setMaximumWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("ACTORS")
        title.setFont(QFont("Monospace", 9, QFont.Bold))
        title.setStyleSheet("color: #888; padding: 2px;")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setFont(QFont("Monospace", 9))
        self._list.setStyleSheet("""
            QListWidget {
                background: #1a1a1a;
                border: 1px solid #333;
                color: #ccc;
            }
            QListWidget::item:selected {
                background: #2a3a55;
                color: #fff;
            }
            QListWidget::item:hover {
                background: #222;
            }
        """)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list)

        self._patch_label = QLabel()
        self._patch_label.setFont(QFont("Monospace", 8))
        self._patch_label.setWordWrap(True)
        self._patch_label.setStyleSheet("color: #aaa; padding: 2px;")
        self._patch_label.setMinimumHeight(60)
        layout.addWidget(self._patch_label)

        self._scene_data: Optional[SceneData] = None
        self._patch_model: Optional[PatchModel] = None
        self._track_ids: list = []

    def load(self, scene_data: SceneData, patch_model: PatchModel):
        self._scene_data = scene_data
        self._patch_model = patch_model
        patch_model.changed.connect(self._on_patch_changed)
        self._populate()

    def _populate(self):
        self._list.blockSignals(True)
        self._list.clear()
        self._track_ids = []

        if not self._scene_data:
            self._list.blockSignals(False)
            return

        # Egos first, then vehicles, walkers, cyclists
        order = ["ego", "vehicle", "cyclist", "walker"]
        tracks = sorted(
            self._scene_data.tracks,
            key=lambda t: (order.index(t.role) if t.role in order else 99, t.track_id),
        )

        for track in tracks:
            icon = ROLE_ICON.get(track.role, "●")
            label = f"{icon} {track.track_id}"

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, track.track_id)
            self._apply_item_style(item, track.track_id)
            self._list.addItem(item)
            self._track_ids.append(track.track_id)

        self._list.blockSignals(False)

    def _apply_item_style(self, item: QListWidgetItem, track_id: str):
        if not self._patch_model:
            return
        ov = self._patch_model.overrides.get(track_id)
        if ov and ov.delete:
            item.setForeground(QColor(180, 80, 80))
            item.setText(item.text().rstrip() + " ✕")
        elif ov and not ov.is_empty():
            item.setForeground(QColor(255, 215, 0))
            item.setText(item.text().rstrip() + " ◆")
        else:
            item.setForeground(QColor(180, 180, 180))

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self._track_ids):
            tid = self._track_ids[row]
            self.actor_selected.emit(tid)
            self._update_patch_label(tid)

    def _on_patch_changed(self, actor_id: str):
        # Refresh the item for this actor
        for i, tid in enumerate(self._track_ids):
            if tid == actor_id:
                item = self._list.item(i)
                if item:
                    # Rebuild text without stale suffixes
                    track = self._scene_data.track_by_id.get(tid) if self._scene_data else None
                    if track:
                        icon = ROLE_ICON.get(track.role, "●")
                        item.setText(f"{icon} {tid}")
                    self._apply_item_style(item, tid)
                break
        # Update patch summary if this is the currently selected actor
        current = self._current_selected_id()
        if current == actor_id:
            self._update_patch_label(actor_id)

    def _current_selected_id(self) -> Optional[str]:
        row = self._list.currentRow()
        if 0 <= row < len(self._track_ids):
            return self._track_ids[row]
        return None

    def _update_patch_label(self, track_id: str):
        if not self._patch_model:
            self._patch_label.setText("")
            return
        ov = self._patch_model.overrides.get(track_id)
        if not ov or ov.is_empty():
            self._patch_label.setText("<no patches>")
            return
        lines = []
        if ov.delete:
            lines.append("✕ deleted")
        if ov.snap_to_outermost:
            lines.append("→ outermost")
        if ov.phase_override:
            lines.append(f"↻ phase: {ov.phase_override}")
        for lso in ov.lane_segment_overrides:
            t0 = f"{lso.start_t:.2f}" if lso.start_t is not None else "start"
            t1 = f"{lso.end_t:.2f}" if lso.end_t is not None else "end"
            lines.append(f"⊞ lane {lso.lane_id} [{t0}–{t1}]")
        for wp in ov.waypoint_overrides:
            lines.append(f"· wpt[{wp.frame_idx}] Δ({wp.dx:.2f},{wp.dy:.2f})")
        self._patch_label.setText("\n".join(lines))

    def select_actor(self, track_id: str):
        """External selection from canvas."""
        self._list.blockSignals(True)
        for i, tid in enumerate(self._track_ids):
            if tid == track_id:
                self._list.setCurrentRow(i)
                self._update_patch_label(track_id)
                break
        self._list.blockSignals(False)

    def next_actor(self):
        row = self._list.currentRow()
        if row < self._list.count() - 1:
            self._list.setCurrentRow(row + 1)

    def prev_actor(self):
        row = self._list.currentRow()
        if row > 0:
            self._list.setCurrentRow(row - 1)

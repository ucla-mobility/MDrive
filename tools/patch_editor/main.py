"""
Scenario Patch Editor
=====================

Usage
-----
  python -m tools.patch_editor.main  <path/to/trajectory_plot.html>  [options]

  python tools/patch_editor/main.py  <html_file>
         [--patch  <override.patch.json>]   # explicit patch path (default: <html>.patch.json)
         [--no-yflip]                        # disable Y-axis flip (if map appears upside-down)

Keyboard shortcuts
------------------
  S            Select tool (default)
  L            Lane snap tool
  O            Snap selected actor to outermost lane
  P            Cycle phase override (approach / turn / exit / clear)
  W            Waypoint adjust tool
  D            Delete selected actor
  Ctrl+Z       Undo
  Ctrl+Y       Redo
  Ctrl+S       Save patch
  Tab          Next actor
  Shift+Tab    Previous actor
  F            Follow selected actor (ego-follow camera)
  A            Fit all
  Space        Play / pause (in follow mode)
  [ / ]        Decrease / increase playback speed
  Esc          Exit tool / follow / deselect

  Shift+drag on timeline  → select time range for lane snap
  Right-click timeline    → clear range
  Middle-drag / Alt+drag  → pan
  Scroll wheel            → zoom (zoom-to-cursor)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QAction, QApplication, QHBoxLayout, QLabel, QMainWindow,
    QShortcut, QSizePolicy, QSplitter, QStatusBar, QToolBar,
    QVBoxLayout, QWidget,
)

from . import cam_server
from .actor_panel import ActorPanel
from .cam_view import CamView
from .canvas import MapCanvas
from .loader import SceneData, find_rightmost_parallel_line, load_from_html
from .patch_model import PatchModel
from .timeline import TimelineScrubber

PHASE_CYCLE = ["approach", "turn", "exit", None]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PatchEditorWindow(QMainWindow):
    def __init__(self, html_path: Path, patch_path: Path, y_flip: bool = True):
        super().__init__()
        self._html_path = html_path
        self._patch_path = patch_path
        self._y_flip = y_flip
        self._scene_data: Optional[SceneData] = None
        self._patch_model: Optional[PatchModel] = None

        self.setWindowTitle("Scenario Patch Editor")
        self.resize(1400, 860)
        self.setStyleSheet("QMainWindow { background: #111; }")

        self._build_ui()
        self._build_shortcuts()
        self._load(html_path, patch_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Toolbar ─────────────────────────────────────────────────
        toolbar = QToolBar("Tools")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar { background: #1c1c1c; border-bottom: 1px solid #333; spacing: 4px; padding: 3px; }
            QToolButton { color: #ccc; background: #2a2a2a; border: 1px solid #444;
                          padding: 4px 10px; font-family: monospace; font-size: 11px; }
            QToolButton:checked { background: #2a4a6a; border-color: #4488cc; color: #fff; }
            QToolButton:hover   { background: #333; }
        """)
        self.addToolBar(toolbar)

        def _btn(label: str, shortcut: str, checkable: bool = False) -> QAction:
            a = QAction(f"{label}  [{shortcut}]", self)
            a.setCheckable(checkable)
            toolbar.addAction(a)
            return a

        self._act_select   = _btn("Select",   "S",   checkable=True)
        self._act_lane     = _btn("Lane",     "L",   checkable=True)
        self._act_outer    = _btn("Outer",    "O")
        self._act_phase    = _btn("Phase",    "P")
        self._act_wpt      = _btn("Waypoint", "W",   checkable=True)
        self._act_delete   = _btn("Delete",   "D")
        toolbar.addSeparator()
        self._act_overlap  = _btn("Overlap",  "Space", checkable=True)
        toolbar.addSeparator()
        self._act_undo = _btn("Undo", "Ctrl+Z")
        self._act_redo = _btn("Redo", "Ctrl+Y")
        toolbar.addSeparator()
        self._act_save = _btn("Save", "Ctrl+S")

        self._act_select.setChecked(True)

        self._act_select.triggered.connect(lambda: self._set_tool("select"))
        self._act_lane.triggered.connect(lambda: self._set_tool("lane"))
        self._act_wpt.triggered.connect(lambda: self._set_tool("waypoint"))
        self._act_outer.triggered.connect(self._do_outer_snap)
        self._act_phase.triggered.connect(self._do_phase_cycle)
        self._act_delete.triggered.connect(self._do_delete)
        self._act_overlap.triggered.connect(self._toggle_overlap)
        self._act_undo.triggered.connect(self._do_undo)
        self._act_redo.triggered.connect(self._do_redo)
        self._act_save.triggered.connect(self._do_save)

        # ── Main area: canvas | (actor panel + cam view) ─────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #333; }")

        self._canvas = MapCanvas(y_flip=self._y_flip)
        self._canvas.actor_selected.connect(self._on_actor_selected_from_canvas)
        self._canvas.t_changed.connect(self._on_t_changed)
        self._canvas.status_message.connect(self._show_status)
        splitter.addWidget(self._canvas)

        # Right panel: actor list (top) + cam view (bottom)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setStyleSheet("QSplitter::handle { background: #333; }")

        self._actor_panel = ActorPanel()
        self._actor_panel.actor_selected.connect(self._on_actor_selected_from_panel)
        right_splitter.addWidget(self._actor_panel)

        self._cam_view = CamView()
        right_splitter.addWidget(self._cam_view)

        right_splitter.setStretchFactor(0, 0)   # actor list: fixed
        right_splitter.setStretchFactor(1, 1)   # cam view: expands
        right_splitter.setSizes([280, 300])

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1150, 250])

        root_layout.addWidget(splitter, 1)

        # ── Timeline ────────────────────────────────────────────────
        tl_container = QWidget()
        tl_container.setStyleSheet("background: #1a1a1a; border-top: 1px solid #333;")
        tl_layout = QVBoxLayout(tl_container)
        tl_layout.setContentsMargins(4, 2, 4, 2)

        self._tl_actor_label = QLabel("No actor selected")
        self._tl_actor_label.setFont(QFont("Monospace", 8))
        self._tl_actor_label.setStyleSheet("color: #777;")
        tl_layout.addWidget(self._tl_actor_label)

        self._timeline = TimelineScrubber()
        self._timeline.t_changed.connect(self._on_timeline_t_changed)
        self._timeline.range_changed.connect(self._on_timeline_range_changed)
        tl_layout.addWidget(self._timeline)

        root_layout.addWidget(tl_container)

        # ── Status bar ──────────────────────────────────────────────
        sb = QStatusBar()
        sb.setStyleSheet("QStatusBar { background: #111; color: #888; font-family: monospace; font-size: 10px; }")
        self.setStatusBar(sb)
        self._status_label = QLabel("Loading…")
        sb.addPermanentWidget(self._status_label)

        self._current_tool = "select"

    def _build_shortcuts(self):
        def _sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)

        _sc("S", lambda: self._set_tool("select"))
        _sc("L", lambda: self._set_tool("lane"))
        _sc("W", lambda: self._set_tool("waypoint"))
        _sc("O", self._do_outer_snap)
        _sc("P", self._do_phase_cycle)
        _sc("D", self._do_delete)
        _sc("Ctrl+Z", self._do_undo)
        _sc("Ctrl+Y", self._do_redo)
        _sc("Ctrl+Shift+Z", self._do_redo)
        _sc("Ctrl+S", self._do_save)
        _sc("Tab", self._actor_panel.next_actor)
        _sc("Shift+Tab", self._actor_panel.prev_actor)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self, html_path: Path, patch_path: Path):
        try:
            scene_data = load_from_html(html_path)
        except Exception as e:
            self._show_status(f"ERROR loading {html_path.name}: {e}")
            return

        patch_model = PatchModel(scene_data.scenario_name, parent=self)
        patch_model.load(patch_path)

        self._scene_data = scene_data
        self._patch_model = patch_model

        # Wire undo/redo button state
        patch_model.undo_stack.canUndoChanged.connect(self._act_undo.setEnabled)
        patch_model.undo_stack.canRedoChanged.connect(self._act_redo.setEnabled)
        self._act_undo.setEnabled(False)
        self._act_redo.setEnabled(False)

        self._canvas.load(scene_data, patch_model)
        self._actor_panel.load(scene_data, patch_model)

        # Start camera HTTP server (no-op if already running)
        port = cam_server.start(8765)
        self._cam_view.set_server_url(cam_server.url())
        self._cam_view.load_scenario(html_path, scene_data)

        n_tracks = len(scene_data.tracks)
        n_lines  = len(scene_data.carla_lines)
        n_patches = sum(1 for ov in patch_model.overrides.values() if not ov.is_empty())
        self._show_status(
            f"{scene_data.scenario_name}  |  {n_tracks} actors  |  {n_lines} CARLA lines"
            + (f"  |  {n_patches} existing patches loaded" if n_patches else "")
        )
        self.setWindowTitle(f"Patch Editor — {html_path.name}")

    # ------------------------------------------------------------------
    # Tool switching
    # ------------------------------------------------------------------

    def _set_tool(self, tool: str):
        self._current_tool = tool
        self._act_select.setChecked(tool == "select")
        self._act_lane.setChecked(tool == "lane")
        self._act_wpt.setChecked(tool == "waypoint")
        self._canvas.set_lane_snap_mode(tool == "lane")
        self._canvas.set_waypoint_mode(tool == "waypoint")
        if tool == "select":
            self._show_status("Select: click actor, double-click to follow")

    def _toggle_overlap(self):
        self._canvas.toggle_overlap()

    # ------------------------------------------------------------------
    # Tool actions
    # ------------------------------------------------------------------

    def _do_outer_snap(self):
        if not self._patch_model or not self._canvas._selected_id:
            self._show_status("Select an actor first (O)")
            return
        tid = self._canvas._selected_id
        if not self._scene_data:
            return
        track = self._scene_data.track_by_id.get(tid)
        if not track or not track.frames:
            return
        # Find current ccli
        mid_f = track.frames[len(track.frames) // 2]
        cur_line = self._scene_data.line_by_idx.get(mid_f.ccli)
        if cur_line is None:
            self._show_status(f"Actor {tid} has no CARLA snap — outermost unavailable")
            return
        outer = find_rightmost_parallel_line(self._scene_data.carla_lines, cur_line)
        if outer:
            # Also apply a lane snap to the outermost line (entire trajectory)
            self._patch_model.cmd_lane_snap(tid, str(outer.idx), None, None)
        self._patch_model.cmd_outermost_snap(tid)
        self._show_status(
            f"Outermost snap applied to {tid}"
            + (f" → lane {outer.idx}" if outer else " (no parallel found)")
        )

    def _do_phase_cycle(self):
        if not self._patch_model or not self._canvas._selected_id:
            self._show_status("Select an actor first (P)")
            return
        tid = self._canvas._selected_id
        ov = self._patch_model.overrides.get(tid)
        current = ov.phase_override if ov else None
        try:
            idx = PHASE_CYCLE.index(current)
        except ValueError:
            idx = -1
        next_phase = PHASE_CYCLE[(idx + 1) % len(PHASE_CYCLE)]
        self._patch_model.cmd_phase_override(tid, next_phase)
        self._show_status(
            f"Phase for {tid}: {next_phase or '(cleared)'}"
        )

    def _do_delete(self):
        if not self._patch_model or not self._canvas._selected_id:
            self._show_status("Select an actor first (D)")
            return
        tid = self._canvas._selected_id
        self._patch_model.cmd_delete(tid)
        self._show_status(f"Deleted {tid}  (Ctrl+Z to undo)")

    def _do_undo(self):
        if self._patch_model:
            self._patch_model.undo_stack.undo()

    def _do_redo(self):
        if self._patch_model:
            self._patch_model.undo_stack.redo()

    def _do_save(self):
        if not self._patch_model:
            return
        self._patch_model.save(self._patch_path)
        n = sum(1 for ov in self._patch_model.overrides.values() if not ov.is_empty())
        self._show_status(f"Saved {n} overrides → {self._patch_path.name}")

    # ------------------------------------------------------------------
    # Signals from canvas / timeline / panel
    # ------------------------------------------------------------------

    def _on_actor_selected_from_canvas(self, track_id: str):
        self._actor_panel.select_actor(track_id)
        self._update_timeline_for(track_id)

    def _on_actor_selected_from_panel(self, track_id: str):
        self._canvas.select_actor(track_id)
        self._update_timeline_for(track_id)
        self._canvas._fit_to_actor(track_id)

    def _update_timeline_for(self, track_id: str):
        if not self._scene_data:
            return
        track = self._scene_data.track_by_id.get(track_id)
        if not track:
            return
        self._tl_actor_label.setText(
            f"{track_id}  [{track.role}]  {len(track.frames)} frames  "
            f"{track.t_start:.2f}–{track.t_end:.2f}s"
        )
        self._timeline.set_track(
            track.t_start, track.t_end, track.ccli_change_times()
        )
        self._timeline.set_current_t(self._canvas._current_t)
        self._canvas.set_t_range(None, None)

    def _on_timeline_t_changed(self, t: float):
        self._canvas.set_current_t(t)
        self._cam_view.show_t(t)

    def _on_timeline_range_changed(self, t0, t1):
        self._canvas.set_t_range(t0, t1)
        if t0 is not None and t1 is not None:
            self._show_status(
                f"Range selected: {t0:.2f}–{t1:.2f}s  — press L then click a lane"
            )
        else:
            self._show_status("")

    def _on_t_changed(self, t: float):
        """Ego follow playback tick — update both timeline and camera."""
        self._timeline.set_current_t(t)
        self._cam_view.show_t(t)

    def _show_status(self, msg: str):
        self._status_label.setText(msg)

    def closeEvent(self, event):  # noqa: N802
        cam_server.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scenario Patch Editor")
    parser.add_argument("html", help="Pipeline HTML output file")
    parser.add_argument("--patch", help="Patch JSON file (default: <html>.patch.json)")
    parser.add_argument("--no-yflip", action="store_true",
                        help="Disable Y-axis flip (use if map appears upside-down)")
    parser.add_argument("--cam-port", type=int, default=8765,
                        help="Port for the camera HTTP server (default: 8765)")
    args = parser.parse_args()

    html_path  = Path(args.html).expanduser().resolve()
    patch_path = Path(args.patch).expanduser().resolve() if args.patch else (
        PatchModel.patch_path_for(html_path)
    )
    y_flip = not args.no_yflip

    # Start camera server early so the URL is printed before the window opens
    port = cam_server.start(args.cam_port)
    print(f"[patch_editor] Camera feed: {cam_server.url()}")
    print(f"[patch_editor] SSH tunnel:  ssh -L {port}:localhost:{port} <user>@<server>")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    win = PatchEditorWindow(html_path, patch_path, y_flip=y_flip)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

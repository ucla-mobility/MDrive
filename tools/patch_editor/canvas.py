"""
canvas.py  —  Map + actor trajectory canvas with ego-follow mode.

Coordinate system: items placed at raw world (x, y).
A Y-flip (scale 1, -1) is baked into all view transforms so that
geographic North (positive Y in CARLA world) renders as screen-up.
If your map appears upside-down, pass y_flip=False to MapCanvas().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import (
    QPointF, QRectF, Qt, QTimer, pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
    QPolygonF, QTransform, QKeyEvent, QWheelEvent, QMouseEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
    QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsTextItem, QGraphicsView,
)

from .loader import ActorFrame, ActorTrack, CarlaLine, SceneData, find_nearest_line, find_rightmost_parallel_line
from .patch_model import PatchModel

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_MAP_LINE       = QColor(80, 80, 80)
C_MAP_TICK       = QColor(110, 110, 110)
C_MAP_LINE_HOVER = QColor(0, 220, 130)
C_LANE_SNAP_CAND = QColor(0, 220, 130, 160)

ROLE_COLORS = {
    "ego":      QColor(0, 170, 255),
    "vehicle":  QColor(255, 179, 0),
    "walker":   QColor(102, 187, 106),
    "cyclist":  QColor(171, 71, 188),
}
C_SELECTED      = QColor(255, 255, 0)
C_CCLI_CHANGE   = QColor(255, 102, 0)
C_DELETED       = QColor(180, 60, 60, 120)
C_MODIFIED_RING = QColor(255, 215, 0)
C_OVERLAP       = QColor(255, 40, 40)
C_EGO_MARKER    = QColor(255, 255, 255)
C_EGO_MARKER_FG = QColor(0, 0, 0)

TICK_EVERY_M = 20.0    # direction tick spacing
TICK_LEN_M   = 3.0     # tick arm length


# ---------------------------------------------------------------------------
# Scene item tags (stored in QGraphicsItem.setData)
# ---------------------------------------------------------------------------
TAG_TYPE   = 0   # "map_line" | "actor_traj" | "ccli_dot" | "frame_dot"
TAG_LINEID = 1   # CarlaLine.idx
TAG_TRKID  = 2   # ActorTrack.track_id
TAG_FRAMEI = 3   # frame index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _actor_color(track: ActorTrack, selected: bool = False) -> QColor:
    if selected:
        return C_SELECTED
    return ROLE_COLORS.get(track.role, ROLE_COLORS["vehicle"])


def _make_path_item(path: QPainterPath, color: QColor,
                    width: float = 1.5, z: float = 0.0) -> QGraphicsPathItem:
    item = QGraphicsPathItem(path)
    pen = QPen(color, width)
    pen.setCosmetic(True)  # width in pixels, not world units
    item.setPen(pen)
    item.setBrush(QBrush(Qt.NoBrush))
    item.setZValue(z)
    return item


def _polyline_to_path(pts) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
    for i in range(1, len(pts)):
        path.lineTo(float(pts[i, 0]), float(pts[i, 1]))
    return path


def _direction_ticks(pts, every_m: float = TICK_EVERY_M,
                     arm: float = TICK_LEN_M) -> QPainterPath:
    """Build arrowhead ticks along a polyline showing travel direction."""
    path = QPainterPath()
    cum = 0.0
    next_tick = every_m * 0.5
    for i in range(len(pts) - 1):
        ax, ay = float(pts[i, 0]),   float(pts[i, 1])
        bx, by = float(pts[i+1, 0]), float(pts[i+1, 1])
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-6:
            continue
        while next_tick <= cum + seg:
            frac = (next_tick - cum) / seg
            mx = ax + frac * (bx - ax)
            my = ay + frac * (by - ay)
            dx = (bx - ax) / seg
            dy = (by - ay) / seg
            rx, ry = -dy, dx  # right-perpendicular
            # small V-arrow
            tip_x, tip_y = mx + dx * arm, my + dy * arm
            l_x = mx - dx * arm * 0.5 + rx * arm * 0.5
            l_y = my - dy * arm * 0.5 + ry * arm * 0.5
            r_x = mx - dx * arm * 0.5 - rx * arm * 0.5
            r_y = my - dy * arm * 0.5 - ry * arm * 0.5
            path.moveTo(l_x, l_y)
            path.lineTo(tip_x, tip_y)
            path.lineTo(r_x, r_y)
            next_tick += every_m
        cum += seg
    return path


# ---------------------------------------------------------------------------
# MapCanvas
# ---------------------------------------------------------------------------

class MapCanvas(QGraphicsView):
    """
    2D canvas with:
    - CARLA polylines + direction ticks (map layer, drawn once)
    - Actor trajectories (actor layer, redrawn on patch change)
    - Ego-follow camera mode (rotating, forward-biased)

    Signals
    -------
    actor_selected(track_id: str)
    lane_clicked(line_idx: int, world_x: float, world_y: float)
    t_changed(t: float)          emitted during ego playback
    """

    actor_selected  = pyqtSignal(str)
    lane_clicked    = pyqtSignal(int, float, float)
    t_changed       = pyqtSignal(float)
    status_message  = pyqtSignal(str)

    def __init__(self, y_flip: bool = True, parent=None):
        super().__init__(parent)
        self._y_flip = y_flip  # flip Y so geographic North = screen up

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(18, 18, 18)))
        self.setFocusPolicy(Qt.StrongFocus)

        self._scene_data: Optional[SceneData] = None
        self._patch_model: Optional[PatchModel] = None

        # view state
        self._zoom: float = 1.0
        self._pan_start: Optional[QPointF] = None
        self._pan_transform_start: Optional[QTransform] = None

        # selection / tool state
        self._selected_id: Optional[str] = None
        self._hover_line_idx: Optional[int] = None
        self._lane_snap_mode: bool = False
        self._waypoint_mode: bool = False
        self._show_overlap: bool = False
        self._selected_t_range: Optional[Tuple[float, float]] = None  # from timeline

        # ego-follow state
        self._follow_id: Optional[str] = None
        self._current_t: float = 0.0
        self._playback_speed: float = 1.0
        self._playing: bool = False
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(80)  # ~12fps
        self._playback_timer.timeout.connect(self._tick_playback)

        # item registries
        self._map_items: List[QGraphicsItem] = []
        self._actor_items: Dict[str, List[QGraphicsItem]] = {}
        self._hover_items: List[QGraphicsItem] = []

        # waypoint drag
        self._drag_frame_item: Optional[QGraphicsEllipseItem] = None
        self._drag_actor_id: Optional[str] = None
        self._drag_frame_idx: int = -1
        self._drag_start_scene: Optional[QPointF] = None
        self._drag_original_xy: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, scene_data: SceneData, patch_model: PatchModel):
        self._scene.clear()
        self._map_items.clear()
        self._actor_items.clear()
        self._hover_items.clear()
        self._scene_data = scene_data
        self._patch_model = patch_model
        patch_model.changed.connect(self._on_patch_changed)

        self._draw_map()
        self._draw_all_actors()
        self._fit_all()

    def _fit_all(self):
        if not self._scene_data:
            return
        bb = self._scene_data.map_bbox
        rect = QRectF(
            float(bb["min_x"]), -float(bb["max_y"]),
            float(bb["max_x"]) - float(bb["min_x"]),
            float(bb["max_y"]) - float(bb["min_y"]),
        ) if self._y_flip else QRectF(
            float(bb["min_x"]), float(bb["min_y"]),
            float(bb["max_x"]) - float(bb["min_x"]),
            float(bb["max_y"]) - float(bb["min_y"]),
        )
        self.fitInView(rect, Qt.KeepAspectRatio)
        # capture resulting zoom from fitInView
        self._zoom = self.transform().m11()

    # ------------------------------------------------------------------
    # Map layer
    # ------------------------------------------------------------------

    def _draw_map(self):
        if not self._scene_data:
            return
        for cl in self._scene_data.carla_lines:
            pts = cl.pts
            # Main polyline
            path = _polyline_to_path(pts if not self._y_flip else self._flip_pts(pts))
            item = _make_path_item(path, C_MAP_LINE, width=1.2, z=0.0)
            item.setData(TAG_TYPE, "map_line")
            item.setData(TAG_LINEID, cl.idx)
            self._scene.addItem(item)
            self._map_items.append(item)
            # Direction ticks
            tick_pts = pts if not self._y_flip else self._flip_pts(pts)
            ticks = _direction_ticks(tick_pts)
            if not ticks.isEmpty():
                tick_item = _make_path_item(ticks, C_MAP_TICK, width=1.0, z=0.1)
                self._scene.addItem(tick_item)
                self._map_items.append(tick_item)

    @staticmethod
    def _flip_pts(pts):
        """Return pts with Y negated (numpy array)."""
        import numpy as np
        flipped = pts.copy()
        flipped[:, 1] = -flipped[:, 1]
        return flipped

    def _world_to_scene(self, x: float, y: float) -> Tuple[float, float]:
        return (x, -y) if self._y_flip else (x, y)

    def _scene_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        return (sx, -sy) if self._y_flip else (sx, sy)

    # ------------------------------------------------------------------
    # Actor layer
    # ------------------------------------------------------------------

    def _draw_all_actors(self):
        if not self._scene_data:
            return
        for track in self._scene_data.tracks:
            self._draw_actor(track)
        if self._show_overlap:
            self._draw_overlap_markers()

    def _draw_actor(self, track: ActorTrack):
        # Remove old items for this actor
        for item in self._actor_items.pop(track.track_id, []):
            self._scene.removeItem(item)

        items: List[QGraphicsItem] = []
        if not track.frames:
            self._actor_items[track.track_id] = items
            return

        is_selected  = track.track_id == self._selected_id
        is_follow    = track.track_id == self._follow_id
        is_deleted   = self._patch_model and self._patch_model.is_deleted(track.track_id)
        is_modified  = self._patch_model and self._patch_model.is_modified(track.track_id)

        base_color = _actor_color(track, selected=is_selected)
        traj_width = 3.0 if (is_selected or is_follow) else 1.8
        traj_z     = 3.0 if (is_selected or is_follow) else 2.0

        # Build trajectory path from cx/cy (CARLA-snapped) preferring snapped
        path = QPainterPath()
        f0 = track.frames[0]
        sx0, sy0 = self._world_to_scene(f0.cx if f0.ccli >= 0 else f0.x,
                                         f0.cy if f0.ccli >= 0 else f0.y)
        path.moveTo(sx0, sy0)
        for f in track.frames[1:]:
            wx = f.cx if f.ccli >= 0 else f.x
            wy = f.cy if f.ccli >= 0 else f.y
            sx, sy = self._world_to_scene(wx, wy)
            path.lineTo(sx, sy)

        color = base_color
        if is_deleted:
            color = C_DELETED
        traj_item = _make_path_item(path, color, width=traj_width, z=traj_z)
        self._scene.addItem(traj_item)
        items.append(traj_item)

        # Modified ring around first point
        if is_modified and not is_deleted:
            sx, sy = self._world_to_scene(f0.cx if f0.ccli >= 0 else f0.x,
                                           f0.cy if f0.ccli >= 0 else f0.y)
            ring = self._scene.addEllipse(
                sx - 4, sy - 4, 8, 8,
                QPen(C_MODIFIED_RING, 1.5, style=Qt.DotLine),
                QBrush(Qt.NoBrush),
            )
            ring.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            items.append(ring)

        # CCLI change markers
        for t_change in track.ccli_change_times():
            f = track.interp(t_change)
            wx = f.cx if f.ccli >= 0 else f.x
            wy = f.cy if f.ccli >= 0 else f.y
            sx, sy = self._world_to_scene(wx, wy)
            dot = self._scene.addEllipse(
                sx - 4, sy - 4, 8, 8,
                QPen(C_CCLI_CHANGE, 1.0),
                QBrush(C_CCLI_CHANGE),
            )
            dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            dot.setZValue(4.0)
            dot.setData(TAG_TYPE, "ccli_dot")
            dot.setData(TAG_TRKID, track.track_id)
            items.append(dot)

        # Frame dots (for waypoint tool, small, only selected actor)
        if is_selected and self._waypoint_mode:
            ov = self._patch_model.overrides.get(track.track_id) if self._patch_model else None
            wp_overrides = {w.frame_idx: (w.dx, w.dy) for w in (ov.waypoint_overrides if ov else [])}
            for fi, f in enumerate(track.frames):
                dx, dy = wp_overrides.get(fi, (0.0, 0.0))
                wx = (f.cx if f.ccli >= 0 else f.x) + dx
                wy = (f.cy if f.ccli >= 0 else f.y) + dy
                sx, sy = self._world_to_scene(wx, wy)
                fr_dot = self._scene.addEllipse(
                    sx - 3, sy - 3, 6, 6,
                    QPen(base_color, 1.0),
                    QBrush(base_color),
                )
                fr_dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                fr_dot.setZValue(5.0)
                fr_dot.setData(TAG_TYPE, "frame_dot")
                fr_dot.setData(TAG_TRKID, track.track_id)
                fr_dot.setData(TAG_FRAMEI, fi)
                fr_dot.setCursor(Qt.SizeAllCursor)
                items.append(fr_dot)

        # Track label at midpoint
        if is_selected or is_follow:
            mid_f = track.frames[len(track.frames) // 2]
            wx = mid_f.cx if mid_f.ccli >= 0 else mid_f.x
            wy = mid_f.cy if mid_f.ccli >= 0 else mid_f.y
            sx, sy = self._world_to_scene(wx, wy)
            label_item = self._scene.addText(
                f"{track.track_id}\n[{track.role}]",
                QFont("Monospace", 8),
            )
            label_item.setDefaultTextColor(base_color)
            label_item.setPos(sx + 6, sy - 8)
            label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            label_item.setZValue(6.0)
            items.append(label_item)

        # Ego playback marker
        if is_follow:
            self._add_ego_marker(items, track)

        self._actor_items[track.track_id] = items

    def _add_ego_marker(self, items: list, track: ActorTrack):
        """Draw a direction triangle at current_t position."""
        f = track.interp(self._current_t)
        wx = f.cx if f.ccli >= 0 else f.x
        wy = f.cy if f.ccli >= 0 else f.y
        sx, sy = self._world_to_scene(wx, wy)

        size = 10
        poly = QPolygonF([
            QPointF(0, -size),
            QPointF(size * 0.6, size * 0.6),
            QPointF(-size * 0.6, size * 0.6),
        ])
        marker = QGraphicsPolygonItem(poly)
        marker.setPen(QPen(C_EGO_MARKER, 1.5))
        marker.setBrush(QBrush(C_EGO_MARKER))
        marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        marker.setZValue(10.0)
        # Rotate marker to face actor heading
        yaw = f.cyaw if f.ccli >= 0 else f.yaw
        # In scene coords with y-flip: heading = -(yaw - 90) to face screen-up when yaw=90
        screen_angle = -(yaw - 90) if self._y_flip else yaw
        marker.setRotation(screen_angle)
        marker.setPos(sx, sy)
        self._scene.addItem(marker)
        items.append(marker)

    def _draw_overlap_markers(self):
        """Red dots where two actors are within 2m at same timestamp."""
        if not self._scene_data:
            return
        tracks = [t for t in self._scene_data.tracks
                  if not (self._patch_model and self._patch_model.is_deleted(t.track_id))]
        for i, ta in enumerate(tracks):
            for j, tb in enumerate(tracks):
                if j <= i:
                    continue
                for fa in ta.frames:
                    fb = tb.interp(fa.t)
                    dist = math.hypot(
                        (fa.cx if fa.ccli >= 0 else fa.x) - (fb.cx if fb.ccli >= 0 else fb.x),
                        (fa.cy if fa.ccli >= 0 else fa.y) - (fb.cy if fb.ccli >= 0 else fb.y),
                    )
                    if dist < 2.0:
                        sx, sy = self._world_to_scene(
                            fa.cx if fa.ccli >= 0 else fa.x,
                            fa.cy if fa.ccli >= 0 else fa.y,
                        )
                        dot = self._scene.addEllipse(
                            sx - 6, sy - 6, 12, 12,
                            QPen(C_OVERLAP, 1.5),
                            QBrush(QColor(255, 40, 40, 80)),
                        )
                        dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                        dot.setZValue(7.0)
                        self._actor_items.setdefault("__overlap__", []).append(dot)

    def _on_patch_changed(self, actor_id: str):
        if not self._scene_data:
            return
        track = self._scene_data.track_by_id.get(actor_id)
        if track:
            self._draw_actor(track)
        # redraw overlap if visible
        if self._show_overlap:
            for item in self._actor_items.pop("__overlap__", []):
                self._scene.removeItem(item)
            self._draw_overlap_markers()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_actor(self, track_id: Optional[str]):
        old_id = self._selected_id
        self._selected_id = track_id
        # Redraw old + new
        for tid in {old_id, track_id}:
            if tid and self._scene_data:
                track = self._scene_data.track_by_id.get(tid)
                if track:
                    self._draw_actor(track)
        if track_id:
            self.actor_selected.emit(track_id)

    def set_follow_actor(self, track_id: Optional[str]):
        old = self._follow_id
        self._follow_id = track_id
        if old and self._scene_data:
            track = self._scene_data.track_by_id.get(old)
            if track:
                self._draw_actor(track)
        if track_id:
            self.select_actor(track_id)
            track = self._scene_data.track_by_id.get(track_id) if self._scene_data else None
            if track:
                self._current_t = track.t_start
                self._draw_actor(track)

    def set_t_range(self, start_t: Optional[float], end_t: Optional[float]):
        self._selected_t_range = (start_t, end_t) if (start_t is not None and end_t is not None) else None

    def set_current_t(self, t: float):
        """Called by timeline scrubber."""
        self._current_t = t
        if self._follow_id:
            self._update_ego_camera()
            track = self._scene_data.track_by_id.get(self._follow_id) if self._scene_data else None
            if track:
                for item in self._actor_items.pop(self._follow_id, []):
                    self._scene.removeItem(item)
                self._draw_actor(track)
                self._actor_items[self._follow_id] = self._actor_items.get(self._follow_id, [])

    # ------------------------------------------------------------------
    # Ego-follow camera
    # ------------------------------------------------------------------

    def toggle_playback(self):
        if not self._follow_id:
            return
        self._playing = not self._playing
        if self._playing:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()

    def _tick_playback(self):
        if not self._follow_id or not self._scene_data:
            return
        track = self._scene_data.track_by_id.get(self._follow_id)
        if not track:
            return
        dt = self._playback_timer.interval() / 1000.0
        self._current_t += dt * self._playback_speed
        if self._current_t > track.t_end:
            self._current_t = track.t_start

        self._update_ego_camera()
        # Redraw ego marker
        for item in self._actor_items.pop(self._follow_id, []):
            self._scene.removeItem(item)
        self._draw_actor(track)
        self.t_changed.emit(self._current_t)

    def _update_ego_camera(self):
        if not self._follow_id or not self._scene_data:
            return
        track = self._scene_data.track_by_id.get(self._follow_id)
        if not track:
            return
        f = track.interp(self._current_t)
        wx = f.cx if f.ccli >= 0 else f.x
        wy = f.cy if f.ccli >= 0 else f.y
        yaw = f.cyaw if f.ccli >= 0 else f.yaw
        sx, sy = self._world_to_scene(wx, wy)

        vw = self.viewport().width()
        vh = self.viewport().height()

        # Actor placed at 35% from top → more view ahead.
        # Rotation formula (derived):
        #   r = 90 - yaw_deg makes heading = screen-up when y-flipped.
        # If y_flip=False, heading convention may differ; adjust as needed.
        if self._y_flip:
            r = 90.0 - yaw
        else:
            r = yaw - 90.0

        t = QTransform()
        t.translate(vw / 2.0, vh * 0.35)
        t.rotate(r)
        t.scale(self._zoom, self._zoom)   # no additional y-flip here; y-flip is in _world_to_scene
        t.translate(-sx, -sy)
        self.setTransform(t)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def set_lane_snap_mode(self, enabled: bool):
        self._lane_snap_mode = enabled
        self._waypoint_mode = False
        self._clear_hover()
        if enabled:
            self.setCursor(Qt.CrossCursor)
            self.status_message.emit("Lane snap: hover lane, click to apply")
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_waypoint_mode(self, enabled: bool):
        self._waypoint_mode = enabled
        self._lane_snap_mode = False
        self._clear_hover()
        if enabled and self._selected_id and self._scene_data:
            track = self._scene_data.track_by_id.get(self._selected_id)
            if track:
                self._draw_actor(track)  # show frame dots
            self.status_message.emit("Waypoint: drag dots to adjust, release to commit")
        elif not enabled and self._selected_id and self._scene_data:
            track = self._scene_data.track_by_id.get(self._selected_id)
            if track:
                self._draw_actor(track)  # hide frame dots

    def toggle_overlap(self):
        self._show_overlap = not self._show_overlap
        for item in self._actor_items.pop("__overlap__", []):
            self._scene.removeItem(item)
        if self._show_overlap:
            self._draw_overlap_markers()

    def _clear_hover(self):
        for item in self._hover_items:
            self._scene.removeItem(item)
        self._hover_items.clear()
        self._hover_line_idx = None
        # Reset map line colors
        for item in self._map_items:
            if item.data(TAG_TYPE) == "map_line":
                pen = item.pen()
                pen.setColor(C_MAP_LINE)
                item.setPen(pen)

    def _highlight_line(self, line_idx: int):
        if line_idx == self._hover_line_idx:
            return
        self._clear_hover()
        self._hover_line_idx = line_idx
        for item in self._map_items:
            if item.data(TAG_TYPE) == "map_line" and item.data(TAG_LINEID) == line_idx:
                pen = item.pen()
                pen.setColor(C_MAP_LINE_HOVER)
                pen.setWidthF(3.0)
                item.setPen(pen)

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        scene_pt = self.mapToScene(event.pos())
        wx, wy = self._scene_to_world(float(scene_pt.x()), float(scene_pt.y()))

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier
        ):
            # Pan start
            self._pan_start = event.pos()
            self._pan_transform_start = self.transform()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            # Check if clicking a frame dot (waypoint mode)
            if self._waypoint_mode:
                items_at = self._scene.items(scene_pt, Qt.IntersectsItemBoundingRect)
                for item in items_at:
                    if item.data(TAG_TYPE) == "frame_dot":
                        self._drag_frame_item = item
                        self._drag_actor_id = item.data(TAG_TRKID)
                        self._drag_frame_idx = item.data(TAG_FRAMEI)
                        self._drag_start_scene = scene_pt
                        track = self._scene_data.track_by_id.get(self._drag_actor_id)
                        if track and self._drag_frame_idx < len(track.frames):
                            f = track.frames[self._drag_frame_idx]
                            cx = f.cx if f.ccli >= 0 else f.x
                            cy = f.cy if f.ccli >= 0 else f.y
                            self._drag_original_xy = (cx, cy)
                        return

            if self._lane_snap_mode:
                # Find nearest CARLA line to click
                if self._scene_data:
                    line = find_nearest_line(
                        self._scene_data.carla_lines, wx, wy, max_dist=20.0,
                    )
                    if line and self._selected_id and self._patch_model:
                        t0, t1 = (self._selected_t_range or (None, None))
                        self._patch_model.cmd_lane_snap(
                            self._selected_id, str(line.idx), t0, t1,
                        )
                        self.lane_clicked.emit(line.idx, wx, wy)
                        self.status_message.emit(
                            f"Snapped {self._selected_id} → lane {line.idx} ({line.label})"
                        )
                        self._clear_hover()
                return

            # Hit-test actors
            hit = self._hit_test_actor(scene_pt)
            if hit:
                self.select_actor(hit)
            else:
                self.select_actor(None)

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click actor → enter ego-follow mode."""
        scene_pt = self.mapToScene(event.pos())
        hit = self._hit_test_actor(scene_pt)
        if hit:
            if self._follow_id == hit:
                # Second double-click exits follow mode
                self._exit_follow()
            else:
                self.set_follow_actor(hit)
                self.status_message.emit(
                    f"Following {hit} — Space=play/pause, Esc=exit follow"
                )
        super().mouseDoubleClickEvent(event)

    def _exit_follow(self):
        old = self._follow_id
        self._playing = False
        self._playback_timer.stop()
        self._follow_id = None
        if old and self._scene_data:
            track = self._scene_data.track_by_id.get(old)
            if track:
                self._draw_actor(track)
        self._fit_all()
        self.status_message.emit("Exited follow mode")

    def mouseMoveEvent(self, event: QMouseEvent):
        scene_pt = self.mapToScene(event.pos())
        wx, wy = self._scene_to_world(float(scene_pt.x()), float(scene_pt.y()))

        # Pan
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            t = QTransform(self._pan_transform_start)
            # Apply pan in viewport space
            new_t = QTransform(
                t.m11(), t.m12(), t.m13(),
                t.m21(), t.m22(), t.m23(),
                t.dx() + delta.x(), t.dy() + delta.y(), t.m33(),
            )
            self.setTransform(new_t)
            self._follow_id = None  # pan exits follow
            return

        # Waypoint drag
        if self._drag_frame_item is not None and self._drag_start_scene is not None:
            delta_s = scene_pt - self._drag_start_scene
            dsx, dsy = float(delta_s.x()), float(delta_s.y())
            dwx, dwy = (dsx, -dsy) if self._y_flip else (dsx, dsy)
            # Clamp to 5m
            mag = math.hypot(dwx, dwy)
            if mag > 5.0:
                dwx, dwy = dwx * 5.0 / mag, dwy * 5.0 / mag
            # Move dot visually
            if self._drag_original_xy:
                ox, oy = self._drag_original_xy
                nsx, nsy = self._world_to_scene(ox + dwx, oy + dwy)
                self._drag_frame_item.setPos(nsx - self._drag_frame_item.rect().x() - 3,
                                              nsy - self._drag_frame_item.rect().y() - 3)
            return

        # Lane snap hover
        if self._lane_snap_mode and self._scene_data:
            line = find_nearest_line(self._scene_data.carla_lines, wx, wy, max_dist=15.0)
            if line:
                self._highlight_line(line.idx)
            else:
                self._clear_hover()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._pan_start is not None:
            self._pan_start = None
            self._pan_transform_start = None
            self.setCursor(Qt.ArrowCursor if not self._lane_snap_mode else Qt.CrossCursor)
            return

        # Commit waypoint drag
        if self._drag_frame_item is not None and event.button() == Qt.LeftButton:
            if self._drag_start_scene is not None and self._drag_actor_id and self._patch_model:
                scene_pt = self.mapToScene(event.pos())
                delta_s = scene_pt - self._drag_start_scene
                dsx, dsy = float(delta_s.x()), float(delta_s.y())
                dwx, dwy = (dsx, -dsy) if self._y_flip else (dsx, dsy)
                mag = math.hypot(dwx, dwy)
                if mag > 5.0:
                    dwx, dwy = dwx * 5.0 / mag, dwy * 5.0 / mag
                if abs(dwx) > 0.01 or abs(dwy) > 0.01:
                    self._patch_model.cmd_waypoint_adjust(
                        self._drag_actor_id, self._drag_frame_idx, dwx, dwy,
                    )
            self._drag_frame_item = None
            self._drag_actor_id = None
            self._drag_frame_idx = -1
            self._drag_start_scene = None
            self._drag_original_xy = None

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self._follow_id:
            # Adjust zoom in follow mode
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            self._zoom = max(0.05, min(200.0, self._zoom * factor))
            self._update_ego_camera()
            return

        # Standard zoom-to-cursor
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        anchor = self.mapToScene(event.pos())
        t = self.transform()
        new_t = QTransform(
            t.m11() * factor, t.m12() * factor, t.m13(),
            t.m21() * factor, t.m22() * factor, t.m23(),
            t.dx() - anchor.x() * t.m11() * (factor - 1),
            t.dy() - anchor.y() * t.m22() * (factor - 1),
            t.m33(),
        )
        self.setTransform(new_t)
        self._zoom = abs(new_t.m11())

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Escape:
            if self._follow_id:
                self._exit_follow()
            elif self._lane_snap_mode:
                self.set_lane_snap_mode(False)
            elif self._waypoint_mode:
                self.set_waypoint_mode(False)
            else:
                self.select_actor(None)
            return

        if key == Qt.Key_Space and self._follow_id:
            self.toggle_playback()
            return

        if key == Qt.Key_F:
            if self._selected_id and not self._follow_id:
                self.set_follow_actor(self._selected_id)
            elif self._follow_id:
                self._exit_follow()
            elif self._selected_id:
                self._fit_to_actor(self._selected_id)
            else:
                self._fit_all()
            return

        if key == Qt.Key_A:
            self._fit_all()
            return

        # Playback speed
        if key == Qt.Key_BracketRight and self._follow_id:
            self._playback_speed = min(5.0, self._playback_speed * 1.5)
            self.status_message.emit(f"Speed: {self._playback_speed:.1f}x")
        if key == Qt.Key_BracketLeft and self._follow_id:
            self._playback_speed = max(0.1, self._playback_speed / 1.5)
            self.status_message.emit(f"Speed: {self._playback_speed:.1f}x")

        super().keyPressEvent(event)

    def _fit_to_actor(self, track_id: str):
        if not self._scene_data:
            return
        track = self._scene_data.track_by_id.get(track_id)
        if not track or not track.frames:
            return
        xs = [f.cx if f.ccli >= 0 else f.x for f in track.frames]
        ys = [f.cy if f.ccli >= 0 else f.y for f in track.frames]
        sx_list = [x for x in xs]
        sy_list = [-y if self._y_flip else y for y in ys]
        pad = 20.0
        rect = QRectF(
            min(sx_list) - pad, min(sy_list) - pad,
            max(sx_list) - min(sx_list) + 2 * pad,
            max(sy_list) - min(sy_list) + 2 * pad,
        )
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom = abs(self.transform().m11())

    def _hit_test_actor(self, scene_pt: QPointF) -> Optional[str]:
        """Find actor whose trajectory is closest to scene_pt (within ~8px)."""
        if not self._scene_data:
            return None
        px, py = float(scene_pt.x()), float(scene_pt.y())
        # pixel threshold in scene units (approx)
        threshold = 8.0 / max(abs(self.transform().m11()), 0.001)

        best_id = None
        best_d = float("inf")
        for track in self._scene_data.tracks:
            for f in track.frames:
                wx = f.cx if f.ccli >= 0 else f.x
                wy = f.cy if f.ccli >= 0 else f.y
                sx, sy = self._world_to_scene(wx, wy)
                d = math.hypot(sx - px, sy - py)
                if d < best_d and d < threshold:
                    best_d = d
                    best_id = track.track_id
        return best_id

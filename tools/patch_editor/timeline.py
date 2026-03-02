"""
timeline.py  —  Scrubber widget showing actor duration, CCLI changes,
                playback cursor, and draggable selection range.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QFontMetrics, QPainter, QPen, QMouseEvent,
)
from PyQt5.QtWidgets import QWidget

C_BG       = QColor(28, 28, 28)
C_RAIL     = QColor(55, 55, 55)
C_CURSOR   = QColor(255, 255, 255)
C_RANGE    = QColor(80, 140, 255, 70)
C_RANGE_BD = QColor(80, 140, 255, 200)
C_CCLI     = QColor(255, 102, 0)
C_TRACK    = QColor(0, 140, 200, 180)
C_TEXT     = QColor(180, 180, 180)
C_LABEL    = QColor(120, 120, 120)

RAIL_H = 14      # height of the main rail bar
MARGIN = 12      # left/right margin px
TOP_PAD = 22     # space for time labels above rail


class TimelineScrubber(QWidget):
    """
    Signals
    -------
    t_changed(t: float)      — user dragged the cursor
    range_changed(t0, t1)    — user dragged a selection range (or None,None to clear)
    """

    t_changed     = pyqtSignal(float)
    range_changed = pyqtSignal(object, object)  # float|None, float|None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setCursor(Qt.SizeHorCursor)

        self._t_min: float = 0.0
        self._t_max: float = 1.0
        self._current_t: float = 0.0
        self._ccli_changes: List[float] = []

        # Selection range
        self._range_start: Optional[float] = None
        self._range_end:   Optional[float] = None
        self._range_drag_start_px: Optional[int] = None
        self._range_drag_anchor_t: Optional[float] = None

        # Cursor drag
        self._cursor_dragging: bool = False

    def set_track(self, t_min: float, t_max: float, ccli_changes: List[float]):
        self._t_min = float(t_min)
        self._t_max = float(max(t_max, t_min + 0.001))
        self._current_t = float(t_min)
        self._ccli_changes = list(ccli_changes)
        self._range_start = None
        self._range_end = None
        self.update()

    def set_current_t(self, t: float):
        self._current_t = max(self._t_min, min(self._t_max, t))
        self.update()

    def get_range(self) -> Tuple[Optional[float], Optional[float]]:
        if self._range_start is None or self._range_end is None:
            return None, None
        a, b = sorted([self._range_start, self._range_end])
        return a, b

    def clear_range(self):
        self._range_start = None
        self._range_end = None
        self.range_changed.emit(None, None)
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _t_to_px(self, t: float) -> float:
        w = self.width() - 2 * MARGIN
        if self._t_max <= self._t_min:
            return float(MARGIN)
        frac = (t - self._t_min) / (self._t_max - self._t_min)
        return MARGIN + frac * w

    def _px_to_t(self, px: int) -> float:
        w = self.width() - 2 * MARGIN
        if w <= 0:
            return self._t_min
        frac = (px - MARGIN) / w
        return self._t_min + frac * (self._t_max - self._t_min)

    def _rail_y(self) -> int:
        return TOP_PAD

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        h = self.height()
        w = self.width()
        rail_y = self._rail_y()

        # Background
        p.fillRect(0, 0, w, h, C_BG)

        # Rail background
        p.fillRect(MARGIN, rail_y, w - 2 * MARGIN, RAIL_H, C_RAIL)

        # Selection range
        if self._range_start is not None and self._range_end is not None:
            a, b = sorted([self._range_start, self._range_end])
            xa = int(self._t_to_px(a))
            xb = int(self._t_to_px(b))
            p.fillRect(xa, rail_y, max(1, xb - xa), RAIL_H, C_RANGE)
            p.setPen(QPen(C_RANGE_BD, 1))
            p.drawLine(xa, rail_y, xa, rail_y + RAIL_H)
            p.drawLine(xb, rail_y, xb, rail_y + RAIL_H)

        # CCLI change markers
        p.setPen(QPen(C_CCLI, 1.5))
        for t_ch in self._ccli_changes:
            x = int(self._t_to_px(t_ch))
            p.drawLine(x, rail_y - 3, x, rail_y + RAIL_H + 3)

        # Track bar
        x0 = int(self._t_to_px(self._t_min))
        x1 = int(self._t_to_px(self._t_max))
        p.fillRect(x0, rail_y + 3, max(1, x1 - x0), RAIL_H - 6, C_TRACK)

        # Time labels
        p.setFont(QFont("Monospace", 8))
        p.setPen(QPen(C_LABEL))
        n_ticks = max(2, (w - 2 * MARGIN) // 60)
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            t_label = self._t_min + frac * (self._t_max - self._t_min)
            x_label = int(self._t_to_px(t_label))
            text = f"{t_label:.1f}s"
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(text)
            # clamp label to widget
            lx = max(0, min(w - tw, x_label - tw // 2))
            p.drawText(lx, rail_y - 4, text)
            # tick mark
            p.drawLine(x_label, rail_y, x_label, rail_y - 2)

        # Current-t cursor
        cx = int(self._t_to_px(self._current_t))
        p.setPen(QPen(C_CURSOR, 2))
        p.drawLine(cx, 0, cx, h)
        # cursor label
        p.setFont(QFont("Monospace", 8))
        p.setPen(QPen(C_TEXT))
        t_text = f"{self._current_t:.2f}s"
        fm = QFontMetrics(p.font())
        tw = fm.horizontalAdvance(t_text)
        lx = max(0, min(w - tw - 2, cx - tw // 2))
        p.drawText(lx, h - 4, t_text)

        # Range label
        if self._range_start is not None and self._range_end is not None:
            a, b = sorted([self._range_start, self._range_end])
            p.setPen(QPen(C_RANGE_BD))
            rtext = f"[{a:.2f} — {b:.2f}s]"
            p.drawText(MARGIN, h - 4, rtext)

        p.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            t = self._px_to_t(event.pos().x())
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+drag = range selection
                self._range_drag_start_px = event.pos().x()
                self._range_drag_anchor_t = t
                self._range_start = t
                self._range_end = t
            else:
                # Plain drag = cursor
                self._cursor_dragging = True
                self._range_start = None
                self._range_end = None
                self._current_t = max(self._t_min, min(self._t_max, t))
                self.t_changed.emit(self._current_t)
                self.range_changed.emit(None, None)
            self.update()

        elif event.button() == Qt.RightButton:
            self.clear_range()

    def mouseMoveEvent(self, event: QMouseEvent):
        t = self._px_to_t(event.pos().x())

        if self._range_drag_anchor_t is not None:
            self._range_start = self._range_drag_anchor_t
            self._range_end = t
            self.range_changed.emit(*sorted([self._range_start, self._range_end]))
            self.update()

        elif self._cursor_dragging:
            self._current_t = max(self._t_min, min(self._t_max, t))
            self.t_changed.emit(self._current_t)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._cursor_dragging = False
            self._range_drag_start_px = None
            self._range_drag_anchor_t = None

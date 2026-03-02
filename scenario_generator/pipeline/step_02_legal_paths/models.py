from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .geometry import cumulative_dist, heading_deg_from_vec


@dataclass
class LaneSegment:
    """Represents a continuous lane segment with geometry and metadata."""
    seg_id: int
    road_id: int
    lane_id: int
    section_id: int
    points: np.ndarray      # (N, 2) x,y coordinates
    yaws: np.ndarray        # (N,) heading at each point
    orig_idx: np.ndarray    # (N,) indices into original node file

    def bbox(self) -> Tuple[float, float, float, float]:
        """Return bounding box (xmin, xmax, ymin, ymax)."""
        mn = self.points.min(axis=0)
        mx = self.points.max(axis=0)
        return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1])

    def heading_at_start(self, k: int = 6) -> float:
        """Compute heading at start using first k+1 points."""
        k = min(k, len(self.points) - 1)
        if k == 0:
            return float(self.yaws[0])
        v = self.points[k] - self.points[0]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return float(self.yaws[0])
        return heading_deg_from_vec(v)

    def heading_at_end(self, k: int = 6) -> float:
        """Compute heading at end using last k+1 points."""
        k = min(k, len(self.points) - 1)
        if k == 0:
            return float(self.yaws[-1])
        v = self.points[-1] - self.points[-(k + 1)]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return float(self.yaws[-1])
        return heading_deg_from_vec(v)

    def length(self) -> float:
        """Total arc length of the segment."""
        return float(cumulative_dist(self.points)[-1])

    def is_straight(self, max_heading_change_deg: float = 30.0) -> bool:
        """
        Check if segment is approximately straight based on heading variation.
        
        A segment is considered straight if the total heading change from start
        to end is less than max_heading_change_deg.
        """
        if len(self.yaws) < 2:
            return True
        
        # Get heading at start and end
        start_heading = float(self.yaws[0])
        end_heading = float(self.yaws[-1])
        
        # Compute angular difference (handle wraparound)
        diff = abs(end_heading - start_heading)
        if diff > 180:
            diff = 360 - diff
        
        return diff < max_heading_change_deg


@dataclass
class CropBox:
    """Rectangular region for filtering segments."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, p: np.ndarray) -> bool:
        """Check if point is inside the crop box."""
        return (self.xmin <= float(p[0]) <= self.xmax and
                self.ymin <= float(p[1]) <= self.ymax)

    def intersects_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if this crop box intersects with another bounding box."""
        x0, x1, y0, y1 = bbox
        return not (x1 < self.xmin or x0 > self.xmax or
                    y1 < self.ymin or y0 > self.ymax)


@dataclass
class LegalPath:
    """A legal multi-segment path through the road network."""
    segments: List[Any]
    total_length: float

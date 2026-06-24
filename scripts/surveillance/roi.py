"""Region-of-interest helpers for zone-based event rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersection_area(self, other: NormalizedBox) -> float:
        x1 = max(self.x_min, other.x_min)
        y1 = max(self.y_min, other.y_min)
        x2 = min(self.x_max, other.x_max)
        y2 = min(self.y_max, other.y_max)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    def iou(self, other: NormalizedBox) -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        area_a = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        area_b = (other.x_max - other.x_min) * (other.y_max - other.y_min)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Zone:
    """Axis-aligned box or normalized polygon (points in 0..1)."""

    box: NormalizedBox | None = None
    points: tuple[tuple[float, float], ...] | None = None

    @property
    def is_polygon(self) -> bool:
        return self.points is not None and len(self.points) >= 3

    def bounding_box(self) -> NormalizedBox:
        if self.box is not None:
            return self.box
        if not self.points:
            raise ValueError("Zone has neither box nor points")
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return NormalizedBox(min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, x: float, y: float) -> bool:
        if self.points:
            return _point_in_polygon(x, y, self.points)
        if self.box is None:
            return False
        return self.box.contains_point(x, y)

    def hits_detection(self, det: dict, min_iou: float = 0.05) -> bool:
        det_box = bbox_from_detection(det)
        if det_box.iou(self.bounding_box()) < min_iou:
            cx, cy = bbox_center(det)
            if not self.contains_point(cx, cy):
                return False
        corners = (
            (det_box.x_min, det_box.y_min),
            (det_box.x_max, det_box.y_min),
            (det_box.x_min, det_box.y_max),
            (det_box.x_max, det_box.y_max),
        )
        for x, y in corners:
            if self.contains_point(x, y):
                return True
        return self.contains_point(*bbox_center(det))


def _point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / max(y2 - y1, 1e-9) + x1
        ):
            inside = not inside
    return inside


def parse_zone(raw: dict[str, Any]) -> Zone:
    if "points" in raw:
        points = tuple((float(p[0]), float(p[1])) for p in raw["points"])
        if len(points) < 3:
            raise ValueError("polygon zone requires at least 3 points")
        return Zone(points=points)
    return Zone(
        box=NormalizedBox(
            x_min=float(raw["x_min"]),
            y_min=float(raw["y_min"]),
            x_max=float(raw["x_max"]),
            y_max=float(raw["y_max"]),
        )
    )


def bbox_from_detection(det: dict) -> NormalizedBox:
    bb = det["bounding_box"]
    return NormalizedBox(
        x_min=float(bb["x_min"]),
        y_min=float(bb["y_min"]),
        x_max=float(bb["x_max"]),
        y_max=float(bb["y_max"]),
    )


def bbox_center(det: dict) -> tuple[float, float]:
    box = bbox_from_detection(det)
    return (box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0


def zone_hits(det: dict, zones: dict[str, Zone], min_iou: float = 0.05) -> set[str]:
    hits: set[str] = set()
    for name, zone in zones.items():
        if zone.hits_detection(det, min_iou=min_iou):
            hits.add(name)
    return hits
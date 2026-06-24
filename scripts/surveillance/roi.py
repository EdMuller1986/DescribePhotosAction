"""Region-of-interest helpers for zone-based event rules."""

from __future__ import annotations

from dataclasses import dataclass


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
    x_min: float | None = None
    y_min: float | None = None
    x_max: float | None = None
    y_max: float | None = None
    points: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def from_config(cls, raw: dict) -> Zone:
        if "points" in raw:
            pts = tuple((float(p[0]), float(p[1])) for p in raw["points"])
            return cls(points=pts)
        return cls(
            x_min=float(raw["x_min"]),
            y_min=float(raw["y_min"]),
            x_max=float(raw["x_max"]),
            y_max=float(raw["y_max"]),
        )

    def bounding_box(self) -> NormalizedBox:
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            return NormalizedBox(min(xs), min(ys), max(xs), max(ys))
        assert self.x_min is not None and self.y_min is not None
        assert self.x_max is not None and self.y_max is not None
        return NormalizedBox(self.x_min, self.y_min, self.x_max, self.y_max)

    def contains_point(self, x: float, y: float) -> bool:
        if self.points:
            return _point_in_polygon(x, y, self.points)
        box = self.bounding_box()
        return box.contains_point(x, y)

    def iou(self, other: NormalizedBox) -> float:
        return self.bounding_box().iou(other)


def _point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


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
    det_box = bbox_from_detection(det)
    center = bbox_center(det)
    hits: set[str] = set()
    for name, zone in zones.items():
        if zone.iou(det_box) >= min_iou or zone.contains_point(*center):
            hits.add(name)
    return hits
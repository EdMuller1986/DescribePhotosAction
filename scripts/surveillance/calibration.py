"""Generate surveillance calibration artifacts from a reference camera frame."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

DEFAULT_ZONES: dict[str, Any] = {
    "road": {
        "type": "polygon",
        "points": [
            [0.00, 0.36],
            [0.58, 0.40],
            [0.78, 0.26],
            [0.02, 0.26],
        ],
    },
    "gate": {"x_min": 0.36, "y_min": 0.38, "x_max": 0.58, "y_max": 0.52},
    "property": {"x_min": 0.12, "y_min": 0.48, "x_max": 0.98, "y_max": 0.98},
}

CALIBRATION_SKIP_IMAGE_NAMES = frozenset({"zones_preview.jpg", "ignore_mask.png"})

REFERENCE_IMAGE_PREFER = (
    "reference_day.jpg",
    "reference.jpg",
    "day.jpg",
    "dayframe.jpg",
)


def build_ignore_mask(shape: tuple[int, int, int]) -> np.ndarray:
    height, width = shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[:, : int(width * 0.28)] = 0
    cx1, cx2 = int(width * 0.47), int(width * 0.53)
    mask[: int(height * 0.55), cx1:cx2] = 0
    return mask


def draw_zones(image: np.ndarray, zones: dict[str, Any]) -> np.ndarray:
    out = image.copy()
    colors = {
        "road": (0, 180, 255),
        "gate": (0, 255, 0),
        "property": (255, 128, 0),
    }
    h, w = out.shape[:2]
    for name, zone in zones.items():
        color = colors.get(name, (255, 255, 255))
        if "points" in zone:
            pts = np.array(
                [[int(p[0] * w), int(p[1] * h)] for p in zone["points"]],
                dtype=np.int32,
            )
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(
                out,
                name,
                (int(pts[0][0]) + 4, max(20, int(pts[0][1]) + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        else:
            x1 = int(zone["x_min"] * w)
            y1 = int(zone["y_min"] * h)
            x2 = int(zone["x_max"] * w)
            y2 = int(zone["y_max"] * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, name, (x1 + 4, max(20, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def build_surveillance_config(
    zones: dict[str, Any] | None = None,
    *,
    gdrive_config_dir: str = "config",
    ignore_mask_file: str = "ignore_mask.png",
    input_dir: str = "",
) -> dict[str, Any]:
    return {
        "gdrive_config_dir": gdrive_config_dir,
        "ignore_mask_file": ignore_mask_file,
        "input_dir": input_dir,
        "output_dir": "",
        "local_model_path": "yolo11n.pt",
        "yolo_confidence": 0.25,
        "yolo_device": "cpu",
        "video_frame_interval_seconds": 1.0,
        "zones": zones or DEFAULT_ZONES,
        "motion": {
            "min_duration_sec": 0.9,
            "merge_gap_sec": 2.0,
            "min_displacement_ratio": 0.18,
            "max_oscillation_ratio": 6.5,
            "sample_every_n_frames": 2,
            "downscale_width": 960,
        },
    }


def _encode_image(image: np.ndarray, ext: str) -> bytes:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {ext}")
    return buf.tobytes()


def calibrate_from_image(image: np.ndarray, *, input_dir: str = "") -> dict[str, bytes]:
    """Return calibration artifacts as {filename: bytes}."""
    ignore_mask = build_ignore_mask(image.shape)
    zones = DEFAULT_ZONES
    preview = draw_zones(image, zones)
    config = build_surveillance_config(zones, input_dir=input_dir)

    return {
        "ignore_mask.png": _encode_image(ignore_mask, ".png"),
        "zones_preview.jpg": _encode_image(preview, ".jpg"),
        "reference_day.jpg": _encode_image(image, ".jpg"),
        "surveillance.json": json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8"),
    }
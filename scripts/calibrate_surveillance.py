#!/usr/bin/env python3
"""Generate ignore mask, zones preview and local config from a reference camera frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Calibrated for IPC 1280x720 elevated yard camera (see Documents/cam samples).
DEFAULT_ZONES = {
    "road": {"x_min": 0.00, "y_min": 0.28, "x_max": 0.72, "y_max": 0.50},
    "gate": {"x_min": 0.36, "y_min": 0.38, "x_max": 0.58, "y_max": 0.52},
    "property": {"x_min": 0.12, "y_min": 0.48, "x_max": 0.98, "y_max": 0.98},
}


def build_ignore_mask(shape: tuple[int, int, int]) -> np.ndarray:
    height, width = shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    # Left foliage — main wind false-positive source.
    mask[:, : int(width * 0.28)] = 0
    # Hanging lights pole near center-top.
    cx1, cx2 = int(width * 0.47), int(width * 0.53)
    mask[: int(height * 0.55), cx1:cx2] = 0
    return mask


def draw_zones(image: np.ndarray, zones: dict[str, dict[str, float]]) -> np.ndarray:
    out = image.copy()
    colors = {
        "road": (0, 180, 255),
        "gate": (0, 255, 0),
        "property": (255, 128, 0),
    }
    h, w = out.shape[:2]
    for name, zone in zones.items():
        x1 = int(zone["x_min"] * w)
        y1 = int(zone["y_min"] * h)
        x2 = int(zone["x_max"] * w)
        y2 = int(zone["y_max"] * h)
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, name, (x1 + 4, max(20, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate surveillance zones and ignore mask")
    parser.add_argument("--reference-image", required=True, help="Daytime reference frame (jpg/png)")
    parser.add_argument("--output-dir", required=True, help="Directory for mask, preview and config")
    parser.add_argument("--input-dir", default="", help="Video input directory for local runs")
    args = parser.parse_args()

    image_path = Path(args.reference_image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    ignore_mask = build_ignore_mask(image.shape)
    zones = DEFAULT_ZONES

    mask_path = output_dir / "ignore_mask.png"
    preview_path = output_dir / "zones_preview.jpg"
    config_path = output_dir / "surveillance.json"

    cv2.imwrite(str(mask_path), ignore_mask)
    cv2.imwrite(str(preview_path), draw_zones(image, zones))

    config: dict[str, Any] = {
        "input_dir": args.input_dir or str(image_path.parent),
        "output_dir": str(output_dir / "summaries"),
        "ignore_mask_path": str(mask_path),
        "local_model_path": "yolo11n.pt",
        "yolo_confidence": 0.25,
        "yolo_device": "cpu",
        "video_frame_interval_seconds": 1.0,
        "zones": zones,
        "motion": {
            "min_duration_sec": 0.9,
            "merge_gap_sec": 2.0,
            "min_displacement_ratio": 0.18,
            "max_oscillation_ratio": 6.5,
            "sample_every_n_frames": 2,
            "downscale_width": 960,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"written: {mask_path}")
    print(f"written: {preview_path}")
    print(f"written: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
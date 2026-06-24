#!/usr/bin/env python3
"""Generate ignore mask, zones preview and config from a reference camera frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from surveillance.calibration import calibrate_from_image


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

    for filename, data in calibrate_from_image(image, input_dir=args.input_dir).items():
        out_path = output_dir / filename
        out_path.write_bytes(data)
        print(f"written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
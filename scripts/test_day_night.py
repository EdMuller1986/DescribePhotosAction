#!/usr/bin/env python3
"""Quick local check: classify day/night frames and transitions in sample videos."""

from __future__ import annotations

import argparse
import glob
import os
import sys

from surveillance.day_night import detect_day_night_from_video, is_night_frame

import cv2


def classify_image(path: str) -> None:
    image = cv2.imread(path)
    if image is None:
        print(f"skip: cannot read {path}")
        return
    mode = "night" if is_night_frame(image) else "day"
    print(f"{os.path.basename(path)}: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-glob", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    args = parser.parse_args()

    if args.images_glob:
        for path in sorted(glob.glob(args.images_glob)):
            classify_image(path)

    if args.video:
        result = detect_day_night_from_video(
            args.video,
            day_label="clip",
            sample_interval_sec=args.sample_interval_sec,
        )
        print(
            f"{os.path.basename(args.video)}: "
            f"dawn={result.dawn} dusk={result.dusk} "
            f"night_ratio={result.night_sample_ratio:.2f}"
        )
    if not args.images_glob and not args.video:
        print("Provide --images-glob or --video", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Daily surveillance summary: motion filter -> YOLO on segments -> event timeline.

Example:
  python scripts/analyze_surveillance_day.py \
    --video /home/gedonist/gdrive/camera/20260622.mkv \
    --config config/surveillance.example.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from analyze_ftp_photos import Settings, extract_detections, run_video_frame
from surveillance.events import infer_events
from surveillance.motion_detector import MotionDetectorConfig, MotionSegment, find_motion_segments
from surveillance.roi import Zone, parse_zone
from surveillance.day_night import detect_day_night_from_video
from surveillance.summary import build_summary_json, build_summary_text
from surveillance.sun_times import parse_video_day


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_zones(config: dict[str, Any]) -> dict[str, Zone]:
    return {name: parse_zone(raw) for name, raw in (config.get("zones") or {}).items()}


def motion_config_from_json(raw: dict[str, Any] | None) -> MotionDetectorConfig:
    raw = raw or {}
    return MotionDetectorConfig(
        min_duration_sec=float(raw.get("min_duration_sec", 0.8)),
        merge_gap_sec=float(raw.get("merge_gap_sec", 2.0)),
        min_displacement_ratio=float(raw.get("min_displacement_ratio", 0.15)),
        max_oscillation_ratio=float(raw.get("max_oscillation_ratio", 8.0)),
        sample_every_n_frames=int(raw.get("sample_every_n_frames", 2)),
        downscale_width=int(raw.get("downscale_width", 960)),
    )


def frame_in_segments(frame_index: int, segments: list[MotionSegment]) -> bool:
    return any(seg.start_frame <= frame_index <= seg.end_frame for seg in segments)


def yolo_settings_from_config(config: dict[str, Any]) -> Settings:
    return Settings(
        ftp_url="local",
        ftp_user="local",
        ftp_pass="local",
        scan_dir=".",
        ftp_protocol="local",
        ftp_port="",
        model_path=str(config.get("local_model_path", "yolo11n.pt")),
        yolo_confidence=float(config.get("yolo_confidence", 0.25)),
        yolo_iou=0.7,
        yolo_image_size=1280,
        yolo_device=str(config.get("yolo_device", "cpu")),
        max_image_edge=1600,
        max_detections=300,
        process_images=False,
        process_videos=True,
        save_yolo_txt=False,
        save_empty_yolo_txt=False,
        create_image_boxes_preview=False,
        enable_video_tracking=True,
        video_tracker="bytetrack",
        video_frame_interval_seconds=float(config.get("video_frame_interval_seconds", 1.0)),
        video_max_frames_per_file=0,
        save_track_paths=True,
        track_path_max_points=1000,
        process_missing_side_outputs=False,
        force_reprocess=True,
        max_files_per_run=0,
        json_indent=2,
        fail_on_errors=False,
    )


def analyze_motion_segments(
    video_path: str,
    model: YOLO,
    segments: list[MotionSegment],
    settings: Settings,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    frame_step = max(1, int(round(fps * max(settings.video_frame_interval_seconds, 0.1))))
    frames_out: list[dict[str, Any]] = []
    frame_index = 0

    try:
        model.predictor = None
    except Exception:
        pass

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0 or not frame_in_segments(frame_index, segments):
            frame_index += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        timestamp_sec = frame_index / fps

        result = run_video_frame(model, frame_rgb, settings)
        detections = extract_detections(result, model.names, width, height)
        frames_out.append(
            {
                "frame_number": frame_index,
                "timestamp_sec": round(timestamp_sec, 3),
                "detections": detections,
            }
        )
        frame_index += 1

    cap.release()
    return frames_out


def segments_to_json(segments: list[MotionSegment]) -> list[dict[str, Any]]:
    return [
        {
            "start_sec": round(seg.start_sec, 3),
            "end_sec": round(seg.end_sec, 3),
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "duration_sec": round(seg.duration_sec, 3),
            "peak_area_ratio": round(seg.peak_area_ratio, 6),
            "net_displacement": round(seg.net_displacement, 6),
            "oscillation_ratio": round(seg.oscillation_ratio, 6),
        }
        for seg in segments
    ]


def resolve_video(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.video:
        return args.video
    input_dir = args.input_dir or config.get("input_dir")
    if not input_dir:
        raise SystemExit("Provide --video or input_dir in config")
    day = args.date
    if not day:
        raise SystemExit("Provide --video or --date with input_dir")
    patterns = [f"*{day}*.mkv", f"*{day}*.mp4", f"*{day}*.avi"]
    for pattern in patterns:
        matches = sorted(Path(input_dir).glob(pattern))
        if matches:
            return str(matches[0])
    raise SystemExit(f"No video found for date {day} in {input_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build surveillance day summary from camera video")
    parser.add_argument("--video", help="Path to daily video file")
    parser.add_argument("--config", required=True, help="Path to surveillance JSON config")
    parser.add_argument("--input-dir", help="Override config input_dir")
    parser.add_argument("--output-dir", help="Override config output_dir")
    parser.add_argument("--date", help="Day in YYYYMMDD format when using input_dir lookup")
    parser.add_argument("--ignore-mask", help="Override ignore mask image path")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = resolve_video(args, config)
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    output_dir = args.output_dir or config.get("output_dir") or str(Path(video_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    day = parse_video_day(video_path, fallback=date.today())
    zones = load_zones(config)
    motion_cfg = motion_config_from_json(config.get("motion"))
    ignore_mask = args.ignore_mask or config.get("ignore_mask_path")
    if not ignore_mask:
        mask_file = config.get("ignore_mask_file")
        if mask_file:
            ignore_mask = str(Path(args.config).resolve().parent / mask_file)
    day_night_interval = float(config.get("day_night_sample_interval_sec", 60))

    print(f"video: {video_path}")
    print(f"day: {day.isoformat()}")
    print("step 1/4: day/night from camera color mode")
    day_night = detect_day_night_from_video(
        video_path,
        day_label=day.isoformat(),
        sample_interval_sec=day_night_interval,
    )
    print(
        "day_night: "
        f"dawn={day_night.dawn} dusk={day_night.dusk} "
        f"night_ratio={day_night.night_sample_ratio:.2f}"
    )

    print("step 2/4: motion detection")
    segments, motion_stats = find_motion_segments(video_path, motion_cfg, ignore_mask)
    print(
        "motion: "
        f"segments={int(motion_stats['segments_merged'])} "
        f"rejected_wind={int(motion_stats['segments_rejected_wind'])}"
    )
    if not segments:
        print("warning: no motion segments after filtering", file=sys.stderr)

    print("step 3/4: YOLO on motion segments")
    yolo_settings = yolo_settings_from_config(config)
    model = YOLO(yolo_settings.model_path)
    frames = analyze_motion_segments(
        video_path=video_path,
        model=model,
        segments=segments,
        settings=yolo_settings,
    )
    print(f"yolo frames analyzed: {len(frames)}")

    print("step 4/4: event inference and summary")
    events = infer_events(frames, zones)
    summary_json = build_summary_json(
        video_path=video_path,
        day_night=day_night,
        events=events,
        motion_stats=motion_stats,
        segments=segments_to_json(segments),
    )

    stem = Path(video_path).stem
    json_path = os.path.join(output_dir, f"{stem}.summary.json")
    txt_path = os.path.join(output_dir, f"{stem}.summary.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(build_summary_text(day_night, events, motion_stats))

    print(f"written: {json_path}")
    print(f"written: {txt_path}")
    print()
    print(build_summary_text(day_night, events, motion_stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
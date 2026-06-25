"""Video scan helpers with frame skipping and progress logging."""

from __future__ import annotations

import sys

import cv2


def video_stats(cap: cv2.VideoCapture) -> tuple[float, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, total_frames


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log_scan_progress(
    label: str,
    frame_index: int,
    total_frames: int,
    fps: float,
    *,
    last_logged_pct: int,
    step_pct: int = 10,
) -> int:
    if total_frames <= 0:
        if frame_index > 0 and frame_index % max(1, int(fps * 300)) == 0:
            ts = format_duration(frame_index / fps)
            print(f"{label}: scanned {ts} of video", flush=True)
        return last_logged_pct

    pct = min(100, int(frame_index * 100 / total_frames))
    if pct < last_logged_pct + step_pct and pct < 100:
        return last_logged_pct
    ts = format_duration(frame_index / fps)
    total_ts = format_duration(total_frames / fps)
    print(f"{label}: {pct}% ({ts} / {total_ts})", flush=True)
    return pct


def advance_frame(cap: cv2.VideoCapture, frame_index: int, *, decode: bool) -> tuple[bool, object | None]:
    if decode:
        ok, frame = cap.read()
        return ok, frame
    return cap.grab(), None
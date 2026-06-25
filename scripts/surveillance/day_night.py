"""Detect dawn/dusk from camera color mode (IR/grayscale night vs color day)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from surveillance.video_io import advance_frame, log_scan_progress, video_stats


@dataclass(frozen=True)
class DayNightTimes:
    date: str
    method: str
    dawn: str | None
    dusk: str | None
    night_sample_ratio: float
    color_sample_ratio: float


def _analysis_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    mask[0 : max(1, int(height * 0.12)), max(0, int(width * 0.70)) :] = False
    mask[max(0, int(height * 0.90)) :, : max(1, int(width * 0.18))] = False
    return mask


def is_night_frame(frame_bgr: np.ndarray, saturation_threshold: float = 18.0) -> bool:
    """True when the camera is in IR/grayscale night mode."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = _analysis_mask(frame_bgr.shape[0], frame_bgr.shape[1])
    sat = hsv[:, :, 1][mask]
    if sat.size == 0:
        return True

    mean_sat = float(np.mean(sat))
    b, g, r = cv2.split(frame_bgr)
    rb = r[mask].astype(np.float32)
    gb = g[mask].astype(np.float32)
    bb = b[mask].astype(np.float32)
    channel_spread = float(
        max(
            np.mean(np.abs(rb - gb)),
            np.mean(np.abs(rb - bb)),
            np.mean(np.abs(gb - bb)),
        )
    )
    return mean_sat < saturation_threshold and channel_spread < 12.0


def _seconds_to_hhmm(seconds: float) -> str:
    total = int(max(0.0, seconds)) % 86_400
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _smooth_labels(labels: list[bool], window: int = 5) -> list[bool]:
    if not labels:
        return []
    radius = max(1, window // 2)
    out: list[bool] = []
    for i in range(len(labels)):
        lo = max(0, i - radius)
        hi = min(len(labels), i + radius + 1)
        chunk = labels[lo:hi]
        out.append(sum(chunk) > len(chunk) / 2)
    return out


def detect_day_night_from_video(
    video_path: str,
    day_label: str,
    sample_interval_sec: float = 60.0,
    min_transition_samples: int = 2,
) -> DayNightTimes:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps, total_frames = video_stats(cap)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    step = max(1, int(round(fps * max(sample_interval_sec, 5.0))))

    sample_times: list[float] = []
    sample_night: list[bool] = []
    frame_index = 0
    last_logged_pct = -1
    while True:
        if total_frames > 0 and frame_index >= total_frames:
            break
        decode = frame_index % step == 0
        ok, frame = advance_frame(cap, frame_index, decode=decode)
        if not ok:
            break
        if decode and frame is not None:
            ts = frame_index / fps
            sample_times.append(ts)
            sample_night.append(is_night_frame(frame))
        last_logged_pct = log_scan_progress(
            "day_night scan",
            frame_index + 1,
            total_frames,
            fps,
            last_logged_pct=last_logged_pct,
        )
        frame_index += 1
    cap.release()

    if not sample_times:
        return DayNightTimes(
            date=day_label,
            method="camera_color_mode",
            dawn=None,
            dusk=None,
            night_sample_ratio=1.0,
            color_sample_ratio=0.0,
        )

    night_flags = _smooth_labels(sample_night, window=5)
    night_count = sum(1 for flag in night_flags if flag)
    night_ratio = night_count / len(night_flags)

    dawn: str | None = None
    dusk: str | None = None

    run = 0
    for idx, is_night in enumerate(night_flags):
        if not is_night:
            run += 1
            if run >= min_transition_samples and dawn is None:
                dawn_idx = idx - min_transition_samples + 1
                dawn = _seconds_to_hhmm(sample_times[dawn_idx])
        else:
            run = 0

    run = 0
    last_color_idx: int | None = None
    for idx, is_night in enumerate(night_flags):
        if not is_night:
            run += 1
            last_color_idx = idx
        else:
            if run >= min_transition_samples and last_color_idx is not None:
                dusk = _seconds_to_hhmm(sample_times[last_color_idx])
            run = 0

    if duration_sec > 0 and night_flags and night_flags[-1] and last_color_idx is not None:
        dusk = _seconds_to_hhmm(sample_times[last_color_idx])

    return DayNightTimes(
        date=day_label,
        method="camera_color_mode",
        dawn=dawn,
        dusk=dusk,
        night_sample_ratio=round(night_ratio, 4),
        color_sample_ratio=round(1.0 - night_ratio, 4),
    )
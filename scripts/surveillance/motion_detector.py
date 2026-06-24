"""Motion segmentation with heuristics to reject wind, branches and laundry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionDetectorConfig:
    min_area_ratio: float = 0.0008
    max_area_ratio: float = 0.35
    min_duration_sec: float = 0.8
    merge_gap_sec: float = 2.0
    min_displacement_ratio: float = 0.15
    max_oscillation_ratio: float = 8.0
    mog2_history: int = 500
    mog2_var_threshold: float = 32.0
    mog2_detect_shadows: bool = False
    downscale_width: int = 960
    sample_every_n_frames: int = 2


@dataclass
class MotionBlobSample:
    frame_index: int
    timestamp_sec: float
    centroid_x: float
    centroid_y: float
    area_ratio: float
    bbox_aspect: float


@dataclass
class MotionSegment:
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    peak_area_ratio: float = 0.0
    net_displacement: float = 0.0
    path_length: float = 0.0
    oscillation_ratio: float = 0.0
    samples: list[MotionBlobSample] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _load_ignore_mask(path: str | None, width: int, height: int) -> np.ndarray | None:
    if not path:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read ignore mask: {path}")
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask < 128).astype(np.uint8)


def _resize_frame(frame_bgr: np.ndarray, target_width: int) -> tuple[np.ndarray, float]:
    height, width = frame_bgr.shape[:2]
    if width <= target_width:
        return frame_bgr, 1.0
    scale = target_width / width
    resized = cv2.resize(frame_bgr, (target_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _largest_motion_blob(mask: np.ndarray) -> tuple[float, tuple[int, int, int, int]] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area <= 1.0:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return area, (x, y, w, h)


def _segment_metrics(samples: list[MotionBlobSample], frame_area: float) -> tuple[float, float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    peak_area_ratio = max(s.area_ratio for s in samples)
    if len(samples) < 2:
        return peak_area_ratio, 0.0, 0.0, 0.0

    start = samples[0]
    end = samples[-1]
    net = math.hypot(end.centroid_x - start.centroid_x, end.centroid_y - start.centroid_y)
    path = 0.0
    aspects: list[float] = []
    for prev, cur in zip(samples, samples[1:]):
        path += math.hypot(cur.centroid_x - prev.centroid_x, cur.centroid_y - prev.centroid_y)
        aspects.append(cur.bbox_aspect)
    aspect_std = float(np.std(aspects)) if aspects else 0.0
    oscillation = path / max(net, 1.0)
    return peak_area_ratio, net / math.sqrt(frame_area), path / math.sqrt(frame_area), oscillation + aspect_std
    # aspect_std nudges windy laundry (flapping aspect) upward


def _accept_segment(segment: MotionSegment, cfg: MotionDetectorConfig) -> bool:
    if segment.duration_sec < cfg.min_duration_sec:
        return False
    if segment.peak_area_ratio < cfg.min_area_ratio or segment.peak_area_ratio > cfg.max_area_ratio:
        return False
    if segment.net_displacement < cfg.min_displacement_ratio:
        return False
    if segment.oscillation_ratio > cfg.max_oscillation_ratio:
        return False
    return True


def _merge_segments(segments: list[MotionSegment], gap_sec: float) -> list[MotionSegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start_sec)
    merged: list[MotionSegment] = [ordered[0]]
    for seg in ordered[1:]:
        prev = merged[-1]
        if seg.start_sec - prev.end_sec <= gap_sec:
            prev.end_sec = max(prev.end_sec, seg.end_sec)
            prev.end_frame = max(prev.end_frame, seg.end_frame)
            prev.samples.extend(seg.samples)
            peak, net, path, osc = _segment_metrics(prev.samples, 1.0)
            prev.peak_area_ratio = max(prev.peak_area_ratio, peak)
            prev.net_displacement = max(prev.net_displacement, net)
            prev.path_length = max(prev.path_length, path)
            prev.oscillation_ratio = max(prev.oscillation_ratio, osc)
        else:
            merged.append(seg)
    return merged


def find_motion_segments(
    video_path: str,
    cfg: MotionDetectorConfig | None = None,
    ignore_mask_path: str | None = None,
) -> tuple[list[MotionSegment], dict[str, float]]:
    cfg = cfg or MotionDetectorConfig()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ignore_mask = _load_ignore_mask(ignore_mask_path, width, height)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=cfg.mog2_history,
        varThreshold=cfg.mog2_var_threshold,
        detectShadows=cfg.mog2_detect_shadows,
    )

    active_samples: list[MotionBlobSample] = []
    segments: list[MotionSegment] = []
    frame_index = 0
    processed = 0
    rejected_wind = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_index % cfg.sample_every_n_frames != 0:
            frame_index += 1
            continue

        resized, scale = _resize_frame(frame_bgr, cfg.downscale_width)
        fg = subtractor.apply(resized)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        if ignore_mask is not None:
            mask_small = cv2.resize(ignore_mask, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_NEAREST)
            fg = cv2.bitwise_and(fg, fg, mask=mask_small)

        frame_area = float(fg.shape[0] * fg.shape[1])
        blob = _largest_motion_blob(fg)
        timestamp_sec = frame_index / fps

        if blob:
            area, (x, y, w, h) = blob
            area_ratio = area / frame_area
            cx = (x + w / 2.0) / fg.shape[1]
            cy = (y + h / 2.0) / fg.shape[0]
            aspect = w / max(h, 1)
            active_samples.append(
                MotionBlobSample(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    centroid_x=cx,
                    centroid_y=cy,
                    area_ratio=area_ratio,
                    bbox_aspect=aspect,
                )
            )
        elif active_samples:
            peak, net_disp, path_len, osc = _segment_metrics(active_samples, frame_area)
            segment = MotionSegment(
                start_sec=active_samples[0].timestamp_sec,
                end_sec=active_samples[-1].timestamp_sec,
                start_frame=active_samples[0].frame_index,
                end_frame=active_samples[-1].frame_index,
                peak_area_ratio=peak,
                net_displacement=net_disp,
                path_length=path_len,
                oscillation_ratio=osc,
                samples=list(active_samples),
            )
            if _accept_segment(segment, cfg):
                segments.append(segment)
            else:
                rejected_wind += 1
            active_samples = []

        processed += 1
        frame_index += 1

    if active_samples:
        peak, net_disp, path_len, osc = _segment_metrics(active_samples, 1.0)
        segment = MotionSegment(
            start_sec=active_samples[0].timestamp_sec,
            end_sec=active_samples[-1].timestamp_sec,
            start_frame=active_samples[0].frame_index,
            end_frame=active_samples[-1].frame_index,
            peak_area_ratio=peak,
            net_displacement=net_disp,
            path_length=path_len,
            oscillation_ratio=osc,
            samples=list(active_samples),
        )
        if _accept_segment(segment, cfg):
            segments.append(segment)
        else:
            rejected_wind += 1

    cap.release()
    merged = _merge_segments(segments, cfg.merge_gap_sec)
    stats = {
        "fps": fps,
        "frames_scanned": float(frame_index),
        "frames_sampled": float(processed),
        "segments_raw": float(len(segments)),
        "segments_merged": float(len(merged)),
        "segments_rejected_wind": float(rejected_wind),
    }
    return merged, stats
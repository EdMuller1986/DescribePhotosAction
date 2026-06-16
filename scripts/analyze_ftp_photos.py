#!/usr/bin/env python3
"""Analyze images and videos from FTP/FTPS/SFTP with a local YOLO model.

Behavior:
- scan FTP_PHOTO_DIR recursively;
- process image and video files that do not have a sibling JSON result;
- optionally reprocess files with JSON when requested side outputs are missing;
- write JSON results next to the source media;
- write YOLO TXT labels in the original YOLO detection format;
- optionally write *.boxes.jpg previews for images only;
- use ByteTrack/BoT-SORT tracking for videos when enabled.

No OpenAI API or external vision API is used.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import posixpath
import re
import stat
import sys
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from ftplib import FTP, FTP_TLS, error_perm
from typing import Any, Iterable
from urllib.parse import urlparse

import cv2
import paramiko
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}

DEFAULT_SCAN_DIR = "photo"
DEFAULT_MODEL_PATH = "yolo11n.pt"
DEFAULT_MAX_IMAGE_EDGE = 1600
DEFAULT_YOLO_IMAGE_SIZE = 1280
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.70
DEFAULT_DEVICE = "cpu"

COCO_LABEL_RU = {
    "person": "человек",
    "bicycle": "велосипед",
    "car": "автомобиль",
    "motorcycle": "мотоцикл",
    "airplane": "самолёт",
    "bus": "автобус",
    "train": "поезд",
    "truck": "грузовик",
    "boat": "лодка",
    "traffic light": "светофор",
    "fire hydrant": "пожарный гидрант",
    "stop sign": "знак стоп",
    "parking meter": "паркомат",
    "bench": "скамейка",
    "bird": "птица",
    "cat": "кошка",
    "dog": "собака",
    "horse": "лошадь",
    "sheep": "овца",
    "cow": "корова",
    "elephant": "слон",
    "bear": "медведь",
    "zebra": "зебра",
    "giraffe": "жираф",
    "backpack": "рюкзак",
    "umbrella": "зонт",
    "handbag": "сумка",
    "tie": "галстук",
    "suitcase": "чемодан",
    "frisbee": "фрисби",
    "skis": "лыжи",
    "snowboard": "сноуборд",
    "sports ball": "мяч",
    "kite": "воздушный змей",
    "baseball bat": "бейсбольная бита",
    "baseball glove": "бейсбольная перчатка",
    "skateboard": "скейтборд",
    "surfboard": "сёрфборд",
    "tennis racket": "теннисная ракетка",
    "bottle": "бутылка",
    "wine glass": "бокал",
    "cup": "чашка",
    "fork": "вилка",
    "knife": "нож",
    "spoon": "ложка",
    "bowl": "миска",
    "banana": "банан",
    "apple": "яблоко",
    "sandwich": "сэндвич",
    "orange": "апельсин",
    "broccoli": "брокколи",
    "carrot": "морковь",
    "hot dog": "хот-дог",
    "pizza": "пицца",
    "donut": "пончик",
    "cake": "торт",
    "chair": "стул",
    "couch": "диван",
    "potted plant": "растение в горшке",
    "bed": "кровать",
    "dining table": "обеденный стол",
    "toilet": "туалет",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "mouse": "мышь",
    "remote": "пульт",
    "keyboard": "клавиатура",
    "cell phone": "телефон",
    "microwave": "микроволновка",
    "oven": "духовка",
    "toaster": "тостер",
    "sink": "раковина",
    "refrigerator": "холодильник",
    "book": "книга",
    "clock": "часы",
    "vase": "ваза",
    "scissors": "ножницы",
    "teddy bear": "плюшевый медведь",
    "hair drier": "фен",
    "toothbrush": "зубная щётка",
}


@dataclass(frozen=True)
class Settings:
    ftp_url: str
    ftp_user: str
    ftp_pass: str
    scan_dir: str
    ftp_protocol: str
    ftp_port: str
    model_path: str
    yolo_confidence: float
    yolo_iou: float
    yolo_image_size: int
    yolo_device: str
    max_image_edge: int
    max_detections: int
    process_images: bool
    process_videos: bool
    save_yolo_txt: bool
    save_empty_yolo_txt: bool
    create_image_boxes_preview: bool
    enable_video_tracking: bool
    video_tracker: str
    video_frame_interval_seconds: float
    video_max_frames_per_file: int
    save_track_paths: bool
    track_path_max_points: int
    process_missing_side_outputs: bool
    force_reprocess: bool
    max_files_per_run: int
    json_indent: int | None
    fail_on_errors: bool


class RemoteClient:
    def list_files(self, root: str) -> list[str]:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def download_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def upload_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FtpClient(RemoteClient):
    def __init__(self, host: str, port: int, username: str, password: str, use_tls: bool) -> None:
        self.ftp: FTP | FTP_TLS
        self.ftp = FTP_TLS() if use_tls else FTP()
        self.ftp.connect(host, port, timeout=60)
        self.ftp.login(username, password)
        if use_tls and isinstance(self.ftp, FTP_TLS):
            self.ftp.prot_p()
        try:
            self.ftp.voidcmd("TYPE I")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.ftp.quit()
        except Exception:
            try:
                self.ftp.close()
            except Exception:
                pass

    def exists(self, path: str) -> bool:
        path = normalize_remote_path(path)
        try:
            self.ftp.size(path)
            return True
        except Exception:
            pass
        try:
            current = self.ftp.pwd()
            self.ftp.cwd(path)
            self.ftp.cwd(current)
            return True
        except Exception:
            try:
                self.ftp.cwd("/")
            except Exception:
                pass
            return False

    def download_bytes(self, path: str) -> bytes:
        buf = io.BytesIO()
        self.ftp.retrbinary(f"RETR {normalize_remote_path(path)}", buf.write)
        return buf.getvalue()

    def upload_bytes(self, path: str, data: bytes) -> None:
        path = normalize_remote_path(path)
        parent = posixpath.dirname(path)
        if parent and parent != ".":
            self._makedirs(parent)
        self.ftp.storbinary(f"STOR {path}", io.BytesIO(data))

    def _makedirs(self, directory: str) -> None:
        directory = normalize_remote_path(directory)
        if not directory or directory == ".":
            return
        parts = [p for p in directory.split("/") if p]
        current = ""
        for part in parts:
            current = part if not current else f"{current}/{part}"
            try:
                self.ftp.mkd(current)
            except error_perm:
                pass
            except Exception:
                pass

    def list_files(self, root: str) -> list[str]:
        root = normalize_remote_path(root)
        out: list[str] = []
        self._walk(root, out)
        return sorted(set(out))

    def _walk(self, directory: str, out: list[str]) -> None:
        if self._walk_mlsd(directory, out):
            return
        self._walk_nlst(directory, out)

    def _walk_mlsd(self, directory: str, out: list[str]) -> bool:
        try:
            entries = list(self.ftp.mlsd(directory))
        except Exception:
            return False
        for name, facts in entries:
            if name in {".", ".."}:
                continue
            path = posixpath.join(directory, name)
            entry_type = facts.get("type", "").lower()
            if entry_type == "dir":
                self._walk(path, out)
            elif entry_type == "file":
                out.append(normalize_remote_path(path))
        return True

    def _walk_nlst(self, directory: str, out: list[str]) -> None:
        try:
            names = self.ftp.nlst(directory)
        except Exception as exc:
            print(f"warning: cannot list {directory}: {exc}", file=sys.stderr)
            return
        for name in names:
            if not name or name in {".", ".."}:
                continue
            if name.rstrip("/") == directory.rstrip("/"):
                continue
            path = name if "/" in name else posixpath.join(directory, name)
            path = normalize_remote_path(path)
            if self._is_dir(path):
                self._walk(path, out)
            else:
                out.append(path)

    def _is_dir(self, path: str) -> bool:
        current = None
        try:
            current = self.ftp.pwd()
            self.ftp.cwd(path)
            if current:
                self.ftp.cwd(current)
            return True
        except Exception:
            if current:
                try:
                    self.ftp.cwd(current)
                except Exception:
                    pass
            return False


class SftpClient(RemoteClient):
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.transport = paramiko.Transport((host, port))
        self.transport.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def close(self) -> None:
        try:
            self.sftp.close()
        finally:
            self.transport.close()

    def exists(self, path: str) -> bool:
        try:
            self.sftp.stat(normalize_remote_path(path))
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def download_bytes(self, path: str) -> bytes:
        with self.sftp.open(normalize_remote_path(path), "rb") as f:
            return f.read()

    def upload_bytes(self, path: str, data: bytes) -> None:
        path = normalize_remote_path(path)
        parent = posixpath.dirname(path)
        if parent and parent != ".":
            self._makedirs(parent)
        with self.sftp.open(path, "wb") as f:
            f.write(data)

    def _makedirs(self, directory: str) -> None:
        directory = normalize_remote_path(directory)
        parts = [p for p in directory.split("/") if p]
        current = ""
        for part in parts:
            current = part if not current else f"{current}/{part}"
            try:
                self.sftp.mkdir(current)
            except OSError:
                pass

    def list_files(self, root: str) -> list[str]:
        root = normalize_remote_path(root)
        out: list[str] = []
        self._walk(root, out)
        return sorted(out)

    def _walk(self, directory: str, out: list[str]) -> None:
        try:
            attrs = self.sftp.listdir_attr(directory)
        except Exception as exc:
            print(f"warning: cannot list {directory}: {exc}", file=sys.stderr)
            return
        for attr in attrs:
            name = attr.filename
            if name in {".", ".."}:
                continue
            path = normalize_remote_path(posixpath.join(directory, name))
            if stat.S_ISDIR(attr.st_mode):
                self._walk(path, out)
            else:
                out.append(path)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"warning: invalid integer {name}={raw!r}; using {default}", file=sys.stderr)
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"warning: invalid float {name}={raw!r}; using {default}", file=sys.stderr)
        return default


def load_settings() -> Settings:
    ftp_url = os.getenv("FTP_URL", "").strip()
    ftp_user = os.getenv("FTP_USER", "").strip()
    ftp_pass = os.getenv("FTP_PASS", "")
    if not ftp_url or not ftp_user or not ftp_pass:
        raise RuntimeError("FTP_URL, FTP_USER and FTP_PASS are required")

    create_image_boxes_preview = env_bool("CREATE_IMAGE_BOXES_PREVIEW", False) or env_bool("CREATE_BOXES_PREVIEW", False)

    json_indent_raw = os.getenv("JSON_INDENT", "2").strip()
    json_indent = None if json_indent_raw.lower() in {"none", "null", "0"} else env_int("JSON_INDENT", 2)

    return Settings(
        ftp_url=ftp_url,
        ftp_user=ftp_user,
        ftp_pass=ftp_pass,
        scan_dir=os.getenv("FTP_PHOTO_DIR", DEFAULT_SCAN_DIR).strip() or DEFAULT_SCAN_DIR,
        ftp_protocol=os.getenv("FTP_PROTOCOL", "ftp").strip().lower() or "ftp",
        ftp_port=os.getenv("FTP_PORT", "").strip(),
        model_path=os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH).strip() or DEFAULT_MODEL_PATH,
        yolo_confidence=env_float("YOLO_CONFIDENCE", DEFAULT_CONFIDENCE),
        yolo_iou=env_float("YOLO_IOU", DEFAULT_IOU),
        yolo_image_size=env_int("YOLO_IMAGE_SIZE", DEFAULT_YOLO_IMAGE_SIZE),
        yolo_device=os.getenv("YOLO_DEVICE", DEFAULT_DEVICE).strip() or DEFAULT_DEVICE,
        max_image_edge=env_int("MAX_IMAGE_EDGE", DEFAULT_MAX_IMAGE_EDGE),
        max_detections=env_int("MAX_DETECTIONS", 300),
        process_images=env_bool("PROCESS_IMAGES", True),
        process_videos=env_bool("PROCESS_VIDEOS", True),
        save_yolo_txt=env_bool("SAVE_YOLO_TXT", True),
        save_empty_yolo_txt=env_bool("SAVE_EMPTY_YOLO_TXT", True),
        create_image_boxes_preview=create_image_boxes_preview,
        enable_video_tracking=env_bool("ENABLE_VIDEO_TRACKING", True),
        video_tracker=os.getenv("VIDEO_TRACKER", "bytetrack").strip().lower() or "bytetrack",
        video_frame_interval_seconds=env_float("VIDEO_FRAME_INTERVAL_SECONDS", 1.0),
        video_max_frames_per_file=env_int("VIDEO_MAX_FRAMES_PER_FILE", 300),
        save_track_paths=env_bool("SAVE_TRACK_PATHS", True),
        track_path_max_points=env_int("TRACK_PATH_MAX_POINTS", 1000),
        process_missing_side_outputs=env_bool("PROCESS_MISSING_SIDE_OUTPUTS", True),
        force_reprocess=env_bool("FORCE_REPROCESS", False),
        max_files_per_run=env_int("MAX_FILES_PER_RUN", 0),
        json_indent=json_indent,
        fail_on_errors=env_bool("FAIL_ON_ERRORS", False),
    )


def normalize_remote_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    if path in {"", "."}:
        return "."
    leading = path.startswith("/")
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return "/" if leading else "."
    return normalized if leading else normalized.lstrip("/")


def split_ext(path: str) -> tuple[str, str]:
    base, ext = posixpath.splitext(path)
    return base, ext.lower()


def replace_ext(path: str, new_ext: str) -> str:
    base, _ = posixpath.splitext(path)
    return base + new_ext


def add_suffix_before_ext(path: str, suffix: str, new_ext: str | None = None) -> str:
    base, ext = posixpath.splitext(path)
    return base + suffix + (new_ext if new_ext is not None else ext)


def video_yolo_txt_dir(path: str) -> str:
    base, _ = posixpath.splitext(path)
    return base + ".yolo_txt"


def video_frame_txt_path(video_path: str, frame_number: int, timestamp_sec: float) -> str:
    ms = int(round(timestamp_sec * 1000.0))
    return posixpath.join(video_yolo_txt_dir(video_path), f"frame_{frame_number:06d}_t{ms:010d}ms.txt")


def is_image(path: str) -> bool:
    return split_ext(path)[1] in IMAGE_EXTENSIONS


def is_video(path: str) -> bool:
    return split_ext(path)[1] in VIDEO_EXTENSIONS


def safe_filename(path: str) -> str:
    return posixpath.basename(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_connection(settings: Settings) -> tuple[str, str, int]:
    raw = settings.ftp_url.strip()
    if "://" not in raw:
        raw = f"{settings.ftp_protocol}://{raw}"
    parsed = urlparse(raw)
    protocol = (parsed.scheme or settings.ftp_protocol or "ftp").lower()
    host = parsed.hostname or raw.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    if not host:
        raise RuntimeError("Cannot parse FTP host from FTP_URL")
    if settings.ftp_port:
        port = int(settings.ftp_port)
    elif parsed.port:
        port = parsed.port
    else:
        port = 22 if protocol == "sftp" else 21
    return protocol, host, port


def connect(settings: Settings) -> RemoteClient:
    protocol, host, port = parse_connection(settings)
    print(f"connect: {protocol}://***:{port}")
    if protocol == "sftp":
        return SftpClient(host, port, settings.ftp_user, settings.ftp_pass)
    if protocol == "ftps":
        return FtpClient(host, port, settings.ftp_user, settings.ftp_pass, use_tls=True)
    if protocol == "ftp":
        return FtpClient(host, port, settings.ftp_user, settings.ftp_pass, use_tls=False)
    raise RuntimeError(f"Unsupported protocol: {protocol}. Use ftp, ftps or sftp.")


def load_model(settings: Settings) -> YOLO:
    print(f"load model: {settings.model_path}")
    return YOLO(settings.model_path)


def prepare_image(data: bytes, max_edge: int) -> tuple[Image.Image, dict[str, Any]]:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    original_width, original_height = img.size
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_edge > 0:
        width, height = img.size
        largest = max(width, height)
        if largest > max_edge:
            scale = max_edge / largest
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    prepared_width, prepared_height = img.size
    return img, {
        "original_bytes": len(data),
        "original_width": original_width,
        "original_height": original_height,
        "prepared_width": prepared_width,
        "prepared_height": prepared_height,
        "prepared_mode": img.mode,
    }


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def label_ru(label: str) -> str:
    return COCO_LABEL_RU.get(label, label)


def clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def extract_detections(result: Any, names: Any, width: int, height: int) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return detections

    boxes_xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(boxes_xyxy)
    classes = boxes.cls.cpu().tolist() if boxes.cls is not None else [None] * len(boxes_xyxy)
    track_ids = boxes.id.cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(boxes_xyxy)

    for xyxy, confidence, class_value, track_id in zip(boxes_xyxy, confidences, classes, track_ids):
        class_id = int(class_value) if class_value is not None else -1
        label = class_name(names, class_id)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bbox = {
            "x_min": round(clamp01(x1 / width), 6),
            "y_min": round(clamp01(y1 / height), 6),
            "x_max": round(clamp01(x2 / width), 6),
            "y_max": round(clamp01(y2 / height), 6),
        }
        det: dict[str, Any] = OrderedDict()
        if track_id is not None:
            det["track_id"] = int(track_id)
        det["label"] = label
        det["label_ru"] = label_ru(label)
        det["class_id"] = class_id
        det["confidence"] = round(float(confidence), 6) if confidence is not None else None
        det["bounding_box"] = bbox
        detections.append(det)
    return detections


def summarize_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
    for det in detections:
        key = (int(det["class_id"]), str(det["label"]))
        if key not in grouped:
            grouped[key] = OrderedDict(
                label=det["label"],
                label_ru=det.get("label_ru", det["label"]),
                class_id=det["class_id"],
                count=0,
                confidence=0.0,
                max_confidence=0.0,
                bounding_boxes=[],
            )
        group = grouped[key]
        confidence = float(det.get("confidence") or 0.0)
        group["count"] += 1
        group["confidence"] += confidence
        group["max_confidence"] = max(group["max_confidence"], confidence)
        group["bounding_boxes"].append({**det["bounding_box"], "confidence": det.get("confidence")})

    result: list[dict[str, Any]] = []
    for group in grouped.values():
        count = max(1, int(group["count"]))
        group["confidence"] = round(float(group["confidence"]) / count, 6)
        group["max_confidence"] = round(float(group["max_confidence"]), 6)
        result.append(group)
    return result


def yolo_txt_from_detections(detections: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for det in detections:
        bbox = det["bounding_box"]
        x_min = float(bbox["x_min"])
        y_min = float(bbox["y_min"])
        x_max = float(bbox["x_max"])
        y_max = float(bbox["y_max"])
        x_center = clamp01((x_min + x_max) / 2.0)
        y_center = clamp01((y_min + y_max) / 2.0)
        width = clamp01(x_max - x_min)
        height = clamp01(y_max - y_min)
        lines.append(f"{int(det['class_id'])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def draw_image_preview(image: Image.Image, detections: list[dict[str, Any]]) -> bytes:
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    w, h = preview.size
    for det in detections:
        bbox = det["bounding_box"]
        x1 = int(float(bbox["x_min"]) * w)
        y1 = int(float(bbox["y_min"]) * h)
        x2 = int(float(bbox["x_max"]) * w)
        y2 = int(float(bbox["y_max"]) * h)
        label = f"{det.get('label')} {float(det.get('confidence') or 0):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        tx1, ty1, tx2, ty2 = text_box
        draw.rectangle([tx1, max(0, ty1 - 2), tx2 + 4, ty2 + 2], fill=(255, 0, 0))
        draw.text((x1 + 2, max(0, y1)), label, fill=(255, 255, 255), font=font)
    out = io.BytesIO()
    preview.save(out, format="JPEG", quality=90, optimize=True)
    return out.getvalue()


def run_predict(model: YOLO, image: Image.Image, settings: Settings) -> Any:
    result = model.predict(
        source=image,
        conf=settings.yolo_confidence,
        iou=settings.yolo_iou,
        imgsz=settings.yolo_image_size,
        device=settings.yolo_device,
        max_det=settings.max_detections,
        verbose=False,
    )[0]
    return result


def run_video_frame(model: YOLO, frame: Any, settings: Settings) -> Any:
    if settings.enable_video_tracking:
        tracker = "botsort.yaml" if settings.video_tracker in {"botsort", "bot-sort", "bot_sort"} else "bytetrack.yaml"
        return model.track(
            source=frame,
            persist=True,
            tracker=tracker,
            conf=settings.yolo_confidence,
            iou=settings.yolo_iou,
            imgsz=settings.yolo_image_size,
            device=settings.yolo_device,
            max_det=settings.max_detections,
            verbose=False,
        )[0]
    return model.predict(
        source=frame,
        conf=settings.yolo_confidence,
        iou=settings.yolo_iou,
        imgsz=settings.yolo_image_size,
        device=settings.yolo_device,
        max_det=settings.max_detections,
        verbose=False,
    )[0]


def inference_info(settings: Settings) -> dict[str, Any]:
    return {
        "engine": "ultralytics-yolo",
        "model_path": settings.model_path,
        "confidence_threshold": settings.yolo_confidence,
        "iou_threshold": settings.yolo_iou,
        "image_size": settings.yolo_image_size,
        "max_detections": settings.max_detections,
        "device": settings.yolo_device,
    }


def process_image(client: RemoteClient, model: YOLO, path: str, settings: Settings) -> None:
    json_path = replace_ext(path, ".json")
    txt_path = replace_ext(path, ".txt")
    preview_path = add_suffix_before_ext(path, ".boxes", ".jpg")

    print(f"analyze image: {path} -> {json_path}")
    data = client.download_bytes(path)
    image, metadata = prepare_image(data, settings.max_image_edge)
    width, height = image.size
    result = run_predict(model, image, settings)
    detections = extract_detections(result, model.names, width, height)
    objects = summarize_detections(detections)

    output = OrderedDict(
        schema_version="3.0",
        media_type="image",
        source=OrderedDict(
            path=path,
            filename=safe_filename(path),
            source_mime_guess=mimetypes.guess_type(path)[0],
            **metadata,
        ),
        analyzed_at=utc_now_iso(),
        model=settings.model_path,
        description=f"Local YOLO detection found {len(detections)} object(s) across {len(objects)} class(es).",
        objects=objects,
        detections=detections,
        outputs=OrderedDict(
            json=json_path,
            yolo_txt=txt_path if settings.save_yolo_txt else None,
            boxes_preview=preview_path if settings.create_image_boxes_preview else None,
        ),
        inference=inference_info(settings),
    )

    payload = json.dumps(output, ensure_ascii=False, indent=settings.json_indent).encode("utf-8")
    client.upload_bytes(json_path, payload)

    if settings.save_yolo_txt and (detections or settings.save_empty_yolo_txt):
        client.upload_bytes(txt_path, yolo_txt_from_detections(detections).encode("utf-8"))

    if settings.create_image_boxes_preview:
        client.upload_bytes(preview_path, draw_image_preview(image, detections))

    print(f"done: {json_path}")


def update_track_summary(
    tracks: dict[int, dict[str, Any]],
    det: dict[str, Any],
    timestamp_sec: float,
    frame_number: int,
    settings: Settings,
) -> None:
    track_id = det.get("track_id")
    if track_id is None:
        return
    tid = int(track_id)
    bbox = det["bounding_box"]
    center_x = round((float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, 6)
    center_y = round((float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0, 6)
    confidence = float(det.get("confidence") or 0.0)

    if tid not in tracks:
        tracks[tid] = OrderedDict(
            track_id=tid,
            label=det["label"],
            label_ru=det.get("label_ru", det["label"]),
            class_id=det["class_id"],
            first_seen_sec=round(timestamp_sec, 3),
            last_seen_sec=round(timestamp_sec, 3),
            duration_sec=0.0,
            first_frame=frame_number,
            last_frame=frame_number,
            frames_seen=0,
            max_confidence=0.0,
            path=[],
        )
    track = tracks[tid]
    track["last_seen_sec"] = round(timestamp_sec, 3)
    track["duration_sec"] = round(float(track["last_seen_sec"]) - float(track["first_seen_sec"]), 3)
    track["last_frame"] = frame_number
    track["frames_seen"] += 1
    track["max_confidence"] = round(max(float(track["max_confidence"]), confidence), 6)
    if settings.save_track_paths and len(track["path"]) < settings.track_path_max_points:
        track["path"].append(
            OrderedDict(
                timestamp_sec=round(timestamp_sec, 3),
                frame_number=frame_number,
                x=center_x,
                y=center_y,
            )
        )


def process_video(client: RemoteClient, model: YOLO, path: str, settings: Settings) -> None:
    json_path = replace_ext(path, ".json")
    txt_dir = video_yolo_txt_dir(path)

    print(f"analyze video: {path} -> {json_path}")
    data = client.download_bytes(path)

    suffix = split_ext(path)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    frames: list[dict[str, Any]] = []
    tracks: dict[int, dict[str, Any]] = {}
    class_frame_hits: dict[str, int] = defaultdict(int)
    class_detection_count: dict[str, int] = defaultdict(int)
    processed_frames = 0
    total_frames = 0
    fps = 0.0
    duration_sec = None
    width = 0
    height = 0

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video file")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps > 0 and total_frames > 0:
            duration_sec = round(total_frames / fps, 3)

        interval = max(0.0, settings.video_frame_interval_seconds)
        frame_step = 1 if interval <= 0 or fps <= 0 else max(1, int(round(fps * interval)))

        if settings.enable_video_tracking:
            # Reset Ultralytics predictor/tracker state between different video files.
            try:
                model.predictor = None
            except Exception:
                pass

        frame_number = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_number % frame_step != 0:
                frame_number += 1
                continue
            if settings.video_max_frames_per_file > 0 and processed_frames >= settings.video_max_frames_per_file:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame_rgb.shape[:2]
            timestamp_sec = frame_number / fps if fps > 0 else processed_frames * max(interval, 1.0)

            result = run_video_frame(model, frame_rgb, settings)
            detections = extract_detections(result, model.names, frame_width, frame_height)

            labels_in_frame: set[str] = set()
            for det in detections:
                labels_in_frame.add(str(det["label"]))
                class_detection_count[str(det["label"])] += 1
                update_track_summary(tracks, det, timestamp_sec, frame_number, settings)
            for label in labels_in_frame:
                class_frame_hits[label] += 1

            frame_entry = OrderedDict(
                frame_number=frame_number,
                timestamp_sec=round(timestamp_sec, 3),
                width=frame_width,
                height=frame_height,
                yolo_txt=video_frame_txt_path(path, frame_number, timestamp_sec) if settings.save_yolo_txt else None,
                detections=detections,
            )
            frames.append(frame_entry)

            if settings.save_yolo_txt and (detections or settings.save_empty_yolo_txt):
                client.upload_bytes(frame_entry["yolo_txt"], yolo_txt_from_detections(detections).encode("utf-8"))

            processed_frames += 1
            frame_number += 1

        cap.release()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    tracks_list = sorted(tracks.values(), key=lambda t: (str(t["label"]), int(t["track_id"])))
    unique_by_class: dict[str, int] = defaultdict(int)
    for track in tracks_list:
        unique_by_class[str(track["label"])] += 1

    output = OrderedDict(
        schema_version="3.0",
        media_type="video",
        source=OrderedDict(
            path=path,
            filename=safe_filename(path),
            original_bytes=len(data),
            source_mime_guess=mimetypes.guess_type(path)[0],
        ),
        video=OrderedDict(
            width=width,
            height=height,
            fps=round(fps, 6) if fps else None,
            total_frames=total_frames,
            duration_sec=duration_sec,
            processed_frames=processed_frames,
            frame_interval_seconds=settings.video_frame_interval_seconds,
            max_frames_per_file=settings.video_max_frames_per_file,
        ),
        analyzed_at=utc_now_iso(),
        model=settings.model_path,
        tracker=OrderedDict(
            enabled=settings.enable_video_tracking,
            name=settings.video_tracker if settings.enable_video_tracking else None,
        ),
        outputs=OrderedDict(
            json=json_path,
            yolo_txt_directory=txt_dir if settings.save_yolo_txt else None,
            boxes_preview=None,
        ),
        summary=OrderedDict(
            detections_by_class=dict(sorted(class_detection_count.items())),
            frames_with_class=dict(sorted(class_frame_hits.items())),
            unique_tracked_objects_by_class=dict(sorted(unique_by_class.items())),
        ),
        tracks=tracks_list,
        frames=frames,
        inference=inference_info(settings),
    )

    payload = json.dumps(output, ensure_ascii=False, indent=settings.json_indent).encode("utf-8")
    client.upload_bytes(json_path, payload)
    print(f"done: {json_path}")


def side_outputs_missing(client: RemoteClient, path: str, settings: Settings) -> bool:
    if is_image(path):
        if settings.save_yolo_txt and not client.exists(replace_ext(path, ".txt")):
            return True
        if settings.create_image_boxes_preview and not client.exists(add_suffix_before_ext(path, ".boxes", ".jpg")):
            return True
    if is_video(path):
        if settings.save_yolo_txt and not client.exists(video_yolo_txt_dir(path)):
            return True
    return False


def should_process(client: RemoteClient, path: str, settings: Settings) -> tuple[bool, str]:
    json_path = replace_ext(path, ".json")
    if settings.force_reprocess:
        return True, "force"
    if not client.exists(json_path):
        return True, "missing-json"
    if settings.process_missing_side_outputs and side_outputs_missing(client, path, settings):
        return True, "missing-side-output"
    return False, "already-processed"


def run() -> int:
    settings = load_settings()
    client = connect(settings)
    processed = 0
    errors = 0
    model: YOLO | None = None

    try:
        root = normalize_remote_path(settings.scan_dir)
        print(f"scan root: {root}")
        all_files = client.list_files(root)
        candidates: list[str] = []
        for path in all_files:
            if settings.process_images and is_image(path):
                candidates.append(path)
            elif settings.process_videos and is_video(path):
                candidates.append(path)

        print(f"media candidates: {len(candidates)}")

        for path in candidates:
            do_process, reason = should_process(client, path, settings)
            if not do_process:
                continue
            if settings.max_files_per_run > 0 and processed >= settings.max_files_per_run:
                print(f"limit reached: MAX_FILES_PER_RUN={settings.max_files_per_run}")
                break

            try:
                print(f"process reason: {reason}")
                if model is None:
                    model = load_model(settings)
                if is_image(path):
                    process_image(client, model, path, settings)
                elif is_video(path):
                    process_video(client, model, path, settings)
                processed += 1
            except Exception as exc:
                errors += 1
                print(f"error: failed to process {path}: {exc}", file=sys.stderr)
                if settings.fail_on_errors:
                    raise

        print(f"processed: {processed}")
        if errors:
            print(f"errors: {errors}", file=sys.stderr)
        return 1 if errors and settings.fail_on_errors else 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(run())


#!/usr/bin/env python3
"""Scan an FTP/FTPS/SFTP photo folder, detect objects locally, and upload JSON results.

An image is considered unprocessed when there is no JSON file with the same basename
in the same remote directory. Example: photo/cat.jpg -> photo/cat.json.

Object detection is performed locally with Ultralytics YOLO on the GitHub Actions runner.
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
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from ftplib import FTP, FTP_TLS, error_perm
from typing import Iterable, Literal
from urllib.parse import urlparse

import paramiko
from PIL import Image, ImageOps
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

DEFAULT_PHOTO_DIR = "photo"
DEFAULT_MODEL_PATH = "yolo11n.pt"
DEFAULT_MAX_IMAGE_EDGE = 1600
DEFAULT_YOLO_IMAGE_SIZE = 1280
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.70
DEFAULT_DEVICE = "cpu"

# Russian labels for the default COCO class set used by common YOLO pretrained models.
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
    "toilet": "унитаз",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "mouse": "мышь",
    "remote": "пульт",
    "keyboard": "клавиатура",
    "cell phone": "мобильный телефон",
    "microwave": "микроволновая печь",
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

Kind = Literal["file", "dir", "other"]
_MODEL: YOLO | None = None
_MODEL_PATH: str | None = None


@dataclass(frozen=True)
class RemoteEntry:
    name: str
    path: str
    kind: Kind


@dataclass(frozen=True)
class ConnectionConfig:
    scheme: str
    host: str
    port: int | None
    username: str
    password: str
    url_path: str


def getenv_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got: {raw!r}") from exc


def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got: {raw!r}") from exc


def parse_connection_config() -> ConnectionConfig:
    raw_url = getenv_required("FTP_URL")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url):
        raw_url = "ftp://" + raw_url

    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "ftp").lower()
    if scheme not in {"ftp", "ftps", "sftp"}:
        raise RuntimeError(
            f"Unsupported FTP_URL scheme {scheme!r}. Use ftp://, ftps://, sftp://, or a bare host."
        )

    username = os.getenv("FTP_USER", "").strip() or (parsed.username or "")
    password = os.getenv("FTP_PASS", "").strip() or (parsed.password or "")
    if not username:
        raise RuntimeError("Missing FTP user. Set FTP_USER secret.")
    if not password:
        raise RuntimeError("Missing FTP password. Set FTP_PASS secret.")
    if not parsed.hostname:
        raise RuntimeError("FTP_URL must contain a host name.")

    return ConnectionConfig(
        scheme=scheme,
        host=parsed.hostname,
        port=parsed.port,
        username=username,
        password=password,
        url_path=parsed.path or "",
    )


def remote_join(directory: str, name: str) -> str:
    if directory in {"", "."}:
        return name
    if directory == "/":
        return "/" + name.lstrip("/")
    return posixpath.join(directory, name)


def normalize_remote_dir(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return DEFAULT_PHOTO_DIR
    if value.startswith("./"):
        value = value[2:]
    value = value.rstrip("/")
    return value or "."


def photo_dir_from_env_or_url(config: ConnectionConfig) -> str:
    env_value = os.getenv("FTP_PHOTO_DIR", "").strip()
    if env_value:
        return normalize_remote_dir(env_value)
    if config.url_path and config.url_path != "/":
        return normalize_remote_dir(config.url_path)
    return DEFAULT_PHOTO_DIR


class RemoteClient:
    def list_dir(self, remote_dir: str) -> list[RemoteEntry]:
        raise NotImplementedError

    def download_bytes(self, remote_path: str) -> bytes:
        raise NotImplementedError

    def upload_bytes(self, remote_path: str, data: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FtpRemoteClient(RemoteClient):
    def __init__(self, config: ConnectionConfig):
        if config.scheme == "ftps":
            ftp: FTP | FTP_TLS = FTP_TLS()
        else:
            ftp = FTP()

        port = config.port or 21
        ftp.encoding = "utf-8"
        ftp.connect(config.host, port, timeout=45)
        ftp.login(config.username, config.password)
        if isinstance(ftp, FTP_TLS):
            ftp.prot_p()

        passive_raw = os.getenv("FTP_PASSIVE", "true").strip().lower()
        ftp.set_pasv(passive_raw not in {"0", "false", "no", "off"})
        self.ftp = ftp

    def list_dir(self, remote_dir: str) -> list[RemoteEntry]:
        entries: list[RemoteEntry] = []
        try:
            for name, facts in self.ftp.mlsd(remote_dir):
                if name in {".", ".."}:
                    continue
                entry_type = (facts.get("type") or "").lower()
                if entry_type == "dir":
                    kind: Kind = "dir"
                elif entry_type == "file":
                    kind = "file"
                else:
                    kind = "other"
                entries.append(RemoteEntry(name=name, path=remote_join(remote_dir, name), kind=kind))
            return entries
        except Exception:
            # Some hosting providers disable MLSD. Fall back to NLST + CWD probing.
            pass

        try:
            names = self.ftp.nlst(remote_dir)
        except error_perm as exc:
            raise RuntimeError(f"Cannot list remote directory {remote_dir!r}: {exc}") from exc

        for raw_name in names:
            name = posixpath.basename(raw_name.rstrip("/"))
            if not name or name in {".", ".."}:
                continue
            path = raw_name if raw_name.startswith(remote_dir.rstrip("/") + "/") else remote_join(remote_dir, name)
            kind = "dir" if self._is_dir(path) else "file"
            entries.append(RemoteEntry(name=name, path=path, kind=kind))
        return entries

    def _is_dir(self, remote_path: str) -> bool:
        old_dir = self.ftp.pwd()
        try:
            self.ftp.cwd(remote_path)
            return True
        except Exception:
            return False
        finally:
            try:
                self.ftp.cwd(old_dir)
            except Exception:
                pass

    def download_bytes(self, remote_path: str) -> bytes:
        buffer = io.BytesIO()
        self.ftp.retrbinary(f"RETR {remote_path}", buffer.write)
        return buffer.getvalue()

    def upload_bytes(self, remote_path: str, data: bytes) -> None:
        self.ftp.storbinary(f"STOR {remote_path}", io.BytesIO(data))

    def close(self) -> None:
        try:
            self.ftp.quit()
        except Exception:
            self.ftp.close()


class SftpRemoteClient(RemoteClient):
    def __init__(self, config: ConnectionConfig):
        port = config.port or 22
        self.transport = paramiko.Transport((config.host, port))
        self.transport.connect(username=config.username, password=config.password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def list_dir(self, remote_dir: str) -> list[RemoteEntry]:
        entries: list[RemoteEntry] = []
        try:
            attrs = self.sftp.listdir_attr(remote_dir)
        except Exception as exc:
            raise RuntimeError(f"Cannot list remote directory {remote_dir!r}: {exc}") from exc

        for item in attrs:
            name = item.filename
            if name in {".", ".."}:
                continue
            mode = item.st_mode or 0
            if stat.S_ISDIR(mode):
                kind: Kind = "dir"
            elif stat.S_ISREG(mode):
                kind = "file"
            else:
                kind = "other"
            entries.append(RemoteEntry(name=name, path=remote_join(remote_dir, name), kind=kind))
        return entries

    def download_bytes(self, remote_path: str) -> bytes:
        with self.sftp.open(remote_path, "rb") as remote_file:
            return remote_file.read()

    def upload_bytes(self, remote_path: str, data: bytes) -> None:
        with self.sftp.open(remote_path, "wb") as remote_file:
            remote_file.write(data)

    def close(self) -> None:
        self.sftp.close()
        self.transport.close()


def connect_remote(config: ConnectionConfig) -> RemoteClient:
    if config.scheme == "sftp":
        return SftpRemoteClient(config)
    return FtpRemoteClient(config)


def is_image_filename(filename: str) -> bool:
    return posixpath.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def json_path_for_image(remote_image_path: str) -> str:
    directory = posixpath.dirname(remote_image_path)
    filename = posixpath.basename(remote_image_path)
    stem = posixpath.splitext(filename)[0]
    return remote_join(directory, stem + ".json")


def iter_pending_images(client: RemoteClient, root_dir: str) -> Iterable[str]:
    stack = [root_dir]
    while stack:
        current_dir = stack.pop()
        entries = client.list_dir(current_dir)
        file_names_lower = {entry.name.lower() for entry in entries if entry.kind == "file"}

        # Add directories in reverse alphabetical order so processing is deterministic.
        child_dirs = sorted((entry.path for entry in entries if entry.kind == "dir"), reverse=True)
        stack.extend(child_dirs)

        for entry in sorted(entries, key=lambda item: item.path.lower()):
            if entry.kind != "file" or not is_image_filename(entry.name):
                continue
            expected_json_name = posixpath.splitext(entry.name)[0].lower() + ".json"
            if expected_json_name in file_names_lower:
                print(f"skip: {entry.path} already has matching JSON")
                continue
            yield entry.path


def prepare_image_for_model(original: bytes, remote_path: str) -> tuple[Image.Image, dict]:
    max_edge = getenv_int("MAX_IMAGE_EDGE", DEFAULT_MAX_IMAGE_EDGE)

    with Image.open(io.BytesIO(original)) as source_image:
        image = ImageOps.exif_transpose(source_image)
        if getattr(image, "is_animated", False):
            image.seek(0)
        original_width, original_height = image.size

        if max_edge > 0:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            background.paste(image.convert("RGBA"), mask=alpha)
            image = background
        else:
            image = image.convert("RGB")

        prepared = image.copy()

    metadata = {
        "original_bytes": len(original),
        "original_width": original_width,
        "original_height": original_height,
        "prepared_width": prepared.width,
        "prepared_height": prepared.height,
        "source_mime_guess": mimetypes.guess_type(remote_path)[0],
        "prepared_mode": prepared.mode,
    }
    return prepared, metadata


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def get_model() -> tuple[YOLO, str]:
    global _MODEL, _MODEL_PATH

    model_path = os.getenv("LOCAL_MODEL_PATH", "").strip() or DEFAULT_MODEL_PATH
    if _MODEL is None or _MODEL_PATH != model_path:
        print(f"load model: {model_path}")
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL, model_path


def analyze_image(image: Image.Image, remote_path: str) -> dict:
    model, model_path = get_model()
    confidence_threshold = getenv_float("YOLO_CONFIDENCE", DEFAULT_CONFIDENCE)
    iou_threshold = getenv_float("YOLO_IOU", DEFAULT_IOU)
    image_size = getenv_int("YOLO_IMAGE_SIZE", DEFAULT_YOLO_IMAGE_SIZE)
    max_detections = getenv_int("MAX_DETECTIONS", 300)
    device = os.getenv("YOLO_DEVICE", "").strip() or DEFAULT_DEVICE

    prediction_kwargs = {
        "source": image,
        "conf": confidence_threshold,
        "iou": iou_threshold,
        "imgsz": image_size,
        "max_det": max_detections,
        "device": device,
        "verbose": False,
    }

    results = model.predict(**prediction_kwargs)
    if not results:
        raise RuntimeError("YOLO returned no result object")

    result = results[0]
    names = result.names or getattr(model, "names", {}) or {}
    width, height = image.size
    detections: list[dict] = []

    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()

        for xyxy, confidence, class_value in zip(boxes_xyxy, confidences, classes):
            class_id = int(class_value)
            label = str(names.get(class_id, class_id))
            x_min, y_min, x_max, y_max = xyxy
            bounding_box = {
                "x_min": round_float(clamp01(x_min / width)),
                "y_min": round_float(clamp01(y_min / height)),
                "x_max": round_float(clamp01(x_max / width)),
                "y_max": round_float(clamp01(y_max / height)),
            }
            detections.append(
                {
                    "label": label,
                    "label_ru": COCO_LABEL_RU.get(label, label),
                    "class_id": class_id,
                    "confidence": round_float(confidence, 4),
                    "bounding_box": bounding_box,
                }
            )

    grouped: OrderedDict[tuple[int, str], dict] = OrderedDict()
    for detection in sorted(detections, key=lambda item: (-item["confidence"], item["label"])):
        key = (detection["class_id"], detection["label"])
        if key not in grouped:
            grouped[key] = {
                "label": detection["label"],
                "label_ru": detection["label_ru"],
                "class_id": detection["class_id"],
                "count": 0,
                "confidence": 0.0,
                "max_confidence": 0.0,
                "bounding_boxes": [],
            }
        group = grouped[key]
        group["count"] += 1
        group["confidence"] += detection["confidence"]
        group["max_confidence"] = max(group["max_confidence"], detection["confidence"])
        group["bounding_boxes"].append({**detection["bounding_box"], "confidence": detection["confidence"]})

    objects = []
    for group in grouped.values():
        group["confidence"] = round_float(group["confidence"] / group["count"], 4)
        group["max_confidence"] = round_float(group["max_confidence"], 4)
        objects.append(group)

    class_count = len(objects)
    object_count = len(detections)
    description = (
        f"Local YOLO detection found {object_count} object(s) "
        f"across {class_count} class(es)."
    )

    return {
        "description": description,
        "objects": objects,
        "detections": detections,
        "inference": {
            "engine": "ultralytics-yolo",
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "image_size": image_size,
            "max_detections": max_detections,
            "device": device,
        },
        "remote_path": remote_path,
    }


def build_output_json(remote_path: str, model_result: dict, image_metadata: dict) -> dict:
    return {
        "schema_version": "2.0",
        "source": {
            "path": remote_path,
            "filename": posixpath.basename(remote_path),
            **image_metadata,
        },
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "model": model_result["inference"]["model_path"],
        "description": model_result["description"],
        "objects": model_result["objects"],
        "detections": model_result["detections"],
        "inference": model_result["inference"],
    }


def main() -> int:
    config = parse_connection_config()
    root_dir = photo_dir_from_env_or_url(config)
    max_files = getenv_int("MAX_FILES_PER_RUN", 0)
    json_indent = getenv_int("JSON_INDENT", 2)
    fail_on_errors = os.getenv("FAIL_ON_ERRORS", "false").strip().lower() in {"1", "true", "yes", "on"}

    print(f"connect: {config.scheme}://{config.host}:{config.port or ('22' if config.scheme == 'sftp' else '21')}")
    print(f"scan root: {root_dir}")

    processed = 0
    errors: list[str] = []
    client = connect_remote(config)
    try:
        for remote_image_path in iter_pending_images(client, root_dir):
            if max_files and processed >= max_files:
                print(f"limit reached: MAX_FILES_PER_RUN={max_files}")
                break

            target_json_path = json_path_for_image(remote_image_path)
            print(f"analyze: {remote_image_path} -> {target_json_path}")
            try:
                original = client.download_bytes(remote_image_path)
                image, image_metadata = prepare_image_for_model(original, remote_image_path)
                model_result = analyze_image(image, remote_image_path)
                output = build_output_json(remote_image_path, model_result, image_metadata)
                payload = json.dumps(output, ensure_ascii=False, indent=json_indent).encode("utf-8")
                client.upload_bytes(target_json_path, payload)
                processed += 1
                print(f"done: {target_json_path}")
            except Exception as exc:
                message = f"error: {remote_image_path}: {exc}"
                errors.append(message)
                print(message, file=sys.stderr)
                if fail_on_errors:
                    break
    finally:
        client.close()

    print(f"processed: {processed}")
    if errors:
        print(f"errors: {len(errors)}", file=sys.stderr)
        for message in errors:
            print(message, file=sys.stderr)
        return 1 if fail_on_errors else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

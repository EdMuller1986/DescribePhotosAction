#!/usr/bin/env python3
"""Generate surveillance calibration files from a reference image on Google Drive."""

from __future__ import annotations

import os
import posixpath

import cv2
import numpy as np

from analyze_gdrive_photos import GDriveClient, build_drive_credentials, is_image, parse_folder_id
from surveillance.calibration import (
    CALIBRATION_SKIP_IMAGE_NAMES,
    REFERENCE_IMAGE_PREFER,
    calibrate_from_image,
)


def _remote_path(config_dir: str, filename: str) -> str:
    config_dir = config_dir.strip().replace("\\", "/").strip("/")
    filename = filename.strip().replace("\\", "/").lstrip("/")
    return filename if not config_dir else f"{config_dir}/{filename}"


def _is_reference_candidate(path: str) -> bool:
    if not is_image(path):
        return False
    name = posixpath.basename(path).lower()
    return name not in CALIBRATION_SKIP_IMAGE_NAMES


def find_reference_image(client: GDriveClient, config_dir: str, override: str) -> str:
    if override:
        remote = override.replace("\\", "/").lstrip("/")
        if not client.exists(remote):
            raise SystemExit(f"Reference image not found on Drive: {remote}")
        return remote

    all_files = client.list_files(".")
    root_images = sorted(path for path in all_files if _is_reference_candidate(path) and "/" not in path)
    for name in REFERENCE_IMAGE_PREFER:
        if name in root_images:
            return name

    config_candidates = [
        _remote_path(config_dir, "reference_input.jpg"),
        _remote_path(config_dir, "reference_day.jpg"),
    ]
    for path in config_candidates:
        if client.exists(path):
            return path

    if root_images:
        return root_images[0]

    raise SystemExit(
        "No reference image found on Drive. Upload a daytime JPG/PNG to the folder root "
        "(for example reference_day.jpg) and run calibration again."
    )


def main() -> int:
    folder_raw = os.getenv("GDRIVE_FOLDER_ID", "").strip() or os.getenv("GDRIVE_URL", "").strip()
    config_dir = os.getenv("SURVEILLANCE_GDRIVE_CONFIG_DIR", "config").strip() or "config"
    reference_override = os.getenv("SURVEILLANCE_REFERENCE_FILE", "").strip()

    if not folder_raw:
        raise SystemExit("GDRIVE_FOLDER_ID is required")

    credentials, auth_mode = build_drive_credentials()
    folder_id = parse_folder_id(folder_raw)
    client = GDriveClient(folder_id, credentials, auth_mode)

    try:
        client.verify_write_access()
        reference_path = find_reference_image(client, config_dir, reference_override)
        print(f"reference: {reference_path}")

        image_bytes = client.download_bytes(reference_path)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Cannot decode reference image: {reference_path}")

        artifacts = calibrate_from_image(image)
        for filename, data in artifacts.items():
            remote_out = _remote_path(config_dir, filename)
            client.upload_bytes(remote_out, data)
            print(f"uploaded: {remote_out}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
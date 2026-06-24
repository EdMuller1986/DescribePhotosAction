#!/usr/bin/env python3
"""Run surveillance day summary on a Google Drive folder video and upload results."""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import sys
import tempfile
from pathlib import Path

from analyze_gdrive_photos import GDriveClient, build_drive_credentials, parse_folder_id


def main() -> int:
    config_path = os.getenv("SURVEILLANCE_CONFIG_PATH", "config/surveillance.example.json").strip()
    folder_raw = os.getenv("GDRIVE_FOLDER_ID", "").strip() or os.getenv("GDRIVE_URL", "").strip()
    video_name = os.getenv("SURVEILLANCE_VIDEO_NAME", "").strip()

    if not folder_raw:
        raise SystemExit("GDRIVE_FOLDER_ID is required")
    if not video_name:
        raise SystemExit("SURVEILLANCE_VIDEO_NAME is required (daily video filename)")

    credentials, auth_mode = build_drive_credentials()
    folder_id = parse_folder_id(folder_raw)
    client = GDriveClient(folder_id, credentials, auth_mode)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    remote_video = video_name.replace("\\", "/").lstrip("/")
    if not client.exists(remote_video):
        raise SystemExit(f"Video not found on Drive: {remote_video}")

    suffix = Path(remote_video).suffix or ".mkv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(client.download_bytes(remote_video))
        local_video = tmp.name

    local_config = config_path
    local_output = tempfile.mkdtemp(prefix="surveillance-out-")
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analyze_surveillance_day.py"),
            "--video",
            local_video,
            "--config",
            local_config,
            "--output-dir",
            local_output,
        ]
        print("run:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        stem = Path(remote_video).stem
        for ext in (".summary.json", ".summary.txt"):
            local_file = Path(local_output) / f"{stem}{ext}"
            if not local_file.is_file():
                continue
            parent = posixpath.dirname(remote_video.replace("\\", "/"))
            remote_out = f"{stem}{ext}" if parent in {"", "."} else f"{parent}/{stem}{ext}"
            client.upload_bytes(remote_out, local_file.read_bytes())
            print(f"uploaded: {remote_out}")
    finally:
        try:
            os.unlink(local_video)
        except OSError:
            pass
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
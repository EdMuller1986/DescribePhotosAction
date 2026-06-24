#!/usr/bin/env python3
"""Run surveillance day summary for videos in a Google Drive folder."""

from __future__ import annotations

import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from analyze_gdrive_photos import GDriveClient, build_drive_credentials, env_bool, is_video, parse_folder_id
from surveillance.gdrive_config import download_surveillance_setup


def _summary_json_path(video_path: str) -> str:
    stem = Path(video_path).stem
    parent = posixpath.dirname(video_path.replace("\\", "/"))
    if parent in {"", "."}:
        return f"{stem}.summary.json"
    return f"{parent}/{stem}.summary.json"


def _list_surveillance_videos(client: GDriveClient, config_dir: str) -> list[str]:
    config_prefix = config_dir.strip().replace("\\", "/").strip("/")
    if config_prefix:
        config_prefix += "/"

    videos: list[str] = []
    for path in client.list_files("."):
        if not is_video(path):
            continue
        normalized = path.replace("\\", "/")
        if normalized.startswith("config/") or (config_prefix and normalized.startswith(config_prefix)):
            continue
        videos.append(path)
    return sorted(videos)


def _should_process_video(client: GDriveClient, video_path: str, *, force: bool) -> tuple[bool, str]:
    if force:
        return True, "force"
    summary_path = _summary_json_path(video_path)
    if client.exists(summary_path):
        return False, "summary-exists"
    return True, "missing-summary"


def _process_video(
    client: GDriveClient,
    *,
    remote_video: str,
    local_config: str,
    output_dir: str,
) -> None:
    video_path: str | None = None
    try:
        suffix = Path(remote_video).suffix or ".mkv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(client.download_bytes(remote_video))
            video_path = tmp.name

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analyze_surveillance_day.py"),
            "--video",
            video_path,
            "--config",
            local_config,
            "--output-dir",
            output_dir,
        ]
        print("run:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        stem = Path(remote_video).stem
        for ext in (".summary.json", ".summary.txt"):
            local_file = Path(output_dir) / f"{stem}{ext}"
            if not local_file.is_file():
                continue
            parent = posixpath.dirname(remote_video.replace("\\", "/"))
            remote_out = f"{stem}{ext}" if parent in {"", "."} else f"{parent}/{stem}{ext}"
            client.upload_bytes(remote_out, local_file.read_bytes())
            print(f"uploaded: {remote_out}")
    finally:
        if video_path:
            try:
                os.unlink(video_path)
            except OSError:
                pass


def main() -> int:
    folder_raw = os.getenv("GDRIVE_FOLDER_ID", "").strip() or os.getenv("GDRIVE_URL", "").strip()
    config_dir = os.getenv("SURVEILLANCE_GDRIVE_CONFIG_DIR", "config").strip() or "config"
    config_file = os.getenv("SURVEILLANCE_GDRIVE_CONFIG_FILE", "surveillance.json").strip()
    mask_file = os.getenv("SURVEILLANCE_GDRIVE_MASK_FILE", "ignore_mask.png").strip()
    force_reprocess = env_bool("SURVEILLANCE_FORCE_REPROCESS", False)
    max_videos = int(os.getenv("SURVEILLANCE_MAX_VIDEOS_PER_RUN", "0") or "0")

    if not folder_raw:
        raise SystemExit("GDRIVE_FOLDER_ID is required")

    credentials, auth_mode = build_drive_credentials()
    folder_id = parse_folder_id(folder_raw)
    client = GDriveClient(folder_id, credentials, auth_mode)

    config_tmpdir: str | None = None
    output_dir = tempfile.mkdtemp(prefix="surveillance-out-")
    processed = 0
    skipped = 0
    errors = 0

    try:
        client.verify_write_access()
        _, _, config_tmpdir = download_surveillance_setup(
            client,
            config_dir=config_dir,
            config_file=config_file,
            ignore_mask_file=mask_file,
        )
        local_config = os.path.join(config_tmpdir, "surveillance.local.json")

        videos = _list_surveillance_videos(client, config_dir)
        print(f"video candidates: {len(videos)}")

        for remote_video in videos:
            do_process, reason = _should_process_video(client, remote_video, force=force_reprocess)
            if not do_process:
                print(f"skip: {remote_video} ({reason})")
                skipped += 1
                continue
            if max_videos > 0 and processed >= max_videos:
                print(f"limit reached: SURVEILLANCE_MAX_VIDEOS_PER_RUN={max_videos}")
                break

            try:
                print(f"process: {remote_video} ({reason})")
                _process_video(
                    client,
                    remote_video=remote_video,
                    local_config=local_config,
                    output_dir=output_dir,
                )
                processed += 1
            except subprocess.CalledProcessError as exc:
                errors += 1
                print(f"error: failed to process {remote_video}: {exc}", file=sys.stderr)
                if env_bool("FAIL_ON_ERRORS", False):
                    raise

        print(f"processed: {processed}")
        print(f"skipped: {skipped}")
        if errors:
            print(f"errors: {errors}", file=sys.stderr)
            return 1
    finally:
        if config_tmpdir:
            shutil.rmtree(config_tmpdir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
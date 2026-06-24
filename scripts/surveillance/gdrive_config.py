"""Download private surveillance calibration files from Google Drive."""

from __future__ import annotations

import json
import os
import posixpath
import tempfile
from typing import Any

from analyze_gdrive_photos import GDriveClient


def _remote_path(config_dir: str, filename: str) -> str:
    config_dir = config_dir.strip().replace("\\", "/").strip("/")
    filename = filename.strip().replace("\\", "/").lstrip("/")
    return filename if not config_dir else f"{config_dir}/{filename}"


def download_surveillance_setup(
    client: GDriveClient,
    config_dir: str = "config",
    config_file: str = "surveillance.json",
    ignore_mask_file: str = "ignore_mask.png",
) -> tuple[dict[str, Any], str, str]:
    """Return config dict, local ignore_mask path, and temp directory to clean up later."""
    remote_config = _remote_path(config_dir, config_file)
    if not client.exists(remote_config):
        raise RuntimeError(
            f"Surveillance config not found on Drive: {remote_config}. "
            "Upload config/surveillance.json to your camera folder."
        )

    config = json.loads(client.download_bytes(remote_config).decode("utf-8"))
    mask_name = str(config.get("ignore_mask_file", ignore_mask_file))
    remote_mask = _remote_path(config_dir, mask_name)
    if not client.exists(remote_mask):
        raise RuntimeError(f"Ignore mask not found on Drive: {remote_mask}")

    tmpdir = tempfile.mkdtemp(prefix="surveillance-config-")
    mask_path = os.path.join(tmpdir, posixpath.basename(mask_name))
    with open(mask_path, "wb") as f:
        f.write(client.download_bytes(remote_mask))

    config["ignore_mask_path"] = mask_path
    local_config_path = os.path.join(tmpdir, "surveillance.local.json")
    with open(local_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"config: loaded from Drive/{remote_config}")
    print(f"mask: loaded from Drive/{remote_mask}")
    return config, mask_path, tmpdir
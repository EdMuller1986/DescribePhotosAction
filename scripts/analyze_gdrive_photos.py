#!/usr/bin/env python3
"""Analyze images and videos from a Google Drive folder with a local YOLO model.

Behavior mirrors analyze_ftp_photos.py:
- scan GDRIVE_FOLDER_ID recursively;
- process image and video files without a sibling JSON result;
- write JSON, YOLO TXT, and optional previews back to the same Drive tree.

Auth options:
- OAuth refresh token via GOOGLE_DRIVE_OAUTH_CREDENTIALS (recommended for My Drive)
- Service account JSON via GOOGLE_DRIVE_CREDENTIALS (Shared Drive / Workspace)
"""

from __future__ import annotations

import io
import json
import os
import posixpath
import re
import sys
from dataclasses import dataclass
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from analyze_ftp_photos import (
    env_bool,
    env_float,
    env_int,
    is_image,
    is_video,
    load_model,
    normalize_remote_path,
    process_image,
    process_video,
    should_process,
)
from analyze_ftp_photos import RemoteClient

DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive",)
FOLDER_MIME = "application/vnd.google-apps.folder"

_FOLDER_ID_PATTERNS = (
    re.compile(r"drive\.google\.com/(?:drive/(?:u/\d+/)?folders|file/d)/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
    re.compile(r"^([A-Za-z0-9_-]{20,})$"),
)

def _owner_emails(meta: dict[str, Any]) -> list[str]:
    owners = meta.get("owners") or []
    return [str(owner.get("emailAddress", "")) for owner in owners if owner.get("emailAddress")]


def _owner_account_match(auth_email: str, owner_emails: list[str]) -> bool | None:
    if not auth_email or auth_email in {"unknown", "service-account"}:
        return None
    auth = auth_email.strip().lower()
    return any(owner.strip().lower() == auth for owner in owner_emails)


def _owner_match_label(match: bool | None) -> str:
    if match is True:
        return "соответствует"
    if match is False:
        return "не соответствует"
    return "н/д"


@dataclass(frozen=True)
class GDriveSettings:
    folder_id: str
    credentials: Any
    auth_mode: str
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


def parse_folder_id(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise RuntimeError("GDRIVE_FOLDER_ID or drive_url is required")
    for pattern in _FOLDER_ID_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    raise RuntimeError(f"Cannot parse Google Drive folder ID from: {value!r}")


def _oauth_parts() -> tuple[str, str, str]:
    refresh_token = os.getenv("GOOGLE_DRIVE_OAUTH_REFRESH_TOKEN", "").strip()
    client_id = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET", "").strip()

    oauth_json = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS", "").strip()
    if oauth_json:
        data = json.loads(oauth_json)
        refresh_token = data.get("refresh_token", refresh_token) or refresh_token
        client_id = data.get("client_id", client_id) or client_id
        client_secret = data.get("client_secret", client_secret) or client_secret
        installed = data.get("installed") or data.get("web") or {}
        client_id = installed.get("client_id", client_id) or client_id
        client_secret = installed.get("client_secret", client_secret) or client_secret

    return refresh_token, client_id, client_secret


def build_drive_credentials() -> tuple[Any, str]:
    refresh_token, client_id, client_secret = _oauth_parts()
    if refresh_token and client_id and client_secret:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=DRIVE_SCOPES,
        )
        creds.refresh(Request())
        print("auth: Google Drive OAuth user credentials")
        return creds, "oauth"

    credentials_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "").strip()
    if credentials_json:
        info = json.loads(credentials_json)
        print("auth: Google Drive service account")
        return (
            service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES),
            "service_account",
        )

    raise RuntimeError(
        "Google Drive auth is required. Set GOOGLE_DRIVE_OAUTH_CREDENTIALS "
        "(recommended for My Drive) or GOOGLE_DRIVE_CREDENTIALS (service account)."
    )


def load_settings() -> GDriveSettings:
    credentials, auth_mode = build_drive_credentials()

    folder_raw = os.getenv("GDRIVE_FOLDER_ID", "").strip() or os.getenv("GDRIVE_URL", "").strip()
    folder_id = parse_folder_id(folder_raw)

    create_image_boxes_preview = env_bool("CREATE_IMAGE_BOXES_PREVIEW", False) or env_bool(
        "CREATE_BOXES_PREVIEW", False
    )
    json_indent_raw = os.getenv("JSON_INDENT", "2").strip()
    json_indent = None if json_indent_raw.lower() in {"none", "null", "0"} else env_int("JSON_INDENT", 2)

    return GDriveSettings(
        folder_id=folder_id,
        credentials=credentials,
        auth_mode=auth_mode,
        model_path=os.getenv("LOCAL_MODEL_PATH", "yolo11n.pt").strip() or "yolo11n.pt",
        yolo_confidence=env_float("YOLO_CONFIDENCE", 0.25),
        yolo_iou=env_float("YOLO_IOU", 0.70),
        yolo_image_size=env_int("YOLO_IMAGE_SIZE", 1280),
        yolo_device=os.getenv("YOLO_DEVICE", "cpu").strip() or "cpu",
        max_image_edge=env_int("MAX_IMAGE_EDGE", 1600),
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


class GDriveClient(RemoteClient):
    def __init__(self, folder_id: str, credentials: Any, auth_mode: str) -> None:
        self.service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.auth_mode = auth_mode
        self.root_folder_id = folder_id
        self._files_by_path: dict[str, dict[str, str]] = {}
        self._folder_ids_by_path: dict[str, str] = {"": folder_id, ".": folder_id}
        self._flat_upload_prefixes: set[str] = set()

    def authenticated_identity(self) -> str:
        if self.auth_mode == "oauth":
            about = self.service.about().get(fields="user/emailAddress").execute()
            return str(about.get("user", {}).get("emailAddress", "unknown"))
        info = getattr(self.service._http.credentials, "service_account_email", None)
        return str(info or "service-account")

    def resolve_target_folder_id(self) -> str:
        meta = (
            self.service.files()
            .get(
                fileId=self.root_folder_id,
                fields="id,name,mimeType,shortcutDetails",
                supportsAllDrives=True,
            )
            .execute()
        )
        mime = meta.get("mimeType", "")
        if mime == FOLDER_MIME:
            return self.root_folder_id
        shortcut = meta.get("shortcutDetails") or {}
        target_id = shortcut.get("targetId")
        target_mime = shortcut.get("targetMimeType", "")
        if target_id and target_mime == FOLDER_MIME:
            print(f"target folder is a shortcut, using targetId: {target_id}")
            self.root_folder_id = target_id
            self._folder_ids_by_path[""] = target_id
            self._folder_ids_by_path["."] = target_id
            return target_id
        raise RuntimeError(
            f"Target {self.root_folder_id!r} is not a folder (mimeType={mime!r}). "
            "Pass a Google Drive folder URL, not a file URL."
        )

    def folder_access_report(self) -> dict[str, Any]:
        meta = (
            self.service.files()
            .get(
                fileId=self.root_folder_id,
                fields="id,name,mimeType,owners,shared,capabilities,driveId",
                supportsAllDrives=True,
            )
            .execute()
        )
        capabilities = meta.get("capabilities") or {}
        owner_emails = _owner_emails(meta)
        auth_email = self.authenticated_identity()
        owner_match = _owner_account_match(auth_email, owner_emails)
        return {
            "folder_id": meta.get("id"),
            "folder_name": meta.get("name"),
            "shared": meta.get("shared"),
            "drive_id": meta.get("driveId"),
            "owner_account_match": owner_match,
            "owner_account_match_label": _owner_match_label(owner_match),
            "can_add_children": capabilities.get("canAddChildren"),
            "can_edit": capabilities.get("canEdit"),
        }

    def write_denied_help(self) -> str:
        report = self.folder_access_report()
        lines = [
            "Google Drive write access denied for the target folder.",
            f"Folder: {report['folder_name']!r}",
            f"owner account match: {report['owner_account_match_label']}",
            f"canAddChildren: {report['can_add_children']}",
            f"canEdit: {report['can_edit']}",
        ]
        if self.auth_mode == "oauth":
            if report["owner_account_match"] is False:
                lines.append(
                    "OAuth account is not the folder owner. Sign in with the owner account "
                    "or grant Editor access to the OAuth account."
                )
            lines.extend(
                [
                    "",
                    "OAuth is configured, but this Google account cannot create files in that folder.",
                    "Fix options:",
                    "1) Run get_gdrive_oauth_token.py again and sign in with the folder owner account.",
                    "2) In Google Drive, share the folder with the OAuth account as Editor.",
                    "3) Do not use a public Viewer link only; API write needs Editor rights.",
                    "4) If the folder is on a Shared Drive, grant that account at least Content manager.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Service accounts cannot write into a regular My Drive folder.",
                    "Use OAuth for My Drive, or move media to a Shared Drive and grant the",
                    "service account Content manager access.",
                ]
            )
        return "\n".join(lines)

    def _list_children(self, folder_id: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        page_token: str | None = None
        query = f"'{folder_id}' in parents and trashed=false"
        while True:
            response = (
                self.service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return items

    def _walk(self, folder_id: str, prefix: str) -> None:
        for item in self._list_children(folder_id):
            name = item["name"]
            path = normalize_remote_path(f"{prefix}/{name}" if prefix else name)
            mime = item.get("mimeType", "")
            if mime == FOLDER_MIME:
                self._folder_ids_by_path[path] = item["id"]
                self._walk(item["id"], path)
            else:
                self._files_by_path[path] = {
                    "id": item["id"],
                    "parent_id": folder_id,
                    "name": name,
                }

    def _parent_folder_id(self, parent_path: str) -> str:
        parent_path = normalize_remote_path(parent_path)
        if parent_path in {"", "."}:
            return self.root_folder_id
        folder_id = self._folder_ids_by_path.get(parent_path)
        if folder_id:
            return folder_id
        return self._ensure_folder(parent_path)

    def _ensure_folder(self, path: str) -> str:
        path = normalize_remote_path(path)
        if path in self._folder_ids_by_path:
            return self._folder_ids_by_path[path]

        parent_path = posixpath.dirname(path)
        name = posixpath.basename(path)
        parent_id = self._parent_folder_id(parent_path)

        for item in self._list_children(parent_id):
            if item.get("mimeType") == FOLDER_MIME and item["name"] == name:
                self._folder_ids_by_path[path] = item["id"]
                return item["id"]

        try:
            created = (
                self.service.files()
                .create(
                    body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except HttpError as exc:
            if exc.resp.status == 403:
                self._flat_upload_prefixes.add(path)
                print(
                    f"warning: cannot create Drive folder {path!r}; "
                    f"nested outputs will use flat names in the parent folder",
                    file=sys.stderr,
                )
                return parent_id
            raise
        folder_id = created["id"]
        self._folder_ids_by_path[path] = folder_id
        return folder_id

    def _resolve_upload_target(self, path: str) -> tuple[str, str]:
        path = normalize_remote_path(path)
        parent_path = posixpath.dirname(path)
        name = posixpath.basename(path)

        for prefix in sorted(self._flat_upload_prefixes, key=len, reverse=True):
            if path == prefix or path.startswith(prefix + "/"):
                flat_parent = posixpath.dirname(prefix)
                suffix = path[len(prefix) :].lstrip("/")
                flat_name = posixpath.basename(prefix) + ("__" + suffix if suffix else "")
                parent_id = self._parent_folder_id(flat_parent)
                return parent_id, flat_name

        parent_id = self._parent_folder_id(parent_path)
        return parent_id, name

    def verify_write_access(self) -> None:
        self.resolve_target_folder_id()
        report = self.folder_access_report()
        print(
            "folder access: "
            f"name={report['folder_name']!r} "
            f"owner_account_match={report['owner_account_match_label']} "
            f"canAddChildren={report['can_add_children']}"
        )
        if report["can_add_children"] is False:
            raise RuntimeError(self.write_denied_help())

        test_name = ".describe-photos-action-write-test"
        media = MediaIoBaseUpload(io.BytesIO(b"ok"), mimetype="text/plain", resumable=False)
        created_id = None
        try:
            created = (
                self.service.files()
                .create(
                    body={"name": test_name, "parents": [self.root_folder_id]},
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            created_id = created["id"]
            print("write check: ok")
        except HttpError as exc:
            if exc.resp.status == 403:
                raise RuntimeError(self.write_denied_help()) from exc
            raise
        finally:
            if created_id:
                try:
                    self.service.files().delete(fileId=created_id, supportsAllDrives=True).execute()
                except Exception:
                    pass

    def list_files(self, root: str) -> list[str]:
        del root
        self._files_by_path.clear()
        self._folder_ids_by_path.clear()
        self._folder_ids_by_path[""] = self.root_folder_id
        self._folder_ids_by_path["."] = self.root_folder_id
        self._walk(self.root_folder_id, "")
        return sorted(self._files_by_path.keys())

    def _lookup_file(self, path: str) -> dict[str, str] | None:
        path = normalize_remote_path(path)
        cached = self._files_by_path.get(path)
        if cached:
            return cached

        parent_path = posixpath.dirname(path)
        name = posixpath.basename(path)
        parent_id = self._parent_folder_id(parent_path)
        for item in self._list_children(parent_id):
            if item.get("mimeType") != FOLDER_MIME and item["name"] == name:
                meta = {"id": item["id"], "parent_id": parent_id, "name": name}
                self._files_by_path[path] = meta
                return meta
        return None

    def exists(self, path: str) -> bool:
        path = normalize_remote_path(path)
        if self._lookup_file(path):
            return True

        parent_path = posixpath.dirname(path)
        name = posixpath.basename(path)
        parent_id = self._parent_folder_id(parent_path)
        for item in self._list_children(parent_id):
            if item.get("mimeType") == FOLDER_MIME and item["name"] == name:
                self._folder_ids_by_path[path] = item["id"]
                return True
        return False

    def download_bytes(self, path: str) -> bytes:
        meta = self._lookup_file(path)
        if not meta:
            raise FileNotFoundError(path)

        request = self.service.files().get_media(fileId=meta["id"])
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    def upload_bytes(self, path: str, data: bytes) -> None:
        path = normalize_remote_path(path)
        parent_id, name = self._resolve_upload_target(path)

        existing = self._lookup_file(path)
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/octet-stream", resumable=True)
        try:
            if existing:
                self.service.files().update(
                    fileId=existing["id"],
                    media_body=media,
                    supportsAllDrives=True,
                ).execute()
                return

            for item in self._list_children(parent_id):
                if item.get("mimeType") != FOLDER_MIME and item["name"] == name:
                    self.service.files().update(
                        fileId=item["id"],
                        media_body=media,
                        supportsAllDrives=True,
                    ).execute()
                    self._files_by_path[path] = {"id": item["id"], "parent_id": parent_id, "name": name}
                    return

            created = (
                self.service.files()
                .create(
                    body={"name": name, "parents": [parent_id]},
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            self._files_by_path[path] = {
                "id": created["id"],
                "parent_id": parent_id,
                "name": name,
            }
        except HttpError as exc:
            if exc.resp.status == 403:
                raise RuntimeError(self.write_denied_help()) from exc
            raise


def to_ftp_settings(settings: GDriveSettings):
    from analyze_ftp_photos import Settings

    return Settings(
        ftp_url="gdrive://local",
        ftp_user="gdrive",
        ftp_pass="gdrive",
        scan_dir=".",
        ftp_protocol="gdrive",
        ftp_port="",
        model_path=settings.model_path,
        yolo_confidence=settings.yolo_confidence,
        yolo_iou=settings.yolo_iou,
        yolo_image_size=settings.yolo_image_size,
        yolo_device=settings.yolo_device,
        max_image_edge=settings.max_image_edge,
        max_detections=settings.max_detections,
        process_images=settings.process_images,
        process_videos=settings.process_videos,
        save_yolo_txt=settings.save_yolo_txt,
        save_empty_yolo_txt=settings.save_empty_yolo_txt,
        create_image_boxes_preview=settings.create_image_boxes_preview,
        enable_video_tracking=settings.enable_video_tracking,
        video_tracker=settings.video_tracker,
        video_frame_interval_seconds=settings.video_frame_interval_seconds,
        video_max_frames_per_file=settings.video_max_frames_per_file,
        save_track_paths=settings.save_track_paths,
        track_path_max_points=settings.track_path_max_points,
        process_missing_side_outputs=settings.process_missing_side_outputs,
        force_reprocess=settings.force_reprocess,
        max_files_per_run=settings.max_files_per_run,
        json_indent=settings.json_indent,
        fail_on_errors=settings.fail_on_errors,
    )


def run() -> int:
    settings = load_settings()
    client = GDriveClient(settings.folder_id, settings.credentials, settings.auth_mode)
    ftp_settings = to_ftp_settings(settings)
    processed = 0
    errors = 0
    model = None

    try:
        print(f"scan root folder: {settings.folder_id}")
        client.verify_write_access()
        all_files = client.list_files(".")
        candidates: list[str] = []
        for path in all_files:
            if settings.process_images and is_image(path):
                candidates.append(path)
            elif settings.process_videos and is_video(path):
                candidates.append(path)

        print(f"media candidates: {len(candidates)}")

        for path in candidates:
            do_process, reason = should_process(client, path, ftp_settings)
            if not do_process:
                continue
            if settings.max_files_per_run > 0 and processed >= settings.max_files_per_run:
                print(f"limit reached: MAX_FILES_PER_RUN={settings.max_files_per_run}")
                break

            try:
                print(f"process reason: {reason}")
                if model is None:
                    model = load_model(ftp_settings)
                if is_image(path):
                    process_image(client, model, path, ftp_settings)
                elif is_video(path):
                    process_video(client, model, path, ftp_settings)
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
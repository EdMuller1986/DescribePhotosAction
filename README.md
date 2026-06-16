# DescribePhotosAction

GitHub Actions workflow for local YOLO object detection on media files stored on FTP/FTPS/SFTP.

The workflow is manual only. It scans a remote folder, finds images and videos, skips media that already has a sibling JSON result, and uploads analysis outputs back to the same FTP tree.

No OpenAI API is used.

## Repository structure

```text
.github/workflows/analyze-ftp-photos.yml
scripts/analyze_ftp_photos.py
requirements.txt
README.md
```

## Required secrets

Use the same FTP secrets as the deploy workflow:

| Secret | Meaning |
|---|---|
| `FTP_URL` | FTP host or URL, for example `example.com`, `ftp://example.com`, `ftps://example.com`, `sftp://example.com` |
| `FTP_USER` | FTP/SFTP username |
| `FTP_PASS` | FTP/SFTP password |

## Main variables

| Variable | Default | Meaning |
|---|---:|---|
| `FTP_PHOTO_DIR` | `photo` | Remote folder to scan recursively |
| `LOCAL_MODEL_PATH` | `yolo11n.pt` | YOLO model path or model name |
| `PROCESS_IMAGES` | `true` | Process image files |
| `PROCESS_VIDEOS` | `true` | Process video files |
| `SAVE_YOLO_TXT` | `true` | Save YOLO TXT labels |
| `CREATE_IMAGE_BOXES_PREVIEW` | `false` | Save `*.boxes.jpg` previews for images only |
| `ENABLE_VIDEO_TRACKING` | `true` | Use tracking for videos |
| `VIDEO_TRACKER` | `bytetrack` | `bytetrack` or `botsort` |
| `VIDEO_FRAME_INTERVAL_SECONDS` | `1` | Analyze one frame every N seconds |
| `VIDEO_MAX_FRAMES_PER_FILE` | `300` | Max analyzed frames per video, `0` means no limit |
| `PROCESS_MISSING_SIDE_OUTPUTS` | `true` | Reprocess when JSON exists but TXT/preview side outputs are missing |
| `FORCE_REPROCESS` | `false` | Reprocess even when JSON exists |

## Supported files

Images:

```text
jpg, jpeg, png, webp, gif, bmp, tif, tiff
```

Videos:

```text
mp4, mov, m4v, avi, mkv, webm
```

## Outputs for images

For `photo/example.jpg`:

```text
photo/example.json
photo/example.txt
photo/example.boxes.jpg   # only when CREATE_IMAGE_BOXES_PREVIEW=true
```

The TXT file uses the original YOLO detection format:

```text
class_id x_center y_center width height
```

All coordinates are normalized from `0` to `1`.

## Outputs for videos

For `photo/camera.mp4`:

```text
photo/camera.json
photo/camera.yolo_txt/frame_000000_t0000000000ms.txt
photo/camera.yolo_txt/frame_000025_t0000001000ms.txt
```

Video preview images are intentionally not generated.

When tracking is enabled, video JSON contains `track_id` values and a `tracks` summary with `first_seen_sec`, `last_seen_sec`, `duration_sec`, `frames_seen`, and optional path points.

## Manual run

Open GitHub Actions, select **Analyze FTP media with local YOLO**, and click **Run workflow**.

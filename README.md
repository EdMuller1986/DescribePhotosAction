# DescribePhotosAction

GitHub Actions workflows for local YOLO object detection on media files stored on FTP/FTPS/SFTP or Google Drive.

The workflows are manual only. They scan a remote folder, find images and videos, skip media that already has a sibling JSON result, and upload analysis outputs back to the same storage tree.

No OpenAI API is used.

## Repository structure

```text
.github/workflows/analyze-ftp-photos.yml
.github/workflows/analyze-gdrive-photos.yml
scripts/analyze_ftp_photos.py
scripts/analyze_gdrive_photos.py
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

## Google Drive workflow

### Required secrets

Use **OAuth** for a regular My Drive folder (recommended):

| Secret | Meaning |
|---|---|
| `GOOGLE_DRIVE_OAUTH_CREDENTIALS` | JSON with `client_id`, `client_secret`, `refresh_token` |

Or use a **service account** only for Shared Drive / Google Workspace folders:

| Secret | Meaning |
|---|---|
| `GOOGLE_DRIVE_CREDENTIALS` | Service account JSON key with Drive API access |

Service accounts can read a shared My Drive folder, but Google Drive returns `403 insufficientParentPermissions` when creating result files there. For My Drive, configure OAuth instead.

### OAuth setup via Google Cloud Shell (no local Python)

#### 1. Google Cloud Console

1. Open [Google Cloud Console](https://console.cloud.google.com/) and select your project.
2. **APIs & Services → Library** → enable **Google Drive API**.
3. **APIs & Services → OAuth consent screen**
   - User type: **External** (or Internal for Workspace)
   - Add scope: `https://www.googleapis.com/auth/drive`
   - **Test users** → add the Google account that owns the Drive folder
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**
   - Application type: **Web application**
   - **Authorized redirect URIs** → add exactly:
     ```text
     http://localhost:8080/
     ```
   - Copy `client_id` and `client_secret`

   Important: a **Desktop app** client often causes `Missing required parameter: redirect_uri`. Use **Web application** with the redirect URI above.

#### 2. Cloud Shell

1. Open [shell.cloud.google.com](https://shell.cloud.google.com/)
2. Clone this repository (or upload only `scripts/get_gdrive_oauth_token.py`):

```bash
git clone https://github.com/EdMuller1986/DescribePhotosAction.git
cd DescribePhotosAction
python scripts/get_gdrive_oauth_token.py
```

3. The script prints an authorization URL. Open it in your browser.
4. Sign in with the Google account that owns the Drive folder and click **Allow**.
5. The browser redirects to `http://localhost/?code=...` and shows a connection error. That is expected.
6. Copy the **full URL** from the address bar (or only the `code` value) and paste it back into Cloud Shell.
7. The script prints JSON like this:

```json
{
  "client_id": "123.apps.googleusercontent.com",
  "client_secret": "YOUR_CLIENT_SECRET",
  "refresh_token": "YOUR_REFRESH_TOKEN"
}
```

#### 3. GitHub secret

1. Repository → **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `GOOGLE_DRIVE_OAUTH_CREDENTIALS`
3. Value: the full JSON from step 2.7

After that, run **Analyze Google Drive media with local YOLO**. The log should contain:

```text
auth: Google Drive OAuth user credentials
write check: ok
```

If `refresh_token` is empty, revoke the app at [myaccount.google.com/permissions](https://myaccount.google.com/permissions) and run the script again.

### Service account alternative (Shared Drive only)

For service accounts, share the target Shared Drive folder with the service account email from JSON (`client_email`) as **Content manager**. Service accounts cannot write results into a regular My Drive folder.

### Manual run

Open GitHub Actions, select **Analyze Google Drive media with local YOLO**, click **Run workflow**, and paste the full folder URL, for example:

```text
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz
```

Bare folder IDs are also accepted.

The workflow uses the same processing variables as the FTP workflow.

## Manual run (FTP)

Open GitHub Actions, select **Analyze FTP media with local YOLO**, and click **Run workflow**.

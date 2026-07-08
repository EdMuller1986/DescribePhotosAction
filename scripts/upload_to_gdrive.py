import json
import os
import sys
import time
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def get_or_create_folder(service, folder_name: str) -> str:
    query = (
        f"name='{folder_name}' "
        "and mimeType='application/vnd.google-apps.folder' "
        "and trashed=false"
    )

    result = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name)",
            pageSize=1,
        )
        .execute()
    )

    files = result.get("files", [])
    if files:
        return files[0]["id"]

    folder = (
        service.files()
        .create(
            body={
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        )
        .execute()
    )

    print(f"Создана папка '{folder_name}'")
    return folder["id"]

def get_free_space(service):
    about = service.about().get(fields="storageQuota").execute()
    quota = about.get("storageQuota", {})
    limit = int(quota.get("limit", 0))
    usage = int(quota.get("usage", 0))
    free = limit - usage
    return free

def upload_file(service, file_path, folder_id):
    file_name = os.path.basename(file_path)
    metadata = {
        "name": file_name,
        "parents": [folder_id],
    }
    media = MediaFileUpload(file_path, resumable=True)
    uploaded = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id",
        )
        .execute()
    )
    print(f"Загружен: {file_name} (ID={uploaded['id']})")
    return uploaded['id']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Режим наблюдения за папкой")
    parser.add_argument("--check-space", action="store_true", help="Проверить свободное место на Google Drive")
    args = parser.parse_args()

    creds_json_str = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS")
    if not creds_json_str:
        print("Ошибка: GOOGLE_DRIVE_OAUTH_CREDENTIALS не настроен", file=sys.stderr)
        sys.exit(1)

    try:
        creds_json = json.loads(creds_json_str)
    except json.JSONDecodeError as ex:
        print(f"Ошибка JSON: {ex}", file=sys.stderr)
        sys.exit(1)

    SCOPES = ["https://www.googleapis.com/auth/drive"]
    creds = None

    # 1. Пытаемся загрузить как авторизованного пользователя
    if "refresh_token" in creds_json:
        try:
            creds = Credentials.from_authorized_user_info(creds_json, SCOPES)
        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить Credentials из JSON: {e}", file=sys.stderr)

    # 2. Если токен просрочен, но есть refresh_token - обновляем
    if creds and creds.expired and creds.refresh_token:
        print("Обновляем просроченный токен...")
        try:
            creds.refresh(Request())
        except Exception as e:
            print(f"Ошибка при обновлении токена: {e}", file=sys.stderr)
            creds = None

    # 3. Если всё еще нет валидных прав и мы в CI (GitHub Actions), выдаем ошибку
    if not creds or not creds.valid:
        if os.getenv("GITHUB_ACTIONS"):
            print("Ошибка: В среде GitHub Actions нет валидного токена и refresh_token.", file=sys.stderr)
            sys.exit(1)
        
        # Интерактивный вход (только локально)
        client_secrets_file = "/tmp/client_secrets.json"
        with open(client_secrets_file, "w") as f:
            json.dump(creds_json, f)
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
        creds = flow.run_local_server(port=0, open_browser=False)

    service = build("drive", "v3", credentials=creds)

    if args.check_space:
        free = get_free_space(service)
        print(f"FREE_SPACE_BYTES={free}")
        print(f"Свободное место на Google Drive: {free / 1024**3:.2f} GB")
        return

    folder_id = get_or_create_folder(service, "Download")
    upload_dir = "download"

    if not os.path.isdir(upload_dir):
        if args.watch:
            os.makedirs(upload_dir, exist_ok=True)
        else:
            print(f"Папка '{upload_dir}' не существует", file=sys.stderr)
            sys.exit(1)

    uploaded_files = set()
    print(f"Начинаем {'наблюдение за' if args.watch else 'загрузку из'} папки '{upload_dir}'...")

    while True:
        current_files_to_upload = []
        for root, _, files in os.walk(upload_dir):
            for file_name in files:
                if file_name.endswith(".aria2"):
                    continue
                file_path = os.path.join(root, file_name)
                aria2_control_file = file_path + ".aria2"
                if file_path not in uploaded_files and not os.path.exists(aria2_control_file):
                    current_files_to_upload.append(file_path)

        for file_path in current_files_to_upload:
            try:
                upload_file(service, file_path, folder_id)
                uploaded_files.add(file_path)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Файл удален локально: {file_path}")
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {e}", file=sys.stderr)

        if not args.watch:
            break
        time.sleep(10)

if __name__ == "__main__":
    main()

import json
import os
import sys
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


import time
import argparse

def get_or_create_folder(service, folder_name: str) -> str:
... (rest of the function remains the same)

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

def get_free_space(service):
    about = service.about().get(fields="storageQuota").execute()
    quota = about.get("storageQuota", {})
    limit = int(quota.get("limit", 0))
    usage = int(quota.get("usage", 0))
    free = limit - usage
    return free

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Режим наблюдения за папкой")
    parser.add_argument("--check-space", action="store_true", help="Проверить свободное место на Google Drive")
    args = parser.parse_args()

    creds_json_str = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS")
... (rest of credentials loading)

    service = build("drive", "v3", credentials=creds)

    if args.check_space:
        free = get_free_space(service)
        print(f"FREE_SPACE_BYTES={free}")
        print(f"Свободное место на Google Drive: {free / 1024**3:.2f} GB")
        return

    folder_id = get_or_create_folder(service, "Download")

    upload_dir = "download"

    if not os.path.isdir(upload_dir):
        # В режиме watch создаем папку если ее нет
        if args.watch:
            os.makedirs(upload_dir, exist_ok=True)
        else:
            print(f"Папка '{upload_dir}' не существует", file=sys.stderr)
            sys.exit(1)

    uploaded_files = set()

    print(f"Начинаем {'наблюдение за' if args.watch else 'загрузку из'} папки '{upload_dir}'...")

    while True:
        current_files_to_upload = []
        
        # Рекурсивно ищем файлы
        for root, _, files in os.walk(upload_dir):
            for file_name in files:
                # Игнорируем сервисные файлы aria2
                if file_name.endswith(".aria2"):
                    continue
                
                file_path = os.path.join(root, file_name)
                aria2_control_file = file_path + ".aria2"
                
                # Если файл еще не загружен и нет контрольного файла aria2
                if file_path not in uploaded_files and not os.path.exists(aria2_control_file):
                    current_files_to_upload.append(file_path)

        for file_path in current_files_to_upload:
            try:
                upload_file(service, file_path, folder_id)
                uploaded_files.add(file_path)
                
                # Удаляем файл после успешной загрузки, чтобы освободить место
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Файл удален локально: {file_path}")
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {e}", file=sys.stderr)

        if not args.watch:
            if not uploaded_files:
                print("Нет готовых файлов для загрузки", file=sys.stderr)
                sys.exit(1)
            break
        
        # В режиме ожидания проверяем каждые 10 секунд
        time.sleep(10)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

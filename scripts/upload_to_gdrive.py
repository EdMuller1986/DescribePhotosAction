import json
import os
import sys

from google.oauth2.credentials import Credentials
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


def main():
    creds_data = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS")
    if not creds_data:
        print("Ошибка: GOOGLE_DRIVE_OAUTH_CREDENTIALS не настроен", file=sys.stderr)
        sys.exit(1)

    try:
        creds_info = json.loads(creds_data)
    except json.JSONDecodeError as ex:
        print(f"Ошибка JSON: {ex}", file=sys.stderr)
        sys.exit(1)

    creds = Credentials.from_authorized_user_info(creds_info)
    service = build("drive", "v3", credentials=creds)

    folder_id = get_or_create_folder(service, "Download")

    upload_dir = "download"

    if not os.path.isdir(upload_dir):
        print(f"Папка '{upload_dir}' не существует", file=sys.stderr)
        sys.exit(1)

    uploaded_count = 0

    for root, _, files in os.walk(upload_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

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

            uploaded_count += 1
            print(f"Загружен: {file_name} (ID={uploaded['id']})")

    if uploaded_count == 0:
        print("Нет файлов для загрузки", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import json
import os
import sys

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def main():
    creds_data = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS")
    if not creds_data:
        print("Ошибка: GOOGLE_DRIVE_OAUTH_CREDENTIALS не настроен", file=sys.stderr)
        sys.exit(1)

    try:
        creds_info = json.loads(creds_data)
    except json.JSONDecodeError as ex:
        print(f"Ошибка: GOOGLE_DRIVE_OAUTH_CREDENTIALS не является валидным JSON: {ex}", file=sys.stderr)
        sys.exit(1)

    creds = Credentials.from_authorized_user_info(creds_info)
    service = build("drive", "v3", credentials=creds)

    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "root")
    upload_dir = "download"

    if not os.path.isdir(upload_dir):
        print(f"Ошибка: папка {upload_dir} не найдена", file=sys.stderr)
        sys.exit(1)

    uploaded_count = 0

    for root_dir, _, files in os.walk(upload_dir):
        for file_name in files:
            file_path = os.path.join(root_dir, file_name)

            file_metadata = {
                "name": file_name,
                "parents": [folder_id],
            }

            media = MediaFileUpload(file_path, resumable=True)

            uploaded = (
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                )
                .execute()
            )

            uploaded_count += 1
            print(f"Загружен: {file_name} (ID: {uploaded.get('id')})")

    if uploaded_count == 0:
        print("Ошибка: нет файлов для загрузки", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

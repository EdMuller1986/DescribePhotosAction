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


def main():
    creds_json_str = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS")
    if not creds_json_str:
        print("Ошибка: GOOGLE_DRIVE_OAUTH_CREDENTIALS не настроен", file=sys.stderr)
        sys.exit(1)

    try:
        creds_json = json.loads(creds_json_str)
    except json.JSONDecodeError as ex:
        print(f"Ошибка JSON: {ex}", file=sys.stderr)
        sys.exit(1)

    # Сохраняем client secrets во временный файл
    client_secrets_file = "/tmp/client_secrets.json"
    with open(client_secrets_file, "w") as f:
        json.dump(creds_json, f)

    SCOPES = ["https://www.googleapis.com/auth/drive"]

    creds = None
    token_file = "/tmp/token.json"

    # Если есть сохраненный token, загружаем его
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except Exception as e:
            print(f"Ошибка при загрузке token.json: {e}", file=sys.stderr)

    # Если нет валидного токена, создаем новый через Flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Обновляем токен...")
            creds.refresh(Request())
        else:
            print("Получаем новый токен через OAuth Flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES
            )
            # В GitHub Actions используем local_server=False
            creds = flow.run_local_server(port=0, open_browser=False)

        # Сохраняем токен для следующего использования
        with open(token_file, "w") as token:
            token.write(creds.to_json())

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

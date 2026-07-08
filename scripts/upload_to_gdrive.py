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

    SCOPES = ["https://www.googleapis.com/auth/drive"]
    creds = None

    # 1. Пытаемся загрузить как авторизованного пользователя (если есть refresh_token)
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
            print("Пожалуйста, получите refresh_token локально с помощью scripts/get_gdrive_oauth_token.py", file=sys.stderr)
            print("и обновите секрет GOOGLE_DRIVE_OAUTH_CREDENTIALS.", file=sys.stderr)
            sys.exit(1)
        
        # Только если мы НЕ в CI, пробуем интерактивный вход
        print("Получаем новый токен через интерактивный OAuth Flow...")
        # (Для этого нужен формат client_secrets.json)
        client_secrets_file = "/tmp/client_secrets.json"
        with open(client_secrets_file, "w") as f:
            json.dump(creds_json, f)
        
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
        creds = flow.run_local_server(port=0, open_browser=False)

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

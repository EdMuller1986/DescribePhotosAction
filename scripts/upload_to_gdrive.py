upload_to_gdrive.pyimport os
import json
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

creds_data = os.getenv("GOOGLE_DRIVE_OAUTH_CREDENTIALS")
if not creds_data:
    print("Ошибка: GOOGLE_DRIVE_OAUTH_CREDENTIALS не настроен", file=sys.stderr)
    sys.exit(1)

creds_info = json.loads(creds_data)
creds = Credentials.from_authorized_user_info(creds_info)

service = build("drive", "v3", credentials=creds)

folder_id = "root"  # Замени на ID нужной папки

for root_dir, dirs, files in os.walk("download"):
    for file_name in files:
        file_path = os.path.join(root_dir, file_name)
        file_metadata = {"name": file_name, "parents": [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        uploaded = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        print(f"Загружен: {file_name} (ID: {uploaded.get('id')})")

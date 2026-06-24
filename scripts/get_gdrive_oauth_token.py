#!/usr/bin/env python3
"""One-time helper to obtain a Google Drive OAuth refresh token.

Designed for Google Cloud Shell (https://shell.cloud.google.com/):
  pip install google-auth-oauthlib
  python scripts/get_gdrive_oauth_token.py

You can also pass client credentials via environment variables:
  GOOGLE_DRIVE_OAUTH_CLIENT_ID
  GOOGLE_DRIVE_OAUTH_CLIENT_SECRET
"""

from __future__ import annotations

import json
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def load_client_config() -> dict:
    client_id = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": ["http://localhost"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

    print("Enter OAuth Desktop client credentials from Google Cloud Console.")
    client_id = input("client_id: ").strip()
    client_secret = input("client_secret: ").strip()
    if not client_id or not client_secret:
        raise SystemExit("client_id and client_secret are required")

    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def extract_code(raw: str) -> str:
    value = raw.strip()
    if "code=" in value:
        return value.split("code=", 1)[1].split("&", 1)[0]
    return value


def main() -> int:
    client_config = load_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, scopes=[DRIVE_SCOPE])
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    print()
    print("1) Open this URL in your browser (same Google account that owns the Drive folder):")
    print(auth_url)
    print()
    print("2) Allow access. The browser will redirect to localhost and show an error page.")
    print("   Copy the FULL address bar URL, or only the value of the 'code' parameter.")
    print()
    raw_code = input("Paste redirect URL or code here: ")
    code = extract_code(raw_code)
    flow.fetch_token(code=code)

    refresh_token = flow.credentials.refresh_token
    if not refresh_token:
        print(
            "ERROR: refresh_token is empty. Revoke app access at "
            "https://myaccount.google.com/permissions and run again with prompt=consent.",
            file=sys.stderr,
        )
        return 1

    payload = {
        "client_id": client_config["installed"]["client_id"],
        "client_secret": client_config["installed"]["client_secret"],
        "refresh_token": refresh_token,
    }

    print()
    print("Save this JSON into GitHub secret GOOGLE_DRIVE_OAUTH_CREDENTIALS:")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
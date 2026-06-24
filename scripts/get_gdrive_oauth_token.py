#!/usr/bin/env python3
"""One-time helper to obtain a Google Drive OAuth refresh token.

Designed for Google Cloud Shell (https://shell.cloud.google.com/):
  python scripts/get_gdrive_oauth_token.py

Uses only Python stdlib (no pip install required).

Before running, configure OAuth client in Google Cloud Console:
  Credentials -> OAuth client ID -> Web application
  Authorized redirect URIs: http://localhost:8080/
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
REDIRECT_URI = "http://localhost:8080/"


def load_credentials() -> tuple[str, str]:
    client_id = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return client_id, client_secret

    print("Enter OAuth client credentials from Google Cloud Console.")
    print("Client type must be: Web application")
    print(f"Authorized redirect URI must include: {REDIRECT_URI}")
    print()
    client_id = input("client_id: ").strip()
    client_secret = input("client_secret: ").strip()
    if not client_id or not client_secret:
        raise SystemExit("client_id and client_secret are required")
    return client_id, client_secret


def build_auth_url(client_id: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": DRIVE_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)


def extract_code(raw: str) -> str:
    value = raw.strip()
    if "code=" in value:
        return value.split("code=", 1)[1].split("&", 1)[0]
    return value


def exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    body = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Token exchange failed ({exc.code}): {details}") from exc


def main() -> int:
    client_id, client_secret = load_credentials()
    auth_url = build_auth_url(client_id)

    print()
    print("Open this URL in your browser (same Google account that owns the Drive folder):")
    print(auth_url)
    print()
    print("After Allow, the browser goes to localhost:8080 and shows a connection error.")
    print("Copy the FULL address bar URL, or only the value of the 'code' parameter.")
    print()
    raw_code = input("Paste redirect URL or code here: ")
    code = extract_code(raw_code)
    token_response = exchange_code(client_id, client_secret, code)

    refresh_token = token_response.get("refresh_token")
    if not refresh_token:
        print(
            "ERROR: refresh_token is empty. Revoke app access at "
            "https://myaccount.google.com/permissions and run again.",
            file=sys.stderr,
        )
        print("Token response:", json.dumps(token_response, indent=2), file=sys.stderr)
        return 1

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }

    print()
    print("Save this JSON into GitHub secret GOOGLE_DRIVE_OAUTH_CREDENTIALS:")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
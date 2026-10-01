"""
modules/google_meet_service.py
==============================
Google Meet / Google Drive Cloud Recording Integration Module

Architecture:
  Google Drive / Workspace API (OAuth 2.0 / Service Account)
                        ↓
         Google Meet Recording Files Retrieval
                        ↓
            Media Stream Download (Temp File)
                        ↓
       Existing process_meeting_input Pipeline
    (Whisper → LLM → Schema → SQLite → Vector Store)
"""

import os
import tempfile
import requests
import time
import logging
from typing import Dict, Any, List, Optional

from modules.database import get_meeting_by_google_id
from modules.meeting_service import process_meeting_input

logger = logging.getLogger(__name__)


class GoogleAuthError(Exception):
    """Raised when Google API authentication or token acquisition fails."""
    pass


class GoogleApiError(Exception):
    """Raised when Google Drive / Meet API requests fail."""
    pass


class GoogleMeetService:
    """Service layer for Google OAuth 2.0, Drive recording retrieval, and pipeline import."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        drive_folder_id: Optional[str] = None,
        service_account_file: Optional[str] = None
    ):
        self.client_id = client_id or os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GOOGLE_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.getenv("GOOGLE_REFRESH_TOKEN")
        self.drive_folder_id = drive_folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        self.service_account_file = service_account_file or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def get_access_token(self) -> str:
        """
        Obtain or refresh Google OAuth 2.0 access token using refresh_token or client credentials.
        Raises GoogleAuthError if credentials are missing or authorization fails.
        """
        # Return cached token if valid (with 60s buffer)
        if self._cached_token and time.time() < (self._token_expires_at - 60):
            return self._cached_token

        if not self.refresh_token and not (self.client_id and self.client_secret):
            raise GoogleAuthError(
                "Missing required Google OAuth credentials. Ensure GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN are configured."
            )

        url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token"
        }

        try:
            resp = requests.post(url, data=payload, timeout=10)
            if resp.status_code != 200:
                raise GoogleAuthError(f"Google OAuth token refresh failed ({resp.status_code}): {resp.text}")

            data = resp.json()
            token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)

            if not token:
                raise GoogleAuthError("Google OAuth response did not contain an access_token.")

            self._cached_token = token
            self._token_expires_at = time.time() + expires_in
            logger.info("Successfully acquired new Google OAuth access token.")
            return token

        except requests.exceptions.RequestException as exc:
            raise GoogleAuthError(f"Network error during Google OAuth authentication: {exc}")

    def list_meet_recordings(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve Google Meet recording files from Google Drive API.
        """
        token = self.get_access_token()
        url = "https://www.googleapis.com/drive/v3/files"
        headers = {"Authorization": f"Bearer {token}"}
        
        target_folder = folder_id or self.drive_folder_id
        q_parts = [
            "(mimeType contains 'video/' or mimeType contains 'audio/' or name contains 'Meet' or name contains 'Recording')",
            "trashed = false"
        ]
        if target_folder:
            q_parts.append(f"'{target_folder}' in parents")

        params = {
            "q": " and ".join(q_parts),
            "fields": "files(id, name, mimeType, size, createdTime, webViewLink)",
            "pageSize": 50
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 403:
                raise GoogleApiError(f"Permission denied accessing Google Drive API ({resp.status_code}): {resp.text}")
            elif resp.status_code != 200:
                raise GoogleApiError(f"Failed to list Google Meet recordings ({resp.status_code}): {resp.text}")
            
            data = resp.json()
            return data.get("files", [])

        except requests.exceptions.RequestException as exc:
            raise GoogleApiError(f"Network error calling Google Drive API: {exc}")

    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Fetch metadata for a specific file ID in Google Drive."""
        token = self.get_access_token()
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"fields": "id, name, mimeType, size, createdTime"}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 404:
                raise GoogleApiError(f"Google Drive file '{file_id}' not found.")
            elif resp.status_code != 200:
                raise GoogleApiError(f"Error fetching Google file metadata ({resp.status_code}): {resp.text}")
            
            return resp.json()
        except requests.exceptions.RequestException as exc:
            raise GoogleApiError(f"Network error fetching file metadata: {exc}")

    def download_recording_file(self, file_id: str, output_file_path: str) -> str:
        """
        Stream and save a Google Drive recording file to disk.
        """
        token = self.get_access_token()
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    raise GoogleApiError(f"Google Drive download failed ({resp.status_code}): {resp.text}")
                with open(output_file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            return output_file_path

        except requests.exceptions.RequestException as exc:
            raise GoogleApiError(f"Failed to download Google Meet recording file '{file_id}': {exc}")

    def import_google_meet_recording(
        self,
        file_id: str,
        title: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a Google Meet recording file into the existing application processing pipeline.
        Enforces duplicate recording prevention using Google Drive file ID.
        """
        clean_id = str(file_id).strip()
        if not clean_id:
            raise ValueError("Invalid or empty Google Meet file ID provided.")

        # 1. Duplicate Check
        existing_meeting = get_meeting_by_google_id(clean_id, db_path=db_path)
        if existing_meeting:
            logger.info(f"Duplicate Google Meet import prevented for file '{clean_id}'.")
            return {
                "status": "duplicate",
                "message": f"Google Meet recording '{clean_id}' has already been imported.",
                "meeting_id": existing_meeting["meeting_id"],
                "title": existing_meeting.get("title")
            }

        # 2. Fetch File Metadata
        meta = self.get_file_metadata(clean_id)
        recording_name = title or meta.get("name") or f"Google Meet Recording {clean_id}"

        # 3. Download & Process via Existing Pipeline
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"gmeet_{clean_id}.mp4")

        try:
            logger.info(f"Downloading Google Meet recording '{clean_id}'...")
            self.download_recording_file(clean_id, temp_path)

            logger.info(f"Passing downloaded Google Meet recording to process_meeting_input pipeline...")
            result = process_meeting_input(
                file_path=temp_path,
                title=recording_name,
                google_meeting_id=clean_id,
                db_path=db_path
            )
            result["google_meeting_id"] = clean_id
            return result

        except Exception as exc:
            logger.exception(f"Processing failed for Google Meet recording '{clean_id}'")
            raise GoogleApiError(f"Google Meet processing pipeline failed: {exc}")

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except Exception:
                    pass

"""
modules/zoom_service.py
=======================
Zoom Cloud Recording Integration Module

Architecture:
  Zoom Server-to-Server OAuth / Webhook
                   ↓
        Zoom Cloud Recordings API
                   ↓
       Recording Download (Temp File)
                   ↓
     Existing process_meeting_input Pipeline
  (Whisper → LLM → Schema → SQLite → Vector Store)
"""

import os
import hmac
import hashlib
import tempfile
import requests
import base64
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from modules.database import get_meeting_by_zoom_id
from modules.meeting_service import process_meeting_input

logger = logging.getLogger(__name__)


class ZoomAuthError(Exception):
    """Raised when Zoom OAuth authentication fails."""
    pass


class ZoomApiError(Exception):
    """Raised when Zoom API requests fail."""
    pass


class ZoomService:
    """Service layer for Zoom S2S OAuth, recording retrieval, and meeting import."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        account_id: Optional[str] = None,
        webhook_secret_token: Optional[str] = None
    ):
        self.client_id = client_id or os.getenv("ZOOM_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ZOOM_CLIENT_SECRET")
        self.account_id = account_id or os.getenv("ZOOM_ACCOUNT_ID")
        self.webhook_secret_token = webhook_secret_token or os.getenv("ZOOM_WEBHOOK_SECRET_TOKEN", "default_secret")
        
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def get_access_token(self) -> str:
        """
        Obtain or return cached Zoom Server-to-Server OAuth access token.
        Raises ZoomAuthError if credentials are missing or authorization fails.
        """
        # Return cached token if still valid (with 60s buffer)
        if self._cached_token and time.time() < (self._token_expires_at - 60):
            return self._cached_token

        if not self.client_id or not self.client_secret or not self.account_id:
            raise ZoomAuthError(
                "Missing required Zoom OAuth credentials. Ensure ZOOM_CLIENT_ID, "
                "ZOOM_CLIENT_SECRET, and ZOOM_ACCOUNT_ID are configured."
            )

        url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={self.account_id}"
        auth_bytes = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        basic_auth = base64.b64encode(auth_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            resp = requests.post(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                err_text = resp.text
                raise ZoomAuthError(f"Zoom OAuth token request failed ({resp.status_code}): {err_text}")

            data = resp.json()
            token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)

            if not token:
                raise ZoomAuthError("Zoom OAuth response did not contain an access_token.")

            self._cached_token = token
            self._token_expires_at = time.time() + expires_in
            logger.info("Successfully acquired new Zoom OAuth access token.")
            return token

        except requests.exceptions.RequestException as exc:
            raise ZoomAuthError(f"Network error during Zoom OAuth authentication: {exc}")

    def list_cloud_recordings(
        self,
        user_id: str = "me",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve cloud recordings list for a given Zoom user.
        """
        token = self.get_access_token()
        url = f"https://api.zoom.us/v2/users/{user_id}/recordings"
        headers = {"Authorization": f"Bearer {token}"}
        
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                raise ZoomApiError(f"Failed to list Zoom recordings ({resp.status_code}): {resp.text}")
            
            data = resp.json()
            return data.get("meetings", [])

        except requests.exceptions.RequestException as exc:
            raise ZoomApiError(f"Network error calling Zoom API: {exc}")

    def download_recording_file(self, download_url: str, output_file_path: str) -> str:
        """
        Download a Zoom cloud recording file stream to local disk.
        """
        token = self.get_access_token()
        sep = "&" if "?" in download_url else "?"
        authenticated_url = f"{download_url}{sep}access_token={token}"

        try:
            with requests.get(authenticated_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(output_file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            return output_file_path

        except requests.exceptions.RequestException as exc:
            raise ZoomApiError(f"Failed to download Zoom recording file from '{download_url}': {exc}")

    def import_zoom_recording(
        self,
        recording_id: str,
        download_url: Optional[str] = None,
        title: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a Zoom cloud recording into the existing application processing pipeline.
        Enforces duplicate recording prevention using Zoom meeting ID.
        """
        clean_id = str(recording_id).strip()
        if not clean_id:
            raise ValueError("Invalid or empty Zoom recording ID provided.")

        # 1. Duplicate Check
        existing_meeting = get_meeting_by_zoom_id(clean_id, db_path=db_path)
        if existing_meeting:
            logger.info(f"Duplicate Zoom import prevented for recording '{clean_id}'.")
            return {
                "status": "duplicate",
                "message": f"Zoom recording '{clean_id}' has already been imported.",
                "meeting_id": existing_meeting["meeting_id"],
                "title": existing_meeting.get("title")
            }

        # 2. Fetch recording metadata if download_url is not provided
        recording_topic = title or f"Zoom Meeting {clean_id}"
        target_download_url = download_url

        if not target_download_url:
            recordings = self.list_cloud_recordings()
            target_meeting = None
            for rec in recordings:
                if str(rec.get("id")) == clean_id or str(rec.get("uuid")) == clean_id:
                    target_meeting = rec
                    break

            if not target_meeting:
                raise ZoomApiError(f"Zoom recording '{clean_id}' not found in accessible account recordings.")

            recording_topic = title or target_meeting.get("topic") or recording_topic

            # Find suitable audio/video file
            files = target_meeting.get("recording_files", [])
            for f in files:
                if f.get("file_type") in ("MP4", "M4A", "WAV", "AUDIO"):
                    target_download_url = f.get("download_url")
                    break

            if not target_download_url and files:
                target_download_url = files[0].get("download_url")

        if not target_download_url:
            raise ZoomApiError(f"No downloadable recording media files found for Zoom meeting '{clean_id}'.")

        # 3. Download & Process via Existing Pipeline
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"zoom_{clean_id}.mp4")

        try:
            logger.info(f"Downloading Zoom recording '{clean_id}' for pipeline processing...")
            self.download_recording_file(target_download_url, temp_path)

            logger.info(f"Passing downloaded Zoom recording to process_meeting_input pipeline...")
            result = process_meeting_input(
                file_path=temp_path,
                title=recording_topic,
                zoom_meeting_id=clean_id,
                db_path=db_path
            )
            result["zoom_meeting_id"] = clean_id
            return result

        except Exception as exc:
            logger.exception(f"Processing failed for Zoom recording '{clean_id}'")
            raise ZoomApiError(f"Zoom recording processing pipeline failed: {exc}")

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

    def validate_webhook_url(self, plain_token: str) -> Dict[str, str]:
        """
        Generate response payload for Zoom Webhook URL Validation challenge.
        Calculates HMAC-SHA256 hash using webhook secret token.
        """
        if not plain_token:
            raise ValueError("plain_token must be provided for webhook validation.")

        secret = self.webhook_secret_token.encode("utf-8")
        encrypted_token = hmac.new(secret, plain_token.encode("utf-8"), hashlib.sha256).hexdigest()

        return {
            "plainToken": plain_token,
            "encryptedToken": encrypted_token
        }

"""
modules/api_client.py
=====================
API Client for AI-Powered Career Intelligence Platform Streamlit Frontend

Communicates with the existing Flask backend API endpoints.
Does not perform local business logic or direct database access.
"""

import os
import requests
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")


class MeetingApiClient:
    """Client for communicating with the Meeting Intelligence Flask API backend."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("AUTH_TOKEN")

    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user account."""
        url = f"{self.base_url}/auth/register"
        resp = requests.post(url, json={"username": username, "email": email, "password": password}, timeout=10)
        data = resp.json()
        if resp.status_code in (200, 201) and data.get("status") == "ok":
            if data.get("token"):
                self.api_key = data["token"]
            return data
        raise RuntimeError(data.get("message", "Registration failed."))

    def login(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """Authenticate user credentials and receive session token."""
        url = f"{self.base_url}/auth/login"
        resp = requests.post(url, json={"username": username_or_email, "password": password}, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and data.get("status") == "ok":
            if data.get("token"):
                self.api_key = data["token"]
            return data
        raise RuntimeError(data.get("message", "Login failed. Invalid credentials."))

    def logout(self) -> bool:
        """Invalidate user session token."""
        url = f"{self.base_url}/auth/logout"
        headers = self._get_headers()
        try:
            requests.post(url, headers=headers, timeout=5)
        except Exception:
            pass
        self.api_key = None
        return True

    def health_check(self) -> Tuple[bool, str]:
        """Check if backend API health endpoint responds with 200 OK."""
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True, "Backend API operational"
            return False, f"Backend returned HTTP {resp.status_code}"
        except Exception as exc:
            return False, f"Backend health check failed: {exc}"

    def verify_connection(self) -> Tuple[bool, str]:
        """
        Verify API server reachability and authentication token.
        Returns (success: bool, message: str).
        """
        url = f"{self.base_url}/meetings"
        headers = self._get_headers()
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                return True, "Successfully connected to backend API."
            elif resp.status_code == 401:
                return False, "Authentication failed: Invalid or missing API key."
            else:
                return False, f"Server returned status code {resp.status_code}: {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, f"Could not connect to API server at {self.base_url}. Ensure app.py is running."
        except Exception as exc:
            return False, f"Connection error: {exc}"

    def list_meetings(self) -> List[Dict[str, Any]]:
        """List all processed meetings."""
        url = f"{self.base_url}/meetings"
        headers = self._get_headers()
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "ok":
            return data.get("meetings", [])
        raise RuntimeError(data.get("message", "Failed to list meetings."))


    def get_meeting_details(self, meeting_id: str) -> Dict[str, Any]:
        """Retrieve complete intelligence for a specific meeting."""
        url = f"{self.base_url}/meetings/{meeting_id}"
        headers = self._get_headers()
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            err_data = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
            err_msg = err_data.get("message", f"Meeting '{meeting_id}' not found.")
            raise RuntimeError(err_msg)
        data = resp.json()
        if data.get("status") == "ok":
            return data
        raise RuntimeError(data.get("message", f"Failed to get meeting {meeting_id}."))

    def get_all_meetings_knowledge(self) -> List[Dict[str, Any]]:
        """Retrieve full historical knowledge for all meetings."""
        url = f"{self.base_url}/meetings/knowledge"
        headers = self._get_headers()
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "ok":
            return data.get("meetings", [])
        raise RuntimeError(data.get("message", "Failed to retrieve meeting knowledge."))


    def process_meeting(
        self,
        file_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        transcript_text: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit media file or raw transcript to backend for processing.
        """
        url = f"{self.base_url}/meetings/process"
        headers = self._get_headers()
        data = {}
        if title:
            data["title"] = title
        if transcript_text:
            data["transcript"] = transcript_text

        files = None
        if file_bytes and filename:
            files = {"media_file": (filename, file_bytes)}

        resp = requests.post(url, headers=headers, data=data, files=files, timeout=300)
        if resp.status_code not in (200, 201):
            err_msg = resp.json().get("message", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"Processing failed ({resp.status_code}): {err_msg}")
        
        return resp.json()

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        content_type: Optional[str] = None,
        meeting_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_score: Optional[float] = None,
        deduplicate: bool = False
    ) -> Dict[str, Any]:
        """Execute semantic vector search via backend API."""
        url = f"{self.base_url}/search"
        headers = self._get_headers()
        payload = {
            "query": query,
            "top_k": top_k,
            "deduplicate": deduplicate
        }
        if content_type:
            payload["content_type"] = content_type
        if meeting_id:
            payload["meeting_id"] = meeting_id
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        if min_score is not None:
            payload["min_score"] = min_score

        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def ask_assistant(
        self,
        question: str,
        top_k: int = 5,
        content_type: Optional[str] = None,
        meeting_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute RAG question answering via backend API."""
        url = f"{self.base_url}/ask"
        headers = self._get_headers()
        payload = {
            "question": question,
            "top_k": top_k
        }
        if content_type:
            payload["content_type"] = content_type
        if meeting_id:
            payload["meeting_id"] = meeting_id
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_zoom_recordings(
        self,
        user_id: str = "me",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available Zoom cloud recordings via backend Zoom API integration."""
        url = f"{self.base_url}/zoom/recordings"
        headers = self._get_headers()
        params = {"user_id": user_id}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            err_msg = resp.json().get("message", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"Zoom list error ({resp.status_code}): {err_msg}")

        data = resp.json()
        if data.get("status") == "ok":
            return data.get("recordings", [])
        raise RuntimeError(data.get("message", "Failed to list Zoom recordings."))

    def import_zoom_recording(
        self,
        recording_id: str,
        download_url: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger backend import of a Zoom cloud recording into the processing pipeline."""
        url = f"{self.base_url}/zoom/import"
        headers = self._get_headers()
        payload = {"recording_id": recording_id}
        if download_url:
            payload["download_url"] = download_url
        if title:
            payload["title"] = title

        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        if resp.status_code not in (200, 201):
            err_msg = resp.json().get("message", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"Zoom import error ({resp.status_code}): {err_msg}")

        return resp.json()

    def list_google_recordings(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available Google Meet cloud recordings via backend API."""
        url = f"{self.base_url}/google/recordings"
        headers = self._get_headers()
        params = {}
        if folder_id:
            params["folder_id"] = folder_id

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            err_msg = resp.json().get("message", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"Google Meet list error ({resp.status_code}): {err_msg}")

        data = resp.json()
        if data.get("status") == "ok":
            return data.get("recordings", [])
        raise RuntimeError(data.get("message", "Failed to list Google Meet recordings."))

    def import_google_recording(self, file_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Trigger backend import of a Google Meet cloud recording into the processing pipeline."""
        url = f"{self.base_url}/google/import"
        headers = self._get_headers()
        payload = {"file_id": file_id}
        if title:
            payload["title"] = title

        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        if resp.status_code not in (200, 201):
            err_msg = resp.json().get("message", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise RuntimeError(f"Google Meet import error ({resp.status_code}): {err_msg}")

        return resp.json()

    def export_meeting_pdf(self, meeting_id: str) -> bytes:
        """Download executive PDF report bytes for a specific meeting."""
        url = f"{self.base_url}/meetings/{meeting_id}/export/pdf"
        headers = self._get_headers()
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"PDF export failed ({resp.status_code}): {resp.text}")
        return resp.content

    def export_meeting_csv(self, meeting_id: str) -> str:
        """Download structured CSV report string for a specific meeting."""
        url = f"{self.base_url}/meetings/{meeting_id}/export/csv"
        headers = self._get_headers()
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"CSV export failed ({resp.status_code}): {resp.text}")
        return resp.text

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze raw text using backend sentiment and NLP pipeline (/api/analyze)."""
        url = f"{self.base_url}/api/analyze"
        headers = self._get_headers()
        resp = requests.post(url, headers=headers, json={"text": text}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Text analysis failed ({resp.status_code}): {resp.text}")
        return resp.json()





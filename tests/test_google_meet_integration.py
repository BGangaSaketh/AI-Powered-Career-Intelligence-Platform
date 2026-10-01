"""
tests/test_google_meet_integration.py
======================================
Unit and Integration Tests for Google Meet Cloud Recording Integration (Milestone 4 Task 5)
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from modules.database import init_db, get_meeting_by_google_id, save_meeting
from modules.google_meet_service import GoogleMeetService, GoogleAuthError, GoogleApiError
from modules.schemas import MeetingIntelligence
from app import app


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


def test_google_auth_missing_credentials():
    """Verify GoogleAuthError is raised when OAuth environment credentials are missing."""
    service = GoogleMeetService(client_id="", client_secret="", refresh_token="")
    with pytest.raises(GoogleAuthError) as exc_info:
        service.get_access_token()
    assert "Missing required Google OAuth credentials" in str(exc_info.value)


@patch("modules.google_meet_service.requests.post")
def test_google_auth_success(mock_post):
    """Verify Google OAuth access token acquisition and caching."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "mock_gmeet_token_999",
        "expires_in": 3600
    }
    mock_post.return_value = mock_resp

    service = GoogleMeetService(client_id="id", client_secret="secret", refresh_token="refresh")
    token = service.get_access_token()
    assert token == "mock_gmeet_token_999"

    # Second call should return cached token without duplicate HTTP POST request
    token_cached = service.get_access_token()
    assert token_cached == "mock_gmeet_token_999"
    assert mock_post.call_count == 1


@patch("modules.google_meet_service.requests.get")
@patch("modules.google_meet_service.GoogleMeetService.get_access_token")
def test_list_meet_recordings(mock_token, mock_get):
    """Verify list_meet_recordings parses Google Drive API response cleanly."""
    mock_token.return_value = "mock_gmeet_token"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "files": [
            {
                "id": "gdrive_file_123",
                "name": "Google Meet - Architecture Review",
                "mimeType": "video/mp4",
                "createdTime": "2026-09-29T14:00:00Z"
            }
        ]
    }
    mock_get.return_value = mock_resp

    service = GoogleMeetService(client_id="id", client_secret="secret", refresh_token="ref")
    recordings = service.list_meet_recordings()
    assert len(recordings) == 1
    assert recordings[0]["name"] == "Google Meet - Architecture Review"


@patch("modules.google_meet_service.process_meeting_input")
@patch("modules.google_meet_service.GoogleMeetService.download_recording_file")
@patch("modules.google_meet_service.GoogleMeetService.get_file_metadata")
def test_import_google_meet_recording_and_duplicate_prevention(mock_meta, mock_download, mock_process, temp_db):
    """
    Verify end-to-end import of a Google Meet recording into the pipeline
    and duplicate import prevention using Google Drive file ID.
    """
    mock_meta.return_value = {
        "id": "gdrive_file_777",
        "name": "Google Meet - Product Sync"
    }
    mock_download.return_value = "/tmp/mock_gmeet.mp4"
    mock_process.return_value = {
        "status": "ok",
        "meeting_id": "mtg_gmeet_100",
        "title": "Google Meet - Product Sync",
        "summary": "Imported Google Meet summary."
    }

    service = GoogleMeetService(client_id="id", client_secret="secret", refresh_token="ref")

    # 1. First Import — Should execute full pipeline
    res1 = service.import_google_meet_recording("gdrive_file_777", db_path=temp_db)
    assert res1["status"] == "ok"
    assert res1["google_meeting_id"] == "gdrive_file_777"
    assert mock_process.call_count == 1

    # Simulate saving meeting with google_meeting_id in DB
    intel = MeetingIntelligence(summary="Imported summary", key_points=[], decisions=[], action_items=[], participants=[])
    save_meeting(intel, "Transcript", title="Google Meet - Product Sync", db_path=temp_db, google_meeting_id="gdrive_file_777")

    # 2. Second Import — Duplicate prevention check should catch existing google_meeting_id
    res2 = service.import_google_meet_recording("gdrive_file_777", db_path=temp_db)
    assert res2["status"] == "duplicate"
    assert "already been imported" in res2["message"]
    # Pipeline should NOT be called a second time
    assert mock_process.call_count == 1


@patch("modules.google_meet_service.requests.get")
@patch("modules.google_meet_service.GoogleMeetService.get_access_token")
def test_google_api_permission_denied_handling(mock_token, mock_get):
    """Verify permission denied error (403) raises GoogleApiError gracefully."""
    mock_token.return_value = "token"
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Insufficient Permission"
    mock_get.return_value = mock_resp

    service = GoogleMeetService(client_id="id", client_secret="secret", refresh_token="ref")
    with pytest.raises(GoogleApiError) as exc_info:
        service.list_meet_recordings()
    assert "Permission denied" in str(exc_info.value)

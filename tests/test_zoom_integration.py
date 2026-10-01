"""
tests/test_zoom_integration.py
===============================
Unit and Integration Tests for Zoom Cloud Recording Integration (Milestone 4 Task 4)
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from modules.database import init_db, get_meeting_by_zoom_id
from modules.zoom_service import ZoomService, ZoomAuthError, ZoomApiError
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from app import app


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


def test_zoom_auth_missing_credentials():
    """Verify ZoomAuthError is raised when OAuth environment credentials are missing."""
    service = ZoomService(client_id="", client_secret="", account_id="")
    with pytest.raises(ZoomAuthError) as exc_info:
        service.get_access_token()
    assert "Missing required Zoom OAuth credentials" in str(exc_info.value)


@patch("modules.zoom_service.requests.post")
def test_zoom_auth_success(mock_post):
    """Verify Zoom OAuth token retrieval and caching."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "mock_access_token_123",
        "expires_in": 3600
    }
    mock_post.return_value = mock_resp

    service = ZoomService(client_id="id", client_secret="secret", account_id="acc")
    token = service.get_access_token()
    assert token == "mock_access_token_123"

    # Second call should use cached token without additional HTTP POST request
    token_cached = service.get_access_token()
    assert token_cached == "mock_access_token_123"
    assert mock_post.call_count == 1


@patch("modules.zoom_service.requests.get")
@patch("modules.zoom_service.ZoomService.get_access_token")
def test_list_cloud_recordings(mock_token, mock_get):
    """Verify list_cloud_recordings parses API response cleanly."""
    mock_token.return_value = "mock_token"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meetings": [
            {
                "id": 84920481920,
                "topic": "Sprint Planning Zoom",
                "start_time": "2026-09-29T10:00:00Z",
                "recording_files": [{"download_url": "https://zoom.us/rec/dl/123", "file_type": "MP4"}]
            }
        ]
    }
    mock_get.return_value = mock_resp

    service = ZoomService(client_id="id", client_secret="secret", account_id="acc")
    recordings = service.list_cloud_recordings()
    assert len(recordings) == 1
    assert recordings[0]["topic"] == "Sprint Planning Zoom"


@patch("modules.zoom_service.process_meeting_input")
@patch("modules.zoom_service.ZoomService.download_recording_file")
@patch("modules.zoom_service.ZoomService.list_cloud_recordings")
def test_import_zoom_recording_and_duplicate_prevention(mock_list, mock_download, mock_process, temp_db):
    """
    Verify end-to-end import of a Zoom recording into the processing pipeline
    and subsequent duplicate import prevention using Zoom meeting ID.
    """
    mock_list.return_value = [
        {
            "id": 9988776655,
            "topic": "Q4 Strategy Meeting",
            "recording_files": [{"download_url": "https://zoom.us/dl/test", "file_type": "MP4"}]
        }
    ]
    mock_download.return_value = "/tmp/mock.mp4"
    mock_process.return_value = {
        "status": "ok",
        "meeting_id": "mtg_zoom_test_1",
        "title": "Q4 Strategy Meeting",
        "summary": "Imported Zoom recording summary."
    }

    service = ZoomService(client_id="id", client_secret="secret", account_id="acc")

    # 1. First Import — Should execute full pipeline
    result1 = service.import_zoom_recording("9988776655", db_path=temp_db)
    assert result1["status"] == "ok"
    assert result1["zoom_meeting_id"] == "9988776655"
    assert mock_process.call_count == 1

    # Simulate saving meeting with zoom_meeting_id in DB
    from modules.database import save_meeting
    intel = MeetingIntelligence(summary="Imported summary", key_points=[], decisions=[], action_items=[], participants=[])
    save_meeting(intel, "Transcript", title="Q4 Strategy Meeting", db_path=temp_db, zoom_meeting_id="9988776655")

    # 2. Second Import — Duplicate prevention check should catch existing zoom_meeting_id
    result2 = service.import_zoom_recording("9988776655", db_path=temp_db)
    assert result2["status"] == "duplicate"
    assert "already been imported" in result2["message"]
    # Pipeline should NOT be called a second time
    assert mock_process.call_count == 1


def test_webhook_url_validation_challenge():
    """Verify HMAC-SHA256 encryption calculation for Zoom Webhook URL Validation."""
    service = ZoomService(webhook_secret_token="test_secret_key")
    val_res = service.validate_webhook_url("plain_token_xyz")

    assert val_res["plainToken"] == "plain_token_xyz"
    assert "encryptedToken" in val_res
    assert len(val_res["encryptedToken"]) == 64  # SHA256 hex string length


def test_zoom_webhook_endpoint_url_validation():
    """Verify Flask endpoint POST /zoom/webhook responds correctly to validation challenge."""
    payload = {
        "event": "endpoint.url_validation",
        "payload": {
            "plainToken": "sample_plain_token_123"
        }
    }

    with app.test_client() as client:
        resp = client.post("/zoom/webhook", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["plainToken"] == "sample_plain_token_123"
        assert "encryptedToken" in data

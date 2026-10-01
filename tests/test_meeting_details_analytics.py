"""
tests/test_meeting_details_analytics.py
========================================
Unit and Integration Tests for Meeting Details & Analytics (Milestone 4 Task 2)
"""

import tempfile
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from modules.database import init_db, save_meeting, get_meeting
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.api_client import MeetingApiClient


@pytest.fixture
def test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


def test_meeting_data_isolation_and_analytics(test_db):
    """
    Verify that 2 distinct meetings saved to the database maintain complete data isolation
    and return correct, isolated meeting details & calculated analytics.
    """
    # 1. Save Meeting A
    intel_a = MeetingIntelligence(
        summary="Summary for Meeting A",
        key_points=["Key point A1", "Key point A2"],
        decisions=["Decision A1"],
        action_items=[
            ActionItem(task="Task A1", assigned_to="Alice", priority="High", status="Completed", deadline="Tomorrow"),
            ActionItem(task="Task A2", assigned_to="Bob", priority="Medium", status="Pending", deadline="Next week")
        ],
        participants=[
            Participant(name="Alice", responsibilities=["Manage product"]),
            Participant(name="Bob", responsibilities=["Develop API"])
        ]
    )
    meeting_id_a = save_meeting(intel_a, "Alice: Let's discuss product. Bob: Sure.", title="Meeting A Title", db_path=test_db)

    # 2. Save Meeting B
    intel_b = MeetingIntelligence(
        summary="Summary for Meeting B",
        key_points=["Key point B1"],
        decisions=["Decision B1", "Decision B2"],
        action_items=[
            ActionItem(task="Task B1", assigned_to="Charlie", priority="Low", status="Completed", deadline="Friday")
        ],
        participants=[
            Participant(name="Charlie", responsibilities=["Security audit"])
        ]
    )
    meeting_id_b = save_meeting(intel_b, "Charlie: Running security tests.", title="Meeting B Title", db_path=test_db)

    # 3. Retrieve Meeting A and verify attributes & analytics
    mtg_a = get_meeting(meeting_id_a, db_path=test_db)
    assert mtg_a["title"] == "Meeting A Title"
    assert mtg_a["summary"] == "Summary for Meeting A"
    assert len(mtg_a["decisions"]) == 1
    assert mtg_a["decisions"][0] == "Decision A1"
    assert len(mtg_a["action_items"]) == 2
    assert len(mtg_a["participants"]) == 2
    assert mtg_a["participants"][0]["name"] == "Alice"
    assert mtg_a["participants"][1]["name"] == "Bob"

    # Compute Meeting A Analytics
    actions_a = mtg_a["action_items"]
    completed_a = sum(1 for a in actions_a if a["status"].lower() == "completed")
    pending_a = len(actions_a) - completed_a
    assert completed_a == 1
    assert pending_a == 1

    # 4. Retrieve Meeting B and verify isolation from Meeting A
    mtg_b = get_meeting(meeting_id_b, db_path=test_db)
    assert mtg_b["title"] == "Meeting B Title"
    assert mtg_b["summary"] == "Summary for Meeting B"
    assert len(mtg_b["decisions"]) == 2
    assert "Decision A1" not in mtg_b["decisions"]
    assert len(mtg_b["action_items"]) == 1
    assert mtg_b["action_items"][0]["task"] == "Task B1"
    assert len(mtg_b["participants"]) == 1
    assert mtg_b["participants"][0]["name"] == "Charlie"


def test_empty_meeting_analytics_resilience(test_db):
    """
    Verify that an empty/minimal meeting object produces clean 0/empty analytics without crashing.
    """
    intel_empty = MeetingIntelligence(
        summary="",
        key_points=[],
        decisions=[],
        action_items=[],
        participants=[]
    )
    m_id = save_meeting(intel_empty, "", title="Empty Meeting", db_path=test_db)
    
    mtg = get_meeting(m_id, db_path=test_db)
    assert mtg is not None
    assert mtg["raw_transcript"] == ""
    assert mtg["metadata"]["word_count"] == 0
    assert len(mtg["action_items"]) == 0
    assert len(mtg["participants"]) == 0
    assert len(mtg["decisions"]) == 0


    # Calculated metrics
    num_action_items = len(mtg["action_items"])
    completed_actions = sum(1 for a in mtg["action_items"] if a.get("status", "").lower() == "completed")
    completion_rate = (completed_actions / num_action_items * 100) if num_action_items > 0 else 0.0

    assert num_action_items == 0
    assert completion_rate == 0.0


@patch("modules.api_client.requests.get")
def test_api_client_get_meeting_details(mock_get):
    """
    Verify MeetingApiClient correctly formats and returns meeting details and metadata.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "meeting_id": "mtg-100",
        "title": "Roadmap Alignment",
        "summary": "Detailed meeting summary",
        "transcript": "Full raw transcript text",
        "word_count": 4,
        "created_at": "2026-09-29 12:00:00",
        "meeting_status": "completed",
        "action_items": [
            {"task": "Deploy app", "assigned_to": "Dave", "priority": "High", "status": "Pending", "deadline": "2026-10-01"}
        ],
        "participants": [
            {"name": "Dave", "responsibilities": ["Deployment"]}
        ],
        "decisions": ["Approve release"],
        "deadlines": [{"task": "Deploy app", "deadline": "2026-10-01"}]
    }
    mock_get.return_value = mock_resp

    client = MeetingApiClient(base_url="http://localhost:5000")
    details = client.get_meeting_details("mtg-100")

    assert details["status"] == "ok"
    assert details["meeting_id"] == "mtg-100"
    assert details["title"] == "Roadmap Alignment"
    assert len(details["action_items"]) == 1
    assert details["action_items"][0]["assigned_to"] == "Dave"
    assert len(details["participants"]) == 1
    assert len(details["decisions"]) == 1

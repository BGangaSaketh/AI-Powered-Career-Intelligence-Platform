"""
tests/test_database.py
======================
Unit tests for SQLite database creation, persistence, relationships, and retrieval.
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import (
    init_db,
    save_meeting,
    get_meeting,
    list_meetings,
    get_db_connection
)


class TestDatabase:

    @pytest.fixture
    def temp_db(self):
        """Fixture providing a temporary isolated SQLite database file."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(db_path)
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_init_db_creates_tables(self, temp_db):
        conn = get_db_connection(temp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cur.fetchall()]
        conn.close()

        expected_tables = ["meetings", "transcripts", "participants", "action_items", "decisions", "key_points"]
        for table in expected_tables:
            assert table in tables

    def test_save_and_get_meeting(self, temp_db):
        intelligence = MeetingIntelligence(
            meeting_id="mtg_test_001",
            summary="Test executive summary.",
            key_points=["Key point 1", "Key point 2"],
            decisions=["Decision A"],
            action_items=[
                ActionItem(
                    task="Setup database schema",
                    assigned_to="Ravi",
                    deadline="Friday",
                    priority="High",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="Ravi", responsibilities=["Setup database schema"])
            ]
        )

        raw_transcript = "Ravi will setup database schema by Friday."
        saved_id = save_meeting(intelligence, raw_transcript, title="Sprint Review", db_path=temp_db)
        assert saved_id == "mtg_test_001"

        retrieved = get_meeting("mtg_test_001", db_path=temp_db)
        assert retrieved is not None
        assert retrieved["meeting_id"] == "mtg_test_001"
        assert retrieved["title"] == "Sprint Review"
        assert retrieved["summary"] == "Test executive summary."
        assert len(retrieved["key_points"]) == 2
        assert len(retrieved["decisions"]) == 1
        assert len(retrieved["action_items"]) == 1
        assert retrieved["action_items"][0]["task"] == "Setup database schema"
        assert retrieved["action_items"][0]["assigned_to"] == "Ravi"
        assert len(retrieved["participants"]) == 1
        assert retrieved["participants"][0]["name"] == "Ravi"
        assert retrieved["raw_transcript"] == raw_transcript

    def test_list_meetings(self, temp_db):
        intelligence = MeetingIntelligence(
            meeting_id="mtg_list_01",
            summary="Summary 1",
            key_points=[],
            decisions=[],
            action_items=[],
            participants=[]
        )
        save_meeting(intelligence, "Transcript 1", title="Meeting 1", db_path=temp_db)

        meetings = list_meetings(db_path=temp_db)
        assert len(meetings) == 1
        assert meetings[0]["id"] == "mtg_list_01"
        assert meetings[0]["title"] == "Meeting 1"

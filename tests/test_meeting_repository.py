"""
tests/test_meeting_repository.py
=================================
Unit & Integration tests for Milestone 3 Task 1 - Meeting Knowledge Repository.

Verifies:
- Meeting retrieval
- Transcript retrieval
- Summary retrieval
- Decision retrieval
- Action item retrieval
- Participant retrieval
- Deadline retrieval
- Correct meeting relationships
- Missing / incomplete records safe handling
- Multiple historical meetings retrieval
- Flask API sub-resource endpoints
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import (
    init_db,
    save_meeting,
    get_meeting,
    get_meeting_metadata,
    get_meeting_transcript,
    get_meeting_summary,
    get_meeting_decisions,
    get_meeting_action_items,
    get_meeting_participants,
    get_meeting_deadlines,
    get_complete_meeting,
    get_all_meetings_knowledge,
    get_db_connection
)


class TestMeetingKnowledgeRepository:

    @pytest.fixture
    def temp_db(self):
        """Provide a temporary isolated SQLite database for repository tests."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(db_path)
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def sample_meetings(self, temp_db):
        """Seed database with two sample historical meetings."""
        mtg1_intel = MeetingIntelligence(
            meeting_id="mtg_hist_101",
            summary="Q1 Roadmap & Tech Stack Alignment.",
            key_points=["Migrate to microservices", "Adopt GraphQL"],
            decisions=["Approved microservice architecture", "Selected PostgreSQL"],
            action_items=[
                ActionItem(
                    task="Draft RFC for microservices",
                    assigned_to="Alice",
                    deadline="2026-04-15",
                    priority="High",
                    status="Completed"
                ),
                ActionItem(
                    task="Setup GraphQL gateway",
                    assigned_to="Bob",
                    deadline="2026-05-01",
                    priority="Medium",
                    status="In Progress"
                ),
                ActionItem(
                    task="General discussions",
                    assigned_to="Charlie",
                    deadline="",
                    priority="Low",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="Alice", responsibilities=["Architecture RFC"]),
                Participant(name="Bob", responsibilities=["Gateway setup"])
            ]
        )
        tx1 = "Alice: Let's finalize the microservices plan. Bob will setup GraphQL by May 1st."
        save_meeting(mtg1_intel, tx1, title="Q1 Tech Planning", db_path=temp_db)

        mtg2_intel = MeetingIntelligence(
            meeting_id="mtg_hist_102",
            summary="Q2 Security & Compliance Audit.",
            key_points=["SOC2 Compliance", "Penetration Testing"],
            decisions=["Authorized 3rd party penetration test"],
            action_items=[
                ActionItem(
                    task="Hire security auditor",
                    assigned_to="Diana",
                    deadline="2026-06-10",
                    priority="High",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="Diana", responsibilities=["Security Lead"])
            ]
        )
        tx2 = "Diana: Security audit is scheduled for June 10. Third party team will run pen tests."
        save_meeting(mtg2_intel, tx2, title="Q2 Security Audit", db_path=temp_db)

        return ["mtg_hist_101", "mtg_hist_102"]

    # 1. Meeting Retrieval & Complete Hierarchical Knowledge
    def test_complete_meeting_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        data = get_complete_meeting(mtg_id, db_path=temp_db)
        assert data is not None
        assert data["meeting_id"] == mtg_id
        assert data["title"] == "Q1 Tech Planning"
        assert data["summary"] == "Q1 Roadmap & Tech Stack Alignment."
        assert "metadata" in data
        assert data["metadata"]["word_count"] > 0
        assert data["transcript"] == "Alice: Let's finalize the microservices plan. Bob will setup GraphQL by May 1st."
        assert len(data["key_points"]) == 2
        assert len(data["decisions"]) == 2
        assert len(data["action_items"]) == 3
        assert len(data["participants"]) == 2
        assert len(data["deadlines"]) == 2

    # 2. Metadata Retrieval
    def test_metadata_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        meta = get_meeting_metadata(mtg_id, db_path=temp_db)
        assert meta is not None
        assert meta["id"] == mtg_id
        assert meta["title"] == "Q1 Tech Planning"
        assert meta["summary"] == "Q1 Roadmap & Tech Stack Alignment."
        assert meta["status"] == "completed"
        assert "created_at" in meta
        assert meta["word_count"] > 0

    # 3. Transcript Retrieval
    def test_transcript_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        tx_data = get_meeting_transcript(mtg_id, db_path=temp_db)
        assert tx_data is not None
        assert tx_data["meeting_id"] == mtg_id
        assert "Alice: Let's finalize" in tx_data["raw_text"]
        assert tx_data["word_count"] > 0

    # 4. Summary Retrieval
    def test_summary_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[1]
        sum_data = get_meeting_summary(mtg_id, db_path=temp_db)
        assert sum_data is not None
        assert sum_data["meeting_id"] == mtg_id
        assert sum_data["summary"] == "Q2 Security & Compliance Audit."

    # 5. Decision Retrieval
    def test_decision_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        decisions = get_meeting_decisions(mtg_id, db_path=temp_db)
        assert decisions is not None
        assert len(decisions) == 2
        dec_texts = [d["decision_text"] for d in decisions]
        assert "Approved microservice architecture" in dec_texts
        assert "Selected PostgreSQL" in dec_texts

    # 6. Action Item Retrieval
    def test_action_item_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        actions = get_meeting_action_items(mtg_id, db_path=temp_db)
        assert actions is not None
        assert len(actions) == 3
        tasks = [a["task"] for a in actions]
        assert "Draft RFC for microservices" in tasks

    # 7. Participant Retrieval
    def test_participant_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        parts = get_meeting_participants(mtg_id, db_path=temp_db)
        assert parts is not None
        assert len(parts) == 2
        p_names = [p["name"] for p in parts]
        assert "Alice" in p_names
        assert "Bob" in p_names

    # 8. Deadline Retrieval
    def test_deadline_retrieval(self, temp_db, sample_meetings):
        mtg_id = sample_meetings[0]
        deadlines = get_meeting_deadlines(mtg_id, db_path=temp_db)
        assert deadlines is not None
        # Should filter out empty deadline
        assert len(deadlines) == 2
        dl_dates = [d["deadline"] for d in deadlines]
        assert "2026-04-15" in dl_dates
        assert "2026-05-01" in dl_dates

    # 9. Correct Meeting Relationships
    def test_meeting_relationships(self, temp_db, sample_meetings):
        mtg1 = sample_meetings[0]
        mtg2 = sample_meetings[1]

        actions_mtg1 = get_meeting_action_items(mtg1, db_path=temp_db)
        actions_mtg2 = get_meeting_action_items(mtg2, db_path=temp_db)

        # Confirm actions belong strictly to their respective meeting
        assert len(actions_mtg1) == 3
        assert len(actions_mtg2) == 1
        assert actions_mtg2[0]["assigned_to"] == "Diana"

    # 10. Multiple Historical Meetings Retrieval
    def test_multiple_historical_meetings(self, temp_db, sample_meetings):
        all_meetings = get_all_meetings_knowledge(db_path=temp_db)
        assert len(all_meetings) == 2
        ids = [m["meeting_id"] for m in all_meetings]
        assert "mtg_hist_101" in ids
        assert "mtg_hist_102" in ids

    # 11. Missing & Incomplete Records Handling
    def test_missing_and_incomplete_records(self, temp_db):
        non_existent = "mtg_non_existent_999"

        # Non-existent meeting returns None across functions
        assert get_meeting(non_existent, db_path=temp_db) is None
        assert get_meeting_metadata(non_existent, db_path=temp_db) is None
        assert get_meeting_transcript(non_existent, db_path=temp_db) is None
        assert get_meeting_summary(non_existent, db_path=temp_db) is None
        assert get_meeting_decisions(non_existent, db_path=temp_db) is None
        assert get_meeting_action_items(non_existent, db_path=temp_db) is None
        assert get_meeting_participants(non_existent, db_path=temp_db) is None
        assert get_meeting_deadlines(non_existent, db_path=temp_db) is None

        # Partial record in database (meeting inserted directly without transcripts or sub-tables)
        conn = get_db_connection(temp_db)
        conn.execute(
            "INSERT INTO meetings (id, title, summary, status) VALUES (?, ?, ?, ?)",
            ("mtg_bare", "Bare Meeting", None, "completed")
        )
        conn.commit()
        conn.close()

        bare = get_meeting("mtg_bare", db_path=temp_db)
        assert bare is not None
        assert bare["summary"] == ""
        assert bare["raw_transcript"] == ""
        assert bare["key_points"] == []
        assert bare["decisions"] == []
        assert bare["action_items"] == []
        assert bare["participants"] == []
        assert bare["deadlines"] == []

    # 12. API Endpoint Verification
    def test_api_repository_endpoints(self, temp_db, sample_meetings):
        # Patch DEFAULT_DB_PATH for app testing
        old_env = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = temp_db

        try:
            with app.test_client() as client:
                mtg_id = sample_meetings[0]

                # Complete Meeting
                res = client.get(f"/api/meetings/{mtg_id}")
                assert res.status_code == 200
                data = res.get_json()
                assert data["status"] == "ok"
                assert data["title"] == "Q1 Tech Planning"

                # Transcript endpoint
                res = client.get(f"/api/meetings/{mtg_id}/transcript")
                assert res.status_code == 200
                assert res.get_json()["word_count"] > 0

                # Summary endpoint
                res = client.get(f"/api/meetings/{mtg_id}/summary")
                assert res.status_code == 200
                assert res.get_json()["summary"] != ""

                # Decisions endpoint
                res = client.get(f"/api/meetings/{mtg_id}/decisions")
                assert res.status_code == 200
                assert len(res.get_json()["decisions"]) == 2

                # Action Items endpoint
                res = client.get(f"/api/meetings/{mtg_id}/action-items")
                assert res.status_code == 200
                assert len(res.get_json()["action_items"]) == 3

                # Participants endpoint
                res = client.get(f"/api/meetings/{mtg_id}/participants")
                assert res.status_code == 200
                assert len(res.get_json()["participants"]) == 2

                # Deadlines endpoint
                res = client.get(f"/api/meetings/{mtg_id}/deadlines")
                assert res.status_code == 200
                assert len(res.get_json()["deadlines"]) == 2

                # Knowledge repository endpoint
                res = client.get("/api/meetings/knowledge")
                assert res.status_code == 200
                assert len(res.get_json()["meetings"]) == 2

                # 404 for missing meeting ID
                res = client.get("/api/meetings/mtg_missing_123/transcript")
                assert res.status_code == 404
        finally:
            if old_env is not None:
                os.environ["DATABASE_PATH"] = old_env
            else:
                os.environ.pop("DATABASE_PATH", None)

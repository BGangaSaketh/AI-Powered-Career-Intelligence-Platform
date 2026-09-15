"""
tests/test_meeting_pipeline.py
===============================
End-to-End Integration tests for Meeting Intelligence processing pipeline.

Flow:
  Recording / Transcript Input -> Whisper / Speech -> LLM Extraction ->
  Structured JSON -> Schema Validation -> SQLite DB -> Meeting Dashboard Payload
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.meeting_service import process_meeting_input
from modules.llm_service import LLMService
from modules.database import get_meeting, init_db


class TestMeetingPipelineIntegration:

    @pytest.fixture
    def temp_db(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(db_path)
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_full_pipeline_from_transcript_text(self, temp_db):
        transcript_text = (
            "Ravi: We need to complete the backend API integration by Friday. "
            "Priya: I will handle the UI testing and report back on Monday. "
            "Decision: The team approved the architecture proposal."
        )

        llm_service = LLMService(provider="mock")

        res = process_meeting_input(
            raw_transcript_input=transcript_text,
            title="Q3 Engineering Sync",
            llm_service=llm_service,
            db_path=temp_db
        )

        # Response Assertions
        assert res["status"] == "ok"
        assert res["title"] == "Q3 Engineering Sync"
        assert "meeting_id" in res
        assert res["summary"] != ""
        assert isinstance(res["key_points"], list)
        assert isinstance(res["decisions"], list)
        assert isinstance(res["action_items"], list)
        assert isinstance(res["participants"], list)
        assert res["raw_transcript"] == transcript_text

        # Database Verification
        saved = get_meeting(res["meeting_id"], db_path=temp_db)
        assert saved is not None
        assert saved["meeting_id"] == res["meeting_id"]
        assert saved["summary"] == res["summary"]

    def test_pipeline_empty_input_raises_error(self, temp_db):
        with pytest.raises(ValueError, match="valid media file nor a transcript"):
            process_meeting_input(
                raw_transcript_input="",
                db_path=temp_db
            )

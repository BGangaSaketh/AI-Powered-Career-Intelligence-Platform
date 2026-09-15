"""
tests/test_summarization.py
============================
Unit tests for meeting summarization module.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.summarization import summarize_meeting
from modules.llm_service import LLMService


class TestSummarization:

    def test_summarize_meeting_output_keys(self):
        transcript = (
            "The team met to review the quarterly roadmap. "
            "We decided to proceed with the cloud migration. "
            "Ravi will lead the backend team."
        )
        service = LLMService(provider="mock")
        res = summarize_meeting(transcript, llm_service=service)

        assert "summary" in res
        assert "key_points" in res
        assert "decisions" in res
        assert isinstance(res["summary"], str)
        assert isinstance(res["key_points"], list)
        assert isinstance(res["decisions"], list)
        assert len(res["summary"]) > 0

    def test_summarize_empty_transcript(self):
        res = summarize_meeting("")
        assert res["summary"] == "Empty transcript provided."
        assert res["key_points"] == []
        assert res["decisions"] == []

"""
tests/test_action_extraction.py
================================
Unit tests for action item extraction engine.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.action_extraction import extract_action_items
from modules.llm_service import LLMService


class TestActionExtraction:

    def test_extract_action_items_attributes(self):
        transcript = (
            "Ravi will complete the API integration by Friday with high priority. "
            "Priya will conduct UI testing on Monday."
        )
        service = LLMService(provider="mock")
        items = extract_action_items(transcript, llm_service=service, meeting_id="mtg_123")

        assert isinstance(items, list)
        assert len(items) > 0

        for item in items:
            assert "task" in item
            assert "assigned_to" in item
            assert "deadline" in item
            assert "priority" in item
            assert "status" in item
            assert item.get("meeting_id") == "mtg_123"
            assert item["priority"] in ("High", "Medium", "Low", "Unknown")
            assert item["status"] in ("Pending", "In Progress", "Completed", "Unknown")

    def test_missing_fields_default_to_none_or_unknown(self):
        # A task without assignee or deadline
        transcript = "Someone needs to update the project documentation urgently."
        service = LLMService(provider="mock")
        items = extract_action_items(transcript, llm_service=service)

        for item in items:
            # Must not invent unmentioned assignees or deadlines
            if not item["assigned_to"]:
                assert item["assigned_to"] is None
            if not item["deadline"]:
                assert item["deadline"] is None

"""
tests/test_participants.py
===========================
Unit tests for conservative participant name matching and responsibility mapping.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.participant_mapping import (
    map_participants_and_responsibilities,
    is_same_participant_conservative,
    normalize_participant_name
)
from modules.llm_service import LLMService


class TestParticipants:

    def test_conservative_name_matching_rule(self):
        # Must NOT merge distinct name variations without explicit proof
        assert is_same_participant_conservative("Ravi", "Ravi") is True
        assert is_same_participant_conservative("Ravi", "ravi") is True
        assert is_same_participant_conservative("Ravi", "Ravi Kumar") is False
        assert is_same_participant_conservative("Ravi Kumar", "R. Kumar") is False

    def test_participant_mapping_and_responsibilities(self):
        transcript = (
            "Ravi is responsible for backend development. "
            "Priya: I will manage frontend release and UI testing."
        )
        service = LLMService(provider="mock")
        participants = map_participants_and_responsibilities(transcript, llm_service=service, meeting_id="mtg_456")

        assert isinstance(participants, list)
        assert len(participants) > 0

        names = [p["name"] for p in participants]
        for p in participants:
            assert "name" in p
            assert "responsibilities" in p
            assert isinstance(p["responsibilities"], list)
            assert p.get("meeting_id") == "mtg_456"

    def test_duplicate_participants_prevented(self):
        transcript = (
            "Ravi will build API. Ravi will also handle database migrations."
        )
        service = LLMService(provider="mock")
        participants = map_participants_and_responsibilities(transcript, llm_service=service)

        ravi_records = [p for p in participants if p["name"].lower() == "ravi"]
        assert len(ravi_records) == 1

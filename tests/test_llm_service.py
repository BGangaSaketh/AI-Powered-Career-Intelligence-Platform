"""
tests/test_llm_service.py
==========================
Unit tests for LLM service layer, schema validation, retries, and error handling.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.llm_service import LLMService, LLMServiceError
from modules.schemas import MeetingIntelligence, parse_and_validate_json


class TestLLMService:

    def test_mock_llm_service_successful_response(self):
        service = LLMService(provider="mock")
        transcript = (
            "Ravi: We need to complete the API integration by Friday. "
            "Priya: I will handle the UI testing on Monday. "
            "Decision: We decided to proceed with the release."
        )
        intelligence = service.analyze_transcript(transcript)
        assert isinstance(intelligence, MeetingIntelligence)
        assert intelligence.summary != ""
        assert len(intelligence.key_points) > 0
        assert len(intelligence.action_items) > 0
        assert len(intelligence.participants) > 0

    def test_empty_transcript_raises_error(self):
        service = LLMService(provider="mock")
        with pytest.raises(LLMServiceError):
            service.analyze_transcript("")

    def test_parse_and_validate_json_success(self):
        valid_json = json.dumps({
            "summary": "Team alignment meeting.",
            "key_points": ["Point 1", "Point 2"],
            "decisions": ["Decision 1"],
            "action_items": [
                {
                    "task": "Build feature",
                    "assigned_to": "Ravi",
                    "deadline": "Friday",
                    "priority": "High",
                    "status": "Pending"
                }
            ],
            "participants": [
                {
                    "name": "Ravi",
                    "responsibilities": ["Build feature"]
                }
            ]
        })
        parsed = parse_and_validate_json(valid_json)
        assert parsed.summary == "Team alignment meeting."
        assert len(parsed.action_items) == 1
        assert parsed.action_items[0].assigned_to == "Ravi"
        assert parsed.action_items[0].priority == "High"

    def test_parse_and_validate_json_markdown_wrapped(self):
        markdown_json = "```json\n" + json.dumps({
            "summary": "Wrapped JSON summary.",
            "key_points": ["Point A"],
            "decisions": [],
            "action_items": [],
            "participants": []
        }) + "\n```"
        parsed = parse_and_validate_json(markdown_json)
        assert parsed.summary == "Wrapped JSON summary."

    def test_parse_and_validate_json_malformed_json(self):
        malformed = "{ summary: 'broken', key_points: "
        with pytest.raises(ValueError, match="Invalid JSON format"):
            parse_and_validate_json(malformed)

    def test_parse_and_validate_json_schema_recovery(self):
        # Missing lists should be automatically coerced to default empty lists
        incomplete_json = json.dumps({
            "summary": "Summary without action items"
        })
        parsed = parse_and_validate_json(incomplete_json)
        assert parsed.summary == "Summary without action items"
        assert parsed.key_points == []
        assert parsed.action_items == []

    def test_retry_behavior_on_failure(self, monkeypatch):
        attempts = 0

        def failing_call(system, user):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("Temporary network timeout")
            return json.dumps({
                "summary": "Recovered after retries",
                "key_points": [],
                "decisions": [],
                "action_items": [],
                "participants": []
            })

        service = LLMService(provider="openai", max_retries=3, backoff_factor=0.01)
        monkeypatch.setattr(service, "_call_provider", failing_call)

        res = service.analyze_transcript("Sample meeting text.")
        assert res.summary == "Recovered after retries"
        assert attempts == 3

    def test_api_failure_raises_llm_service_error(self, monkeypatch):
        def failing_call(system, user):
            raise RuntimeError("API Authentication Error")

        service = LLMService(provider="openai", max_retries=2, backoff_factor=0.01)
        monkeypatch.setattr(service, "_call_provider", failing_call)

        with pytest.raises(LLMServiceError, match="failed after 2 retries"):
            service.analyze_transcript("Sample text.")

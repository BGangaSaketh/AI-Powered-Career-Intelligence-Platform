"""
tests/test_streamlit_dashboard.py
==================================
Integration & Verification Tests for Streamlit Dashboard Application
"""

import os
import pytest
from unittest.mock import MagicMock, patch
import streamlit_app
from modules.api_client import MeetingApiClient


def test_streamlit_app_imports():
    """Verify streamlit_app imports and initial setup functions properly."""
    assert hasattr(streamlit_app, "main")
    assert hasattr(streamlit_app, "render_login_view")
    assert hasattr(streamlit_app, "render_dashboard_page")
    assert hasattr(streamlit_app, "render_meetings_explorer_page")
    assert hasattr(streamlit_app, "render_upload_page")
    assert hasattr(streamlit_app, "render_search_page")
    assert hasattr(streamlit_app, "render_ai_assistant_page")


@patch("modules.api_client.requests.get")
def test_streamlit_api_client_integration(mock_get):
    """Verify Streamlit API client integration with backend mock."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "meetings": [
            {
                "id": "m-test-1",
                "title": "Strategy Sync",
                "summary": "Discussed roadmap.",
                "created_at": "2026-09-29",
                "participants": [{"name": "Alice"}],
                "action_items": [{"task": "Update docs", "priority": "High"}],
                "decisions": ["Approved plan"]
            }
        ]
    }
    mock_get.return_value = mock_resp

    client = MeetingApiClient(base_url="http://localhost:5000", api_key="valid-key")
    meetings = client.get_all_meetings_knowledge()

    assert len(meetings) == 1
    assert meetings[0]["id"] == "m-test-1"
    assert meetings[0]["title"] == "Strategy Sync"


@patch("modules.api_client.requests.post")
def test_streamlit_upload_integration(mock_post):
    """Verify meeting processing submission via API client."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "meeting_id": "m-new-123",
        "intelligence": {
            "summary": "Extracted meeting summary",
            "action_items": [],
            "decisions": []
        }
    }
    mock_post.return_value = mock_resp

    client = MeetingApiClient(base_url="http://localhost:5000")
    res = client.process_meeting(transcript_text="Speaker 1: Let's launch tomorrow.", title="Launch Sync")

    assert res["status"] == "ok"
    assert res["meeting_id"] == "m-new-123"
    assert res["intelligence"]["summary"] == "Extracted meeting summary"


@patch("modules.api_client.requests.post")
def test_streamlit_search_rag_integration(mock_post):
    """Verify semantic search and RAG API integration."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "question": "What is the launch date?",
        "answer": "The launch date is October 15.",
        "sources": [
            {
                "meeting_id": "m1",
                "meeting_title": "Launch Sync",
                "similarity_score": 0.92,
                "text": "Target launch date is set for October 15."
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = MeetingApiClient(base_url="http://localhost:5000")
    res = client.ask_assistant(question="What is the launch date?")

    assert res["status"] == "ok"
    assert "October 15" in res["answer"]
    assert len(res["sources"]) == 1

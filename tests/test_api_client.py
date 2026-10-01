"""
tests/test_api_client.py
========================
Unit tests for MeetingApiClient in modules/api_client.py
"""

import pytest
from unittest.mock import MagicMock, patch
from modules.api_client import MeetingApiClient


def test_api_client_initialization():
    client = MeetingApiClient(base_url="http://testserver:5000", api_key="secret-key")
    assert client.base_url == "http://testserver:5000"
    assert client.api_key == "secret-key"
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["X-API-Key"] == "secret-key"


@patch("requests.get")
def test_verify_connection_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    client = MeetingApiClient()
    success, message = client.verify_connection()
    assert success is True
    assert "Successfully connected" in message


@patch("requests.get")
def test_verify_connection_unauthorized(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_get.return_value = mock_response

    client = MeetingApiClient(api_key="wrong-key")
    success, message = client.verify_connection()
    assert success is False
    assert "Authentication failed" in message


@patch("requests.get")
def test_list_meetings(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "meetings": [{"id": "m1", "title": "Test Meeting"}]
    }
    mock_get.return_value = mock_response

    client = MeetingApiClient()
    meetings = client.list_meetings()
    assert len(meetings) == 1
    assert meetings[0]["id"] == "m1"


@patch("requests.post")
def test_semantic_search(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "query": "project roadmap",
        "total_results": 1,
        "results": [{"text": "roadmap discussion", "similarity_score": 0.85}]
    }
    mock_post.return_value = mock_response

    client = MeetingApiClient()
    res = client.semantic_search("project roadmap", top_k=3)
    assert res["status"] == "ok"
    assert len(res["results"]) == 1


@patch("requests.post")
def test_ask_assistant(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "question": "What were the key decisions?",
        "answer": "The project deadline was extended.",
        "sources": []
    }
    mock_post.return_value = mock_response

    client = MeetingApiClient()
    res = client.ask_assistant("What were the key decisions?")
    assert res["status"] == "ok"
    assert "extended" in res["answer"]

"""
tests/test_rag_search_ui.py
============================
Unit and Integration Tests for RAG Search & AI Assistant UI (Milestone 4 Task 3)
"""

import tempfile
import pytest
from unittest.mock import MagicMock, patch
from modules.database import init_db, save_meeting
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.api_client import MeetingApiClient
from modules.semantic_search import SemanticSearchService
from modules.rag_service import RAGService


from modules.embedding_service import generate_meeting_embeddings


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


def test_semantic_search_date_and_metadata_filtering(temp_db):
    """
    Verify semantic search service filters correctly by date and content type.
    """
    intel = MeetingIntelligence(
        summary="Database migration project kickoff meeting.",
        key_points=["Migrate PostgreSQL to SQLite for local development"],
        decisions=["DB migration set for November 15"],
        action_items=[
            ActionItem(task="Prepare schema scripts", assigned_to="Dave", priority="High", status="Pending")
        ],
        participants=[
            Participant(name="Dave", responsibilities=["Schema migration"])
        ]
    )
    m_id = save_meeting(intel, "Dave: We are starting database migration.", title="DB Migration Kickoff", db_path=temp_db)
    generate_meeting_embeddings(m_id, db_path=temp_db)

    search_svc = SemanticSearchService()
    
    # 1. Search with matching content type
    res_decisions = search_svc.search("migration", content_type="decision", db_path=temp_db)
    assert res_decisions["status"] == "ok"
    assert len(res_decisions["results"]) >= 1
    assert any("decision" in r["content_type"] for r in res_decisions["results"])

    # 2. Search with target meeting ID filter
    res_meeting = search_svc.search("migration", meeting_id=m_id, db_path=temp_db)
    assert res_meeting["status"] == "ok"
    assert len(res_meeting["results"]) >= 1
    assert res_meeting["results"][0]["meeting_id"] == m_id


def test_rag_assistant_grounding_and_unsupported_question(temp_db):
    """
    Verify RAG service generates grounded answers for relevant questions
    and returns a clean fallback for irrelevant/unsupported questions.
    """
    intel = MeetingIntelligence(
        summary="Discussion on mobile app rollout schedule.",
        key_points=["iOS and Android app launch date finalized"],
        decisions=["Mobile application launch deadline is December 10"],
        action_items=[
            ActionItem(task="Publish to App Store", assigned_to="Eve", priority="High", status="Pending")
        ],
        participants=[
            Participant(name="Eve", responsibilities=["App store submission"])
        ]
    )
    m_id = save_meeting(intel, "Eve: The mobile app launch deadline is December 10.", title="Mobile App Sync", db_path=temp_db)
    generate_meeting_embeddings(m_id, db_path=temp_db)


    rag_svc = RAGService()

    # 1. Relevant question
    res_relevant = rag_svc.answer_question("What deadline was decided for the mobile application?", db_path=temp_db)
    assert res_relevant["status"] == "ok"
    assert len(res_relevant["sources"]) > 0
    assert any("December 10" in (s.get("text") or s.get("relevant_snippet") or "") or "Mobile" in (s.get("meeting_title") or s.get("title") or "") for s in res_relevant["sources"])


    # 2. Irrelevant / Unsupported question (No matching context in DB)
    res_irrelevant = rag_svc.answer_question("What is the quantum computing budget for 2030?", similarity_threshold=0.3, db_path=temp_db)
    assert res_irrelevant["status"] == "ok"
    assert "couldn't find enough information" in res_irrelevant["answer"].lower()
    assert len(res_irrelevant["sources"]) == 0



@patch("modules.api_client.requests.post")
def test_api_client_rag_and_search(mock_post):
    """
    Verify MeetingApiClient sends and parses search and RAG API requests cleanly.
    """
    # Test Search API endpoint call
    mock_resp_search = MagicMock()
    mock_resp_search.status_code = 200
    mock_resp_search.json.return_value = {
        "status": "ok",
        "query": "database migration",
        "total_results": 1,
        "results": [
            {
                "meeting_id": "m1",
                "meeting_title": "DB Migration Kickoff",
                "created_at": "2026-09-29",
                "content_type": "decision",
                "similarity_score": 0.89,
                "text": "DB migration set for November 15"
            }
        ]
    }
    mock_post.return_value = mock_resp_search

    client = MeetingApiClient(base_url="http://localhost:5000")
    search_res = client.semantic_search(query="database migration", content_type="decision", top_k=3)
    assert search_res["status"] == "ok"
    assert len(search_res["results"]) == 1
    assert search_res["results"][0]["meeting_title"] == "DB Migration Kickoff"

    # Test Ask RAG API endpoint call
    mock_resp_ask = MagicMock()
    mock_resp_ask.status_code = 200
    mock_resp_ask.json.return_value = {
        "status": "ok",
        "question": "What deadline was decided for the mobile application?",
        "answer": "The mobile application launch deadline is December 10.",
        "sources": [
            {
                "meeting_id": "m2",
                "meeting_title": "Mobile App Sync",
                "created_at": "2026-09-29",
                "content_type": "decision",
                "similarity_score": 0.94,
                "text": "Mobile application launch deadline is December 10"
            }
        ]
    }
    mock_post.return_value = mock_resp_ask

    ask_res = client.ask_assistant(question="What deadline was decided for the mobile application?")
    assert ask_res["status"] == "ok"
    assert "December 10" in ask_res["answer"]
    assert len(ask_res["sources"]) == 1
    assert ask_res["sources"][0]["meeting_title"] == "Mobile App Sync"

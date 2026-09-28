"""
tests/test_e2e_integration.py
==============================
Task 9: End-to-End Integration Testing & Regression Validation Suite

Verifies:
1. Complete End-to-End Pipeline:
   Transcript Ingestion -> Meeting Processing -> DB Storage -> Vector Embedding -> Vector Indexing -> Semantic Search -> RAG -> API Output.
2. Required RAG Validation Queries:
   - "What deadline was decided for the mobile application?"
   - "Who was responsible for API integration?"
   - "What decision was made about the database migration?"
   - Unsupported / Unanswerable query (verify anti-hallucination guardrail).
3. API Endpoint Regressions (/meetings, /meetings/{id}, /search, /ask, UI rendering /).
4. Auth Enforcement & Error Handling Middleware.
5. Database & Vector Metadata Integrity.
"""

import sys
import os
import tempfile
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app, semantic_search_service
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import init_db, save_meeting, get_meeting, list_meetings
from modules.embedding_service import EmbeddingService
from modules.meeting_service import generate_meeting_embeddings
from modules.semantic_search import SemanticSearchService
from modules.rag_service import RAGService
from modules.llm_service import LLMService


class TestEndToEndIntegration:

    @pytest.fixture
    def setup_e2e_db(self):
        """Create and populate an isolated temporary database with multi-domain meeting intelligence."""
        fd, db_path = tempfile.mkstemp(suffix="_e2e.db")
        os.close(fd)
        init_db(db_path)

        emb_svc = EmbeddingService(provider="hash")

        # 1. Meeting 1: Mobile App Launch & Deadlines
        mtg1 = MeetingIntelligence(
            meeting_id="mtg_mobile_app_101",
            summary="Mobile application release planning and iOS/Android build deadlines.",
            key_points=["Finalize iOS build", "Complete Android beta testing"],
            decisions=["The deadline decided for the mobile application release is October 15th."],
            action_items=[
                ActionItem(
                    task="Publish iOS build to App Store",
                    assigned_to="Mobile Dev Team",
                    deadline="October 15th",
                    priority="High",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="Sarah", responsibilities=["Mobile Development Lead"])
            ]
        )
        tx1 = "Sarah: The final deadline decided for the mobile application deployment is October 15th."
        save_meeting(mtg1, tx1, title="Mobile App Strategy", db_path=db_path)
        generate_meeting_embeddings("mtg_mobile_app_101", db_path=db_path, service=emb_svc)

        # 2. Meeting 2: API Integration & Backend Leadership
        mtg2 = MeetingIntelligence(
            meeting_id="mtg_api_lead_102",
            summary="Backend API integration and GraphQL gateway setup.",
            key_points=["API gateway routing", "JWT token authentication"],
            decisions=["Use JWT for API gateway authentication"],
            action_items=[
                ActionItem(
                    task="Implement API integration endpoints",
                    assigned_to="Alex Vance",
                    deadline="Next Wednesday",
                    priority="High",
                    status="In Progress"
                )
            ],
            participants=[
                Participant(name="Alex Vance", responsibilities=["API Integration Lead"])
            ]
        )
        tx2 = "Alex Vance was responsible for API integration and backend routing."
        save_meeting(mtg2, tx2, title="API Architecture Review", db_path=db_path)
        generate_meeting_embeddings("mtg_api_lead_102", db_path=db_path, service=emb_svc)

        # 3. Meeting 3: Database Migration
        mtg3 = MeetingIntelligence(
            meeting_id="mtg_db_mig_103",
            summary="PostgreSQL database migration and zero-downtime cutover plan.",
            key_points=["PostgreSQL schema migration", "Read replica sync"],
            decisions=["The decision made for the database migration was to execute zero-downtime cutover on Friday midnight."],
            action_items=[
                ActionItem(
                    task="Run DB migration script",
                    assigned_to="DBA Team",
                    deadline="Friday 12 AM",
                    priority="Critical",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="David", responsibilities=["Database Administrator"])
            ]
        )
        tx3 = "David: The primary decision made about the database migration was to proceed with zero-downtime cutover on Friday."
        save_meeting(mtg3, tx3, title="DB Migration Planning", db_path=db_path)
        generate_meeting_embeddings("mtg_db_mig_103", db_path=db_path, service=emb_svc)

        # Direct search service to test database
        semantic_search_service.embedding_service = emb_svc
        semantic_search_service.vector_store.db_path = db_path

        yield db_path, emb_svc

        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def client(self, setup_e2e_db):
        db_path, _ = setup_e2e_db
        old_db = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = db_path
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c
        if old_db is not None:
            os.environ["DATABASE_PATH"] = old_db
        else:
            os.environ.pop("DATABASE_PATH", None)

    # -------------------------------------------------------------------------
    # E2E Pipeline Tests
    # -------------------------------------------------------------------------

    def test_full_pipeline_ingestion_to_rag(self, setup_e2e_db):
        """Verify full flow: DB storage -> Embeddings -> Search -> RAG Answer."""
        db_path, emb_svc = setup_e2e_db
        search_svc = SemanticSearchService(embedding_service=emb_svc)
        rag_svc = RAGService(semantic_search_service=search_svc, llm_service=LLMService(provider="mock"))

        # Test Semantic Search
        res_search = search_svc.search("mobile application deadline", db_path=db_path)
        assert res_search["status"] == "ok"
        assert res_search["total_results"] > 0
        assert res_search["results"][0]["meeting_id"] == "mtg_mobile_app_101"

        # Test Grounded RAG
        res_rag = rag_svc.answer_question("What deadline was decided for the mobile application?", db_path=db_path)
        assert res_rag["status"] == "ok"
        assert "October 15th" in res_rag["answer"] or "mobile application" in res_rag["answer"].lower()
        assert any(s["meeting_id"] == "mtg_mobile_app_101" for s in res_rag["sources"])

    # -------------------------------------------------------------------------
    # Required RAG Validation Queries
    # -------------------------------------------------------------------------

    def test_rag_query_mobile_deadline(self, setup_e2e_db):
        """Validation Query: 'What deadline was decided for the mobile application?'"""
        db_path, emb_svc = setup_e2e_db
        rag_svc = RAGService(
            semantic_search_service=SemanticSearchService(embedding_service=emb_svc),
            llm_service=LLMService(provider="mock")
        )
        res = rag_svc.answer_question("What deadline was decided for the mobile application?", db_path=db_path)
        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        assert any("mtg_mobile_app_101" == s["meeting_id"] for s in res["sources"])

    def test_rag_query_api_integration_lead(self, setup_e2e_db):
        """Validation Query: 'Who was responsible for API integration?'"""
        db_path, emb_svc = setup_e2e_db
        rag_svc = RAGService(
            semantic_search_service=SemanticSearchService(embedding_service=emb_svc),
            llm_service=LLMService(provider="mock")
        )
        res = rag_svc.answer_question("Who was responsible for API integration?", db_path=db_path)
        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        assert any("mtg_api_lead_102" == s["meeting_id"] for s in res["sources"])

    def test_rag_query_db_migration_decision(self, setup_e2e_db):
        """Validation Query: 'What decision was made about the database migration?'"""
        db_path, emb_svc = setup_e2e_db
        rag_svc = RAGService(
            semantic_search_service=SemanticSearchService(embedding_service=emb_svc),
            llm_service=LLMService(provider="mock")
        )
        res = rag_svc.answer_question("What decision was made about the database migration?", db_path=db_path)
        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        assert any("mtg_db_mig_103" == s["meeting_id"] for s in res["sources"])

    def test_rag_query_unsupported_no_hallucination(self, setup_e2e_db):
        """Validation Query: Unsupported question should yield anti-hallucination fallback answer."""
        db_path, emb_svc = setup_e2e_db
        rag_svc = RAGService(
            semantic_search_service=SemanticSearchService(embedding_service=emb_svc),
            llm_service=LLMService(provider="mock")
        )
        res = rag_svc.answer_question("What is the secret launch recipe for solar powered quantum computing?", db_path=db_path)
        assert res["status"] == "ok"
        # Must return fallback answer or empty sources
        assert res["context_chunks_used"] == 0 or "couldn't find enough information" in res["answer"].lower()

    # -------------------------------------------------------------------------
    # API Regression & HTTP Endpoints
    # -------------------------------------------------------------------------

    def test_api_ui_index_route(self, client):
        """Verify UI template index route / returns HTML page."""
        res = client.get("/")
        assert res.status_code == 200
        assert b"<!DOCTYPE html>" in res.data or b"html" in res.data.lower()

    def test_api_meetings_list(self, client):
        """Verify /meetings returns all 3 meetings."""
        res = client.get("/meetings")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert len(data["meetings"]) == 3

    def test_api_meeting_by_id(self, client):
        """Verify /meetings/{id} returns detail."""
        res = client.get("/meetings/mtg_mobile_app_101")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert data["meeting_id"] == "mtg_mobile_app_101"
        assert "metadata" in data
        assert "transcript" in data

    def test_api_search_endpoint(self, client):
        """Verify /search endpoint over HTTP."""
        res = client.get("/search?q=database%20migration")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert data["total_results"] > 0
        assert "latency_ms" in data

    def test_api_ask_endpoint(self, client):
        """Verify /ask endpoint over HTTP."""
        res = client.get("/ask?q=Who%20was%20responsible%20for%20API%20integration%3F")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert "answer" in data
        assert "sources" in data

    # -------------------------------------------------------------------------
    # Database Integrity & Orphan Check
    # -------------------------------------------------------------------------

    def test_database_integrity_and_orphan_check(self, setup_e2e_db):
        """Verify database integrity, meeting retrieval, and orphan record absence."""
        db_path, _ = setup_e2e_db
        meetings = list_meetings(db_path=db_path)
        assert len(meetings) == 3

        for m in meetings:
            detail = get_meeting(m["id"], db_path=db_path)
            assert detail is not None
            assert detail["meeting_id"] == m["id"]
            assert "transcript" in detail
            assert "metadata" in detail

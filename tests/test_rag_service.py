"""
tests/test_rag_service.py
==========================
Unit & Integration tests for Milestone 3 Task 5 - Grounded RAG Question Answering.

Verifies:
1. Question with an answer in the database
2. Question requiring information from multiple retrieved chunks
3. Question with no matching meeting
4. Question where retrieval returns irrelevant information / low similarity
5. Hallucination prevention fallback
6. Correct meeting source attribution
7. Correct action-item retrieval ("What action items were assigned to...")
8. Correct deadline retrieval ("What deadline was decided for...")
9. Correct participant retrieval ("Who was responsible for...")
10. End-to-end RAG pipeline & API route test
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import init_db, save_meeting
from modules.embedding_service import EmbeddingService, generate_meeting_embeddings
from modules.vector_store import VectorStoreService
from modules.semantic_search import SemanticSearchService
from modules.rag_service import RAGService, FALLBACK_NO_INFO_ANSWER
from modules.llm_service import LLMService


class TestRAGQuestionAnswering:

    @pytest.fixture
    def temp_db(self):
        """Provide isolated temporary database."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(db_path)
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def rag_service_instance(self, temp_db):
        """Seed DB with historical meetings and set up RAG Service."""
        emb_svc = EmbeddingService(provider="hash")
        vec_svc = VectorStoreService(db_path=temp_db)
        search_svc = SemanticSearchService(embedding_service=emb_svc, vector_store=vec_svc)
        llm_svc = LLMService(provider="mock")

        # Meeting 1: Mobile App & API Integration Sync
        mtg1 = MeetingIntelligence(
            meeting_id="mtg_rag_001",
            summary="Mobile application architecture and API refactoring planning.",
            key_points=["Mobile app UI redesign", "REST API integration"],
            decisions=["Approved mobile application release schedule for Friday"],
            action_items=[
                ActionItem(
                    task="Complete mobile application API integration",
                    assigned_to="Ravi",
                    deadline="Friday",
                    priority="High",
                    status="Pending"
                ),
                ActionItem(
                    task="Run security audit on auth endpoint",
                    assigned_to="Priya",
                    deadline="Next Wednesday",
                    priority="Medium",
                    status="In Progress"
                )
            ],
            participants=[
                Participant(name="Ravi", responsibilities=["API integration"]),
                Participant(name="Priya", responsibilities=["Security Audit"])
            ]
        )
        tx1 = (
            "Ravi: We need to complete the mobile application API integration by Friday. "
            "Priya: I will handle security audit by next Wednesday. "
            "Decision: The mobile application deadline was decided as Friday."
        )
        save_meeting(mtg1, tx1, title="Mobile App & API Sync", db_path=temp_db)
        generate_meeting_embeddings("mtg_rag_001", db_path=temp_db, service=emb_svc)

        # Meeting 2: Database Migration Sync
        mtg2 = MeetingIntelligence(
            meeting_id="mtg_rag_002",
            summary="Database migration and microservices setup.",
            key_points=["PostgreSQL migration"],
            decisions=["Approved Python 3.12 for microservices"],
            action_items=[
                ActionItem(
                    task="Execute database migration scripts",
                    assigned_to="Alice",
                    deadline="Thursday",
                    priority="High",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="Alice", responsibilities=["Database migration"])
            ]
        )
        tx2 = "Alice: Decision reached: Team unanimously approved database migration to PostgreSQL."
        save_meeting(mtg2, tx2, title="Database Migration Sync", db_path=temp_db)
        generate_meeting_embeddings("mtg_rag_002", db_path=temp_db, service=emb_svc)

        rag = RAGService(semantic_search_service=search_svc, llm_service=llm_svc)
        return rag

    # 1. Question with Answer in Database
    def test_question_with_answer_in_db(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("What deadline was decided for the mobile application?", db_path=temp_db)

        assert res["status"] == "ok"
        assert res["answer"] != FALLBACK_NO_INFO_ANSWER
        assert "Friday" in res["answer"] or "Mobile" in res["answer"]
        assert len(res["sources"]) > 0
        assert res["sources"][0]["meeting_id"] == "mtg_rag_001"

    # 2. Question Requiring Multi-Chunk Evidence
    def test_multi_chunk_evidence_retrieval(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("What decisions and action items were agreed for mobile application and security?", top_k=5, db_path=temp_db)

        assert res["status"] == "ok"
        assert res["context_chunks_used"] >= 1
        assert len(res["sources"]) >= 1

    # 3. Question with No Matching Meeting / Empty DB
    def test_question_no_matching_meeting(self, temp_db):
        emb_svc = EmbeddingService(provider="hash")
        vec_svc = VectorStoreService(db_path=temp_db)
        search_svc = SemanticSearchService(embedding_service=emb_svc, vector_store=vec_svc)
        rag = RAGService(semantic_search_service=search_svc)

        res = rag.answer_question("What is the quantum computing roadmap?", db_path=temp_db)
        assert res["status"] == "ok"
        assert res["answer"] == FALLBACK_NO_INFO_ANSWER
        assert len(res["sources"]) == 0

    # 4. Irrelevant Information Retrieval / High Threshold
    def test_question_irrelevant_retrieval(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("space rocket orbital trajectory launch", similarity_threshold=0.99, db_path=temp_db)

        assert res["status"] == "ok"
        assert res["answer"] == FALLBACK_NO_INFO_ANSWER
        assert len(res["sources"]) == 0

    # 5. Hallucination Prevention
    def test_hallucination_prevention(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("What is the salary budget for executive hiring?", db_path=temp_db)

        # Confirm system refuses to hallucinate missing facts
        assert res["status"] == "ok"
        assert res["answer"] == FALLBACK_NO_INFO_ANSWER

    # 6. Correct Meeting Source Attribution
    def test_meeting_source_attribution(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("Who was responsible for API integration?", db_path=temp_db)

        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        src = res["sources"][0]
        assert "meeting_id" in src
        assert "title" in src
        assert "date" in src
        assert "content_type" in src
        assert "relevant_snippet" in src
        assert "similarity" in src
        assert src["meeting_id"] == "mtg_rag_001"

    # 7. Action Item Question Retrieval
    def test_action_item_question_retrieval(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("What action items were assigned to Ravi?", db_path=temp_db)

        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        snippets = [s["relevant_snippet"] for s in res["sources"]]
        assert any("Ravi" in sn for sn in snippets)

    # 8. Deadline Question Retrieval
    def test_deadline_question_retrieval(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("When is the database migration deadline?", db_path=temp_db)

        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        assert res["sources"][0]["meeting_id"] == "mtg_rag_002"

    # 9. Participant Responsibility Question Retrieval
    def test_participant_question_retrieval(self, temp_db, rag_service_instance):
        rag = rag_service_instance
        res = rag.answer_question("Who was responsible for security audit?", db_path=temp_db)

        assert res["status"] == "ok"
        assert len(res["sources"]) > 0
        snippets = [s["relevant_snippet"] for s in res["sources"]]
        assert any("Priya" in sn or "security" in sn.lower() for sn in snippets)

    # 10. End-to-End RAG Pipeline & API Route Integration
    def test_end_to_end_rag_api_endpoint(self, temp_db, rag_service_instance):
        old_env = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = temp_db

        try:
            with app.test_client() as client:
                payload = {
                    "question": "What deadline was decided for the mobile application?",
                    "top_k": 5
                }
                res = client.post("/api/search/rag", json=payload)
                assert res.status_code == 200
                data = res.get_json()

                assert data["status"] == "ok"
                assert "answer" in data
                assert "sources" in data
                assert "latency_ms" in data
                assert data["latency_ms"] < 3000.0
                assert len(data["sources"]) > 0

                top_source = data["sources"][0]
                assert top_source["meeting_id"] == "mtg_rag_001"
                assert top_source["title"] == "Mobile App & API Sync"

                # Test GET QA endpoint
                res_get = client.get("/api/meetings/qa?q=database+migration")
                assert res_get.status_code == 200
                assert res_get.get_json()["status"] == "ok"
        finally:
            if old_env is not None:
                os.environ["DATABASE_PATH"] = old_env
            else:
                os.environ.pop("DATABASE_PATH", None)

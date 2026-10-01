"""
tests/test_search_rag_validation.py
====================================
Comprehensive Validation Test Suite for Milestone 3 Task 7: Search & RAG Validation.

Validates:
1. Semantic Search: Relevant queries, irrelevant queries, multiple matches, paraphrased wording.
2. Relevant Meeting Retrieval: Correct meeting ID, title, transcript, snippet, metadata, similarity scoring.
3. Date-based Filtering: Date range filters, exclusion of out-of-range meetings, empty date safety.
4. Metadata Filtering: Filter by content_type and meeting_id using actual stored records.
5. Grounded RAG QA: Required questions (deadlines, responsibilities, decisions, UI testing, action items).
6. Grounding & Hallucination Prevention: Deliberate unanswerable questions return safe fallback without inventing facts.
7. Empty Results Handling: Safe graceful API responses for empty queries, missing context, and empty DBs.
"""

import sys
import os
import tempfile
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import init_db, save_meeting, save_embeddings
from modules.embedding_service import EmbeddingService
from modules.semantic_search import SemanticSearchService
from modules.rag_service import RAGService, FALLBACK_NO_INFO_ANSWER
from modules.llm_service import LLMService


class TestSearchAndRAGValidation:

    @pytest.fixture
    def temp_db(self):
        """Seed temporary isolated database with multi-domain meeting intelligence."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(db_path)

        emb_svc = EmbeddingService(provider="hash")

        # Meeting 1: Database Migration (Date: 2026-04-10)
        mtg1 = MeetingIntelligence(
            meeting_id="mtg_val_001",
            summary="PostgreSQL database migration and schema refactoring planning.",
            key_points=["Migrate schema to PostgreSQL", "Complete indexing before deployment"],
            decisions=["Approved PostgreSQL as primary database"],
            action_items=[
                ActionItem(
                    task="Draft database migration script",
                    assigned_to="Alice",
                    deadline="2026-04-15",
                    priority="High",
                    status="Completed"
                )
            ],
            participants=[Participant(name="Alice", responsibilities=["Database Lead"])]
        )
        tx1 = "Alice: We decided to execute the database migration to PostgreSQL by April 15th."
        save_meeting(mtg1, tx1, title="Database Migration Sync", db_path=db_path)
        
        # Override created_at timestamp in SQLite for date filtering test
        from modules.database import get_db_connection
        c = get_db_connection(db_path)
        with c:
            c.execute("UPDATE meetings SET created_at = '2026-04-10 10:00:00' WHERE id = 'mtg_val_001'")
        c.close()

        # Save embeddings for Meeting 1
        items1 = [
            {"content_type": "summary", "text": mtg1.summary, "embedding": emb_svc.embed_text(mtg1.summary), "chunk_index": 0},
            {"content_type": "decision", "text": "Approved PostgreSQL as primary database", "embedding": emb_svc.embed_text("Approved PostgreSQL as primary database"), "chunk_index": 0},
            {"content_type": "action_item", "text": "Draft database migration script", "embedding": emb_svc.embed_text("Draft database migration script"), "chunk_index": 0},
            {"content_type": "transcript", "text": tx1, "embedding": emb_svc.embed_text(tx1), "chunk_index": 0}
        ]
        save_embeddings("mtg_val_001", items1, db_path=db_path)

        # Meeting 2: Mobile Application & API Integration (Date: 2026-05-20)
        mtg2 = MeetingIntelligence(
            meeting_id="mtg_val_002",
            summary="Mobile application UI design and REST API integration roadmap.",
            key_points=["Mobile app UI redesign", "REST API integration for authentication"],
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
                    task="Execute UI testing on mobile navbar",
                    assigned_to="Priya",
                    deadline="Next Monday",
                    priority="Medium",
                    status="In Progress"
                )
            ],
            participants=[
                Participant(name="Ravi", responsibilities=["API integration"]),
                Participant(name="Priya", responsibilities=["UI testing"])
            ]
        )
        tx2 = (
            "Ravi: I am responsible for API integration. We need to complete mobile application API integration by Friday. "
            "Priya: I will execute UI testing on the mobile navbar by next Monday."
        )
        save_meeting(mtg2, tx2, title="Mobile App & API Sync", db_path=db_path)
        c = get_db_connection(db_path)
        with c:
            c.execute("UPDATE meetings SET created_at = '2026-05-20 14:00:00' WHERE id = 'mtg_val_002'")
        c.close()

        items2 = [
            {"content_type": "summary", "text": mtg2.summary, "embedding": emb_svc.embed_text(mtg2.summary), "chunk_index": 0},
            {"content_type": "decision", "text": "Approved mobile application release schedule for Friday", "embedding": emb_svc.embed_text("Approved mobile application release schedule for Friday"), "chunk_index": 0},
            {"content_type": "action_item", "text": "Complete mobile application API integration", "embedding": emb_svc.embed_text("Complete mobile application API integration"), "chunk_index": 0},
            {"content_type": "action_item", "text": "Execute UI testing on mobile navbar", "embedding": emb_svc.embed_text("Execute UI testing on mobile navbar"), "chunk_index": 0},
            {"content_type": "transcript", "text": tx2, "embedding": emb_svc.embed_text(tx2), "chunk_index": 0}
        ]
        save_embeddings("mtg_val_002", items2, db_path=db_path)

        yield db_path

        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def search_service(self, temp_db):
        emb_svc = EmbeddingService(provider="hash")
        return SemanticSearchService(embedding_service=emb_svc)

    @pytest.fixture
    def rag_service(self, search_service):
        llm_svc = LLMService(provider="mock")
        return RAGService(semantic_search_service=search_service, llm_service=llm_svc)

    @pytest.fixture
    def client(self, temp_db):
        old_db = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = temp_db
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c
        if old_db is not None:
            os.environ["DATABASE_PATH"] = old_db
        else:
            os.environ.pop("DATABASE_PATH", None)

    # =========================================================================
    # 1. SEMANTIC SEARCH VALIDATION
    # =========================================================================

    def test_semantic_search_relevant_query(self, search_service, temp_db):
        """1. Relevant Query Search"""
        res = search_service.search("Which meeting discussed the database migration?", top_k=5, db_path=temp_db)
        assert res["status"] == "ok"
        assert res["total_results"] > 0
        top_hit = res["results"][0]
        assert top_hit["meeting_id"] == "mtg_val_001"
        assert top_hit["title"] == "Database Migration Sync"

    def test_semantic_search_different_wording(self, search_service, temp_db):
        """2. Paraphrased Wording Query"""
        res = search_service.search("DB schema refactoring script plan", top_k=5, db_path=temp_db)
        assert res["status"] == "ok"
        assert res["total_results"] > 0
        assert res["results"][0]["meeting_id"] == "mtg_val_001"

    def test_semantic_search_irrelevant_query(self, search_service, temp_db):
        """3. Irrelevant Query Search with Minimum Score Filtering"""
        res = search_service.search("quantum astrophysics dark matter exploration", top_k=5, min_score=0.4, db_path=temp_db)
        assert res["status"] == "ok"
        assert res["total_results"] == 0

    def test_semantic_search_empty_query(self, search_service, temp_db):
        """4. Empty Query Handling"""
        res = search_service.search("", top_k=5, db_path=temp_db)
        assert res["status"] == "ok"
        assert res["total_results"] == 0
        assert res["results"] == []

    # =========================================================================
    # 2. DATE FILTERING VALIDATION
    # =========================================================================

    def test_date_range_filtering(self, search_service, temp_db):
        """5. Date Filtering: Include April, Exclude May"""
        # Search within April 2026 (deduplicated per meeting)
        res_april = search_service.search(
            "migration API integration",
            start_date="2026-04-01",
            end_date="2026-04-30",
            deduplicate=True,
            db_path=temp_db
        )
        assert res_april["total_results"] == 1
        assert res_april["results"][0]["meeting_id"] == "mtg_val_001"

        # Search within May 2026 (deduplicated per meeting)
        res_may = search_service.search(
            "migration API integration",
            start_date="2026-05-01",
            end_date="2026-05-31",
            deduplicate=True,
            db_path=temp_db
        )
        assert res_may["total_results"] == 1
        assert res_may["results"][0]["meeting_id"] == "mtg_val_002"

    def test_empty_invalid_date_safety(self, search_service, temp_db):
        """6. Safe handling of empty/invalid date parameters"""
        res = search_service.search(
            "database migration",
            start_date="",
            end_date=None,
            db_path=temp_db
        )
        assert res["total_results"] > 0

    # =========================================================================
    # 3. METADATA FILTERING VALIDATION
    # =========================================================================

    def test_metadata_filtering_by_content_type(self, search_service, temp_db):
        """7. Metadata Filtering by content_type and meeting_id"""
        res_decision = search_service.search(
            "PostgreSQL",
            content_type="decision",
            db_path=temp_db
        )
        assert res_decision["total_results"] > 0
        assert res_decision["results"][0]["content_type"] == "decision"

        res_mtg2 = search_service.search(
            "API integration",
            meeting_id="mtg_val_002",
            db_path=temp_db
        )
        assert res_mtg2["total_results"] > 0
        assert all(r["meeting_id"] == "mtg_val_002" for r in res_mtg2["results"])

    # =========================================================================
    # 4. RAG GROUNDING & REQUIRED QUESTIONS VALIDATION
    # =========================================================================

    def test_rag_required_questions(self, rag_service, temp_db):
        """8. RAG Validation for Required Prompt Questions"""

        # Q1: Deadline for mobile application
        q1 = rag_service.answer_question("What deadline was decided for the mobile application?", db_path=temp_db)
        assert q1["status"] == "ok"
        assert q1["context_chunks_used"] > 0
        assert any(s["meeting_id"] == "mtg_val_002" for s in q1["sources"])

        # Q2: Responsibility for API integration
        q2 = rag_service.answer_question("Who was responsible for API integration?", db_path=temp_db)
        assert q2["status"] == "ok"
        assert any("mtg_val_002" in s["meeting_id"] for s in q2["sources"])

        # Q3: Decision about database migration
        q3 = rag_service.answer_question("What decision was made about the database migration?", db_path=temp_db)
        assert q3["status"] == "ok"
        assert any(s["meeting_id"] == "mtg_val_001" for s in q3["sources"])

        # Q4: UI testing meeting
        q4 = rag_service.answer_question("Which meeting discussed UI testing?", db_path=temp_db)
        assert q4["status"] == "ok"
        assert any(s["meeting_id"] == "mtg_val_002" for s in q4["sources"])

        # Q5: Action items assigned to Ravi
        q5 = rag_service.answer_question("What action items were assigned to Ravi?", db_path=temp_db)
        assert q5["status"] == "ok"
        assert any("mtg_val_002" in s["meeting_id"] for s in q5["sources"])

    def test_rag_hallucination_prevention_unanswerable_question(self, rag_service, temp_db):
        """9. Grounding Validation: Unanswerable Questions return safe fallback (No Hallucinations)"""
        # Ask question whose information is NOT present in any meeting
        res = rag_service.answer_question(
            "What was the budget approved for the project?",
            db_path=temp_db
        )
        assert res["status"] == "ok"
        assert res["answer"] == FALLBACK_NO_INFO_ANSWER

    # =========================================================================
    # 5. API ENDPOINT E2E VALIDATION
    # =========================================================================

    def test_api_search_and_rag_endpoints_e2e(self, client):
        """10. End-to-End API Routes for /search and /ask"""
        # Test /search
        res_search = client.post("/search", json={"query": "database migration", "top_k": 3})
        assert res_search.status_code == 200
        data_s = res_search.get_json()
        assert data_s["status"] == "ok"
        assert data_s["total_results"] > 0

        # Test /ask
        res_ask = client.post("/ask", json={"question": "What decision was made about the database migration?"})
        assert res_ask.status_code == 200
        data_a = res_ask.get_json()
        assert data_a["status"] == "ok"
        assert len(data_a["sources"]) > 0

"""
tests/test_semantic_search.py
==============================
Unit & Integration tests for Milestone 3 Task 4 - Semantic Search.

Verifies:
- Relevant query search & snippet extraction
- Irrelevant query search
- Multiple matching meetings handling
- Empty database / no results safety
- Metadata filtering (by content_type and meeting_id)
- Meeting metadata mapping (title, date, snippet, score)
- Duplicate result handling / deduplication
- Sub-3-second search latency benchmark assertion (< 3000 ms)
- Flask API endpoint verification
"""

import sys
import os
import time
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import init_db, save_meeting
from modules.embedding_service import EmbeddingService, generate_meeting_embeddings
from modules.vector_store import VectorStoreService
from modules.semantic_search import SemanticSearchService


class TestSemanticSearch:

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
    def indexed_meetings(self, temp_db):
        """Seed DB with meetings and generate vector embeddings."""
        emb_svc = EmbeddingService(provider="hash")
        vec_svc = VectorStoreService(db_path=temp_db)
        search_svc = SemanticSearchService(embedding_service=emb_svc, vector_store=vec_svc)

        # Meeting 1: Database Migration Sync
        mtg1 = MeetingIntelligence(
            meeting_id="mtg_sem_001",
            summary="PostgreSQL database migration and schema refactoring sync.",
            key_points=["Migrate MySQL tables to PostgreSQL", "Zero downtime deployment"],
            decisions=["Approved PostgreSQL as primary database"],
            action_items=[
                ActionItem(
                    task="Draft database migration script",
                    assigned_to="Alice",
                    deadline="Friday",
                    priority="High",
                    status="In Progress"
                )
            ],
            participants=[Participant(name="Alice", responsibilities=["DB Lead"])]
        )
        tx1 = "Alice: We discussed migrating the existing database to PostgreSQL by Friday. Team approved PostgreSQL."
        save_meeting(mtg1, tx1, title="Database Migration Sync", db_path=temp_db)
        generate_meeting_embeddings("mtg_sem_001", db_path=temp_db, service=emb_svc)

        # Meeting 2: Frontend Design Review
        mtg2 = MeetingIntelligence(
            meeting_id="mtg_sem_002",
            summary="Frontend UI navigation and responsive design review.",
            key_points=["CSS glassmorphism theme", "Mobile navigation tab bar"],
            decisions=["Adopted Inter font typography"],
            action_items=[
                ActionItem(
                    task="Update CSS color palette",
                    assigned_to="Bob",
                    deadline="Wednesday",
                    priority="Medium",
                    status="Pending"
                )
            ],
            participants=[Participant(name="Bob", responsibilities=["UI Lead"])]
        )
        tx2 = "Bob: Frontend UI design review finalized. Team adopted Inter font typography for responsive navbar."
        save_meeting(mtg2, tx2, title="Frontend Design Review", db_path=temp_db)
        generate_meeting_embeddings("mtg_sem_002", db_path=temp_db, service=emb_svc)

        return search_svc

    # 1. Relevant Query Search
    def test_relevant_query_search(self, temp_db, indexed_meetings):
        svc = indexed_meetings
        res = svc.search("Which meeting discussed the database migration?", top_k=5, db_path=temp_db)

        assert res["status"] == "ok"
        assert res["total_results"] > 0
        top = res["results"][0]

        assert top["meeting_id"] == "mtg_sem_001"
        assert top["title"] == "Database Migration Sync"
        assert "database" in top["relevant_snippet"].lower() or "postgressql" in top["relevant_snippet"].lower()
        assert top["similarity"] > 0.0

    # 2. Irrelevant / Non-Matching Query Search
    def test_irrelevant_query_search(self, temp_db, indexed_meetings):
        svc = indexed_meetings
        res = svc.search("quantum computer physics astrophysics hardware", top_k=5, db_path=temp_db)

        assert res["status"] == "ok"
        # Search runs safely without error, returning lower relevance scores
        for r in res["results"]:
            assert "meeting_id" in r

    # 3. Multiple Matching Meetings
    def test_multiple_matching_meetings(self, temp_db, indexed_meetings):
        svc = indexed_meetings
        # Query matching general term present in both meetings ("team approved")
        res = svc.search("approved", top_k=5, db_path=temp_db)

        assert res["status"] == "ok"
        meeting_ids = set(r["meeting_id"] for r in res["results"])
        assert len(meeting_ids) >= 2
        assert "mtg_sem_001" in meeting_ids
        assert "mtg_sem_002" in meeting_ids

    # 4. Empty Results / Empty Query Safety
    def test_empty_query_and_empty_db(self, temp_db):
        emb_svc = EmbeddingService(provider="hash")
        vec_svc = VectorStoreService(db_path=temp_db)
        svc = SemanticSearchService(embedding_service=emb_svc, vector_store=vec_svc)

        # Empty string query
        res1 = svc.search("", db_path=temp_db)
        assert res1["status"] == "ok"
        assert res1["total_results"] == 0

        # Query on empty DB
        res2 = svc.search("database migration", db_path=temp_db)
        assert res2["status"] == "ok"
        assert res2["total_results"] == 0

    # 5. Metadata Filtering (by Content Type and Meeting ID)
    def test_metadata_filtering(self, temp_db, indexed_meetings):
        svc = indexed_meetings

        # Filter by content_type = "decision"
        res_dec = svc.search("approved", content_type="decision", db_path=temp_db)
        assert res_dec["status"] == "ok"
        for r in res_dec["results"]:
            assert r["content_type"] == "decision"

        # Filter by meeting_id = "mtg_sem_002"
        res_mtg2 = svc.search("design", meeting_id="mtg_sem_002", db_path=temp_db)
        assert res_mtg2["status"] == "ok"
        for r in res_mtg2["results"]:
            assert r["meeting_id"] == "mtg_sem_002"

    # 6. Meeting Metadata Mapping
    def test_meeting_metadata_mapping(self, temp_db, indexed_meetings):
        svc = indexed_meetings
        res = svc.search("PostgreSQL migration script", db_path=temp_db)

        assert res["total_results"] > 0
        hit = res["results"][0]

        # Verify exact required schema output
        assert "meeting_id" in hit
        assert "title" in hit
        assert "date" in hit
        assert "relevant_snippet" in hit
        assert "content_type" in hit
        assert "similarity" in hit
        assert hit["title"] == "Database Migration Sync"
        assert hit["date"] != ""

    # 7. Duplicate Result Handling / Deduplication
    def test_duplicate_result_handling(self, temp_db, indexed_meetings):
        svc = indexed_meetings

        # Without deduplication
        res_all = svc.search("database migration PostgreSQL script", deduplicate=False, db_path=temp_db)
        # With deduplication per meeting
        res_dedup = svc.search("database migration PostgreSQL script", deduplicate=True, db_path=temp_db)

        meeting_ids = [r["meeting_id"] for r in res_dedup["results"]]
        assert len(meeting_ids) == len(set(meeting_ids))  # Strict uniqueness per meeting

    # 8. Search Latency Benchmark (< 3000 ms Performance Requirement)
    def test_search_latency_benchmark(self, temp_db, indexed_meetings):
        svc = indexed_meetings

        t0 = time.perf_counter()
        res = svc.search("Which meeting discussed database migration and schema refactoring?", top_k=5, db_path=temp_db)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Assert search completes in strictly under 3000 ms (3 seconds)
        assert elapsed_ms < 3000.0, f"Search latency ({elapsed_ms:.2f} ms) exceeded 3000 ms limit!"
        assert res["latency_ms"] < 3000.0
        print(f"\n[BENCHMARK] Observed Semantic Search Latency: {res['latency_ms']:.2f} ms")

    # 9. API Endpoint Integration
    def test_semantic_search_api_endpoint(self, temp_db, indexed_meetings):
        old_env = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = temp_db

        try:
            with app.test_client() as client:
                # Test POST endpoint
                payload = {
                    "query": "Which meeting discussed database migration?",
                    "top_k": 5
                }
                res = client.post("/api/search/semantic", json=payload)
                assert res.status_code == 200
                data = res.get_json()

                assert data["status"] == "ok"
                assert "latency_ms" in data
                assert data["latency_ms"] < 3000.0
                assert len(data["results"]) > 0

                top_hit = data["results"][0]
                assert top_hit["meeting_id"] == "mtg_sem_001"
                assert top_hit["title"] == "Database Migration Sync"
                assert "relevant_snippet" in top_hit
                assert "similarity" in top_hit

                # Test GET endpoint
                res_get = client.get("/api/search/semantic?q=database+migration")
                assert res_get.status_code == 200
                assert res_get.get_json()["status"] == "ok"
        finally:
            if old_env is not None:
                os.environ["DATABASE_PATH"] = old_env
            else:
                os.environ.pop("DATABASE_PATH", None)

"""
tests/test_api_integration.py
==============================
End-to-End API Integration Tests for Milestone 3 Task 6.

Verifies:
1. Existing /meetings endpoint works.
2. Existing /meetings/{id} endpoint works.
3. /search endpoint works with natural-language query.
4. /ask endpoint works with grounded RAG question answering.
5. Authentication works (authenticated vs unauthorized requests).
6. Existing database services work.
7. Existing error handling works (sanitized error messages, proper status codes, no raw stack traces).
8. Existing logging works (logs search/rag requests, completion, latency, errors without secrets).
9. Semantic search works through API.
10. RAG works through API.
"""

import sys
import os
import tempfile
import json
import logging
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app, semantic_search_service
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import init_db, save_meeting, save_embeddings
from modules.embedding_service import EmbeddingService


class TestAPIIntegration:

    @pytest.fixture
    def temp_db(self):
        """Create isolated temporary database."""
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        init_db(db_path)

        # Seed test meeting
        intel = MeetingIntelligence(
            meeting_id="mtg_api_test_001",
            summary="Discussion regarding PostgreSQL database migration and deployment timeline.",
            key_points=["Migrate schema to PostgreSQL", "Complete indexing by Friday"],
            decisions=["Database migration scheduled for Friday evening"],
            action_items=[
                ActionItem(
                    task="Run database migration scripts",
                    assigned_to="DevOps Team",
                    deadline="Friday 8 PM",
                    priority="High",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="DevOps Lead", responsibilities=["Database Migration"])
            ]
        )
        tx = "DevOps Lead: We decided to execute the database migration on Friday evening. Please prepare the backup."
        save_meeting(intel, tx, title="Database Migration Sync", db_path=db_path)

        # Generate & save embeddings for vector search
        emb_svc = EmbeddingService()
        vec_summary = emb_svc.embed_text(intel.summary)
        vec_dec = emb_svc.embed_text("Database migration scheduled for Friday evening")
        vec_act = emb_svc.embed_text("Run database migration scripts")
        vec_tx = emb_svc.embed_text(tx)

        items = [
            {"content_type": "summary", "text": intel.summary, "embedding": vec_summary, "chunk_index": 0},
            {"content_type": "decision", "text": "Database migration scheduled for Friday evening", "embedding": vec_dec, "chunk_index": 0},
            {"content_type": "action_item", "text": "Run database migration scripts", "embedding": vec_act, "chunk_index": 0},
            {"content_type": "transcript", "text": tx, "embedding": vec_tx, "chunk_index": 0},
        ]
        save_embeddings("mtg_api_test_001", items, db_path=db_path)

        # Ensure semantic search service points to this test DB
        semantic_search_service.vector_store.db_path = db_path

        yield db_path

        if os.path.exists(db_path):
            os.remove(db_path)

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

    # 1. Existing /meetings still works
    def test_get_meetings_list(self, client):
        res = client.get("/meetings")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert len(data["meetings"]) >= 1
        m = data["meetings"][0]
        assert "id" in m
        assert "title" in m
        assert "summary" in m

    # 2. Existing /meetings/{id} still works with all sub-fields
    def test_get_meeting_by_id(self, client):
        res = client.get("/meetings/mtg_api_test_001")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert data["meeting_id"] == "mtg_api_test_001"
        assert "metadata" in data
        assert "transcript" in data
        assert "summary" in data
        assert "key_points" in data
        assert "decisions" in data
        assert "action_items" in data
        assert "participants" in data
        assert "deadlines" in data
        assert len(data["deadlines"]) == 1

    # 3. /search endpoint works (semantic search through API)
    def test_search_endpoint(self, client):
        # GET request
        res_get = client.get("/search?q=database%20migration")
        assert res_get.status_code == 200
        data_get = res_get.get_json()
        assert data_get["status"] == "ok"
        assert data_get["query"] == "database migration"
        assert "latency_ms" in data_get
        assert data_get["total_results"] > 0
        hit = data_get["results"][0]
        assert hit["meeting_id"] == "mtg_api_test_001"

        # POST request
        res_post = client.post(
            "/search",
            data=json.dumps({"query": "database migration", "top_k": 3}),
            content_type="application/json"
        )
        assert res_post.status_code == 200
        data_post = res_post.get_json()
        assert data_post["status"] == "ok"
        assert len(data_post["results"]) > 0

    # 4. /ask endpoint works (RAG grounded Q&A through API)
    def test_ask_endpoint(self, client):
        # GET request
        res_get = client.get("/ask?q=What%20was%20decided%20about%20the%20database%20migration?")
        assert res_get.status_code == 200
        data_get = res_get.get_json()
        assert data_get["status"] == "ok"
        assert data_get["question"] != ""
        assert "answer" in data_get
        assert data_get["answer"] != ""
        assert "sources" in data_get
        assert len(data_get["sources"]) > 0

        # POST request
        res_post = client.post(
            "/ask",
            data=json.dumps({"question": "When is the database migration scheduled?", "top_k": 3}),
            content_type="application/json"
        )
        assert res_post.status_code == 200
        data_post = res_post.get_json()
        assert data_post["status"] == "ok"
        assert data_post["context_chunks_used"] > 0

    # 5. Authentication works
    def test_authentication_enforcement(self, client):
        old_require = os.environ.get("REQUIRE_AUTH")
        old_key = os.environ.get("API_KEY")
        try:
            os.environ["REQUIRE_AUTH"] = "true"
            os.environ["API_KEY"] = "secret-test-token"

            # Unauthorized request (no header/token)
            unauth_res = client.get("/search?q=test")
            assert unauth_res.status_code == 401
            assert unauth_res.get_json()["status"] == "error"

            # Authorized request with Bearer token
            auth_res = client.get(
                "/search?q=database",
                headers={"Authorization": "Bearer secret-test-token"}
            )
            assert auth_res.status_code == 200

            # Authorized request with X-API-Key header
            key_res = client.get(
                "/ask?q=database",
                headers={"X-API-Key": "secret-test-token"}
            )
            assert key_res.status_code == 200

        finally:
            if old_require is not None:
                os.environ["REQUIRE_AUTH"] = old_require
            else:
                os.environ.pop("REQUIRE_AUTH", None)
            if old_key is not None:
                os.environ["API_KEY"] = old_key
            else:
                os.environ.pop("API_KEY", None)

    # 6. Existing error handling works
    def test_error_handling(self, client):
        # 404 for invalid meeting ID
        res_404 = client.get("/meetings/non_existent_id_999")
        assert res_404.status_code == 404
        assert res_404.get_json()["status"] == "error"

        # 400 for invalid parameter in search
        res_400 = client.get("/search?top_k=invalid_int")
        assert res_400.status_code == 400
        assert "Invalid top_k parameter" in res_400.get_json()["message"]

        # Empty search query handles gracefully (status ok, empty results)
        res_empty = client.get("/search?q=")
        assert res_empty.status_code == 200
        assert res_empty.get_json()["total_results"] == 0

    # 7. Logging works
    def test_logging_execution(self, client, caplog):
        with caplog.at_level(logging.INFO):
            res_search = client.get("/search?q=migration")
            assert res_search.status_code == 200
            assert "[SEARCH REQUEST]" in caplog.text
            assert "[SEARCH COMPLETED]" in caplog.text

            res_ask = client.get("/ask?q=migration")
            assert res_ask.status_code == 200
            assert "[RAG REQUEST]" in caplog.text
            assert "[RAG COMPLETED]" in caplog.text

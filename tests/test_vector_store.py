"""
tests/test_vector_store.py
===========================
Unit & Integration tests for Milestone 3 Task 3 - Vector Database Integration.

Tests the 10 core requirements:
1. Insert embedding
2. Retrieve embedding
3. Update embedding
4. Similarity search
5. Filter by meeting
6. Filter by content type
7. Delete embedding
8. Verify deletion
9. Verify meeting-to-vector mapping
10. Verify orphan vector prevention (ON DELETE CASCADE)
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.schemas import MeetingIntelligence
from modules.database import (
    init_db,
    save_meeting,
    get_meeting,
    delete_meeting,
    get_all_embeddings
)
from modules.embedding_service import EmbeddingService
from modules.vector_store import VectorStoreService, compute_cosine_similarity


class TestVectorDatabaseIntegration:

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
    def vector_service(self, temp_db):
        return VectorStoreService(db_path=temp_db)

    @pytest.fixture
    def seed_meeting(self, temp_db):
        intel = MeetingIntelligence(
            meeting_id="mtg_vec_101",
            summary="Q4 Product Strategy.",
            key_points=["Launch AI feature"],
            decisions=["Approved Q4 roadmap"],
            action_items=[],
            participants=[]
        )
        save_meeting(intel, "Transcript content for vector store test.", title="Product Sync", db_path=temp_db)
        return "mtg_vec_101"

    # 1. Cosine Similarity Math
    def test_cosine_similarity_math(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        assert compute_cosine_similarity(v1, v2) == 1.0
        assert compute_cosine_similarity(v1, v3) == 0.0
        assert compute_cosine_similarity([], v1) == 0.0

    # 2. Insert Embedding & Retrieve Embedding
    def test_insert_and_retrieve_embedding(self, vector_service, seed_meeting, temp_db):
        emb_service = EmbeddingService(provider="hash")
        vec = emb_service.embed_text("Database migration deadline is Friday.")

        vid = vector_service.add_embedding(
            meeting_id=seed_meeting,
            content_type="action_item",
            text="Database migration deadline is Friday.",
            embedding=vec,
            source_id="act_001",
            chunk_index=0
        )

        assert vid is not None

        # Retrieve embedding
        retrieved = vector_service.get_embedding(vid)
        assert retrieved is not None
        assert retrieved["id"] == vid
        assert retrieved["meeting_id"] == seed_meeting
        assert retrieved["content_type"] == "action_item"
        assert retrieved["source_id"] == "act_001"
        assert retrieved["text"] == "Database migration deadline is Friday."
        assert len(retrieved["embedding"]) == len(vec)

    # 3. Update Embedding
    def test_update_embedding(self, vector_service, seed_meeting):
        emb_service = EmbeddingService(provider="hash")
        vec1 = emb_service.embed_text("Initial text")
        vec2 = emb_service.embed_text("Updated text content")

        vid = vector_service.add_embedding(
            meeting_id=seed_meeting,
            content_type="summary",
            text="Initial text",
            embedding=vec1
        )

        # Perform Update
        success = vector_service.update_embedding(vid, text="Updated text content", embedding=vec2)
        assert success is True

        updated = vector_service.get_embedding(vid)
        assert updated["text"] == "Updated text content"
        assert updated["embedding"] == vec2

    # 4. Similarity Search
    def test_similarity_search(self, vector_service, seed_meeting):
        emb_service = EmbeddingService(provider="hash")

        # Insert 3 vectors with known texts
        t1 = "Python microservices refactoring"
        t2 = "UI design system redesign Figma"
        t3 = "Security penetration audit"

        v1 = emb_service.embed_text(t1)
        v2 = emb_service.embed_text(t2)
        v3 = emb_service.embed_text(t3)

        vector_service.add_embedding(seed_meeting, "transcript", t1, v1, source_id="tx_1", chunk_index=0)
        vector_service.add_embedding(seed_meeting, "transcript", t2, v2, source_id="tx_2", chunk_index=1)
        vector_service.add_embedding(seed_meeting, "transcript", t3, v3, source_id="tx_3", chunk_index=2)

        # Query vector matching t1
        query_vec = emb_service.embed_text("Python refactoring microservice")
        results = vector_service.similarity_search(query_vec, top_k=2)

        assert len(results) == 2
        top_result = results[0]
        assert top_result["meeting_id"] == seed_meeting
        assert top_result["text"] == t1
        assert "score" in top_result
        assert top_result["score"] > 0.5  # Strong similarity score

    # 5. Filter by Meeting
    def test_filter_by_meeting(self, vector_service, temp_db, seed_meeting):
        # Create second meeting
        intel2 = MeetingIntelligence(
            meeting_id="mtg_vec_102",
            summary="Meeting 2",
            key_points=[],
            decisions=[],
            action_items=[],
            participants=[]
        )
        save_meeting(intel2, "Transcript 2", title="Meeting 2", db_path=temp_db)

        emb_service = EmbeddingService(provider="hash")
        vec = emb_service.embed_text("Sample text")

        vector_service.add_embedding(seed_meeting, "summary", "Summary 1", vec)
        vector_service.add_embedding("mtg_vec_102", "summary", "Summary 2", vec)

        mtg1_vecs = vector_service.filter_by_meeting(seed_meeting)
        mtg2_vecs = vector_service.filter_by_meeting("mtg_vec_102")

        assert len(mtg1_vecs) == 1
        assert mtg1_vecs[0]["text"] == "Summary 1"

        assert len(mtg2_vecs) == 1
        assert mtg2_vecs[0]["text"] == "Summary 2"

    # 6. Filter by Content Type
    def test_filter_by_content_type(self, vector_service, seed_meeting):
        emb_service = EmbeddingService(provider="hash")
        vec = emb_service.embed_text("Sample vector text")

        vector_service.add_embedding(seed_meeting, "summary", "Summary text", vec)
        vector_service.add_embedding(seed_meeting, "decision", "Decision text", vec)
        vector_service.add_embedding(seed_meeting, "action_item", "Action item text", vec)

        summaries = vector_service.filter_by_content_type("summary", meeting_id=seed_meeting)
        decisions = vector_service.filter_by_content_type("decision", meeting_id=seed_meeting)
        actions = vector_service.filter_by_content_type("action_item", meeting_id=seed_meeting)

        assert len(summaries) == 1
        assert summaries[0]["text"] == "Summary text"

        assert len(decisions) == 1
        assert decisions[0]["text"] == "Decision text"

        assert len(actions) == 1
        assert actions[0]["text"] == "Action item text"

    # 7. Delete Embedding & Verify Deletion
    def test_delete_embedding_and_verify(self, vector_service, seed_meeting):
        emb_service = EmbeddingService(provider="hash")
        vec = emb_service.embed_text("Temporary vector text")

        vid = vector_service.add_embedding(seed_meeting, "transcript", "Temporary vector text", vec)
        assert vector_service.get_embedding(vid) is not None

        # Delete
        del_success = vector_service.delete_embedding(vid)
        assert del_success is True

        # Verify deletion
        assert vector_service.get_embedding(vid) is None

    # 8. Meeting-to-Vector Mapping Integrity
    def test_meeting_to_vector_mapping(self, vector_service, seed_meeting):
        emb_service = EmbeddingService(provider="hash")
        vec = emb_service.embed_text("Strict mapping test")

        vid = vector_service.add_embedding(seed_meeting, "summary", "Strict mapping test", vec)
        rec = vector_service.get_embedding(vid)

        # Confirm exact metadata linkage
        assert rec["meeting_id"] == seed_meeting
        assert rec["content_type"] == "summary"
        assert rec["text"] == "Strict mapping test"

    # 9. Orphan Vector Prevention on Meeting Deletion
    def test_prevent_orphan_vectors_on_meeting_delete(self, vector_service, temp_db, seed_meeting):
        emb_service = EmbeddingService(provider="hash")
        vec = emb_service.embed_text("Orphan prevention text")

        vector_service.add_embedding(seed_meeting, "summary", "Orphan prevention text", vec)
        vector_service.add_embedding(seed_meeting, "decision", "Orphan decision text", vec)

        # Confirm vectors exist prior to meeting deletion
        assert len(vector_service.filter_by_meeting(seed_meeting)) == 2

        # Delete the parent meeting record
        del_meeting_success = delete_meeting(seed_meeting, db_path=temp_db)
        assert del_meeting_success is True

        # Verify all associated vectors were automatically cascaded and deleted (no orphan vectors!)
        remaining_vectors = vector_service.filter_by_meeting(seed_meeting)
        assert len(remaining_vectors) == 0

        all_embs = get_all_embeddings(db_path=temp_db)
        assert len(all_embs) == 0

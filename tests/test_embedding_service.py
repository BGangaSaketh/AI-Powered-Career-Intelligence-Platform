"""
tests/test_embedding_service.py
================================
Unit & Integration tests for Milestone 3 Task 2 - Embedding Generation.

Verifies:
- Summary embedding generation & metadata
- Decision embedding generation & metadata
- Action-item embedding generation & metadata
- Transcript chunking & section embedding generation
- Skipping empty / whitespace content safely
- Multiple meetings isolation and foreign key mapping
- Persistence and retrieval of embeddings from SQLite
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.database import (
    init_db,
    save_meeting,
    get_meeting_embeddings,
    get_all_embeddings
)
from modules.embedding_service import (
    EmbeddingService,
    chunk_transcript,
    extract_searchable_items,
    generate_meeting_embeddings
)


class TestEmbeddingGeneration:

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
    def sample_meeting_1(self, temp_db):
        intel = MeetingIntelligence(
            meeting_id="mtg_emb_101",
            summary="Q3 Engineering Sprint Planning.",
            key_points=["Refactor auth service", "Setup CI/CD pipeline"],
            decisions=["Adopt JWT authentication standard"],
            action_items=[
                ActionItem(
                    task="Implement JWT token validation",
                    assigned_to="Alice",
                    deadline="Friday",
                    priority="High",
                    status="Pending"
                ),
                ActionItem(
                    task="Setup GitHub Actions workflow",
                    assigned_to="Bob",
                    deadline="Next Monday",
                    priority="Medium",
                    status="In Progress"
                )
            ],
            participants=[
                Participant(name="Alice", responsibilities=["JWT Auth"]),
                Participant(name="Bob", responsibilities=["CI/CD"])
            ]
        )
        tx = (
            "Alice: Let's discuss the JWT authentication implementation. We need it finalized by Friday. "
            "Bob: I will configure the GitHub Actions workflow for CI/CD automation by next Monday. "
            "Decision: The team approved JWT auth standard."
        )
        save_meeting(intel, tx, title="Sprint Planning", db_path=temp_db)
        return "mtg_emb_101"

    @pytest.fixture
    def sample_meeting_2(self, temp_db):
        intel = MeetingIntelligence(
            meeting_id="mtg_emb_102",
            summary="Design Review & UI Mockups.",
            key_points=["Redesign landing page"],
            decisions=["Approved Figma design system v2"],
            action_items=[
                ActionItem(
                    task="Update CSS color tokens",
                    assigned_to="Charlie",
                    deadline="Wednesday",
                    priority="Low",
                    status="Pending"
                )
            ],
            participants=[
                Participant(name="Charlie", responsibilities=["UI Design"])
            ]
        )
        tx = "Charlie presented the new design system. Team approved Figma design system v2."
        save_meeting(intel, tx, title="Design Sync", db_path=temp_db)
        return "mtg_emb_102"

    # 1. Transcript Chunking
    def test_chunk_transcript(self):
        long_tx = " ".join([f"Word{i}" for i in range(250)])
        chunks = chunk_transcript(long_tx, max_words=100, overlap=20)
        assert len(chunks) > 1
        for idx, c in enumerate(chunks):
            assert c["chunk_index"] == idx
            assert c["word_count"] > 0
            assert "Word" in c["text"]

    # 2. Summary Embedding
    def test_summary_embedding(self, temp_db, sample_meeting_1):
        svc = EmbeddingService(provider="hash")
        embeddings = generate_meeting_embeddings(sample_meeting_1, db_path=temp_db, service=svc)
        
        sum_embs = [e for e in embeddings if e["content_type"] == "summary"]
        assert len(sum_embs) == 1
        e = sum_embs[0]
        assert e["meeting_id"] == sample_meeting_1
        assert e["source_id"] == sample_meeting_1
        assert e["text"] == "Q3 Engineering Sprint Planning."
        assert len(e["embedding"]) == 384
        assert any(v != 0 for v in e["embedding"])

    # 3. Decision Embedding
    def test_decision_embedding(self, temp_db, sample_meeting_1):
        svc = EmbeddingService(provider="hash")
        embeddings = generate_meeting_embeddings(sample_meeting_1, db_path=temp_db, service=svc)

        dec_embs = [e for e in embeddings if e["content_type"] == "decision"]
        assert len(dec_embs) == 1
        e = dec_embs[0]
        assert e["meeting_id"] == sample_meeting_1
        assert "Adopt JWT authentication standard" in e["text"]
        assert len(e["embedding"]) == 384

    # 4. Action Item Embedding
    def test_action_item_embedding(self, temp_db, sample_meeting_1):
        svc = EmbeddingService(provider="hash")
        embeddings = generate_meeting_embeddings(sample_meeting_1, db_path=temp_db, service=svc)

        act_embs = [e for e in embeddings if e["content_type"] == "action_item"]
        assert len(act_embs) == 2
        for e in act_embs:
            assert e["meeting_id"] == sample_meeting_1
            assert "Task:" in e["text"]
            assert len(e["embedding"]) == 384

    # 5. Transcript Chunk Embedding
    def test_transcript_chunk_embedding(self, temp_db, sample_meeting_1):
        svc = EmbeddingService(provider="hash")
        embeddings = generate_meeting_embeddings(sample_meeting_1, db_path=temp_db, service=svc)

        tx_embs = [e for e in embeddings if e["content_type"] == "transcript"]
        assert len(tx_embs) >= 1
        for e in tx_embs:
            assert e["meeting_id"] == sample_meeting_1
            assert "chunk_index" in e
            assert len(e["embedding"]) == 384

    # 6. Empty Content Skipping
    def test_empty_content_skipping(self, temp_db):
        intel = MeetingIntelligence(
            meeting_id="mtg_empty_test",
            summary="",
            key_points=[],
            decisions=[],
            action_items=[],
            participants=[]
        )
        save_meeting(intel, "", title="Empty Meeting", db_path=temp_db)

        svc = EmbeddingService(provider="hash")
        embeddings = generate_meeting_embeddings("mtg_empty_test", db_path=temp_db, service=svc)
        assert len(embeddings) == 0

    # 7. Multiple Meetings & Correct Mapping Isolation
    def test_multiple_meetings_isolation(self, temp_db, sample_meeting_1, sample_meeting_2):
        svc = EmbeddingService(provider="hash")
        generate_meeting_embeddings(sample_meeting_1, db_path=temp_db, service=svc)
        generate_meeting_embeddings(sample_meeting_2, db_path=temp_db, service=svc)

        embs_1 = get_meeting_embeddings(sample_meeting_1, db_path=temp_db)
        embs_2 = get_meeting_embeddings(sample_meeting_2, db_path=temp_db)
        all_embs = get_all_embeddings(db_path=temp_db)

        assert len(embs_1) > 0
        assert len(embs_2) > 0
        assert len(all_embs) == len(embs_1) + len(embs_2)

        for e in embs_1:
            assert e["meeting_id"] == sample_meeting_1
            assert e["meeting_id"] != sample_meeting_2

        for e in embs_2:
            assert e["meeting_id"] == sample_meeting_2
            assert e["meeting_id"] != sample_meeting_1

    # 8. Embedding Service Providers
    def test_embedding_providers(self):
        svc_hash = EmbeddingService(provider="hash")
        vec1 = svc_hash.embed_text("Hello world")
        assert len(vec1) == 384
        assert vec1 != []

        svc_mock = EmbeddingService(provider="mock")
        vec2 = svc_mock.embed_text("Hello world")
        assert len(vec2) == 384
        assert vec2 != []

        # Empty text yields empty vector
        assert svc_hash.embed_text("") == []
        assert svc_hash.embed_text("   ") == []

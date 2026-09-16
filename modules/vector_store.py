"""
modules/vector_store.py
=======================
Dedicated Vector Store Service for Meeting Intelligence Platform

Provides vector store management, CRUD operations, similarity search, metadata filtering,
and strict meeting-to-vector relational integrity.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from modules.database import (
    save_embeddings,
    get_meeting_embeddings,
    get_all_embeddings,
    get_embedding_by_id,
    update_embedding_by_id,
    delete_embedding_by_id,
    delete_meeting_embeddings
)

logger = logging.getLogger(__name__)


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Cosine Similarity score between two vector lists.
    Returns float score in [-1.0, 1.0]. Returns 0.0 if either vector is empty or zero-norm.
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)

    if a.shape != b.shape:
        # Pad or trim if dimensions mismatch
        min_dim = min(len(a), len(b))
        a = a[:min_dim]
        b = b[:min_dim]

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    score = np.dot(a, b) / (norm_a * norm_b)
    return round(float(score), 6)


class VectorStoreService:
    """
    Dedicated Vector Store Service providing complete vector database operations.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def add_embedding(
        self,
        meeting_id: str,
        content_type: str,
        text: str,
        embedding: List[float],
        source_id: Optional[str] = None,
        chunk_index: int = 0,
        vector_id: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> str:
        """
        Insert a vector embedding linked strictly to a meeting_id into the vector store.
        """
        if not meeting_id or not meeting_id.strip():
            raise ValueError("meeting_id is required to insert vector.")
        if not text or not text.strip():
            raise ValueError("text content cannot be empty.")
        if not embedding:
            raise ValueError("embedding vector cannot be empty.")

        target_db = db_path or self.db_path
        vid = vector_id or f"emb_{uuid.uuid4().hex[:12]}"
        src_id = source_id or vid

        item = {
            "id": vid,
            "meeting_id": meeting_id,
            "content_type": content_type,
            "source_id": src_id,
            "chunk_index": chunk_index,
            "text": text.strip(),
            "embedding": embedding
        }

        save_embeddings(meeting_id, [item], db_path=target_db)
        return vid

    def get_embedding(
        self,
        vector_id: str,
        db_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a vector record by its ID."""
        target_db = db_path or self.db_path
        return get_embedding_by_id(vector_id, db_path=target_db)

    def update_embedding(
        self,
        vector_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        db_path: Optional[str] = None
    ) -> bool:
        """Update text and/or vector embedding for an existing vector ID."""
        target_db = db_path or self.db_path
        return update_embedding_by_id(vector_id, text=text, embedding=embedding, db_path=target_db)

    def delete_embedding(
        self,
        vector_id: str,
        db_path: Optional[str] = None
    ) -> bool:
        """Delete a vector record by vector ID."""
        target_db = db_path or self.db_path
        return delete_embedding_by_id(vector_id, db_path=target_db)

    def filter_by_meeting(
        self,
        meeting_id: str,
        db_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all vector embeddings associated with a specific meeting."""
        target_db = db_path or self.db_path
        return get_meeting_embeddings(meeting_id, db_path=target_db)

    def filter_by_content_type(
        self,
        content_type: str,
        meeting_id: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all vectors matching content_type, optionally filtered by meeting_id."""
        target_db = db_path or self.db_path
        if meeting_id:
            return get_meeting_embeddings(meeting_id, content_type=content_type, db_path=target_db)
        else:
            all_vecs = get_all_embeddings(db_path=target_db)
            return [v for v in all_vecs if v["content_type"] == content_type]

    def delete_meeting_vectors(
        self,
        meeting_id: str,
        db_path: Optional[str] = None
    ) -> int:
        """Delete all vector embeddings for a specific meeting."""
        target_db = db_path or self.db_path
        return delete_meeting_embeddings(meeting_id, db_path=target_db)

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        meeting_id: Optional[str] = None,
        content_type: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute cosine similarity search over vector store with optional metadata filtering.

        Parameters
        ----------
        query_vector : List[float]
            The input query vector.
        top_k : int
            Maximum number of top results to return.
        meeting_id : str, optional
            Filter search space to a specific meeting.
        content_type : str, optional
            Filter search space to a specific content type ("summary", "transcript", "decision", "action_item").
        db_path : str, optional
            Database path.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked list of vector records containing metadata and calculated `score`.
        """
        if not query_vector:
            return []

        target_db = db_path or self.db_path

        # 1. Retrieve candidates based on metadata filters
        if meeting_id and content_type:
            candidates = get_meeting_embeddings(meeting_id, content_type=content_type, db_path=target_db)
        elif meeting_id:
            candidates = get_meeting_embeddings(meeting_id, db_path=target_db)
        elif content_type:
            candidates = self.filter_by_content_type(content_type, db_path=target_db)
        else:
            candidates = get_all_embeddings(db_path=target_db)

        if not candidates:
            return []

        # 2. Compute similarity score for each candidate
        scored_results = []
        for cand in candidates:
            vec = cand.get("embedding", [])
            score = compute_cosine_similarity(query_vector, vec)
            item = dict(cand)
            item["score"] = score
            scored_results.append(item)

        # 3. Rank results by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

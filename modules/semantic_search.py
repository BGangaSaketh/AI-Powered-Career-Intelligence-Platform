"""
modules/semantic_search.py
===========================
Dedicated Semantic Search Service for Meeting Intelligence Platform

Features:
- Natural language query processing & embedding generation
- Vector similarity search over historical meeting embeddings
- Meeting context & metadata enrichment (Title, Date, Summary, Snippet, Content Type, Score)
- Sub-3-second Latency Measurement & Performance Benchmarking
- Metadata Filtering & Deduplication per meeting
"""

import time
import logging
from typing import Dict, Any, List, Optional

from modules.embedding_service import EmbeddingService
from modules.vector_store import VectorStoreService
from modules.database import get_meeting_metadata

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    Service for executing natural language semantic searches across historical meetings.
    Maintains cached service instances to eliminate model reloading overhead.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()

    def search(
        self,
        query: str,
        top_k: int = 5,
        meeting_id: Optional[str] = None,
        content_type: Optional[str] = None,
        deduplicate: bool = False,
        db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute semantic search for a natural language query across historical meeting knowledge.

        Parameters
        ----------
        query : str
            Natural language search prompt.
        top_k : int
            Maximum number of top results to return.
        meeting_id : str, optional
            Filter search space to a specific meeting.
        content_type : str, optional
            Filter search space to a specific content type ("summary", "transcript", "decision", "action_item").
        deduplicate : bool
            If True, returns only the single highest-scoring snippet per meeting.
        db_path : str, optional
            Database path.

        Returns
        -------
        Dict[str, Any]
            Search response payload containing latency_ms and list of enriched meeting results.
        """
        t_start = time.perf_counter()

        query_str = (query or "").strip()
        if not query_str:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "ok",
                "query": "",
                "latency_ms": latency_ms,
                "total_results": 0,
                "results": []
            }

        # 1. Generate Query Vector Embedding
        query_vec = self.embedding_service.embed_text(query_str)
        if not query_vec:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "ok",
                "query": query_str,
                "latency_ms": latency_ms,
                "total_results": 0,
                "results": []
            }

        # 2. Perform Vector Similarity Search
        fetch_limit = top_k * 3 if deduplicate else top_k
        raw_hits = self.vector_store.similarity_search(
            query_vector=query_vec,
            top_k=fetch_limit,
            meeting_id=meeting_id,
            content_type=content_type,
            db_path=db_path
        )

        # 3. Enrich with Meeting Metadata
        enriched_results = []
        seen_meetings = set()

        for hit in raw_hits:
            m_id = hit.get("meeting_id")
            if deduplicate and m_id in seen_meetings:
                continue

            meta = get_meeting_metadata(m_id, db_path=db_path) if m_id else None

            title = meta.get("title", "Meeting") if meta else "Meeting"
            created_at = meta.get("created_at", "") if meta else ""
            summary = meta.get("summary", "") if meta else ""

            item = {
                "meeting_id": m_id,
                "title": title,
                "date": created_at,
                "created_at": created_at,
                "summary": summary,
                "relevant_snippet": hit.get("text", ""),
                "text": hit.get("text", ""),
                "content_type": hit.get("content_type", "unknown"),
                "source_id": hit.get("source_id"),
                "chunk_index": hit.get("chunk_index", 0),
                "similarity": hit.get("score", 0.0),
                "score": hit.get("score", 0.0),
                "vector_id": hit.get("id")
            }

            enriched_results.append(item)
            if m_id:
                seen_meetings.add(m_id)

            if len(enriched_results) >= top_k:
                break

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "status": "ok",
            "query": query_str,
            "latency_ms": latency_ms,
            "total_results": len(enriched_results),
            "results": enriched_results
        }

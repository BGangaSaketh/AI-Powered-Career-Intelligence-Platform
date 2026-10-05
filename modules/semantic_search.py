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

from modules.embedding_service import EmbeddingService, HashEmbeddingProvider
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
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_score: Optional[float] = None,
        deduplicate: bool = False,
        db_path: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute semantic search for a natural language query across historical meeting knowledge,
        restricted to the authenticated user_id.
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

        # 2. Perform Vector Similarity Search with User Isolation
        fetch_limit = top_k * 5 if (deduplicate or start_date or end_date) else top_k
        try:
            raw_hits = self.vector_store.similarity_search(
                query_vector=query_vec,
                top_k=fetch_limit,
                meeting_id=meeting_id,
                content_type=content_type,
                db_path=db_path,
                user_id=user_id
            )
        except Exception as exc:
            logger.error(f"Vector Database Error during search: {exc}")
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "ok",
                "query": query_str,
                "latency_ms": latency_ms,
                "total_results": 0,
                "results": []
            }

        # Smart Embedding Provider Fallback Alignment:
        # If top score is low (< 0.4), check if DB records were created with feature hash embeddings
        top_score = raw_hits[0].get("score", 0.0) if raw_hits else 0.0
        if top_score < 0.4 and getattr(self.embedding_service, "provider_name", "") != "hash":
            dim = len(query_vec) if query_vec else 384
            hash_provider = HashEmbeddingProvider(dimension=dim)
            hash_query_vec = hash_provider.embed_text(query_str)
            if hash_query_vec:
                hash_hits = self.vector_store.similarity_search(
                    query_vector=hash_query_vec,
                    top_k=fetch_limit,
                    meeting_id=meeting_id,
                    content_type=content_type,
                    db_path=db_path,
                    user_id=user_id
                )
                if hash_hits and hash_hits[0].get("score", 0.0) > top_score:
                    raw_hits = hash_hits

        # 3. Enrich with Meeting Metadata & Apply Metadata/Date Filters
        enriched_results = []
        seen_meetings = set()

        for hit in raw_hits:
            m_id = hit.get("meeting_id")
            if deduplicate and m_id in seen_meetings:
                continue

            # Minimum Similarity Score Filter
            score_val = hit.get("score", 0.0)
            if min_score is not None and score_val < min_score:
                continue

            meta = get_meeting_metadata(m_id, db_path=db_path) if m_id else None

            title = meta.get("title", "Meeting") if meta else "Meeting"
            created_at = meta.get("created_at", "") if meta else ""
            summary = meta.get("summary", "") if meta else ""

            # Date Range Filtering
            if start_date and str(start_date).strip():
                s_date = str(start_date).strip()[:10]
                m_date = str(created_at).strip()[:10] if created_at else ""
                if m_date and m_date < s_date:
                    continue

            if end_date and str(end_date).strip():
                e_date = str(end_date).strip()[:10]
                m_date = str(created_at).strip()[:10] if created_at else ""
                if m_date and m_date > e_date:
                    continue

            item = {
                "meeting_id": m_id,
                "title": title,
                "meeting_title": title,
                "date": created_at,
                "created_at": created_at,
                "summary": summary,
                "relevant_snippet": hit.get("text", ""),
                "text": hit.get("text", ""),
                "content_type": hit.get("content_type", "unknown"),
                "source_id": hit.get("source_id"),
                "chunk_index": hit.get("chunk_index", 0),
                "similarity": score_val,
                "score": score_val,
                "similarity_score": score_val,
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

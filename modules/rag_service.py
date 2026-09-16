"""
modules/rag_service.py
======================
Dedicated RAG (Retrieval-Augmented Generation) Service for Meeting Intelligence Platform

Architecture:
  User Question
        ↓
  Query Embedding (via SemanticSearchService)
        ↓
  Vector Similarity Search & Retrieval
        ↓
  Top-K Relevant Chunks & Evidence Assembly
        ↓
  LLM Generation (via LLMService with Grounded RAG Prompt)
        ↓
  Grounded Answer + Source Attribution
"""

import time
import logging
from typing import Dict, Any, List, Optional

from modules.semantic_search import SemanticSearchService
from modules.llm_service import LLMService
from modules.prompts import build_rag_prompt

logger = logging.getLogger(__name__)

FALLBACK_NO_INFO_ANSWER = "I couldn't find enough information in the available meeting records to answer this question."


class RAGService:
    """
    Retrieval-Augmented Generation Service.
    Enforces strict grounding on retrieved meeting records and builds clear source attributions.
    """

    def __init__(
        self,
        semantic_search_service: Optional[SemanticSearchService] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.semantic_search_service = semantic_search_service or SemanticSearchService()
        self.llm_service = llm_service or LLMService()

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        meeting_id: Optional[str] = None,
        content_type: Optional[str] = None,
        db_path: Optional[str] = None,
        similarity_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Execute full Grounded RAG pipeline for a user question.

        Parameters
        ----------
        question : str
            Natural language question prompt.
        top_k : int
            Maximum number of context chunks to retrieve.
        meeting_id : str, optional
            Filter context retrieval to a specific meeting.
        content_type : str, optional
            Filter context retrieval to a specific content type ("summary", "transcript", "decision", "action_item").
        db_path : str, optional
            Database path.
        similarity_threshold : float
            Minimum similarity score required to include a context chunk.

        Returns
        -------
        Dict[str, Any]
            RAG response payload containing answer, sources, latency_ms, and context count.
        """
        t_start = time.perf_counter()
        q_str = (question or "").strip()

        if not q_str:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "ok",
                "question": "",
                "answer": FALLBACK_NO_INFO_ANSWER,
                "sources": [],
                "latency_ms": latency_ms,
                "context_chunks_used": 0
            }

        # 1. Semantic Retrieval
        search_res = self.semantic_search_service.search(
            query=q_str,
            top_k=top_k,
            meeting_id=meeting_id,
            content_type=content_type,
            db_path=db_path
        )

        hits = search_res.get("results", [])

        # Filter hits by similarity threshold
        valid_hits = [
            h for h in hits
            if (h.get("similarity") or h.get("score") or 0.0) >= similarity_threshold
            and (h.get("relevant_snippet") or h.get("text") or "").strip()
        ]

        if not valid_hits:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "ok",
                "question": q_str,
                "answer": FALLBACK_NO_INFO_ANSWER,
                "sources": [],
                "latency_ms": latency_ms,
                "context_chunks_used": 0
            }

        # 2. Build Grounded RAG Prompt
        rag_prompt_dict = build_rag_prompt(q_str, valid_hits)
        system_prompt = rag_prompt_dict["system"]
        user_prompt = rag_prompt_dict["user"]

        # 3. Call LLM Service
        raw_answer = ""
        try:
            raw_answer = self.llm_service.call_llm(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning(f"LLM call failed for RAG query ({exc}). Generating direct contextual extraction answer.")
            raw_answer = ""

        # Fallback if LLM output is empty or unavailable
        if not raw_answer or not raw_answer.strip():
            # Synthesize factual direct answer from top retrieved evidence
            top_hit = valid_hits[0]
            snippet = top_hit.get("relevant_snippet") or top_hit.get("text") or ""
            m_title = top_hit.get("title") or "Meeting"
            raw_answer = f"Based on historical meeting \"{m_title}\": {snippet}"

        # 4. Construct Source Attribution Evidence
        sources = []
        for hit in valid_hits:
            sources.append({
                "meeting_id": hit.get("meeting_id"),
                "title": hit.get("title") or "Meeting",
                "date": hit.get("date") or hit.get("created_at") or "",
                "content_type": hit.get("content_type", "transcript"),
                "relevant_snippet": hit.get("relevant_snippet") or hit.get("text") or "",
                "similarity": hit.get("similarity") or hit.get("score") or 0.0
            })

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "status": "ok",
            "question": q_str,
            "answer": raw_answer.strip(),
            "sources": sources,
            "latency_ms": latency_ms,
            "context_chunks_used": len(valid_hits)
        }

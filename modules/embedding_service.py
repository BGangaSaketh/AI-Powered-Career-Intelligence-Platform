"""
modules/embedding_service.py
=============================
Dedicated Embedding Generation Service for Meeting Intelligence Platform

Features:
- Configurable Embedding Models (SentenceTransformers, Feature Hash vectorizer fallback, Mock)
- Transcript Chunking / Sectioning with sequence index preservation
- Searchable Content Extraction (Summaries, Transcript Chunks, Decisions, Action Items)
- Metadata Tagging & Meeting Association (meeting_id, content_type, source_id, chunk_index)
- SQLite Persistence Integration
"""

import os
import re
import json
import uuid
import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from modules.database import (
    get_complete_meeting,
    save_embeddings,
    get_meeting_embeddings,
    delete_meeting_embeddings
)

logger = logging.getLogger(__name__)


# ── Embedding Model Providers ───────────────────────────────────────────────

class BaseEmbeddingProvider:
    """Base interface for embedding model providers."""

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class HashEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic feature hashing vectorizer fallback.
    Generates unit-normalized dense float vectors of fixed dimension using character n-grams
    and token hashing with numpy. Zero external download required.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []

        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return []

        vec = np.zeros(self.dimension, dtype=np.float32)

        # Token hashing
        for t in tokens:
            idx = abs(hash(t)) % self.dimension
            val = (abs(hash(t + "_val")) % 200 - 100) / 100.0
            vec[idx] += val

        # Character n-gram hashing
        clean_str = text.lower().replace(" ", "")
        for i in range(len(clean_str) - 2):
            gram = clean_str[i:i+3]
            idx = abs(hash(gram)) % self.dimension
            vec[idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return [round(float(v), 6) for v in vec]


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Embedding provider using PyTorch / SentenceTransformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(f"SentenceTransformers failed to load ({exc}). Falling back to HashEmbeddingProvider.")
            self.model = None
            self.fallback = HashEmbeddingProvider()

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []

        if self.model is None:
            return self.fallback.embed_text(text)

        vec = self.model.encode(text, normalize_embeddings=True)
        return [round(float(v), 6) for v in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return [[] for _ in texts]

        if self.model is None:
            return [self.fallback.embed_text(t) for t in texts]

        valid_texts = [texts[i] for i in valid_indices]
        encoded = self.model.encode(valid_texts, normalize_embeddings=True)

        results: List[List[float]] = [[] for _ in texts]
        for idx, vec in zip(valid_indices, encoded):
            results[idx] = [round(float(v), 6) for v in vec]
        return results


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for lightweight testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        seed = abs(hash(text)) % 10000
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension)
        vec = vec / np.linalg.norm(vec)
        return [round(float(v), 6) for v in vec]


class EmbeddingService:
    """
    Main Embedding Generation Service.
    Selects provider dynamically based on environment configuration or explicit parameters.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: int = 384
    ):
        self.provider_name = (
            provider or os.getenv("EMBEDDING_PROVIDER") or "sentence-transformers"
        ).lower()
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME") or "all-MiniLM-L6-v2"
        self.dimension = dimension

        if self.provider_name == "mock":
            self.provider = MockEmbeddingProvider(dimension=dimension)
        elif self.provider_name in ("hash", "tfidf"):
            self.provider = HashEmbeddingProvider(dimension=dimension)
        elif self.provider_name == "sentence-transformers":
            try:
                import sentence_transformers
                self.provider = SentenceTransformersProvider(model_name=self.model_name)
            except ImportError:
                logger.info("sentence_transformers not installed. Using HashEmbeddingProvider fallback.")
                self.provider = HashEmbeddingProvider(dimension=dimension)
        else:
            self.provider = HashEmbeddingProvider(dimension=dimension)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        return self.provider.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.provider.embed_batch(texts)


# ── Transcript Chunking Helper ──────────────────────────────────────────────

def chunk_transcript(
    transcript: str,
    max_words: int = 80,
    overlap: int = 15
) -> List[Dict[str, Any]]:
    """
    Split a raw transcript into sequential sections/chunks with sequence index preservation.

    Parameters
    ----------
    transcript : str
        The raw meeting transcript.
    max_words : int
        Maximum number of words per chunk.
    overlap : int
        Word overlap between consecutive chunks.

    Returns
    -------
    List[Dict[str, Any]]
        List of chunk objects containing `chunk_index`, `text`, and `word_count`.
    """
    if not transcript or not transcript.strip():
        return []

    words = transcript.strip().split()
    if not words:
        return []

    chunks = []
    chunk_index = 0
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words).strip()

        if chunk_text_str:
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text_str,
                "word_count": len(chunk_words)
            })
            chunk_index += 1

        if end >= len(words):
            break

        start += max(1, max_words - overlap)

    return chunks


# ── Searchable Content Extraction & Embedding Generation ───────────────────

def extract_searchable_items(
    meeting_data: Dict[str, Any],
    max_chunk_words: int = 80,
    overlap: int = 15
) -> List[Dict[str, Any]]:
    """
    Extract all embeddable items (summary, transcript chunks, decisions, action items)
    from a complete meeting dictionary.
    """
    meeting_id = meeting_data.get("meeting_id") or meeting_data.get("id") or ""
    items = []

    # 1. Summary
    summary_text = (meeting_data.get("summary") or "").strip()
    if summary_text:
        items.append({
            "meeting_id": meeting_id,
            "content_type": "summary",
            "source_id": meeting_id,
            "chunk_index": 0,
            "text": summary_text
        })

    # 2. Transcript Chunks
    raw_tx = (meeting_data.get("transcript") or meeting_data.get("raw_transcript") or "").strip()
    if raw_tx:
        chunks = chunk_transcript(raw_tx, max_words=max_chunk_words, overlap=overlap)
        for c in chunks:
            items.append({
                "meeting_id": meeting_id,
                "content_type": "transcript",
                "source_id": f"{meeting_id}_tx_{c['chunk_index']}",
                "chunk_index": c["chunk_index"],
                "text": c["text"]
            })

    # 3. Decisions
    decisions = meeting_data.get("decisions") or []
    for idx, dec in enumerate(decisions):
        if isinstance(dec, dict):
            dec_text = (dec.get("decision_text") or dec.get("text") or "").strip()
            source_id = dec.get("id") or f"{meeting_id}_dec_{idx}"
        else:
            dec_text = str(dec).strip()
            source_id = f"{meeting_id}_dec_{idx}"

        if dec_text:
            items.append({
                "meeting_id": meeting_id,
                "content_type": "decision",
                "source_id": source_id,
                "chunk_index": idx,
                "text": dec_text
            })

    # 4. Action Items
    action_items = meeting_data.get("action_items") or []
    for idx, item in enumerate(action_items):
        if isinstance(item, dict):
            task = item.get("task", "").strip()
            assigned_to = item.get("assigned_to") or "Unassigned"
            deadline = item.get("deadline") or "None"
            priority = item.get("priority") or "Normal"
            status = item.get("status") or "Pending"
            source_id = item.get("id") or f"{meeting_id}_act_{idx}"

            if task:
                text_repr = f"Task: {task} | Assigned to: {assigned_to} | Deadline: {deadline} | Priority: {priority} | Status: {status}"
                items.append({
                    "meeting_id": meeting_id,
                    "content_type": "action_item",
                    "source_id": source_id,
                    "chunk_index": idx,
                    "text": text_repr
                })

    return items


def generate_meeting_embeddings(
    meeting_id: str,
    db_path: Optional[str] = None,
    service: Optional[EmbeddingService] = None,
    max_chunk_words: int = 80,
    overlap: int = 15
) -> List[Dict[str, Any]]:
    """
    End-to-End embedding pipeline for a historical meeting:
    1. Fetch complete meeting knowledge from database.
    2. Extract searchable content items (summary, transcript chunks, decisions, action items).
    3. Generate embedding vectors dynamically.
    4. Attach metadata (meeting_id, content_type, source_id, chunk_index, text).
    5. Persist to database linked by meeting_id.
    """
    meeting_data = get_complete_meeting(meeting_id, db_path=db_path)
    if not meeting_data:
        raise ValueError(f"Meeting '{meeting_id}' not found in database.")

    svc = service or EmbeddingService()

    # Extract searchable items (skips empty content)
    raw_items = extract_searchable_items(
        meeting_data,
        max_chunk_words=max_chunk_words,
        overlap=overlap
    )

    if not raw_items:
        logger.info(f"No non-empty searchable content found for meeting '{meeting_id}'.")
        delete_meeting_embeddings(meeting_id, db_path=db_path)
        return []

    # Batch generate embeddings
    texts = [it["text"] for it in raw_items]
    embeddings_list = svc.embed_batch(texts)

    processed_items = []
    for item, vec in zip(raw_items, embeddings_list):
        if not vec:
            continue
        processed_items.append({
            "id": f"emb_{uuid.uuid4().hex[:12]}",
            "meeting_id": meeting_id,
            "content_type": item["content_type"],
            "source_id": item["source_id"],
            "chunk_index": item["chunk_index"],
            "text": item["text"],
            "embedding": vec,
            "dimension": len(vec)
        })

    # Clear old embeddings & persist new embeddings atomically
    delete_meeting_embeddings(meeting_id, db_path=db_path)
    save_embeddings(meeting_id, processed_items, db_path=db_path)

    logger.info(f"Generated and saved {len(processed_items)} embeddings for meeting '{meeting_id}'.")
    return get_meeting_embeddings(meeting_id, db_path=db_path)

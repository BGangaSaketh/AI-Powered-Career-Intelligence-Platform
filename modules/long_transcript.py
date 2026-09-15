"""
modules/long_transcript.py
===========================
Long Transcript Chunking & Aggregation Engine

Chunking Strategy:
  1. Size Estimation: Estimates total tokens assuming ~4 characters per token.
  2. Context Decision: If estimated tokens <= max_chunk_tokens, process in a single pass.
  3. Sentence-Aware Chunking: If tokens > max_chunk_tokens, splits transcript into logical 
     chunks along sentence boundaries with configurable token overlaps.
  4. Parallel/Sequential Processing: Calls LLM service for each chunk to extract intermediate 
     intelligence (key points, decisions, action items, participants).
  5. Aggregation: Merges intermediate chunk results into a unified, deduplicated 
     MeetingIntelligence object without losing decisions, action items, or participants.
"""

import os
import re
import logging
from typing import List
from dotenv import load_dotenv

from modules.schemas import MeetingIntelligence
from modules.llm_service import LLMService

load_dotenv()
logger = logging.getLogger(__name__)

# Default token thresholds
DEFAULT_MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "3000"))
DEFAULT_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "200"))


def estimate_token_count(text: str) -> int:
    """
    Estimate the token count of a text string.
    Rule of thumb: 1 token ≈ 4 characters (or ~0.75 words).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def split_transcript_into_chunks(
    transcript: str,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split a long transcript into sentence-aware text chunks.

    Parameters
    ----------
    transcript : str
        The complete meeting transcript text.
    max_chunk_tokens : int
        Maximum estimated tokens per chunk.
    overlap_tokens : int
        Overlapping token window between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    if not transcript or not transcript.strip():
        return []

    total_tokens = estimate_token_count(transcript)
    if total_tokens <= max_chunk_tokens:
        return [transcript.strip()]

    # Split into sentences using punctuation boundaries
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    if not sentences:
        sentences = [transcript.strip()]

    max_chunk_chars = max_chunk_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks = []
    current_chunk_sentences = []
    current_char_length = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_char_length + sent_len > max_chunk_chars and current_chunk_sentences:
            # Emit current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Retain trailing sentences for overlap window
            overlap_sentences = []
            overlap_len = 0
            for prev_s in reversed(current_chunk_sentences):
                if overlap_len + len(prev_s) <= overlap_chars:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break

            current_chunk_sentences = overlap_sentences
            current_char_length = overlap_len

        current_chunk_sentences.append(sentence)
        current_char_length += sent_len

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    logger.info(
        f"Split transcript ({total_tokens} est. tokens) into {len(chunks)} chunk(s) "
        f"(max {max_chunk_tokens} tokens/chunk, {overlap_tokens} tokens overlap)."
    )
    return chunks


def process_long_transcript(
    transcript: str,
    llm_service: LLMService,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    meeting_context: str = "",
) -> MeetingIntelligence:
    """
    Process long transcripts safely with chunking and intermediate aggregation.

    Parameters
    ----------
    transcript : str
        Complete transcript text.
    llm_service : LLMService
        Initialized LLM service instance.
    max_chunk_tokens : int
        Maximum tokens per chunk.
    overlap_tokens : int
        Token overlap between consecutive chunks.
    meeting_context : str
        Optional title/context.

    Returns
    -------
    MeetingIntelligence
        Unified final meeting intelligence result.
    """
    est_tokens = estimate_token_count(transcript)
    logger.info(f"Processing transcript of estimated {est_tokens} tokens.")

    chunks = split_transcript_into_chunks(transcript, max_chunk_tokens, overlap_tokens)

    if len(chunks) <= 1:
        # Fits inside single context window
        return llm_service.analyze_transcript(transcript, meeting_context)

    # Process each chunk individually
    intermediate_results = []
    for idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {idx + 1}/{len(chunks)} ({estimate_token_count(chunk)} est. tokens)...")
        result = llm_service.analyze_chunk(chunk, idx, len(chunks))
        intermediate_results.append(result)

    # Aggregate intermediate chunk extractions into unified result
    logger.info(f"Aggregating {len(intermediate_results)} intermediate chunk extractions...")
    aggregated_result = llm_service.aggregate_chunks(intermediate_results)
    return aggregated_result

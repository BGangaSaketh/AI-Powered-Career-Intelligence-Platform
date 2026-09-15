"""
tests/test_long_transcript.py
==============================
Unit tests for long transcript token estimation, chunking, and aggregation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.long_transcript import (
    estimate_token_count,
    split_transcript_into_chunks,
    process_long_transcript
)
from modules.llm_service import LLMService


class TestLongTranscriptHandling:

    def test_token_count_estimation(self):
        text = "Hello world! This is a test sentence."
        tokens = estimate_token_count(text)
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_short_transcript_no_chunking(self):
        short_text = "Ravi assigned API integration to Priya."
        chunks = split_transcript_into_chunks(short_text, max_chunk_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_long_transcript_chunking_strategy(self):
        # Create long text exceeding chunk token threshold
        sentences = [f"Sentence number {i} discusses item {i} in detail." for i in range(100)]
        long_text = " ".join(sentences)

        chunks = split_transcript_into_chunks(long_text, max_chunk_tokens=100, overlap_tokens=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_process_long_transcript_end_to_end(self):
        sentences = [
            f"Speaker {i % 3}: Discussing milestone phase {i}. Action item for Ravi to complete step {i} by Friday."
            for i in range(30)
        ]
        long_text = " ".join(sentences)

        llm_service = LLMService(provider="mock")
        res = process_long_transcript(long_text, llm_service, max_chunk_tokens=100, overlap_tokens=20)

        assert res.summary != ""
        assert len(res.action_items) > 0
        assert len(res.participants) > 0

"""
tests/test_prompts.py
======================
Unit tests for prompt construction and template formatting.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.prompts import (
    construct_meeting_analysis_prompt,
    construct_chunk_analysis_prompt,
    construct_aggregation_prompt,
    SYSTEM_INSTRUCTION,
    OUTPUT_FORMAT_INSTRUCTIONS
)


class TestPromptEngineering:

    def test_meeting_analysis_prompt_construction(self):
        transcript = "Ravi discussed API timelines."
        context = "Sprint 14 Standup"
        prompts = construct_meeting_analysis_prompt(transcript, context)

        assert "system" in prompts
        assert "user" in prompts
        assert prompts["system"] == SYSTEM_INSTRUCTION
        assert transcript in prompts["user"]
        assert context in prompts["user"]
        assert "Required Output JSON Schema" in prompts["user"]

    def test_chunk_analysis_prompt_construction(self):
        chunk_text = "Part of long discussion..."
        prompts = construct_chunk_analysis_prompt(chunk_text, 0, 3)

        assert "Part 1 of 3" in prompts["user"]
        assert chunk_text in prompts["user"]

    def test_aggregation_prompt_construction(self):
        intermediate_json = '[{"summary": "Part 1"}]'
        prompts = construct_aggregation_prompt(intermediate_json)

        assert intermediate_json in prompts["user"]
        assert "Synthesize all intermediate chunk findings" in prompts["user"]

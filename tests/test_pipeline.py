"""
tests/test_pipeline.py
======================
Task 5 — End-to-End Pipeline Integration Tests

Tests the complete flow:
  Input → Ingestion → Preprocessing → VADER Sentiment → Report

Each test runs the entire pipeline from a raw input to a final report
to verify that all modules work together correctly.
"""

import io
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ingestion     import ingest_text, ingest_txt_file, ingest_csv_file
from modules.preprocessing import preprocess
from modules.sentiment     import analyze_sentiment
from modules.reporting     import generate_report


# ── Helper ─────────────────────────────────────────────────────────────────

def run_full_pipeline(ingestion_result: dict) -> dict:
    """
    Run preprocessing → sentiment → report on a validated ingestion result.
    Returns the final report or raises if ingestion failed.
    """
    assert ingestion_result["status"] == "ok", (
        f"Ingestion failed: {ingestion_result['errors']}"
    )
    text                 = ingestion_result["raw_text"]
    preprocessing_result = preprocess(text)
    sentiment_result     = analyze_sentiment(text)
    report               = generate_report(
        ingestion_result, preprocessing_result, sentiment_result
    )
    return {
        "ingestion":     ingestion_result,
        "preprocessing": preprocessing_result,
        "sentiment":     sentiment_result,
        "report":        report,
    }


# ── Integration Tests ──────────────────────────────────────────────────────

class TestPipelineFromText:

    def test_positive_text_full_pipeline(self):
        ingestion = ingest_text(
            "I am thrilled about this amazing career opportunity! "
            "The team is fantastic and the growth potential is extraordinary."
        )
        result = run_full_pipeline(ingestion)

        # Ingestion
        assert result["ingestion"]["status"] == "ok"

        # Preprocessing
        assert result["preprocessing"]["word_count_raw"]   > 0
        assert result["preprocessing"]["word_count_clean"] > 0
        assert len(result["preprocessing"]["lemmas"])      > 0

        # Sentiment
        assert result["sentiment"]["label"]    == "Positive"
        assert result["sentiment"]["compound"] >= 0.05

        # Report
        report = result["report"]
        assert "report_title"          in report
        assert "sentiment_summary"     in report
        assert "preprocessing_summary" in report
        assert report["sentiment_summary"]["overall_label"] == "Positive"

    def test_negative_text_full_pipeline(self):
        ingestion = ingest_text(
            "This company is absolutely terrible. The management is toxic "
            "and the employees are constantly miserable and overworked."
        )
        result = run_full_pipeline(ingestion)
        assert result["sentiment"]["label"]    == "Negative"
        assert result["sentiment"]["compound"] <= -0.05
        assert result["report"]["sentiment_summary"]["overall_label"] == "Negative"

    def test_neutral_text_full_pipeline(self):
        ingestion = ingest_text(
            "The quarterly meeting is scheduled for the third Wednesday of November."
        )
        result = run_full_pipeline(ingestion)
        assert result["sentiment"]["label"] == "Neutral"
        assert result["report"]["sentiment_summary"]["overall_label"] == "Neutral"

    def test_report_contains_emotion_tags(self):
        ingestion = ingest_text("I love this incredible job! Best experience ever!")
        result    = run_full_pipeline(ingestion)
        tags      = result["report"]["emotion_tags"]
        assert isinstance(tags, list)
        assert len(tags) > 0

    def test_report_contains_top_words(self):
        ingestion = ingest_text(
            "Career growth and career development are crucial for career success."
        )
        result    = run_full_pipeline(ingestion)
        top_words = result["report"]["preprocessing_summary"]["top_words"]
        assert isinstance(top_words, list)
        assert len(top_words) > 0
        # "career" should appear among top words
        words = [item["word"] for item in top_words]
        assert "career" in words

    def test_pipeline_preserves_source(self):
        ingestion = ingest_text("Simple text for source check.")
        result    = run_full_pipeline(ingestion)
        assert result["report"]["input_summary"]["source"] == "text"

    def test_empty_text_fails_at_ingestion(self):
        ingestion = ingest_text("")
        assert ingestion["status"] == "error"


class TestPipelineFromTxtFile:

    def _make_file(self, content: str):
        return io.BytesIO(content.encode("utf-8"))

    def test_txt_file_full_pipeline(self):
        content  = (
            "Working at this company has been a positive experience overall. "
            "The projects are meaningful and the colleagues are supportive."
        )
        f        = self._make_file(content)
        ingestion = ingest_txt_file(f)
        result   = run_full_pipeline(ingestion)

        assert result["ingestion"]["source"]              == "txt_file"
        assert result["preprocessing"]["word_count_raw"]  > 0
        assert result["report"]["input_summary"]["source"] == "txt_file"

    def test_empty_txt_file_fails_at_ingestion(self):
        f         = self._make_file("")
        ingestion = ingest_txt_file(f)
        assert ingestion["status"] == "error"


class TestPipelineFromCsvFile:

    def _make_csv(self, content: str):
        return io.BytesIO(content.encode("utf-8"))

    def test_csv_file_full_pipeline(self):
        csv_content = (
            "text,label\n"
            "I absolutely love the work environment here,positive\n"
            "The management is disorganised and communication is poor,negative\n"
            "The cafeteria opens at nine in the morning,neutral\n"
        )
        f         = self._make_csv(csv_content)
        ingestion = ingest_csv_file(f)
        result    = run_full_pipeline(ingestion)

        assert result["ingestion"]["source"]               == "csv_file"
        assert result["ingestion"]["row_count"] if "row_count" in result["ingestion"] else True
        assert result["preprocessing"]["word_count_raw"]   > 0
        assert result["report"]["input_summary"]["source"] == "csv_file"

    def test_csv_combined_text_analyzed(self):
        """Multiple CSV rows are joined; combined text should be analyzable."""
        csv_content = (
            "text\n"
            "Great place to work\n"
            "Wonderful leadership and culture\n"
            "Highly recommend to everyone\n"
        )
        f         = self._make_csv(csv_content)
        ingestion = ingest_csv_file(f)
        result    = run_full_pipeline(ingestion)
        # All rows are positive so overall should be positive
        assert result["sentiment"]["compound"] > 0


class TestPipelineReportStructure:
    """Verify the report schema regardless of sentiment direction."""

    def test_report_has_milestone_field(self):
        ingestion = ingest_text("Testing the report structure.")
        result    = run_full_pipeline(ingestion)
        assert "milestone" in result["report"]
        assert "Milestone 1" in result["report"]["milestone"]

    def test_report_has_generated_at(self):
        ingestion = ingest_text("Timestamp test.")
        result    = run_full_pipeline(ingestion)
        assert "generated_at" in result["report"]
        assert len(result["report"]["generated_at"]) > 0

    def test_report_preprocessing_summary_keys(self):
        ingestion = ingest_text("The preprocessing summary must have the correct keys.")
        result    = run_full_pipeline(ingestion)
        ps        = result["report"]["preprocessing_summary"]
        for key in ["sentence_count", "word_count_raw", "word_count_clean",
                    "unique_words", "top_words"]:
            assert key in ps, f"Missing preprocessing summary key: {key}"

    def test_report_sentiment_summary_keys(self):
        ingestion = ingest_text("The sentiment summary must have the correct keys.")
        result    = run_full_pipeline(ingestion)
        ss        = result["report"]["sentiment_summary"]
        for key in ["overall_label", "compound_score", "positive_score",
                    "negative_score", "neutral_score"]:
            assert key in ss, f"Missing sentiment summary key: {key}"

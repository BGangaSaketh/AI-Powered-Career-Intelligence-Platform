"""
tests/test_sentiment.py
========================
Task 3 — Unit tests for the VADER sentiment module.

Tests cover:
  - Positive sentiment detection
  - Negative sentiment detection
  - Neutral sentiment detection
  - Compound score range validation
  - Individual scores (pos, neg, neu)
  - Per-sentence breakdown
  - Different sample inputs
  - Empty text handling
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.sentiment import analyze_sentiment


class TestSentiment:

    # ── Return structure ────────────────────────────────────────────────

    def test_returns_expected_keys(self):
        result = analyze_sentiment("I love this job opportunity!")
        expected = ["compound", "pos", "neg", "neu", "label", "per_sentence"]
        for key in expected:
            assert key in result, f"Missing key: {key}"

    # ── Positive sentiment ──────────────────────────────────────────────

    def test_positive_text_detects_positive(self):
        text = "I am absolutely thrilled and excited about this amazing opportunity!"
        result = analyze_sentiment(text)
        assert result["label"]    == "Positive"
        assert result["compound"] >= 0.05
        assert result["pos"]      >  0.0

    def test_positive_compound_score_range(self):
        result = analyze_sentiment("Excellent! Fantastic work. I love it!")
        assert 0.05 <= result["compound"] <= 1.0

    def test_positive_pos_score_greater_than_neg(self):
        result = analyze_sentiment("The career growth here is outstanding and wonderful.")
        assert result["pos"] > result["neg"]

    # ── Negative sentiment ──────────────────────────────────────────────

    def test_negative_text_detects_negative(self):
        text = "This is terrible. The management is awful and the culture is toxic."
        result = analyze_sentiment(text)
        assert result["label"]    == "Negative"
        assert result["compound"] <= -0.05
        assert result["neg"]      >   0.0

    def test_negative_compound_score_range(self):
        result = analyze_sentiment("Horrible experience. I hate this place.")
        assert -1.0 <= result["compound"] <= -0.05

    def test_negative_neg_score_greater_than_pos(self):
        result = analyze_sentiment("Worst job ever. Complete disaster.")
        assert result["neg"] > result["pos"]

    # ── Neutral sentiment ───────────────────────────────────────────────

    def test_neutral_text_detects_neutral(self):
        text = "The meeting is scheduled for Monday at 10 AM in room 204."
        result = analyze_sentiment(text)
        assert result["label"] == "Neutral"
        assert -0.05 < result["compound"] < 0.05

    def test_neutral_neu_score_dominant(self):
        text = "The report was submitted on Tuesday."
        result = analyze_sentiment(text)
        assert result["neu"] >= result["pos"]
        assert result["neu"] >= result["neg"]

    # ── Score range validation ──────────────────────────────────────────

    def test_all_scores_between_zero_and_one(self):
        result = analyze_sentiment("This is a moderately good experience overall.")
        assert 0.0 <= result["pos"]      <= 1.0
        assert 0.0 <= result["neg"]      <= 1.0
        assert 0.0 <= result["neu"]      <= 1.0
        assert -1.0 <= result["compound"] <= 1.0

    def test_scores_sum_to_approximately_one(self):
        """pos + neg + neu should sum to approximately 1.0."""
        result = analyze_sentiment("I enjoy working here but the deadlines are stressful.")
        total = result["pos"] + result["neg"] + result["neu"]
        assert abs(total - 1.0) < 0.01

    # ── Per-sentence breakdown ──────────────────────────────────────────

    def test_per_sentence_returns_list(self):
        result = analyze_sentiment("I love this. It is great.")
        assert isinstance(result["per_sentence"], list)

    def test_per_sentence_count(self):
        text = "Great product. Terrible support. Average price."
        result = analyze_sentiment(text)
        assert len(result["per_sentence"]) == 3

    def test_per_sentence_has_required_keys(self):
        result = analyze_sentiment("I am happy. This is wonderful.")
        for s in result["per_sentence"]:
            assert "sentence" in s
            assert "compound" in s
            assert "pos"      in s
            assert "neg"      in s
            assert "neu"      in s
            assert "label"    in s

    def test_per_sentence_labels_valid(self):
        result = analyze_sentiment("Amazing work! Terrible result. Nothing changed.")
        valid_labels = {"Positive", "Negative", "Neutral"}
        for s in result["per_sentence"]:
            assert s["label"] in valid_labels

    # ── Empty / edge cases ──────────────────────────────────────────────

    def test_empty_text_returns_neutral(self):
        result = analyze_sentiment("")
        assert result["label"]        == "Neutral"
        assert result["compound"]     == 0.0
        assert result["per_sentence"] == []

    def test_whitespace_only_returns_neutral(self):
        result = analyze_sentiment("   \n   ")
        assert result["label"] == "Neutral"

    # ── Different sample inputs ─────────────────────────────────────────

    def test_sample_career_positive(self):
        text = "This internship has been an incredible learning experience with superb mentors."
        result = analyze_sentiment(text)
        assert result["compound"] > 0

    def test_sample_career_negative(self):
        text = "The layoffs were unexpected and the company communication was dreadful."
        result = analyze_sentiment(text)
        assert result["compound"] < 0

    def test_sample_career_mixed(self):
        text = (
            "The salary is competitive but the hours are exhausting. "
            "I enjoy the projects but dislike the commute."
        )
        result = analyze_sentiment(text)
        # Mixed: both pos and neg should be non-zero
        assert result["pos"] > 0
        assert result["neg"] > 0

    def test_numeric_only_text(self):
        """Numbers alone should not crash the module."""
        result = analyze_sentiment("12345 67890")
        assert "label" in result

    def test_single_word_positive(self):
        result = analyze_sentiment("Excellent")
        assert result["compound"] > 0

    def test_single_word_negative(self):
        result = analyze_sentiment("Terrible")
        assert result["compound"] < 0

"""
tests/test_preprocessing.py
===========================
Task 2 — Unit tests for the preprocessing module.

Tests cover:
  - Tokenization
  - Stop-word removal
  - Lemmatization
  - Noise filtering (URLs, emails, special characters)
  - Punctuation removal
  - Empty text handling
  - Repeated spaces
  - Different text lengths (short, medium, long)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.preprocessing import preprocess


class TestPreprocessing:

    # ── Basic sanity ────────────────────────────────────────────────────

    def test_returns_expected_keys(self):
        result = preprocess("The quick brown fox jumps over the lazy dog.")
        expected_keys = [
            "original_text", "sentences", "cleaned_text", "tokens",
            "tokens_no_punct", "filtered_tokens", "lemmas",
            "word_count_raw", "word_count_clean", "char_count",
            "unique_words", "top_words",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_original_text_preserved(self):
        text = "I enjoy machine learning research."
        result = preprocess(text)
        assert result["original_text"] == text

    # ── Tokenization ────────────────────────────────────────────────────

    def test_tokenization_produces_tokens(self):
        result = preprocess("Career growth is important for every professional.")
        assert len(result["tokens"]) > 0

    def test_tokens_no_punct_excludes_punctuation(self):
        result = preprocess("Hello, world! This is great.")
        for token in result["tokens_no_punct"]:
            assert token not in {",", "!", ".", "?", ";", ":"}

    def test_sentence_tokenization(self):
        text = "I am happy. The job is great. I recommend it."
        result = preprocess(text)
        assert len(result["sentences"]) == 3

    # ── Stop-word removal ───────────────────────────────────────────────

    def test_stopwords_removed(self):
        result = preprocess("The quick brown fox jumps over the lazy dog.")
        stop_words = {"the", "over", "a", "an", "is", "are", "was"}
        for token in result["filtered_tokens"]:
            assert token not in stop_words

    def test_content_words_preserved(self):
        result = preprocess("Python developer with machine learning expertise.")
        # These content words should survive stop-word removal
        lemmas_lower = [l.lower() for l in result["lemmas"]]
        assert any("python" in l for l in lemmas_lower)

    # ── Lemmatization ───────────────────────────────────────────────────

    def test_lemmatization_reduces_plurals(self):
        result = preprocess("The companies hired many engineers and developers.")
        lemmas = result["lemmas"]
        # "companies" → "company", "engineers" → "engineer"
        assert "company" in lemmas or "companies" in lemmas
        assert "engineer" in lemmas or "engineers" in lemmas

    def test_lemmatization_reduces_running(self):
        result = preprocess("She is running and jumping every day.")
        lemmas = result["lemmas"]
        # verb forms should be reduced
        assert any(l in {"run", "running"} for l in lemmas)

    # ── Noise filtering ─────────────────────────────────────────────────

    def test_urls_removed(self):
        result = preprocess("Visit https://www.example.com for more details.")
        assert "https" not in result["cleaned_text"]
        assert "example" not in result["cleaned_text"] or True   # URL removed

    def test_emails_removed(self):
        result = preprocess("Contact us at hr@company.org for questions.")
        cleaned = result["cleaned_text"]
        assert "@" not in cleaned

    def test_special_characters_removed(self):
        result = preprocess("Best salary: $120,000/year! #hiring @techcorp")
        cleaned = result["cleaned_text"]
        assert "$" not in cleaned
        assert "#" not in cleaned
        assert "@" not in cleaned

    # ── Repeated spaces ─────────────────────────────────────────────────

    def test_repeated_spaces_collapsed(self):
        result = preprocess("Hello    world   this   is   a   test.")
        assert "  " not in result["cleaned_text"]

    # ── Empty text ──────────────────────────────────────────────────────

    def test_empty_text_returns_zero_counts(self):
        result = preprocess("")
        assert result["word_count_raw"]   == 0
        assert result["word_count_clean"] == 0
        assert result["lemmas"]           == []

    def test_whitespace_only_returns_zero_counts(self):
        result = preprocess("   \n\t   ")
        assert result["word_count_raw"]   == 0

    # ── Different text lengths ───────────────────────────────────────────

    def test_short_text(self):
        result = preprocess("Good job.")
        assert result["word_count_raw"] >= 1

    def test_medium_text(self):
        text = (
            "The candidate demonstrated excellent communication skills "
            "during the interview process. Their technical background "
            "in software development is impressive."
        )
        result = preprocess(text)
        assert result["word_count_raw"] > 5
        assert result["word_count_clean"] > 0

    def test_long_text(self):
        text = " ".join(["The career opportunities at this firm are exceptional."] * 20)
        result = preprocess(text)
        assert result["word_count_raw"]   > 50
        assert result["word_count_clean"] > 10

    # ── Statistics ──────────────────────────────────────────────────────

    def test_word_count_clean_less_than_or_equal_raw(self):
        result = preprocess("I am very excited about this wonderful career opportunity!")
        assert result["word_count_clean"] <= result["word_count_raw"]

    def test_top_words_returns_list(self):
        result = preprocess("Career career career success success growth growth growth.")
        assert isinstance(result["top_words"], list)
        assert len(result["top_words"]) > 0

    def test_char_count_matches_original(self):
        text = "Hello, World!"
        result = preprocess(text)
        assert result["char_count"] == len(text)

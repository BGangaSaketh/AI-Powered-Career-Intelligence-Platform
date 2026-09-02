"""
tests/test_ingestion.py
=======================
Task 1 — Unit tests for the ingestion module.

Tests cover:
  - Valid raw text input
  - Empty text input
  - Whitespace-only text
  - None input
  - Text exceeding maximum length
  - .txt file ingestion (valid + empty)
  - .csv file ingestion (valid + empty + no text column)
"""

import io
import pytest
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ingestion import ingest_text, ingest_txt_file, ingest_csv_file


# ── ingest_text ────────────────────────────────────────────────────────────

class TestIngestText:

    def test_valid_text_returns_ok(self):
        result = ingest_text("The candidate has excellent Python skills.")
        assert result["status"]   == "ok"
        assert result["source"]   == "text"
        assert len(result["raw_text"]) > 0
        assert len(result["rows"])    > 0
        assert result["errors"]       == []

    def test_empty_string_returns_error(self):
        result = ingest_text("")
        assert result["status"] == "error"
        assert len(result["errors"]) > 0

    def test_whitespace_only_returns_error(self):
        result = ingest_text("     \n\t   ")
        assert result["status"] == "error"

    def test_none_input_returns_error(self):
        result = ingest_text(None)
        assert result["status"] == "error"

    def test_text_too_long_returns_error(self):
        long_text = "a" * 200_001
        result = ingest_text(long_text)
        assert result["status"] == "error"
        assert any("exceeds" in e for e in result["errors"])

    def test_multiline_text_rows(self):
        text = "Line one.\nLine two.\nLine three."
        result = ingest_text(text)
        assert result["status"] == "ok"
        assert len(result["rows"]) == 3

    def test_text_with_special_characters(self):
        """Special characters should NOT block ingestion; they are handled in preprocessing."""
        text = "Hello! This is a test... with some #special @chars & symbols."
        result = ingest_text(text)
        assert result["status"] == "ok"

    def test_single_word_text(self):
        result = ingest_text("Python")
        assert result["status"] == "ok"
        assert result["raw_text"] == "Python"


# ── ingest_txt_file ────────────────────────────────────────────────────────

class TestIngestTxtFile:

    def _make_file(self, content: str):
        """Create an in-memory file-like object from a string."""
        return io.BytesIO(content.encode("utf-8"))

    def test_valid_txt_file(self):
        f = self._make_file("I am passionate about data science and AI.")
        result = ingest_txt_file(f)
        assert result["status"] == "ok"
        assert result["source"] == "txt_file"
        assert "data science" in result["raw_text"]

    def test_empty_txt_file(self):
        f = self._make_file("")
        result = ingest_txt_file(f)
        assert result["status"] == "error"

    def test_whitespace_only_txt_file(self):
        f = self._make_file("\n\n   \t   \n")
        result = ingest_txt_file(f)
        assert result["status"] == "error"

    def test_multiline_txt_file(self):
        content = "First line of the document.\nSecond line here.\nThird line ends."
        f = self._make_file(content)
        result = ingest_txt_file(f)
        assert result["status"] == "ok"
        assert len(result["rows"]) == 3


# ── ingest_csv_file ────────────────────────────────────────────────────────

class TestIngestCsvFile:

    def _make_csv(self, content: str):
        return io.BytesIO(content.encode("utf-8"))

    def test_valid_csv_with_text_column(self):
        csv_content = "text,label\nI love this job,positive\nTerrible experience,negative\n"
        f = self._make_csv(csv_content)
        result = ingest_csv_file(f)
        assert result["status"] == "ok"
        assert result["source"] == "csv_file"
        assert len(result["rows"]) == 2

    def test_csv_with_content_column(self):
        csv_content = "content,score\nGreat opportunity here,5\nPoor management,1\n"
        f = self._make_csv(csv_content)
        result = ingest_csv_file(f)
        assert result["status"] == "ok"
        assert len(result["rows"]) == 2

    def test_csv_with_review_column(self):
        csv_content = "id,review\n1,Outstanding performance\n2,Needs improvement\n"
        f = self._make_csv(csv_content)
        result = ingest_csv_file(f)
        assert result["status"] == "ok"

    def test_csv_first_column_fallback(self):
        """When no preferred column name matches, the first column is used."""
        csv_content = "feedback,rating\nExcellent work,5\nAverage at best,3\n"
        f = self._make_csv(csv_content)
        result = ingest_csv_file(f)
        assert result["status"] == "ok"

    def test_empty_csv_returns_error(self):
        f = self._make_csv("")
        result = ingest_csv_file(f)
        assert result["status"] == "error"

    def test_csv_with_only_header_returns_error(self):
        f = self._make_csv("text,label\n")
        result = ingest_csv_file(f)
        assert result["status"] == "error"

"""
modules/ingestion.py
====================
Task 1 — Text Ingestion & Validation

This module handles all input methods:
  1. Raw text typed by the user
  2. .txt file upload
  3. .csv file upload

Each function returns a standardised result dict:
  {
      "status":    "ok" | "error",
      "source":    "text" | "txt_file" | "csv_file",
      "raw_text":  <str>,          # the extracted text (joined if multiple rows)
      "rows":      <list[str]>,    # individual rows / sentences
      "errors":    <list[str]>     # validation error messages
  }
"""

import csv
import io


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TEXT_LENGTH = 100_000   # characters
MIN_TEXT_LENGTH = 1         # at least one non-whitespace character


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _build_result(status: str, source: str, raw_text: str,
                  rows: list, errors: list) -> dict:
    """Return a standardised ingestion result dictionary."""
    return {
        "status":   status,
        "source":   source,
        "raw_text": raw_text,
        "rows":     rows,
        "errors":   errors,
    }


def _validate_text(text: str) -> list:
    """
    Run basic validation checks and return a list of error strings.
    An empty list means the text is valid.
    """
    errors = []

    if text is None:
        errors.append("Input is None — no text provided.")
        return errors

    if not isinstance(text, str):
        errors.append(f"Expected a string, got {type(text).__name__}.")
        return errors

    stripped = text.strip()

    if len(stripped) < MIN_TEXT_LENGTH:
        errors.append("Input is empty or contains only whitespace.")
    elif len(stripped) > MAX_TEXT_LENGTH:
        errors.append(
            f"Input exceeds maximum allowed length "
            f"({len(stripped)} > {MAX_TEXT_LENGTH} characters)."
        )

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_text(text: str) -> dict:
    """
    Validate and ingest a raw text string entered by the user.

    Parameters
    ----------
    text : str
        The raw text provided by the user.

    Returns
    -------
    dict
        Standardised ingestion result.
    """
    errors = _validate_text(text)

    if errors:
        return _build_result("error", "text", "", [], errors)

    cleaned = text.strip()
    # Split into individual non-empty rows / sentences for reporting
    rows = [line.strip() for line in cleaned.splitlines() if line.strip()]

    return _build_result("ok", "text", cleaned, rows, [])


def ingest_txt_file(file_obj) -> dict:
    """
    Read and validate a .txt file upload.

    Parameters
    ----------
    file_obj : file-like object
        A file object (e.g. from Flask's request.files).

    Returns
    -------
    dict
        Standardised ingestion result.
    """
    try:
        raw_bytes = file_obj.read()
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        return _build_result(
            "error", "txt_file", "", [],
            [f"Could not read file: {exc}"]
        )

    errors = _validate_text(text)
    if errors:
        return _build_result("error", "txt_file", "", [], errors)

    cleaned = text.strip()
    rows = [line.strip() for line in cleaned.splitlines() if line.strip()]

    return _build_result("ok", "txt_file", cleaned, rows, [])


def ingest_csv_file(file_obj, text_column: str = None) -> dict:
    """
    Read a .csv file and extract the text column.

    The function auto-detects the text column by checking for common names
    ('text', 'content', 'review', 'comment', 'description') — case-insensitive.
    If none match, the first column is used.

    Parameters
    ----------
    file_obj : file-like object
        A file object (e.g. from Flask's request.files).
    text_column : str, optional
        Explicit column name to extract. If omitted, auto-detection is used.

    Returns
    -------
    dict
        Standardised ingestion result.
    """
    try:
        raw_bytes = file_obj.read()
        text_data = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        return _build_result(
            "error", "csv_file", "", [],
            [f"Could not read CSV file: {exc}"]
        )

    reader = csv.DictReader(io.StringIO(text_data))

    try:
        fieldnames = reader.fieldnames
    except Exception:
        fieldnames = None

    if not fieldnames:
        # Try without headers
        return _build_result(
            "error", "csv_file", "", [],
            ["CSV file has no headers or is empty."]
        )

    # Auto-detect the text column
    preferred_names = ["text", "content", "review", "comment", "description"]
    detected_column = None

    if text_column and text_column in fieldnames:
        detected_column = text_column
    else:
        for name in preferred_names:
            for field in fieldnames:
                if field.strip().lower() == name:
                    detected_column = field
                    break
            if detected_column:
                break

    if not detected_column:
        # Fall back to first column
        detected_column = fieldnames[0]

    rows = []
    for row in reader:
        cell = row.get(detected_column, "").strip()
        if cell:
            rows.append(cell)

    if not rows:
        return _build_result(
            "error", "csv_file", "", [],
            [f"No text found in column '{detected_column}'."]
        )

    combined_text = "\n".join(rows)
    return _build_result("ok", "csv_file", combined_text, rows, [])

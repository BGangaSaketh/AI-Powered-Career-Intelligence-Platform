"""
modules/reporting.py
====================
Task 4 — Initial Emotion/Sentiment Report Generation

Combines the outputs from the ingestion, preprocessing and sentiment
modules into a single structured report.  The report is serialisable
to JSON so the Flask API can return it directly to the frontend.
"""

from datetime import datetime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    ingestion_result:     dict,
    preprocessing_result: dict,
    sentiment_result:     dict,
) -> dict:
    """
    Build a comprehensive analysis report from the pipeline outputs.

    Parameters
    ----------
    ingestion_result     : dict  from modules.ingestion
    preprocessing_result : dict  from modules.preprocessing
    sentiment_result     : dict  from modules.sentiment

    Returns
    -------
    dict
        A structured report suitable for JSON serialisation.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ #
    # 1.  Input summary
    # ------------------------------------------------------------------ #
    input_summary = {
        "source":         ingestion_result.get("source", "unknown"),
        "status":         ingestion_result.get("status", "unknown"),
        "char_count":     len(ingestion_result.get("raw_text", "")),
        "row_count":      len(ingestion_result.get("rows", [])),
        "errors":         ingestion_result.get("errors", []),
    }

    # ------------------------------------------------------------------ #
    # 2.  Preprocessing summary
    # ------------------------------------------------------------------ #
    preprocessing_summary = {
        "sentence_count":   len(preprocessing_result.get("sentences", [])),
        "word_count_raw":   preprocessing_result.get("word_count_raw", 0),
        "word_count_clean": preprocessing_result.get("word_count_clean", 0),
        "unique_words":     preprocessing_result.get("unique_words", 0),
        "top_words":        [
            {"word": w, "count": c}
            for w, c in preprocessing_result.get("top_words", [])
        ],
    }

    # ------------------------------------------------------------------ #
    # 3.  Sentiment summary
    # ------------------------------------------------------------------ #
    sentiment_summary = {
        "overall_label":    sentiment_result.get("label", "Neutral"),
        "compound_score":   sentiment_result.get("compound", 0.0),
        "positive_score":   sentiment_result.get("pos", 0.0),
        "negative_score":   sentiment_result.get("neg", 0.0),
        "neutral_score":    sentiment_result.get("neu", 0.0),
        "sentence_count":   len(sentiment_result.get("per_sentence", [])),
        "per_sentence":     sentiment_result.get("per_sentence", []),
    }

    # ------------------------------------------------------------------ #
    # 4.  Emotion-like classification (derived from scores)
    # ------------------------------------------------------------------ #
    emotion_tags = _derive_emotion_tags(sentiment_result)

    # ------------------------------------------------------------------ #
    # 5.  Assemble final report
    # ------------------------------------------------------------------ #
    report = {
        "report_title":          "AI Career Intelligence — Sentiment Analysis Report",
        "milestone":             "Milestone 1: Text Ingestion & Baseline Sentiment",
        "generated_at":          timestamp,
        "input_summary":         input_summary,
        "preprocessing_summary": preprocessing_summary,
        "sentiment_summary":     sentiment_summary,
        "emotion_tags":          emotion_tags,
    }

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_emotion_tags(sentiment_result: dict) -> list:
    """
    Derive simple emotion-like tags from the VADER scores.

    This is a lightweight heuristic (not ML-based) used for demonstration.
    It maps score ranges to readable emotion labels.
    """
    compound = sentiment_result.get("compound", 0.0)
    pos      = sentiment_result.get("pos",      0.0)
    neg      = sentiment_result.get("neg",      0.0)

    tags = []

    if compound >= 0.5:
        tags.append("Very Positive")
    elif compound >= 0.05:
        tags.append("Mildly Positive")
    elif compound <= -0.5:
        tags.append("Very Negative")
    elif compound <= -0.05:
        tags.append("Mildly Negative")
    else:
        tags.append("Neutral / Balanced")

    if pos > 0.3:
        tags.append("Encouraging")
    if neg > 0.3:
        tags.append("Concerning")
    if pos > 0.15 and neg > 0.15:
        tags.append("Mixed Sentiment")
    if compound >= 0.8:
        tags.append("Highly Enthusiastic")
    if compound <= -0.8:
        tags.append("Strongly Negative")

    return tags if tags else ["Neutral"]

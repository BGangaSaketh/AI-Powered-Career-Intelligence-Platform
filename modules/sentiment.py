"""
modules/sentiment.py
====================
Task 3 — VADER Sentiment Analysis

Uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) model
from the vaderSentiment library.  VADER is specifically attuned to
sentiments expressed in social media and short texts.

Per-sentence sentiment is also computed for granular reporting.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_from_compound(compound: float) -> str:
    """
    Convert a compound score to a human-readable sentiment label.

    VADER convention:
      compound >= 0.05  → Positive
      compound <= -0.05 → Negative
      otherwise         → Neutral
    """
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> dict:
    """
    Run VADER sentiment analysis on the provided text.

    Parameters
    ----------
    text : str
        The (possibly preprocessed) text to analyse.

    Returns
    -------
    dict
        {
            "compound":        float,  # overall score in [-1, 1]
            "pos":             float,  # positive proportion
            "neg":             float,  # negative proportion
            "neu":             float,  # neutral proportion
            "label":           str,    # "Positive" | "Negative" | "Neutral"
            "per_sentence":    list[dict]  # per-sentence breakdown
        }
    """
    if not text or not text.strip():
        return {
            "compound":     0.0,
            "pos":          0.0,
            "neg":          0.0,
            "neu":          1.0,
            "label":        "Neutral",
            "per_sentence": [],
        }

    # Overall document-level scores
    scores = _analyzer.polarity_scores(text)

    compound = scores["compound"]
    pos      = scores["pos"]
    neg      = scores["neg"]
    neu      = scores["neu"]
    label    = _label_from_compound(compound)

    # Per-sentence breakdown
    sentences     = sent_tokenize(text)
    per_sentence  = []
    for sentence in sentences:
        s_scores  = _analyzer.polarity_scores(sentence)
        s_compound = s_scores["compound"]
        per_sentence.append({
            "sentence": sentence,
            "compound": round(s_compound, 4),
            "pos":      round(s_scores["pos"],  4),
            "neg":      round(s_scores["neg"],  4),
            "neu":      round(s_scores["neu"],  4),
            "label":    _label_from_compound(s_compound),
        })

    return {
        "compound":     round(compound, 4),
        "pos":          round(pos,      4),
        "neg":          round(neg,      4),
        "neu":          round(neu,      4),
        "label":        label,
        "per_sentence": per_sentence,
    }

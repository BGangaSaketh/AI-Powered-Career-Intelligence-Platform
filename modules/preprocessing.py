"""
modules/preprocessing.py
========================
Task 2 — NLP Preprocessing Pipeline

Steps performed on raw text:
  1. Noise filtering   — remove URLs, emails, special characters
  2. Lowercasing
  3. Tokenization      — split into word tokens
  4. Punctuation removal
  5. Stop-word removal
  6. Lemmatization     — reduce tokens to their base form

Returns a dict with all intermediate and final results so the UI can
display each step for demonstration purposes.
"""

import re
import string

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available (downloaded once, then cached)
# ---------------------------------------------------------------------------
_NLTK_PACKAGES = [
    ("tokenizers/punkt",          "punkt"),
    ("tokenizers/punkt_tab",      "punkt_tab"),
    ("corpora/stopwords",         "stopwords"),
    ("corpora/wordnet",           "wordnet"),
    ("corpora/omw-1.4",          "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

def _ensure_nltk_data():
    for resource_path, package_name in _NLTK_PACKAGES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once)
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Noise Filtering Patterns
# ---------------------------------------------------------------------------
_URL_PATTERN     = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN   = re.compile(r"\S+@\S+\.\S+")
_HTML_PATTERN    = re.compile(r"<[^>]+>")
_SPECIAL_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")   # keep only alphanumeric + spaces
_MULTI_SPACE     = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(text: str) -> dict:
    """
    Run the full NLP preprocessing pipeline on the input text.

    Parameters
    ----------
    text : str
        Raw input text (already validated by the ingestion module).

    Returns
    -------
    dict
        {
            "original_text":    str,
            "sentences":        list[str],
            "cleaned_text":     str,       # after noise removal
            "tokens":           list[str], # after tokenization
            "tokens_no_punct":  list[str], # after punctuation removal
            "filtered_tokens":  list[str], # after stop-word removal
            "lemmas":           list[str], # after lemmatization
            "word_count_raw":   int,
            "word_count_clean": int,
            "char_count":       int,
            "unique_words":     int,
            "top_words":        list[tuple] # (word, count) top 10
        }
    """
    if not text or not text.strip():
        return _empty_result(text or "")

    original_text = text

    # Step 1 — Sentence tokenization (before cleaning, for context)
    sentences = sent_tokenize(original_text)

    # Step 2 — Noise filtering
    cleaned = _filter_noise(original_text)

    # Step 3 — Lowercasing
    cleaned = cleaned.lower()

    # Step 4 — Tokenize into words
    tokens = word_tokenize(cleaned)

    # Step 5 — Remove punctuation tokens
    tokens_no_punct = [t for t in tokens if t not in string.punctuation and t.strip()]

    # Step 6 — Remove stop-words
    filtered_tokens = [t for t in tokens_no_punct if t not in _stop_words]

    # Step 7 — Lemmatization
    lemmas = [_lemmatizer.lemmatize(t) for t in filtered_tokens]

    # Step 8 — Statistics
    word_count_raw   = len(tokens_no_punct)
    word_count_clean = len(lemmas)
    char_count       = len(original_text)
    unique_words     = len(set(lemmas))
    top_words        = _top_n_words(lemmas, n=10)

    return {
        "original_text":    original_text,
        "sentences":        sentences,
        "cleaned_text":     cleaned,
        "tokens":           tokens,
        "tokens_no_punct":  tokens_no_punct,
        "filtered_tokens":  filtered_tokens,
        "lemmas":           lemmas,
        "word_count_raw":   word_count_raw,
        "word_count_clean": word_count_clean,
        "char_count":       char_count,
        "unique_words":     unique_words,
        "top_words":        top_words,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_noise(text: str) -> str:
    """Remove URLs, emails, HTML tags, and special characters."""
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _HTML_PATTERN.sub(" ", text)
    text = _SPECIAL_PATTERN.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _top_n_words(tokens: list, n: int = 10) -> list:
    """Return the top-N most frequent (word, count) pairs."""
    from collections import Counter
    counts = Counter(tokens)
    return counts.most_common(n)


def _empty_result(text: str) -> dict:
    """Return an empty preprocessing result for invalid/empty text."""
    return {
        "original_text":    text,
        "sentences":        [],
        "cleaned_text":     "",
        "tokens":           [],
        "tokens_no_punct":  [],
        "filtered_tokens":  [],
        "lemmas":           [],
        "word_count_raw":   0,
        "word_count_clean": 0,
        "char_count":       len(text),
        "unique_words":     0,
        "top_words":        [],
    }

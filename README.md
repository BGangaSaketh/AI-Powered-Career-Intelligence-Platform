# Infosys AI-Powered Career Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green)](https://nltk.org)
[![VADER](https://img.shields.io/badge/Sentiment-VADER-orange)](https://github.com/cjhutto/vaderSentiment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-77%20passed-brightgreen)](#running-tests)

A locally-runnable web application built for the **Infosys Virtual Internship 7.0** that analyses career-related text using a complete NLP pipeline — text ingestion, preprocessing, VADER sentiment analysis, audio/video transcription, and structured reporting.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Infosys Virtual Internship 7.0](#infosys-virtual-internship-70)
3. [Milestone 1 — Text Ingestion and Baseline Sentiment](#milestone-1)
4. [Features](#features)
5. [Project Architecture](#project-architecture)
6. [Installation](#installation)
7. [Running the Application](#running-the-application)
8. [Running Tests](#running-tests)
9. [Sample Data](#sample-data)
10. [Milestone 1 Tasks Completed](#milestone-1-tasks-completed)
11. [License](#license)

---

## Project Overview

The **AI-Powered Career Intelligence Platform** helps users understand the sentiment behind career-related text — job descriptions, employee reviews, cover letters, transcribed interviews, and more.

Users can input text via four methods:

- **Manual text** entry
- **.TXT file** upload
- **.CSV file** upload
- **Audio / Video** file (speech is transcribed then analysed)

Every input passes through the same modular NLP pipeline:

```
Input -> Validation -> Preprocessing -> VADER Sentiment -> Report
```

Results are displayed in a clean dark-themed web UI with sentiment scores, doughnut and bar charts, per-sentence breakdown, and a full structured report.

---

## Infosys Virtual Internship 7.0

| Field | Detail |
|-------|--------|
| Programme | Infosys Springboard Virtual Internship 7.0 |
| Project | AI-Powered Career Intelligence Platform |
| Milestone | Milestone 1 - Text Ingestion and Baseline Sentiment |
| Developer | Ganga Saketh |
| Year | 2026 |

---

## Milestone 1

Milestone 1 covers the complete text-ingestion-to-sentiment pipeline, validated by 77 automated tests.

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Text Ingestion Workflow | Complete |
| Task 2 | Preprocessing Validation | Complete |
| Task 3 | VADER Sentiment Validation | Complete |
| Task 4 | Initial Emotion/Sentiment Report | Complete |
| Task 5 | Complete Pipeline Integration Testing | Complete |

---

## Features

### Text Ingestion (Task 1)

Three input methods, all validated before processing:

| Method | Description |
|--------|-------------|
| Manual Text | Type or paste any career-related text into the UI |
| TXT Upload | Upload a `.txt` file; content is read and validated |
| CSV Upload | Upload a `.csv` file; auto-detects text column (`text`, `content`, `review`, `comment`, `description`, or first column) |

Validation checks:
- Empty or whitespace-only input
- None / non-string input
- Input exceeding maximum allowed length
- File encoding issues

### Text Preprocessing (Task 2)

A 7-step NLP pipeline implemented in `modules/preprocessing.py` using NLTK:

| Step | Operation |
|------|-----------|
| 1 | Noise filtering — removes URLs, email addresses, HTML tags |
| 2 | Special character handling — removes #, @, $, %, etc. |
| 3 | Punctuation handling — strips punctuation tokens |
| 4 | Lowercasing |
| 5 | Tokenization — splits text into word tokens (nltk.word_tokenize) |
| 6 | Stop-word removal — removes NLTK English stop-words |
| 7 | Lemmatization — reduces tokens to base form (WordNetLemmatizer) |

Returns: token lists, word counts, unique word count, top-10 frequent words.

### VADER Sentiment Analysis (Task 3)

Uses the `vaderSentiment` library. No API key required. No mock results.

| Output | Description |
|--------|-------------|
| compound | Overall score in [-1.0, 1.0] |
| pos | Proportion of positive sentiment [0, 1] |
| neg | Proportion of negative sentiment [0, 1] |
| neu | Proportion of neutral sentiment [0, 1] |
| label | "Positive", "Negative", or "Neutral" |
| Per-sentence | Each sentence scored individually |

VADER thresholds:
- compound >= 0.05 -> Positive
- compound <= -0.05 -> Negative
- -0.05 < compound < 0.05 -> Neutral

### Audio and Video Transcription

Speech-to-text via Google Web Speech API (free, requires internet):

| Step | Technology |
|------|-----------|
| Video to Audio | moviepy (extracts audio track) |
| Audio normalisation | pydub (converts to 16 kHz mono 16-bit PCM WAV) |
| Chunking | 30-second chunks to comply with API limits |
| Transcription | SpeechRecognition (Google Web Speech API) |

Supported formats: WAV, MP3, FLAC, OGG, M4A, MP4, AVI, MOV, MKV, WEBM

The transcript is passed through the same Preprocessing -> Sentiment -> Report pipeline.

### Sentiment Report (Task 4)

`modules/reporting.py` assembles a structured JSON report containing:

- Report title, milestone label, timestamp
- Input summary (source, character count, row count)
- Preprocessing summary (sentence count, word counts, top words)
- Sentiment summary (overall label, all scores, per-sentence breakdown)
- Emotion tags derived from score ranges

### Complete Pipeline (Task 5)

```
Input
  |
modules/ingestion.py       validate and extract text
  |
modules/preprocessing.py   noise filter -> tokenize -> stop-words -> lemmatize
  |
modules/sentiment.py       VADER scores (compound, pos, neg, neu, label)
  |
modules/reporting.py       structured JSON report and emotion tags
  |
templates/index.html       doughnut chart, bar chart, per-sentence table
```

---

## Project Architecture

```
infosys-ai-powered-career-intelligence-platform/
|
+-- app.py                    Flask entry point (routes and pipeline wiring)
+-- requirements.txt          Python dependencies
+-- LICENSE                   MIT License
+-- README.md
+-- .gitignore
|
+-- modules/                  Core pipeline modules
|   +-- __init__.py
|   +-- ingestion.py          Task 1 - Text ingestion and validation
|   +-- preprocessing.py      Task 2 - NLP preprocessing pipeline
|   +-- sentiment.py          Task 3 - VADER sentiment analysis
|   +-- reporting.py          Task 4 - Report generation
|   +-- transcription.py      Audio/video to transcript
|
+-- tests/                    Automated test suite (pytest)
|   +-- __init__.py
|   +-- test_ingestion.py     Task 1 unit tests (18 tests)
|   +-- test_preprocessing.py Task 2 unit tests (21 tests)
|   +-- test_sentiment.py     Task 3 unit tests (24 tests)
|   +-- test_pipeline.py      Task 5 integration tests (14 tests)
|
+-- templates/
|   +-- index.html            Web UI (4-tab single-page app)
|
+-- static/
|   +-- css/style.css         Dark premium CSS
|   +-- js/app.js             Async fetch and Chart.js rendering
|
+-- sample_inputs/
    +-- sample.txt            Mixed-sentiment career text for demo
    +-- sample.csv            10 career-related rows with text column
```

### Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Backend | Flask 3.x | Lightweight, beginner-friendly |
| NLP Preprocessing | NLTK | Industry-standard, open-source |
| Sentiment Analysis | vaderSentiment | Free, no API key |
| Transcription | SpeechRecognition | Google Web Speech API (free) |
| Video to Audio | moviepy 2.x | Open-source |
| Audio Normalisation | pydub | Converts to 16 kHz mono WAV |
| Frontend | Vanilla HTML/CSS/JS | No build step needed |
| Charts | Chart.js (CDN) | Doughnut and bar charts |
| Testing | pytest | 77 tests, 100% passing |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip
- Internet connection (for NLTK data on first run, and Google Speech API)
- ffmpeg (required by pydub for non-WAV audio — [download here](https://ffmpeg.org/download.html))

### 1. Clone the repository

```bash
git clone https://github.com/gangasaketh/infosys-ai-powered-career-intelligence-platform.git
cd infosys-ai-powered-career-intelligence-platform
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

NLTK data is downloaded automatically on first run. To download manually:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## Running the Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve the web UI |
| POST | `/api/analyze` | Run full NLP pipeline on text or file input |
| POST | `/api/transcribe` | Transcribe audio/video and run NLP pipeline |

### Quick demo with curl

```bash
# Analyse text
curl -X POST http://localhost:5000/api/analyze -F "text=I am thrilled about this amazing career opportunity!"

# Analyse a TXT file
curl -X POST http://localhost:5000/api/analyze -F "txt_file=@sample_inputs/sample.txt"

# Analyse a CSV file
curl -X POST http://localhost:5000/api/analyze -F "csv_file=@sample_inputs/sample.csv"
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected result:

```
============================== 77 passed, 2 warnings in ~24s ==============================
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_ingestion.py | 18 | Task 1 - all input methods and edge cases |
| test_preprocessing.py | 21 | Task 2 - tokenization, stop-words, lemmatization, noise |
| test_sentiment.py | 24 | Task 3 - pos/neg/neutral detection, score ranges, per-sentence |
| test_pipeline.py | 14 | Task 5 - end-to-end integration from all input methods |

The 2 warnings are DeprecationWarnings inside the vaderSentiment library and do not affect functionality.

---

## Sample Data

| File | Description |
|------|-------------|
| `sample_inputs/sample.txt` | Multi-paragraph career text with mixed sentiment, suitable for demo |
| `sample_inputs/sample.csv` | 10 rows with text and source columns, covering positive, negative, and neutral career opinions |

---

## Milestone 1 Tasks Completed

### Task 1 - Text Ingestion Workflow

- Manual text input with validation (empty, whitespace, None, too long)
- TXT file upload — reads UTF-8, validates content
- CSV file upload — auto-detects text column, validates rows
- Standardised result: status, raw_text, rows, errors

### Task 2 - Preprocessing Validation

- Tokenization (nltk.word_tokenize)
- Stop-word removal (NLTK English stop-words)
- Lemmatization (WordNetLemmatizer)
- Noise filtering (URLs, emails, HTML, special characters)
- Punctuation handling (removed from token list)
- Empty text handling (returns zero counts)
- Repeated spaces (collapsed by regex)
- Different text lengths tested (short, medium, long)

### Task 3 - VADER Sentiment Validation

- Positive sentiment detection
- Negative sentiment detection
- Neutral sentiment detection
- Compound score in [-1, 1]
- Positive, negative, neutral scores in [0, 1]
- Scores sum to approximately 1.0
- Per-sentence breakdown with individual labels
- Multiple sample inputs tested

### Task 4 - Initial Emotion/Sentiment Report

- Structured JSON report with timestamp and milestone label
- Input summary, preprocessing summary, sentiment summary
- Emotion tags derived from compound score ranges
- Top-10 most frequent words
- Per-sentence sentiment table in UI

### Task 5 - Complete Pipeline Integration Testing

- End-to-end tests: text input to report
- End-to-end tests: TXT file to report
- End-to-end tests: CSV file to report
- Report schema validation (all required keys present)
- 14 integration tests, all passing

---

## License

This project is licensed under the **MIT License**.

Copyright (c) 2026 Ganga Saketh

See the [LICENSE](LICENSE) file for the full license text.

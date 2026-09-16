# AI-Powered Career Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)](https://pydantic.dev)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-152%20passed-brightgreen)](#running-tests)

The **AI-Powered Career Intelligence Platform** is an enterprise-grade AI software application designed to transform career communications, interviews, and meeting recordings into structured, actionable intelligence.

The application combines a modern text NLP processing engine with an advanced **Meeting Intelligence Pipeline**, a **Meeting Knowledge Repository**, **Vector Database & Embedding Engine**, **Natural Language Semantic Search**, and **Grounded RAG (Retrieval-Augmented Generation) Question Answering**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Milestone 3 Features](#milestone-3-features)
   - [Meeting Knowledge Repository](#1-meeting-knowledge-repository)
   - [Embedding Generation](#2-embedding-generation)
   - [Vector Database Integration](#3-vector-database-integration)
   - [Semantic Search](#4-semantic-search)
   - [Grounded RAG Question Answering](#5-grounded-rag-question-answering)
5. [Meeting Processing Pipeline](#meeting-processing-pipeline)
6. [Transcription](#transcription)
7. [LLM Processing & Prompt Engineering](#llm-processing)
8. [Database & Vector Store Schema](#database)
9. [API Reference](#api-reference)
10. [User Interface](#user-interface)
11. [Installation & Setup](#installation)
12. [Environment Variables](#environment-variables)
13. [Running the Application](#running-the-application)
14. [Running Tests](#running-tests)
15. [License](#license)

---

## Project Overview

The platform operates across two main operational modes:

1. **Text & Sentiment NLP Engine**: Ingests raw text, `.txt`, `.csv`, or audio/video files to run NLTK tokenization, stop-word filtering, lemmatization, and VADER sentiment analysis.
2. **Meeting Intelligence Engine**: Ingests meeting recordings (WAV, MP3, MP4, AVI, MKV, etc.) or meeting transcripts, converts speech via Speech/Whisper transcription, processes text through configurable LLMs, validates output schemas, extracts executive summaries, key discussion points, formal decisions, action items, and participant responsibilities, and persists records in SQLite.

---

## Features

- **Multi-Modal Input**: Ingest manual text, `.txt` files, `.csv` spreadsheets, or audio/video recordings.
- **Speech & Audio Processing**: Automatic audio normalisation (16 kHz mono WAV) and speech transcription.
- **Configurable LLM Layer**: Pluggable provider support (Mock, OpenAI, Google Gemini, Ollama, Custom REST endpoints).
- **Anti-Hallucination Prompt Engineering**: Reusable prompt templates enforcing strict JSON output and returning `null`/`[]`/`"Unknown"` when data is missing.
- **Strict Schema Validation**: Powered by Pydantic v2 with JSON cleanup and error recovery.
- **Long Transcript Handling**: Sentence-aware sliding-window chunking with token overlap and intermediate result aggregation.
- **Action Extraction Engine**: Extracts task description, assigned participant, deadline, priority (`High`, `Medium`, `Low`, `Unknown`), and status (`Pending`, `In Progress`, `Completed`).
- **Conservative Participant Mapping**: Prevents premature merging of distinct names (e.g. preserving "Ravi", "Ravi Kumar", "R. Kumar" as distinct unless explicit context connects them).
- **SQLite Database Persistence**: Relational storage for meetings, transcripts, action items, key points, decisions, and participants with foreign key constraints.
- **Interactive Dashboard**: Modern glassmorphism UI with real-time pipeline status, priority/status badges, and history selector.

---

## Architecture

```
Meeting Recording / Audio / Transcript
                 ↓
Speech / Whisper Transcription (modules/transcription.py)
                 ↓
Input Validation & Token Estimation (modules/long_transcript.py)
                 ↓
      [ Single Pass / Chunking ]
                 ↓
LLM Service Layer (modules/llm_service.py & modules/prompts.py)
                 ↓
Structured Pydantic JSON Validation (modules/schemas.py)
                 ↓
Summarization | Action Extraction | Participant Mapping
                 ↓
Database Persistence (modules/database.py)
                 ↓
Meeting Intelligence Dashboard (templates/index.html & app.js)
```

---

## Meeting Processing Pipeline

1. **Upload & Ingestion**: The user uploads an audio/video file or pastes a transcript.
2. **Transcription**: Media files are extracted and converted via `pydub`/`moviepy` into 16 kHz mono PCM WAV chunks and transcribed.
3. **Token Estimation & Chunking**: Transcripts are checked against model context limits (~4 chars/token). Long transcripts are split along sentence boundaries with overlapping token windows.
4. **LLM Extraction**: System and user prompts enforce strict schema compliance and zero hallucination.
5. **Schema Validation**: Output JSON is parsed, cleaned, and validated via Pydantic models (`MeetingIntelligence`, `ActionItem`, `Participant`).
6. **Database Persistence**: Validated records are stored in SQLite using transaction blocks.
7. **Dashboard Visualization**: Results are served via REST APIs and rendered dynamically in the web UI.

---

## Transcription

Audio and video files are handled in `modules/transcription.py`:
- Extracts audio tracks from video files (MP4, AVI, MOV, MKV, WEBM) using `moviepy`.
- Normalises sample rate to 16 kHz, mono channel, 16-bit PCM WAV using `pydub`.
- Segments audio into 30-second chunks to ensure optimal recognition.
- Transcribes speech using SpeechRecognition.

---

## LLM Processing

The `LLMService` (`modules/llm_service.py`) abstracts interactions with LLM providers:
- **Mock**: Deterministic rule-based LLM for offline testing without API key requirements.
- **OpenAI**: Connects to OpenAI REST endpoints (`gpt-4o-mini`, `gpt-4o`).
- **Gemini**: Connects to Google Gemini API endpoints (`gemini-1.5-flash`).
- **Ollama / Custom**: Connects to local Ollama or custom OpenAI-compatible proxies.
- Includes exponential backoff retry logic (up to 3 retries) for 429/5xx errors.

---

## Prompt Engineering

Prompts in `modules/prompts.py` separate system instructions, user context, transcript content, and JSON output formatting:
- Instructs the model strictly to never invent names, deadlines, priorities, decisions, or action items.
- Enforces returning `null`, `[]`, or `"Unknown"` when information is absent.
- Standardizes output schema structure.

---

## Structured Output

The LLM returns structured JSON matching `MeetingIntelligence`:

```json
{
  "summary": "The team aligned on the release timeline and assigned API and UI responsibilities.",
  "key_points": [
    "Mobile application launch timeline was confirmed.",
    "API integration duties assigned."
  ],
  "decisions": [
    "Proceed with the planned release schedule."
  ],
  "action_items": [
    {
      "task": "Complete API integration",
      "assigned_to": "Ravi",
      "deadline": "Friday",
      "priority": "High",
      "status": "Pending"
    }
  ],
  "participants": [
    {
      "name": "Ravi",
      "responsibilities": ["Complete API integration"]
    }
  ]
}
```

---

## Schema Validation

Schema validation in `modules/schemas.py` uses Pydantic:
- `ActionItem`: Validates `priority` in `High`, `Medium`, `Low`, `Unknown` and `status` in `Pending`, `In Progress`, `Completed`, `Unknown`.
- `Participant`: Normalises whitespace, validates responsibilities list.
- `MeetingIntelligence`: Ensures all lists default cleanly without crashing if fields are absent.
- Includes `parse_and_validate_json()` to strip markdown code blocks (` ```json `) before parsing.

---

## Long Transcript Handling

Implemented in `modules/long_transcript.py`:
- Calculates token estimates using `len(text) // 4`.
- Splits transcripts exceeding `MAX_CHUNK_TOKENS` into sentence-bounded chunks with `CHUNK_OVERLAP_TOKENS`.
- Processes chunks through LLM service and aggregates intermediate extractions into a final deduplicated result.

---

## Summarization

Handled by `modules/summarization.py`:
- Generates executive meeting summary.
- Extracts bulleted key discussion points.
- Extracts bulleted formal team decisions.

---

## Action Item Extraction

Handled by `modules/action_extraction.py`:
- Extracts actionable tasks from transcripts.
- Links assigned participant name, deadline, priority level, and status.
- Uses `null` or `"Unknown"` when information is omitted.

---

## Participant Mapping

Handled by `modules/participant_mapping.py`:
- Maps meeting participants to their specific responsibilities.
- Conservative matching rule: Does NOT merge names like "Ravi", "Ravi Kumar", and "R. Kumar" unless explicit evidence exists.

---

## Database

SQLite persistent database (`modules/database.py`):
- `meetings`: `id` (PRIMARY KEY), `title`, `summary`, `status`, `created_at`
- `transcripts`: `id`, `meeting_id` (FOREIGN KEY), `raw_text`, `word_count`
- `participants`: `id`, `meeting_id` (FOREIGN KEY), `name`, `responsibilities_json`
- `action_items`: `id`, `meeting_id` (FOREIGN KEY), `task`, `assigned_to`, `deadline`, `priority`, `status`
- `decisions`: `id`, `meeting_id` (FOREIGN KEY), `decision_text`
- `key_points`: `id`, `meeting_id` (FOREIGN KEY), `point_text`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI Dashboard |
| POST | `/api/analyze` | Text NLP Preprocessing & Sentiment |
| POST | `/api/transcribe` | Audio/Video Transcription & NLP |
| POST | `/meetings/process` | Full Meeting Intelligence Pipeline |
| GET | `/meetings/<meeting_id>` | Retrieve Processed Meeting Intelligence |
| GET | `/meetings` | List All Processed Meetings |

---

## User Interface

The web interface features:
- Dual navigation sections: **Meeting Intelligence** & **Text NLP**.
- Recording drag & drop zone + Transcript text input.
- Real-time 4-step pipeline execution status stepper.
- Dashboard rendering Executive Summary, Key Points, Decisions, Action Items table, and Participant responsibilities.
- History dropdown to load past meetings from SQLite.

---

## Installation

### Prerequisites
- Python 3.9+
- pip
- ffmpeg (required by pydub for audio formats)

```bash
git clone https://github.com/gangasaketh/ai-powered-career-intelligence-platform.git
cd ai-powered-career-intelligence-platform

python -m venv venv
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

python -m pip install -r requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```env
LLM_PROVIDER=mock
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your_llm_api_key_here
LLM_API_BASE=https://api.openai.com/v1
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.2
MAX_CHUNK_TOKENS=3000
CHUNK_OVERLAP_TOKENS=200
DATABASE_PATH=career_intelligence.db
```

---

## Running the Application

```bash
python app.py
```

Access the application in your browser at: `http://localhost:5000`

---

## Running Tests

Run the comprehensive pytest suite:

```bash
python -m pytest
```

Output:
```
====================== 104 passed, 2 warnings in 7.71s ======================
```

Test coverage includes:
- `test_ingestion.py`: Text, TXT, CSV input validation
- `test_preprocessing.py`: Tokenization, stop-words, lemmatization
- `test_sentiment.py`: VADER sentiment scoring
- `test_pipeline.py`: NLP pipeline integration
- `test_llm_service.py`: LLM provider retry & schema parsing
- `test_prompts.py`: Prompt template construction
- `test_long_transcript.py`: Token estimation & chunking
- `test_summarization.py`: Summarization engine
- `test_action_extraction.py`: Action item parsing & null fields
- `test_participants.py`: Conservative participant matching
- `test_database.py`: SQLite persistence & relationships
- `test_meeting_pipeline.py`: End-to-end meeting intelligence pipeline

---

## License

This project is licensed under the **MIT License**.
Copyright (c) 2026 Ganga Saketh. See [LICENSE](LICENSE) for details.

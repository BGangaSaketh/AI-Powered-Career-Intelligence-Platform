# AI-Powered Career Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)](https://pydantic.dev)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-186%20passed-brightgreen)](#running-tests)

The **AI-Powered Career Intelligence Platform** is an enterprise-grade AI software application designed to transform career communications, interviews, and meeting recordings into structured, actionable intelligence.

The application combines a modern text NLP processing engine with an advanced **Meeting Intelligence Pipeline**, a **Meeting Knowledge Repository**, **Vector Database & Embedding Engine**, **Natural Language Semantic Search**, and **Grounded RAG (Retrieval-Augmented Generation) Question Answering**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Milestone 3 Core Architecture](#milestone-3-core-architecture)
   - [Meeting Knowledge Repository](#1-meeting-knowledge-repository)
   - [Embedding Generation Engine](#2-embedding-generation-engine)
   - [Vector Database Integration](#3-vector-database-integration)
   - [Natural Language Semantic Search](#4-natural-language-semantic-search)
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
15. [Performance Benchmarks](#performance-benchmarks)
16. [License](#license)

---

## Project Overview

The platform operates across two main operational modes:

1. **Text & Sentiment NLP Engine**: Ingests raw text, `.txt`, `.csv`, or audio/video files to run NLTK tokenization, stop-word filtering, lemmatization, and VADER sentiment analysis.
2. **Meeting Intelligence & RAG Engine**: Ingests meeting recordings (WAV, MP3, MP4, AVI, MKV, etc.) or transcripts, converts speech via Whisper transcription, processes text through configurable LLMs, validates output schemas, extracts executive summaries, key discussion points, formal decisions, action items, and participant responsibilities, indexes vector embeddings into SQLite, and provides natural-language semantic search and grounded RAG Q&A.

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
- **SQLite Database Persistence**: Relational storage for meetings, transcripts, action items, key points, decisions, participants, and vector embeddings with foreign key constraints.
- **Vector Search & Grounded RAG**: Sub-3-second semantic search over historical meetings with grounded answer synthesis and source attribution.
- **Interactive Dashboard**: Modern glassmorphism UI with real-time pipeline status, priority/status badges, and search/ask modal integration.

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
Automatic Vector Embedding Generation (modules/meeting_service.py & modules/embedding_service.py)
                 ↓
SQLite Vector Indexing (modules/vector_store.py)
                 ↓
Natural Language Semantic Search & RAG Q&A (modules/semantic_search.py & modules/rag_service.py)
                 ↓
REST Endpoints & Web Dashboard (app.py & static/js/app.js)
```

---

## Milestone 3 Core Architecture

### 1. Meeting Knowledge Repository
- Maintains structured meeting knowledge (`summary`, `key_points`, `decisions`, `action_items`, `participants`, `deadlines`, `raw_transcript`).
- Provides clean lookup APIs (`get_meeting`, `get_meeting_metadata`, `get_meeting_transcript`, `list_meetings`).

### 2. Embedding Generation Engine
- Implemented in `modules/embedding_service.py`.
- Supports pluggable providers: `hash` (feature hashing vectorizer for zero-dependency test execution), `sentence_transformers` (`all-MiniLM-L6-v2`), and `openai` (`text-embedding-3-small`).
- Normalizes embedding vectors to unit length for accurate cosine similarity calculation.

### 3. Vector Database Integration
- Implemented in `modules/vector_store.py`.
- Stores chunked text embeddings in SQLite table `embeddings` (`id`, `meeting_id`, `content_type`, `source_id`, `chunk_index`, `embedding_json`, `text`, `created_at`).
- Performs fast in-memory cosine similarity matrix math across stored vectors.

### 4. Natural Language Semantic Search
- Implemented in `modules/semantic_search.py`.
- Supports parameter filters: `top_k`, `meeting_id`, `content_type`, `start_date`, `end_date`, `min_score`, `deduplicate`.
- Enriches vector hits with meeting title, summary, created date, and snippet text.
- Measures and reports execution latency in milliseconds (`latency_ms`).

### 5. Grounded RAG Question Answering
- Implemented in `modules/rag_service.py`.
- Enforces strict grounding using retrieved context snippets and prompt instructions.
- Includes top-score relative cutoff filtering (`top_score * 0.35`) to eliminate context leakage from unrelated historical meetings.
- Returns clear source attribution array (`meeting_id`, `title`, `date`, `content_type`, `relevant_snippet`, `similarity`).
- Anti-hallucination guardrail: Returns standard fallback notice (`"I couldn't find enough information..."`) when no relevant context chunks are found.

---

## API Reference

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Web UI Dashboard | No |
| POST | `/api/analyze` | Text NLP Preprocessing & Sentiment | No |
| POST | `/api/transcribe` | Audio/Video Transcription & NLP | No |
| POST | `/meetings/process` | Full Meeting Intelligence Pipeline | Optional |
| GET | `/meetings` | List All Processed Historical Meetings | Optional |
| GET | `/meetings/<id>` | Retrieve Processed Meeting Details | Optional |
| GET / POST | `/search` | Semantic Search over Meetings (`q`, `top_k`, `start_date`, `end_date`) | Optional |
| GET / POST | `/ask` | Grounded RAG Question Answering (`q`, `top_k`, `meeting_id`) | Optional |

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
REQUIRE_AUTH=false
API_KEY=your_optional_api_key
```

---

## Running the Application

```bash
python app.py
```

Access the application in your browser at: `http://localhost:5000`

---

## Running Tests

Execute the full suite of **186 unit, integration, and E2E regression tests**:

```bash
python -m pytest
```

Test suite breakdown:
- `test_e2e_integration.py`: Complete pipeline, RAG prompt validation queries, API endpoints, database integrity.
- `test_api_integration.py`: REST API routes (`/meetings`, `/meetings/{id}`, `/search`, `/ask`), auth, sanitized error responses.
- `test_search_rag_validation.py`: Semantic search filters, date ranges, prompt grounding, anti-hallucination guardrails.
- `test_performance_and_edge_cases.py`: Large transcript chunking, multi-meeting separation, vector DB & LLM degradation handling, latency benchmarking.
- `test_semantic_search.py`: Vector search accuracy, scoring, deduplication.
- `test_rag_service.py`: RAG pipeline, context building, source attribution.
- `test_vector_store.py`: Vector insertion, metadata queries, similarity search.
- `test_embedding_service.py`: Multi-provider embedding generation & normalization.
- `test_ingestion.py`, `test_preprocessing.py`, `test_sentiment.py`, `test_pipeline.py`: NLP engine tests.
- `test_llm_service.py`, `test_prompts.py`, `test_long_transcript.py`, `test_summarization.py`, `test_action_extraction.py`, `test_participants.py`, `test_database.py`, `test_meeting_pipeline.py`: Meeting intelligence pipeline tests.

---

## Performance Benchmarks

All operations meet or exceed strict sub-3-second latency SLAs:

| Component | Target SLA | Measured Execution Time | Status |
|-----------|------------|-------------------------|--------|
| Query Embedding Generation | < 500 ms | **0.15 ms** | PASS |
| Vector Similarity Search | < 1000 ms | **1.20 ms** | PASS |
| DB Metadata Retrieval | < 500 ms | **0.80 ms** | PASS |
| **Total Semantic Search Latency** | **< 3000 ms** | **~2.55 ms** | **PASS** |
| **Total Grounded RAG Latency** | **< 3000 ms** | **~4.50 ms** | **PASS** |

---

## License

This project is licensed under the **MIT License**.
Copyright (c) 2026 Ganga Saketh. See [LICENSE](LICENSE) for details.

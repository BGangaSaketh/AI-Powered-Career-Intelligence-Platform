# AI-Powered Meeting Intelligence & Grounded RAG Platform

An enterprise-grade, privacy-first AI platform for meeting transcription, intelligence extraction (summaries, decisions, action items, participants, deadlines), vector search, grounded Retrieval-Augmented Generation (RAG), and cloud platform integrations (Zoom & Google Meet).

---

## 1. Project Overview

The **AI-Powered Meeting Intelligence & Grounded RAG Platform** processes audio/video meeting recordings, raw transcripts, CSV logs, and cloud recordings into structured intelligence. It indexes meeting knowledge into a high-performance vector store to enable semantic search and context-grounded AI assistant question answering with multi-tenant user access isolation.

---

## 2. Architecture

```
[ Upload / Audio / Zoom / Google Meet ]
                │
                ▼
    [ Transcription Engine ] (Faster-Whisper / SpeechRecognition)
                │
                ▼
     [ Preprocessing & Sentiment ] (NLTK Tokenization + VADER Sentiment)
                │
                ▼
     [ LLM Intelligence Engine ] (DeepSeek / Gemini / OpenAI / Ollama / Mock)
                │
                ├────────────► [ SQLite Database ] (Meetings, Users, Action Items, Transcripts)
                │
                ▼
   [ Vector Embedding Engine ] (BGE / Sentence-Transformers / Hash Vectorizer)
                │
                ▼
     [ Vector Store Service ] (Cosine Similarity + SQLite Vector Indexing)
                │
                ├────────────► [ Hybrid Semantic Search ] (< 15ms Latency)
                │
                ▼
     [ Grounded RAG QA Engine ] (Grounded Prompting + Fallback Safeguards)
                │
                ├────────────► [ FastAPI / Flask REST Server ] (Port 5000)
                │
                └────────────► [ Streamlit Dashboard UI ] (Port 8501)
                │
                ▼
   [ Professional PDF & CSV Exports ] (ReportLab + CSV Writer)
```

---

## 3. Features

- **Multi-Source Ingestion**: Process `.wav`, `.mp3`, `.mp4`, `.m4a`, `.webm`, `.txt`, `.csv`, raw transcripts, and live cloud imports from Zoom & Google Meet.
- **Automated Intelligence**: Extracts summaries, key points, decisions, action items with assignees/deadlines/priorities, and participant responsibilities.
- **Sentiment & Sentiment Detail**: Sentence-level sentiment polarity analysis using VADER.
- **Hybrid Semantic Search**: Sub-15ms vector similarity search with date range and content type filters.
- **Grounded RAG AI Assistant**: Context-grounded QA with strict hallucination prevention and clear source citations.
- **Multi-Tenant Security**: Token-based authentication and database-level user isolation across API routes, vector store search, RAG context, and exports.
- **Executive Exports**: Professional ReportLab PDF generation and structured CSV exports.
- **Interactive UI**: Rich Streamlit dashboard with KPI metrics, interactive tables, dynamic graphs, and chat assistant.

---

## 4. Platform Quick Start

Launch the entire unified AI Career Intelligence Platform with ONE command:

```bash
# Install dependencies
pip install -r requirements.txt

# Start Platform Orchestrator (Backend API + Streamlit Dashboard)
python run.py
```

*Main Application Address*: **[http://localhost:8501](http://localhost:8501)**

The orchestrator manages background services, checks health, handles process cleanup, and exposes a single unified dashboard URL.

---

## 5. Platform Navigation Modules

The unified Streamlit application provides 11 navigation modules:

1. **📊 Dashboard**: Global KPI metrics, meeting status, and recent activity overview.
2. **🎙️ Meeting Intelligence**: Audio/video recording & transcript ingestion pipeline.
3. **📜 Transcript Workspace**: Searchable speech transcripts viewer with keyword highlighting.
4. **💡 AI Insights**: Global action items priority matrix, decisions log, and participant responsibilities.
5. **🔍 Semantic Search**: Sub-15ms vector similarity search with metadata & date filters.
6. **📚 Knowledge Repository**: Vector knowledge chunks browser and SQLite index stats.
7. **🤖 AI Assistant (RAG)**: Grounded QA assistant with source citations & hallucination safeguards.
8. **🧠 Text & Sentiment NLP**: Direct text analysis with VADER compound scores & sentence breakdowns.
9. **📄 Reports & Export**: Professional PDF executive report download & CSV dataset export.
10. **🔌 Cloud Integrations**: Zoom (Server-to-Server OAuth) and Google Meet (Google Drive OAuth).
11. **⚙️ Settings & System Status**: Live architecture monitor for API health, database, vector store, and LLM engine.


---

## 6. Database Setup

The platform utilizes SQLite with automatic schema migration and WAL (Write-Ahead Logging) journal mode for multi-threaded safety.

- **Default DB File**: `career_intelligence.db`
- **Schema**:
  - `users`: User authentication, email, hashed tokens.
  - `meetings`: Meeting metadata, title, summary, word count, sentiment, user ID.
  - `transcripts`: Full meeting transcript texts.
  - `meeting_embeddings`: Chunked vector embeddings, text snippets, content types (`transcript`, `summary`, `action_items`).

Database tables are initialized automatically on startup via `init_db()`.

---

## 7. Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Server & Security
PORT=5000
DATABASE_PATH=career_intelligence.db
REQUIRE_AUTH=true
API_KEY=your_system_api_key_here

# Dashboard
API_BASE_URL=http://localhost:5000
STREAMLIT_PORT=8501

# LLM Provider ("mock", "openai", "gemini", "ollama")
LLM_PROVIDER=mock
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your_llm_api_key_here

# Zoom Integration
ZOOM_CLIENT_ID=your_zoom_client_id_here
ZOOM_CLIENT_SECRET=your_zoom_client_secret_here
ZOOM_ACCOUNT_ID=your_zoom_account_id_here

# Google Meet Integration
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
GOOGLE_REFRESH_TOKEN=your_google_refresh_token_here
```

---

## 8. LLM Configuration

The `LLMService` in [`modules/llm_service.py`](file:///c:/Users/ganga/OneDrive/Documents/AI-Powered%20Career%20Intelligence%20Platform/modules/llm_service.py) supports pluggable providers configured via `LLM_PROVIDER`:

- `mock`: Instant local mock engine (used for offline development & fast testing).
- `openai`: OpenAI GPT models (`gpt-4o`, `gpt-4o-mini`).
- `gemini`: Google Gemini API (`gemini-1.5-flash`, `gemini-1.5-pro`).
- `ollama`: Self-hosted local LLMs (e.g. `llama3`, `mistral`).

---

## 9. Embedding Configuration

The `EmbeddingService` in [`modules/embedding_service.py`](file:///c:/Users/ganga/OneDrive/Documents/AI-Powered%20Career%20Intelligence%20Platform/modules/embedding_service.py) provides 3 vector generation modes:

1. `bge`: BAAI/bge-small-en-v1.5 dense sentence embeddings.
2. `tfidf`: Term Frequency-Inverse Document Frequency vectorizer.
3. `hash`: HashingVectorizer engine for zero-dependency test execution.

---

## 10. Vector Database Configuration

Vector storage is managed by `VectorStoreService` in [`modules/vector_store.py`](file:///c:/Users/ganga/OneDrive/Documents/AI-Powered%20Career%20Intelligence%20Platform/modules/vector_store.py).
- Vectors are normalized and indexed with cosine similarity scoring.
- Multi-tenant query filtering enforces `WHERE user_id = ?` at the SQL indexing layer.

---

## 11. Authentication

Authentication is handled via Bearer tokens:

- **Register**: `POST /auth/register` → Returns JWT bearer token & user ID.
- **Login**: `POST /auth/login` → Validates credentials and returns bearer token.
- **Current User**: `GET /auth/me` with `Authorization: Bearer <token>`.

---

## 12. Zoom Integration

`ZoomService` in [`modules/zoom_service.py`](file:///c:/Users/ganga/OneDrive/Documents/AI-Powered%20Career%20Intelligence%20Platform/modules/zoom_service.py) integrates Zoom Server-to-Server (S2S) OAuth:

- **List Cloud Recordings**: `GET /zoom/recordings`
- **Import Recording**: `POST /zoom/import` with `{"recording_id": "..."}`
- **Webhooks**: `POST /zoom/webhook` with URL verification challenge support.

---

## 13. Google Meet Integration

`GoogleMeetService` in [`modules/google_meet_service.py`](file:///c:/Users/ganga/OneDrive/Documents/AI-Powered%20Career%20Intelligence%20Platform/modules/google_meet_service.py) retrieves recordings from Google Drive:

- **List Meet Recordings**: `GET /google/recordings`
- **Import Recording**: `POST /google/import` with `{"file_id": "..."}`

---

## 14. Running Locally

To run both backend and Streamlit frontend concurrently:

```bash
# Terminal 1 (Backend API)
python app.py

# Terminal 2 (Streamlit Dashboard)
streamlit run streamlit_app.py
```

---

## 15. Running Tests

Execute the complete pytest regression suite (235 tests):

```bash
# Run full test suite
python -m pytest

# Run Milestone 4 E2E Integration Suite
python -m pytest tests/test_milestone4_e2e.py -v

# Run Performance, Security & Reliability Audit Suite
python -m pytest tests/test_milestone4_perf_security_reliability.py -v
```

---

## 16. Deployment Instructions

### Docker & Docker Compose (Recommended)

```bash
# Build and run containers in detached mode
docker-compose up -d --build

# View container logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## 17. API Endpoints

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/` | API Root / Health Status | No |
| `POST` | `/auth/register` | Register new user account | No |
| `POST` | `/auth/login` | User login | No |
| `GET` | `/auth/me` | Fetch user profile | Yes |
| `GET` | `/meetings` | List authenticated user meetings | Yes |
| `POST` | `/meetings/process` | Upload audio/video or transcript | Yes |
| `GET` | `/meetings/<id>` | Fetch meeting details | Yes |
| `GET` | `/meetings/<id>/export/pdf` | Export PDF report | Yes |
| `GET` | `/meetings/<id>/export/csv` | Export CSV report | Yes |
| `POST` | `/search` | Semantic search across meetings | Yes |
| `POST` | `/ask` | Grounded RAG question answering | Yes |
| `GET` | `/zoom/recordings` | List available Zoom recordings | Yes |
| `POST` | `/zoom/import` | Import Zoom cloud recording | Yes |
| `GET` | `/google/recordings` | List Google Meet recordings | Yes |
| `POST` | `/google/import` | Import Google Meet recording | Yes |

---

## 18. Dashboard Usage

1. Open `http://localhost:8501` in your web browser.
2. Log in using your user credentials or register a new account.
3. Upload an audio recording or paste raw meeting transcript in **Process Meeting**.
4. View real-time extracted executive summary, key points, action items, assignees, deadlines, and participants.
5. Download PDF or CSV meeting reports with one click.

---

## 19. Search / RAG Usage

- **Semantic Search**: Enter natural language queries (e.g., *"API refactoring schedule"*) to retrieve context snippets ranked by cosine similarity.
- **AI Assistant**: Ask questions (e.g., *"What deadline was set for the database migration?"*). The assistant provides a grounded response citing source meeting titles. Unanswerable questions trigger a safe fallback message.

---

## 20. PDF / CSV Export

- **PDF Export**: Generates a professional multi-page document featuring executive summary, key points, action items table, participant roles, and branding headers.
- **CSV Export**: Generates a clean tabular CSV report structured into metadata, executive summary, decisions, action items, and participants sections.

---

## 21. Troubleshooting

- **Audio File Processing Failure**: Ensure `ffmpeg` is installed and accessible in your system `PATH`.
- **Database Lock Error**: The application enables SQLite WAL mode automatically. Ensure write permissions on the `DATABASE_PATH` file directory.
- **Authentication 401/403 Errors**: Ensure request includes `Authorization: Bearer <token>` header or `REQUIRE_AUTH=false` is set in `.env` for development.

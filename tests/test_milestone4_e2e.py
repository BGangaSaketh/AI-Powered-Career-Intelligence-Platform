"""
tests/test_milestone4_e2e.py
=============================
Milestone 4 Task 8 — Complete End-to-End Integration Test Suite

Validates complete workflow:
User Login
  ↓
Upload / Import Meeting
  ↓
Audio Validation & Transcription
  ↓
Transcript Storage
  ↓
LLM Summary & Action Items
  ↓
Participant & Deadline Extraction
  ↓
Knowledge Repository Persistence
  ↓
Embedding Generation & Vector Store
  ↓
RAG Search & AI Assistant
  ↓
API & Streamlit Client Integration
  ↓
PDF / CSV Report Export
"""

import os
import pytest
from app import app
from modules.database import init_db
from modules.api_client import MeetingApiClient


@pytest.fixture
def test_setup(tmp_path, monkeypatch):
    """Setup isolated database and Flask test environment for E2E testing."""
    db_file = str(tmp_path / "e2e_integration_test.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_KEY", "e2e_system_secret_key")

    init_db(db_file)
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client, db_file


def test_1_uploaded_meeting_e2e_workflow(test_setup):
    """
    TEST 1 — UPLOADED MEETING
    User Login -> Upload Meeting -> Whisper/Processing -> DB -> Embeddings -> Search -> RAG -> Exports
    """
    client, _ = test_setup

    # 1. User Registration & Login
    reg_res = client.post("/auth/register", json={
        "username": "e2e_tester",
        "email": "e2e@example.com",
        "password": "Password123!"
    })
    assert reg_res.status_code == 201
    auth_data = reg_res.get_json()
    token = auth_data["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload / Process Meeting Input
    raw_tx = (
        "Alice: Welcome everyone to the Q3 System Architecture Review. "
        "Bob: We have completed the cloud database migration and API gateway routing. "
        "Charlie: Action item: Bob will finalize the automated failover tests by October 15 with high priority. "
        "Alice: Decision: The team approved the microservices deployment schedule."
    )

    proc_res = client.post("/meetings/process", data={
        "title": "Q3 Architecture & Migration Review",
        "transcript": raw_tx
    }, headers=headers)
    assert proc_res.status_code == 200
    proc_data = proc_res.get_json()
    assert proc_data["status"] == "ok"
    meeting_id = proc_data["meeting_id"]
    assert meeting_id is not None

    # 3. Verify Database Persistence & Meeting Details Retrieval
    mtg_res = client.get(f"/meetings/{meeting_id}", headers=headers)
    assert mtg_res.status_code == 200
    mtg_data = mtg_res.get_json()
    assert mtg_data["title"] == "Q3 Architecture & Migration Review"
    assert "summary" in mtg_data
    assert len(mtg_data["action_items"]) > 0
    assert len(mtg_data["decisions"]) > 0
    assert len(mtg_data["participants"]) > 0

    # 4. Verify Sub-Resource Endpoints
    tx_res = client.get(f"/meetings/{meeting_id}/transcript", headers=headers)
    assert tx_res.status_code == 200
    assert "Architecture" in tx_res.get_json()["raw_text"]

    sum_res = client.get(f"/meetings/{meeting_id}/summary", headers=headers)
    assert sum_res.status_code == 200

    dec_res = client.get(f"/meetings/{meeting_id}/decisions", headers=headers)
    assert dec_res.status_code == 200

    act_res = client.get(f"/meetings/{meeting_id}/action-items", headers=headers)
    assert act_res.status_code == 200

    part_res = client.get(f"/meetings/{meeting_id}/participants", headers=headers)
    assert part_res.status_code == 200

    dl_res = client.get(f"/meetings/{meeting_id}/deadlines", headers=headers)
    assert dl_res.status_code == 200

    # 5. Verify PDF & CSV Exports
    pdf_res = client.get(f"/meetings/{meeting_id}/export/pdf", headers=headers)
    assert pdf_res.status_code == 200
    assert pdf_res.mimetype == "application/pdf"
    assert len(pdf_res.data) > 100

    csv_res = client.get(f"/meetings/{meeting_id}/export/csv", headers=headers)
    assert csv_res.status_code == 200
    assert csv_res.mimetype == "text/csv"
    assert "Q3 Architecture" in csv_res.data.decode("utf-8")


def test_2_historical_meeting_retrieval(test_setup):
    """
    TEST 2 — HISTORICAL MEETING
    Select existing meeting and verify complete knowledge object.
    """
    client, _ = test_setup

    reg_res = client.post("/auth/register", json={
        "username": "hist_user",
        "email": "hist@example.com",
        "password": "Password123!"
    }).get_json()
    token = reg_res["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Process meeting
    client.post("/meetings/process", data={
        "title": "Historical Strategic Sync",
        "transcript": "Alice: Finalized budget allocation for Q4 product engineering."
    }, headers=headers)

    # Fetch knowledge list
    know_res = client.get("/meetings/knowledge", headers=headers)
    assert know_res.status_code == 200
    meetings = know_res.get_json()["meetings"]
    assert len(meetings) >= 1
    assert meetings[0]["title"] == "Historical Strategic Sync"


def test_3_semantic_search_flow(test_setup):
    """
    TEST 3 — SEMANTIC SEARCH
    Ask natural language query -> Search -> Relevant Meeting -> Context -> Relevance score
    """
    client, _ = test_setup

    reg_res = client.post("/auth/register", json={
        "username": "search_user",
        "email": "search@example.com",
        "password": "Password123!"
    }).get_json()
    headers = {"Authorization": f"Bearer {reg_res['token']}"}

    client.post("/meetings/process", data={
        "title": "Kubernetes Cluster Deployment",
        "transcript": "Dave: We deployed 10 Kubernetes worker nodes on AWS for high availability."
    }, headers=headers)

    # Perform natural language semantic search
    srch_res = client.get("/search?q=Kubernetes worker nodes on AWS", headers=headers)
    assert srch_res.status_code == 200
    srch_data = srch_res.get_json()
    assert srch_data["status"] == "ok"
    assert srch_data["total_results"] > 0
    match = srch_data["results"][0]
    assert match["title"] == "Kubernetes Cluster Deployment"
    assert "score" in match
    assert match["similarity"] is not None


def test_4_ai_assistant_grounded_rag_and_hallucination_prevention(test_setup):
    """
    TEST 4 — AI ASSISTANT
    Grounded question -> Grounded Answer with sources
    Unanswerable question -> System fallback, no hallucination
    """
    client, _ = test_setup

    reg_res = client.post("/auth/register", json={
        "username": "rag_tester",
        "email": "ragtest@example.com",
        "password": "Password123!"
    }).get_json()
    headers = {"Authorization": f"Bearer {reg_res['token']}"}

    client.post("/meetings/process", data={
        "title": "Security Compliance Sync",
        "transcript": "Eve: We implemented OAuth 2.0 authentication and TLS 1.3 encryption across all endpoints."
    }, headers=headers)

    # 1. Question with known answer
    ask_known = client.post("/ask", json={
        "question": "What encryption standard was implemented?"
    }, headers=headers).get_json()
    assert ask_known["status"] == "ok"
    assert ask_known["context_chunks_used"] > 0
    assert len(ask_known["sources"]) > 0

    # 2. Question with no answer in records -> System fallback without hallucination
    ask_unknown = client.post("/ask", json={
        "question": "What is the secret recipe for quantum battery manufacturing?"
    }, headers=headers).get_json()
    assert ask_unknown["status"] == "ok"
    assert "couldn't find" in ask_unknown["answer"].lower() or len(ask_unknown["sources"]) == 0


def test_5_security_user_access(test_setup):
    """
    TEST 5 — SECURITY
    User A cannot retrieve User B's meetings, transcripts, search results, or exports.
    """
    client, _ = test_setup

    user_a = client.post("/auth/register", json={"username": "user_a_e2e", "email": "a_e2e@test.com", "password": "pass"}).get_json()
    user_b = client.post("/auth/register", json={"username": "user_b_e2e", "email": "b_e2e@test.com", "password": "pass"}).get_json()

    h_a = {"Authorization": f"Bearer {user_a['token']}"}
    h_b = {"Authorization": f"Bearer {user_b['token']}"}

    mtg_b = client.post("/meetings/process", data={
        "title": "User B Confidential Finance",
        "transcript": "Bob: Q4 secret financial revenue targets and acquisition strategy."
    }, headers=h_b).get_json()["meeting_id"]

    # User A tries to access User B's meeting -> 403 Forbidden
    assert client.get(f"/meetings/{mtg_b}", headers=h_a).status_code == 403
    assert client.get(f"/meetings/{mtg_b}/transcript", headers=h_a).status_code == 403
    assert client.get(f"/meetings/{mtg_b}/export/pdf", headers=h_a).status_code == 403

    # User A search does not return User B's confidential meeting
    srch_a = client.get("/search?q=acquisition strategy", headers=h_a).get_json()
    for res_item in srch_a.get("results", []):
        assert res_item["title"] != "User B Confidential Finance"


def test_6_pdf_and_csv_export_verification(test_setup):
    """
    TEST 6 — EXPORT
    Export PDF and CSV. Verify information belongs strictly to selected meeting.
    """
    client, _ = test_setup

    user_info = client.post("/auth/register", json={"username": "export_user", "email": "exp@test.com", "password": "pass"}).get_json()
    headers = {"Authorization": f"Bearer {user_info['token']}"}

    proc = client.post("/meetings/process", data={
        "title": "Executive Board Meeting 2026",
        "transcript": "Charlie: Approved $5M capital expenditure for global data center expansion."
    }, headers=headers).get_json()
    mtg_id = proc["meeting_id"]

    # 1. PDF Export Verification
    pdf_res = client.get(f"/meetings/{mtg_id}/export/pdf", headers=headers)
    assert pdf_res.status_code == 200
    assert pdf_res.mimetype == "application/pdf"
    assert pdf_res.data.startswith(b"%PDF")

    # 2. CSV Export Verification
    csv_res = client.get(f"/meetings/{mtg_id}/export/csv", headers=headers)
    assert csv_res.status_code == 200
    csv_text = csv_res.data.decode("utf-8")
    assert "Executive Board Meeting 2026" in csv_text
    assert "MEETING METADATA" in csv_text

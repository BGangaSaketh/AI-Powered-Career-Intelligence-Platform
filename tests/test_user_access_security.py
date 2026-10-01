"""
tests/test_user_access_security.py
===================================
Security & User Isolation Validation Test Suite for MILESTONE 4 - TASK 7.

Validates:
1. Registration & Login
2. Authentication & Session Management
3. User Isolation across meetings, transcripts, summaries, action items, participants, decisions, deadlines
4. Report Export Security (PDF & CSV access control)
5. RAG & Semantic Search Vector Store Ownership Isolation
6. Unauthenticated Access Rejection
"""

import os
import pytest
from app import app
from modules.database import init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client fixture with isolated SQLite database and enforced auth."""
    db_file = str(tmp_path / "test_security.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_KEY", "test_system_key_999")

    init_db(db_file)
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_user_registration_and_login(client):
    """Test user registration, duplicate prevention, and login flow."""
    # 1. Register User A
    res_reg_a = client.post("/auth/register", json={
        "username": "user_a",
        "email": "usera@example.com",
        "password": "Password123!"
    })
    assert res_reg_a.status_code == 201
    data_reg_a = res_reg_a.get_json()
    assert data_reg_a["status"] == "ok"
    assert "token" in data_reg_a
    token_a = data_reg_a["token"]

    # 2. Duplicate registration attempt
    res_dup = client.post("/auth/register", json={
        "username": "user_a",
        "email": "usera2@example.com",
        "password": "Password123!"
    })
    assert res_dup.status_code == 400

    # 3. Login User A with correct credentials
    res_login_a = client.post("/auth/login", json={
        "username": "user_a",
        "password": "Password123!"
    })
    assert res_login_a.status_code == 200
    assert res_login_a.get_json()["token"] == token_a

    # 4. Login with invalid password
    res_bad_login = client.post("/auth/login", json={
        "username": "user_a",
        "password": "WrongPassword"
    })
    assert res_bad_login.status_code == 401


def test_session_management_and_logout(client):
    """Test token authentication, session profile retrieval, and logout invalidation."""
    # Register user
    reg_res = client.post("/auth/register", json={
        "username": "session_user",
        "email": "session@example.com",
        "password": "SecretPassword123"
    })
    token = reg_res.get_json()["token"]

    headers = {"Authorization": f"Bearer {token}"}

    # Verify session profile
    me_res = client.get("/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.get_json()["user"]["username"] == "session_user"

    # Logout
    logout_res = client.post("/auth/logout", headers=headers)
    assert logout_res.status_code == 200

    # Verify token is invalidated
    me_after = client.get("/auth/me", headers=headers)
    assert me_after.status_code == 401


def test_user_meeting_isolation_and_access_control(client):
    """
    Test strict user isolation:
    User A cannot retrieve User B's meetings, transcripts, summaries, action items,
    participants, decisions, or deadlines.
    """
    # 1. Register User A and User B
    reg_a = client.post("/auth/register", json={"username": "alice", "email": "alice@test.com", "password": "passA"}).get_json()
    reg_b = client.post("/auth/register", json={"username": "bob", "email": "bob@test.com", "password": "passB"}).get_json()

    headers_a = {"Authorization": f"Bearer {reg_a['token']}"}
    headers_b = {"Authorization": f"Bearer {reg_b['token']}"}

    # 2. User A creates Meeting A
    tx_a = "Alice: Let's discuss the Q4 marketing campaign strategy and budget allocation."
    proc_a = client.post("/meetings/process", data={
        "title": "Alice Q4 Strategy Sync",
        "transcript": tx_a
    }, headers=headers_a).get_json()
    meeting_a_id = proc_a["meeting_id"]

    # 3. User B creates Meeting B
    tx_b = "Bob: Confidential database migration planning meeting for engineering team."
    proc_b = client.post("/meetings/process", data={
        "title": "Bob Confidential DB Migration",
        "transcript": tx_b
    }, headers=headers_b).get_json()
    meeting_b_id = proc_b["meeting_id"]

    # 4. Verify User A list_meetings contains ONLY Meeting A
    list_a = client.get("/meetings", headers=headers_a).get_json()
    meeting_ids_a = [m["id"] for m in list_a["meetings"]]
    assert meeting_a_id in meeting_ids_a
    assert meeting_b_id not in meeting_ids_a

    # 5. Verify User B list_meetings contains ONLY Meeting B
    list_b = client.get("/meetings", headers=headers_b).get_json()
    meeting_ids_b = [m["id"] for m in list_b["meetings"]]
    assert meeting_b_id in meeting_ids_b
    assert meeting_a_id not in meeting_ids_b

    # 6. User A attempting to access User B's endpoints -> 403 Forbidden
    endpoints_to_test = [
        f"/meetings/{meeting_b_id}",
        f"/meetings/{meeting_b_id}/transcript",
        f"/meetings/{meeting_b_id}/summary",
        f"/meetings/{meeting_b_id}/decisions",
        f"/meetings/{meeting_b_id}/action-items",
        f"/meetings/{meeting_b_id}/participants",
        f"/meetings/{meeting_b_id}/deadlines"
    ]

    for ep in endpoints_to_test:
        res = client.get(ep, headers=headers_a)
        assert res.status_code == 403, f"Endpoint '{ep}' should reject User A with 403"
        assert res.get_json()["status"] == "error"


def test_report_export_security(client):
    """Verify PDF/CSV report exports cannot be generated for another user's meeting."""
    # Register User A & User B
    reg_a = client.post("/auth/register", json={"username": "pdf_user_a", "email": "pdf_a@test.com", "password": "pass"}).get_json()
    reg_b = client.post("/auth/register", json={"username": "pdf_user_b", "email": "pdf_b@test.com", "password": "pass"}).get_json()

    headers_a = {"Authorization": f"Bearer {reg_a['token']}"}
    headers_b = {"Authorization": f"Bearer {reg_b['token']}"}

    # User B processes meeting
    proc_b = client.post("/meetings/process", data={
        "title": "Bob Private Report Meeting",
        "transcript": "Bob: Strategic partnership terms and revenue sharing agreement."
    }, headers=headers_b).get_json()
    mtg_b_id = proc_b["meeting_id"]

    # User A tries to export User B's PDF & CSV reports -> 403 Forbidden
    pdf_res_unauth = client.get(f"/meetings/{mtg_b_id}/export/pdf", headers=headers_a)
    assert pdf_res_unauth.status_code == 403

    csv_res_unauth = client.get(f"/meetings/{mtg_b_id}/export/csv", headers=headers_a)
    assert csv_res_unauth.status_code == 403

    # User B exports own PDF & CSV reports -> 200 OK
    pdf_res_auth = client.get(f"/meetings/{mtg_b_id}/export/pdf", headers=headers_b)
    assert pdf_res_auth.status_code == 200
    assert pdf_res_auth.mimetype == "application/pdf"

    csv_res_auth = client.get(f"/meetings/{mtg_b_id}/export/csv", headers=headers_b)
    assert csv_res_auth.status_code == 200
    assert csv_res_auth.mimetype == "text/csv"


def test_rag_and_semantic_search_user_isolation(client):
    """Verify semantic search and RAG Q&A respect user ownership metadata."""
    reg_a = client.post("/auth/register", json={"username": "rag_user_a", "email": "rag_a@test.com", "password": "pass"}).get_json()
    reg_b = client.post("/auth/register", json={"username": "rag_user_b", "email": "rag_b@test.com", "password": "pass"}).get_json()

    headers_a = {"Authorization": f"Bearer {reg_a['token']}"}
    headers_b = {"Authorization": f"Bearer {reg_b['token']}"}

    # User A meeting about Project Alpha
    client.post("/meetings/process", data={
        "title": "Project Alpha Tech Sync",
        "transcript": "Alice: Project Alpha will use PostgreSQL for core data storage and Redis for caching."
    }, headers=headers_a)

    # User B meeting about Project Beta
    client.post("/meetings/process", data={
        "title": "Project Beta Infrastructure",
        "transcript": "Bob: Project Beta will use MongoDB for document storage and Kafka for streaming."
    }, headers=headers_b)

    # 1. Search isolation: User A searching for User B's term gets no User B results
    search_a = client.get("/search?q=MongoDB", headers=headers_a).get_json()
    # Verify User B's meeting is completely absent from User A's search results
    for res_item in search_a.get("results", []):
        assert res_item["title"] != "Project Beta Infrastructure"
        assert "MongoDB" not in res_item["relevant_snippet"]

    # User A searching with min_score threshold gets 0 results
    search_a_filtered = client.get("/search?q=MongoDB&min_score=0.1", headers=headers_a).get_json()
    assert search_a_filtered["total_results"] == 0

    # User B searching for MongoDB gets Project Beta
    search_b = client.get("/search?q=MongoDB", headers=headers_b).get_json()
    assert search_b["total_results"] > 0
    assert "MongoDB" in search_b["results"][0]["text"]
    assert search_b["results"][0]["title"] == "Project Beta Infrastructure"

    # 2. RAG isolation: User A asking RAG does NOT receive User B's MongoDB or Project Beta context
    ask_a = client.post("/ask", json={"q": "What database does Project Beta use?"}, headers=headers_a).get_json()
    assert "MongoDB" not in ask_a.get("answer", "")
    assert all(src.get("title") != "Project Beta Infrastructure" for src in ask_a.get("sources", []))


def test_unauthenticated_api_rejection(client):
    """Verify all protected endpoints reject unauthenticated requests with 401 Unauthorized."""
    unauth_endpoints = [
        ("GET", "/meetings"),
        ("GET", "/meetings/mtg_fake123"),
        ("POST", "/meetings/process"),
        ("GET", "/search?q=test"),
        ("POST", "/ask"),
        ("GET", "/meetings/mtg_fake123/export/pdf"),
        ("GET", "/meetings/mtg_fake123/export/csv")
    ]

    for method, path in unauth_endpoints:
        if method == "GET":
            res = client.get(path)
        else:
            res = client.post(path, json={"q": "test"})
        assert res.status_code == 401, f"Unauthenticated request to '{path}' should return 401"

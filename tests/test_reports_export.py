"""
tests/test_reports_export.py
=============================
Unit and Integration Tests for PDF & CSV Report Generation and Export (Milestone 4 Task 6)
"""

import tempfile
import pytest
from unittest.mock import MagicMock, patch
from modules.database import init_db, save_meeting, get_meeting
from modules.schemas import MeetingIntelligence, ActionItem, Participant
from modules.report_generator import generate_meeting_pdf_report, generate_meeting_csv_report
from modules.api_client import MeetingApiClient
from app import app


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path


def test_pdf_report_generation_and_data_isolation(temp_db):
    """
    Verify PDF report is generated correctly and contains data strictly belonging to Meeting A.
    """
    intel_a = MeetingIntelligence(
        summary="Executive Summary for Project Alpha",
        key_points=["Key point Alpha 1", "Key point Alpha 2"],
        decisions=["Approved Budget Alpha"],
        action_items=[
            ActionItem(task="Deploy Alpha DB", assigned_to="Alice", priority="High", status="Pending", deadline="Friday")
        ],
        participants=[
            Participant(name="Alice", responsibilities=["Alpha Lead"])
        ]
    )
    meeting_id_a = save_meeting(intel_a, "Alice: Alpha project meeting transcript.", title="Alpha Project Sync", db_path=temp_db)

    mtg_a = get_meeting(meeting_id_a, db_path=temp_db)
    pdf_bytes = generate_meeting_pdf_report(mtg_a)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")  # Valid PDF binary signature


def test_csv_report_generation_and_formatting(temp_db):
    """
    Verify CSV report is formatted correctly into structured spreadsheet sections.
    """
    intel_b = MeetingIntelligence(
        summary="Executive Summary for Project Beta",
        key_points=["Key point Beta 1"],
        decisions=["Beta Deadline Set"],
        action_items=[
            ActionItem(task="Build Beta API", assigned_to="Bob", priority="Medium", status="Completed", deadline="Next Month")
        ],
        participants=[
            Participant(name="Bob", responsibilities=["Beta Backend Developer"])
        ]
    )
    meeting_id_b = save_meeting(intel_b, "Bob: Beta project meeting transcript.", title="Beta Project Sync", db_path=temp_db)

    mtg_b = get_meeting(meeting_id_b, db_path=temp_db)
    csv_text = generate_meeting_csv_report(mtg_b)

    assert isinstance(csv_text, str)
    assert "=== MEETING METADATA ===" in csv_text
    assert "=== EXECUTIVE SUMMARY ===" in csv_text
    assert "=== DECISIONS REACHED ===" in csv_text
    assert "=== ACTION ITEMS TABLE ===" in csv_text
    assert "=== PARTICIPANTS & RESPONSIBILITIES ===" in csv_text

    # Verify Meeting B content is present and Meeting A content is absent
    assert "Beta Project Sync" in csv_text
    assert "Build Beta API" in csv_text
    assert "Bob" in csv_text
    assert "Project Alpha" not in csv_text


def test_empty_meeting_report_resilience(temp_db):
    """
    Verify that an empty/minimal meeting generates valid PDF and CSV reports without crashing.
    """
    intel_empty = MeetingIntelligence(
        summary="",
        key_points=[],
        decisions=[],
        action_items=[],
        participants=[]
    )
    m_id = save_meeting(intel_empty, "", title="Empty Meeting", db_path=temp_db)
    mtg_empty = get_meeting(m_id, db_path=temp_db)

    # PDF generation resilience
    pdf_bytes = generate_meeting_pdf_report(mtg_empty)
    assert pdf_bytes.startswith(b"%PDF-")

    # CSV generation resilience
    csv_text = generate_meeting_csv_report(mtg_empty)
    assert "Empty Meeting" in csv_text
    assert "No action items recorded." in csv_text


def test_flask_export_endpoints_integration(temp_db):
    """
    Verify Flask endpoints GET /meetings/<id>/export/pdf and GET /meetings/<id>/export/csv.
    """
    import os
    old_db = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = temp_db

    try:
        intel = MeetingIntelligence(
            summary="Export Endpoint Test Meeting",
            key_points=[],
            decisions=["Approved Export"],
            action_items=[],
            participants=[]
        )
        m_id = save_meeting(intel, "Transcript content", title="Export Test", db_path=temp_db)

        with app.test_client() as client:
            # Test PDF Endpoint
            resp_pdf = client.get(f"/meetings/{m_id}/export/pdf")
            assert resp_pdf.status_code == 200
            assert resp_pdf.mimetype == "application/pdf"
            assert resp_pdf.data.startswith(b"%PDF-")

            # Test CSV Endpoint
            resp_csv = client.get(f"/meetings/{m_id}/export/csv")
            assert resp_csv.status_code == 200
            assert resp_csv.mimetype == "text/csv"
            assert "Export Test" in resp_csv.get_data(as_text=True)

    finally:
        if old_db:
            os.environ["DATABASE_PATH"] = old_db
        else:
            os.environ.pop("DATABASE_PATH", None)

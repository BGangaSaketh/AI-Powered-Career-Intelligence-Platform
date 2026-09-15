"""
modules/database.py
===================
SQLite Persistent Storage Layer for Meeting Intelligence Platform

Models / Tables:
  - meetings (id, title, summary, created_at, status)
  - transcripts (id, meeting_id, raw_text, word_count)
  - participants (id, meeting_id, name, responsibilities_json)
  - action_items (id, meeting_id, task, assigned_to, deadline, priority, status)
  - decisions (id, meeting_id, decision_text)
  - key_points (id, meeting_id, point_text)
"""

import os
import sqlite3
import json
import uuid
import datetime
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from modules.schemas import MeetingIntelligence

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "career_intelligence.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory."""
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes if they do not exist."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT,
                    status TEXT NOT NULL DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS participants (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    responsibilities_json TEXT NOT NULL,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS action_items (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    assigned_to TEXT,
                    deadline TEXT,
                    priority TEXT DEFAULT 'Unknown',
                    status TEXT DEFAULT 'Pending',
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    decision_text TEXT NOT NULL,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS key_points (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    point_text TEXT NOT NULL,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
            """)
        logger.info(f"Database initialized at '{db_path or DEFAULT_DB_PATH}'.")
    finally:
        conn.close()


def save_meeting(
    intelligence: MeetingIntelligence,
    raw_transcript: str,
    title: str = "Meeting Recording",
    db_path: Optional[str] = None
) -> str:
    """
    Persist a validated MeetingIntelligence object and transcript into SQLite.

    Only validated data is stored.

    Parameters
    ----------
    intelligence : MeetingIntelligence
        Validated Pydantic object.
    raw_transcript : str
        Original raw transcript text.
    title : str
        Title for the meeting.
    db_path : str, optional
        Database file path.

    Returns
    -------
    str
        The assigned unique meeting_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    meeting_id = intelligence.meeting_id or f"mtg_{uuid.uuid4().hex[:12]}"
    intelligence.meeting_id = meeting_id

    word_count = len(raw_transcript.split()) if raw_transcript else 0
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        with conn:
            # 1. Insert Meeting
            conn.execute(
                "INSERT INTO meetings (id, title, summary, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (meeting_id, title, intelligence.summary, "completed", now_iso)
            )

            # 2. Insert Transcript
            conn.execute(
                "INSERT INTO transcripts (id, meeting_id, raw_text, word_count) VALUES (?, ?, ?, ?)",
                (f"tx_{uuid.uuid4().hex[:12]}", meeting_id, raw_transcript, word_count)
            )

            # 3. Insert Key Points
            for point in intelligence.key_points:
                conn.execute(
                    "INSERT INTO key_points (id, meeting_id, point_text) VALUES (?, ?, ?)",
                    (f"kp_{uuid.uuid4().hex[:12]}", meeting_id, point)
                )

            # 4. Insert Decisions
            for dec in intelligence.decisions:
                conn.execute(
                    "INSERT INTO decisions (id, meeting_id, decision_text) VALUES (?, ?, ?)",
                    (f"dec_{uuid.uuid4().hex[:12]}", meeting_id, dec)
                )

            # 5. Insert Action Items
            for item in intelligence.action_items:
                conn.execute(
                    """INSERT INTO action_items 
                       (id, meeting_id, task, assigned_to, deadline, priority, status) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"act_{uuid.uuid4().hex[:12]}",
                        meeting_id,
                        item.task,
                        item.assigned_to,
                        item.deadline,
                        item.priority,
                        item.status
                    )
                )

            # 6. Insert Participants
            for p in intelligence.participants:
                conn.execute(
                    "INSERT INTO participants (id, meeting_id, name, responsibilities_json) VALUES (?, ?, ?, ?)",
                    (
                        f"part_{uuid.uuid4().hex[:12]}",
                        meeting_id,
                        p.name,
                        json.dumps(p.responsibilities)
                    )
                )

        logger.info(f"Successfully saved meeting '{meeting_id}' to database.")
        return meeting_id
    except Exception as exc:
        logger.exception(f"Failed to save meeting '{meeting_id}' to database: {exc}")
        raise
    finally:
        conn.close()


def get_meeting(meeting_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve a processed meeting by ID with all associated tables.

    Returns
    -------
    Dict[str, Any] or None
        Full meeting details matching API schema.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    try:
        cur = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
        meeting_row = cur.fetchone()
        if not meeting_row:
            return None

        # Fetch Transcript
        cur = conn.execute("SELECT raw_text FROM transcripts WHERE meeting_id = ?", (meeting_id,))
        t_row = cur.fetchone()
        raw_transcript = t_row["raw_text"] if t_row else ""

        # Fetch Key Points
        cur = conn.execute("SELECT point_text FROM key_points WHERE meeting_id = ?", (meeting_id,))
        key_points = [row["point_text"] for row in cur.fetchall()]

        # Fetch Decisions
        cur = conn.execute("SELECT decision_text FROM decisions WHERE meeting_id = ?", (meeting_id,))
        decisions = [row["decision_text"] for row in cur.fetchall()]

        # Fetch Action Items
        cur = conn.execute("SELECT task, assigned_to, deadline, priority, status FROM action_items WHERE meeting_id = ?", (meeting_id,))
        action_items = [
            {
                "task": r["task"],
                "assigned_to": r["assigned_to"],
                "deadline": r["deadline"],
                "priority": r["priority"],
                "status": r["status"]
            }
            for r in cur.fetchall()
        ]

        # Fetch Participants
        cur = conn.execute("SELECT name, responsibilities_json FROM participants WHERE meeting_id = ?", (meeting_id,))
        participants = [
            {
                "name": r["name"],
                "responsibilities": json.loads(r["responsibilities_json"]) if r["responsibilities_json"] else []
            }
            for r in cur.fetchall()
        ]

        return {
            "meeting_id": meeting_row["id"],
            "title": meeting_row["title"],
            "summary": meeting_row["summary"],
            "status": meeting_row["status"],
            "created_at": meeting_row["created_at"],
            "raw_transcript": raw_transcript,
            "key_points": key_points,
            "decisions": decisions,
            "action_items": action_items,
            "participants": participants
        }
    finally:
        conn.close()


def list_meetings(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List summary details of all processed meetings ordered by creation date."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id, title, summary, created_at, status FROM meetings ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

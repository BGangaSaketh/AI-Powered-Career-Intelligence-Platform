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
import hashlib
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from modules.schemas import MeetingIntelligence

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "career_intelligence.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory."""
    path = db_path or os.getenv("DATABASE_PATH") or DEFAULT_DB_PATH
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
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    token TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT,
                    status TEXT NOT NULL DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    zoom_meeting_id TEXT,
                    google_meeting_id TEXT,
                    user_id TEXT
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

                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    meeting_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source_id TEXT,
                    chunk_index INTEGER DEFAULT 0,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
            """)
            # Safely alter tables to add columns if upgrading existing database
            for alter_cmd in [
                "ALTER TABLE meetings ADD COLUMN zoom_meeting_id TEXT;",
                "ALTER TABLE meetings ADD COLUMN google_meeting_id TEXT;",
                "ALTER TABLE meetings ADD COLUMN user_id TEXT;",
                "ALTER TABLE embeddings ADD COLUMN user_id TEXT;"
            ]:
                try:
                    conn.execute(alter_cmd)
                except Exception:
                    pass

            for idx_cmd in [
                "CREATE INDEX IF NOT EXISTS idx_meetings_zoom_id ON meetings(zoom_meeting_id);",
                "CREATE INDEX IF NOT EXISTS idx_meetings_google_id ON meetings(google_meeting_id);",
                "CREATE INDEX IF NOT EXISTS idx_meetings_user_id ON meetings(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON embeddings(user_id);"
            ]:
                try:
                    conn.execute(idx_cmd)
                except Exception:
                    pass

        logger.info(f"Database initialized at '{db_path or DEFAULT_DB_PATH}'.")
    finally:
        conn.close()


def hash_password(password: str) -> str:
    """Generate SHA-256 hash for a user password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(
    username: str,
    email: str,
    password: str,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new user record in SQLite.
    Raises ValueError if username or email already exists.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    username_clean = username.strip()
    email_clean = email.strip().lower()
    
    if not username_clean or not email_clean or not password:
        raise ValueError("Username, email, and password are required.")

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    token = f"tok_{uuid.uuid4().hex}"
    pwd_hash = hash_password(password)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    try:
        with conn:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, token, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, username_clean, email_clean, pwd_hash, token, now_iso)
            )
        return {
            "id": user_id,
            "username": username_clean,
            "email": email_clean,
            "token": token,
            "created_at": now_iso
        }
    except sqlite3.IntegrityError:
        raise ValueError("User with this username or email already exists.")
    finally:
        conn.close()


def authenticate_user(
    username_or_email: str,
    password: str,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by username/email and password.
    Returns user dict with active token if valid, else None.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    ident = username_or_email.strip().lower()
    pwd_hash = hash_password(password)

    try:
        cur = conn.execute(
            "SELECT * FROM users WHERE (LOWER(username) = ? OR LOWER(email) = ?) AND password_hash = ?",
            (ident, ident, pwd_hash)
        )
        row = cur.fetchone()
        if not row:
            return None

        # Ensure active token exists
        token = row["token"]
        if not token:
            token = f"tok_{uuid.uuid4().hex}"
            with conn:
                conn.execute("UPDATE users SET token = ? WHERE id = ?", (token, row["id"]))

        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "token": token,
            "created_at": row["created_at"]
        }
    finally:
        conn.close()


def get_user_by_token(token: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user record by auth token."""
    if not token or not token.strip():
        return None
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id, username, email, token, created_at FROM users WHERE token = ?", (token.strip(),))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user record by user ID."""
    if not user_id or not user_id.strip():
        return None
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id, username, email, token, created_at FROM users WHERE id = ?", (user_id.strip(),))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def invalidate_user_token(token: str, db_path: Optional[str] = None) -> bool:
    """Invalidate a user session token upon logout."""
    if not token or not token.strip():
        return False
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute("UPDATE users SET token = NULL WHERE token = ?", (token.strip(),))
            return cur.rowcount > 0
    finally:
        conn.close()


def save_meeting(
    intelligence: MeetingIntelligence,
    raw_transcript: str,
    title: str = "Meeting Recording",
    db_path: Optional[str] = None,
    zoom_meeting_id: Optional[str] = None,
    google_meeting_id: Optional[str] = None,
    user_id: Optional[str] = None
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
    zoom_meeting_id : str, optional
        Zoom recording/meeting ID to prevent duplicate imports.
    google_meeting_id : str, optional
        Google Meet recording/file ID to prevent duplicate imports.

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
                "INSERT INTO meetings (id, title, summary, status, created_at, zoom_meeting_id, google_meeting_id, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (meeting_id, title, intelligence.summary, "completed", now_iso, zoom_meeting_id, google_meeting_id, user_id)
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

        logger.info(f"Successfully saved meeting '{meeting_id}' for user '{user_id or 'anonymous'}' to database.")
        return meeting_id
    except Exception as exc:
        logger.exception(f"Failed to save meeting '{meeting_id}' to database: {exc}")
        raise
    finally:
        conn.close()


def get_meeting(meeting_id: str, db_path: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve a processed meeting by ID with all associated tables.
    If user_id is provided, verifies meeting ownership.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    try:
        cur = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
        meeting_row = cur.fetchone()
        if not meeting_row:
            return None

        # Ownership enforcement if user_id is passed and not admin/system
        m_user_id = meeting_row["user_id"] if "user_id" in meeting_row.keys() else None
        if user_id and user_id != "system" and m_user_id and m_user_id != user_id and m_user_id != "system":
            logger.warning(f"User '{user_id}' attempted to access meeting '{meeting_id}' belonging to '{m_user_id}'")
            return None

        # Fetch Transcript
        cur = conn.execute("SELECT raw_text FROM transcripts WHERE meeting_id = ?", (meeting_id,))
        t_row = cur.fetchone()
        raw_transcript = t_row["raw_text"] if t_row else ""
        word_count = len(raw_transcript.split()) if raw_transcript else 0

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

        # Fetch Participants and parse safely
        cur = conn.execute("SELECT id, name, responsibilities_json FROM participants WHERE meeting_id = ?", (meeting_id,))
        participants = []
        for r in cur.fetchall():
            res_list = []
            if r["responsibilities_json"]:
                try:
                    res_list = json.loads(r["responsibilities_json"])
                except Exception:
                    res_list = []
            participants.append({
                "id": r["id"],
                "name": r["name"],
                "responsibilities": res_list if isinstance(res_list, list) else []
            })

        # Extract deadlines from action items
        deadlines = [
            {
                "task": item["task"],
                "assigned_to": item["assigned_to"],
                "deadline": item["deadline"],
                "priority": item["priority"],
                "status": item["status"]
            }
            for item in action_items if item.get("deadline") and item["deadline"].strip()
        ]

        m_user_id = meeting_row["user_id"] if "user_id" in meeting_row.keys() else None

        metadata = {
            "id": meeting_row["id"],
            "title": meeting_row["title"],
            "summary": meeting_row["summary"] or "",
            "status": meeting_row["status"],
            "created_at": meeting_row["created_at"],
            "word_count": word_count,
            "user_id": m_user_id
        }

        return {
            "id": meeting_row["id"],
            "meeting_id": meeting_row["id"],
            "user_id": m_user_id,
            "title": meeting_row["title"],
            "summary": meeting_row["summary"] or "",
            "status": meeting_row["status"],
            "created_at": meeting_row["created_at"],
            "metadata": metadata,
            "raw_transcript": raw_transcript,
            "transcript": raw_transcript,
            "key_points": key_points,
            "decisions": decisions,
            "action_items": action_items,
            "participants": participants,
            "deadlines": deadlines
        }
    finally:
        conn.close()


def get_meeting_by_zoom_id(zoom_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve meeting data by Zoom meeting/recording ID if already imported.
    Used for duplicate import prevention.
    """
    if not zoom_id or not str(zoom_id).strip():
        return None
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id FROM meetings WHERE zoom_meeting_id = ?", (str(zoom_id).strip(),))
        row = cur.fetchone()
        if not row:
            return None
        return get_meeting(row["id"], db_path=db_path)
    finally:
        conn.close()


def get_meeting_by_google_id(google_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve meeting data by Google Meet recording/file ID if already imported.
    Used for duplicate import prevention.
    """
    if not google_id or not str(google_id).strip():
        return None
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id FROM meetings WHERE google_meeting_id = ?", (str(google_id).strip(),))
        row = cur.fetchone()
        if not row:
            return None
        return get_meeting(row["id"], db_path=db_path)
    finally:
        conn.close()



def get_meeting_metadata(meeting_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:

    """Retrieve metadata for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id, title, summary, status, created_at FROM meetings WHERE id = ?", (meeting_id,))
        row = cur.fetchone()
        if not row:
            return None
        cur = conn.execute("SELECT word_count FROM transcripts WHERE meeting_id = ?", (meeting_id,))
        tx_row = cur.fetchone()
        word_count = tx_row["word_count"] if tx_row else 0
        return {
            "id": row["id"],
            "meeting_id": row["id"],
            "title": row["title"],
            "summary": row["summary"] or "",
            "status": row["status"],
            "created_at": row["created_at"],
            "word_count": word_count
        }
    finally:
        conn.close()


def get_meeting_transcript(meeting_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve raw transcript and word count for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,))
        if not cur.fetchone():
            return None
        cur = conn.execute("SELECT id, raw_text, word_count FROM transcripts WHERE meeting_id = ?", (meeting_id,))
        tx_row = cur.fetchone()
        if not tx_row:
            return {"meeting_id": meeting_id, "raw_text": "", "transcript": "", "word_count": 0}
        return {
            "meeting_id": meeting_id,
            "id": tx_row["id"],
            "raw_text": tx_row["raw_text"],
            "transcript": tx_row["raw_text"],
            "word_count": tx_row["word_count"]
        }
    finally:
        conn.close()


def get_meeting_summary(meeting_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve summary for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id, title, summary FROM meetings WHERE id = ?", (meeting_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "meeting_id": row["id"],
            "title": row["title"],
            "summary": row["summary"] or ""
        }
    finally:
        conn.close()


def get_meeting_decisions(meeting_id: str, db_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Retrieve all decisions recorded for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,))
        if not cur.fetchone():
            return None
        cur = conn.execute("SELECT id, decision_text FROM decisions WHERE meeting_id = ?", (meeting_id,))
        return [{"id": r["id"], "decision_text": r["decision_text"]} for r in cur.fetchall()]
    finally:
        conn.close()


def get_meeting_action_items(meeting_id: str, db_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Retrieve all action items recorded for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,))
        if not cur.fetchone():
            return None
        cur = conn.execute(
            "SELECT id, task, assigned_to, deadline, priority, status FROM action_items WHERE meeting_id = ?",
            (meeting_id,)
        )
        return [
            {
                "id": r["id"],
                "task": r["task"],
                "assigned_to": r["assigned_to"],
                "deadline": r["deadline"],
                "priority": r["priority"],
                "status": r["status"]
            }
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


def get_meeting_participants(meeting_id: str, db_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Retrieve all participants recorded for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,))
        if not cur.fetchone():
            return None
        cur = conn.execute("SELECT id, name, responsibilities_json FROM participants WHERE meeting_id = ?", (meeting_id,))
        participants = []
        for r in cur.fetchall():
            res_list = []
            if r["responsibilities_json"]:
                try:
                    res_list = json.loads(r["responsibilities_json"])
                except Exception:
                    res_list = []
            participants.append({
                "id": r["id"],
                "name": r["name"],
                "responsibilities": res_list if isinstance(res_list, list) else []
            })
        return participants
    finally:
        conn.close()


def get_meeting_deadlines(meeting_id: str, db_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Retrieve all deadlines associated with a specific meeting."""
    items = get_meeting_action_items(meeting_id, db_path)
    if items is None:
        return None
    return [
        {
            "id": item["id"],
            "task": item["task"],
            "assigned_to": item["assigned_to"],
            "deadline": item["deadline"],
            "priority": item["priority"],
            "status": item["status"]
        }
        for item in items if item.get("deadline") and str(item["deadline"]).strip()
    ]


def get_complete_meeting(meeting_id: str, db_path: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Alias for get_meeting returning complete hierarchical meeting knowledge."""
    return get_meeting(meeting_id, db_path=db_path, user_id=user_id)


def get_all_meetings_knowledge(db_path: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve complete historical knowledge for all meetings accessible to user_id."""
    summaries = list_meetings(db_path=db_path, user_id=user_id)
    result = []
    for s in summaries:
        mtg = get_meeting(s["id"], db_path=db_path, user_id=user_id)
        if mtg:
            result.append(mtg)
    return result


def list_meetings(db_path: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List summary details of processed meetings ordered by creation date."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        if user_id and user_id != "system":
            cur = conn.execute(
                "SELECT id, title, summary, created_at, status, user_id FROM meetings WHERE user_id = ? OR user_id = 'system' OR user_id IS NULL ORDER BY created_at DESC",
                (user_id,)
            )
        else:
            cur = conn.execute("SELECT id, title, summary, created_at, status, user_id FROM meetings ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def save_embeddings(
    meeting_id: str,
    items: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    user_id: Optional[str] = None
) -> int:
    """
    Persist generated embedding records to SQLite with optional user_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    
    # If user_id is not passed directly, look up owner user_id from meeting record
    if not user_id:
        try:
            m_cur = conn.execute("SELECT user_id FROM meetings WHERE id = ?", (meeting_id,))
            m_row = m_cur.fetchone()
            if m_row and m_row["user_id"]:
                user_id = m_row["user_id"]
        except Exception:
            pass

    inserted_count = 0
    try:
        with conn:
            for item in items:
                text_content = item.get("text", "").strip()
                if not text_content:
                    continue  # Do not store embeddings for empty content

                emb_vec = item.get("embedding", [])
                if not emb_vec:
                    continue

                emb_id = item.get("id") or f"emb_{uuid.uuid4().hex[:12]}"
                content_type = item.get("content_type", "unknown")
                source_id = item.get("source_id")
                chunk_index = item.get("chunk_index", 0)
                dim = len(emb_vec)
                emb_json = json.dumps(emb_vec)
                u_id = item.get("user_id") or user_id

                conn.execute(
                    """INSERT INTO embeddings
                       (id, meeting_id, content_type, source_id, chunk_index, text, embedding_json, dimension, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (emb_id, meeting_id, content_type, source_id, chunk_index, text_content, emb_json, dim, u_id)
                )
                inserted_count += 1
        return inserted_count
    finally:
        conn.close()


def get_meeting_embeddings(
    meeting_id: str,
    content_type: Optional[str] = None,
    db_path: Optional[str] = None,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve embeddings for a specific meeting, optionally filtered by content_type and user_id."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        if content_type:
            cur = conn.execute(
                "SELECT * FROM embeddings WHERE meeting_id = ? AND content_type = ? ORDER BY chunk_index ASC",
                (meeting_id, content_type)
            )
        else:
            cur = conn.execute(
                "SELECT * FROM embeddings WHERE meeting_id = ? ORDER BY content_type ASC, chunk_index ASC",
                (meeting_id,)
            )

        results = []
        for r in cur.fetchall():
            rec_user = r["user_id"] if "user_id" in r.keys() else None
            if user_id and user_id != "system" and rec_user and rec_user != user_id:
                continue
            results.append({
                "id": r["id"],
                "meeting_id": r["meeting_id"],
                "content_type": r["content_type"],
                "source_id": r["source_id"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "embedding": json.loads(r["embedding_json"]),
                "dimension": r["dimension"],
                "created_at": r["created_at"],
                "user_id": rec_user
            })
        return results
    finally:
        conn.close()


def get_all_embeddings(db_path: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all embeddings, filtered by user_id if supplied."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        if user_id and user_id != "system":
            cur = conn.execute("SELECT * FROM embeddings WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        else:
            cur = conn.execute("SELECT * FROM embeddings ORDER BY created_at DESC")
        results = []
        for r in cur.fetchall():
            rec_user = r["user_id"] if "user_id" in r.keys() else None
            results.append({
                "id": r["id"],
                "meeting_id": r["meeting_id"],
                "content_type": r["content_type"],
                "source_id": r["source_id"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "embedding": json.loads(r["embedding_json"]),
                "dimension": r["dimension"],
                "created_at": r["created_at"],
                "user_id": rec_user
            })
        return results
    finally:
        conn.close()


def delete_meeting_embeddings(meeting_id: str, db_path: Optional[str] = None) -> int:
    """Delete all embeddings for a specific meeting."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute("DELETE FROM embeddings WHERE meeting_id = ?", (meeting_id,))
            return cur.rowcount
    finally:
        conn.close()


def get_embedding_by_id(vector_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single embedding record by vector ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM embeddings WHERE id = ?", (vector_id,))
        r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "meeting_id": r["meeting_id"],
            "content_type": r["content_type"],
            "source_id": r["source_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "embedding": json.loads(r["embedding_json"]),
            "dimension": r["dimension"],
            "created_at": r["created_at"]
        }
    finally:
        conn.close()


def update_embedding_by_id(
    vector_id: str,
    text: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    db_path: Optional[str] = None
) -> bool:
    """Update text and/or vector for a specific embedding record."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute("SELECT text, embedding_json, dimension FROM embeddings WHERE id = ?", (vector_id,))
        existing = cur.fetchone()
        if not existing:
            return False

        new_text = text.strip() if (text is not None and text.strip()) else existing["text"]
        new_emb = embedding if (embedding is not None and len(embedding) > 0) else json.loads(existing["embedding_json"])
        new_dim = len(new_emb)
        new_emb_json = json.dumps(new_emb)

        with conn:
            conn.execute(
                "UPDATE embeddings SET text = ?, embedding_json = ?, dimension = ? WHERE id = ?",
                (new_text, new_emb_json, new_dim, vector_id)
            )
        return True
    finally:
        conn.close()


def delete_embedding_by_id(vector_id: str, db_path: Optional[str] = None) -> bool:
    """Delete a single embedding record by vector ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute("DELETE FROM embeddings WHERE id = ?", (vector_id,))
            return cur.rowcount > 0
    finally:
        conn.close()


def delete_meeting(meeting_id: str, db_path: Optional[str] = None) -> bool:
    """
    Delete a meeting and all associated foreign-key records (transcripts, decisions,
    action items, participants, key points, embeddings) via ON DELETE CASCADE.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            return cur.rowcount > 0
    finally:
        conn.close()

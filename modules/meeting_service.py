"""
modules/meeting_service.py
===========================
Integrated Meeting Intelligence Pipeline Service

End-to-End Flow:
  Meeting Recording / Audio / Transcript
          ↓
  Speech / Whisper Transcription (via modules.transcription)
          ↓
  Input Validation
          ↓
  Long Transcript Estimation & Chunking (via modules.long_transcript)
          ↓
  LLM Service Processing (via modules.llm_service & prompts)
          ↓
  Structured Pydantic JSON Validation (via modules.schemas)
          ↓
  Database Persistence (via modules.database)
          ↓
  Structured Output for Meeting Intelligence Dashboard
"""

import logging
from typing import Dict, Any, Optional

from modules.transcription import transcribe_file
from modules.llm_service import LLMService, LLMServiceError
from modules.long_transcript import process_long_transcript
from modules.database import save_meeting, get_meeting

logger = logging.getLogger(__name__)


def process_meeting_input(
    file_path: Optional[str] = None,
    raw_transcript_input: Optional[str] = None,
    title: str = "Meeting Recording",
    llm_service: Optional[LLMService] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute the complete end-to-end Meeting Intelligence pipeline.

    Parameters
    ----------
    file_path : str, optional
        Path to uploaded audio or video recording file.
    raw_transcript_input : str, optional
        Raw transcript text pasted directly.
    title : str
        Title for the meeting.
    llm_service : LLMService, optional
        LLM service instance.
    db_path : str, optional
        Database file path.

    Returns
    -------
    Dict[str, Any]
        Dictionary matching the required POST /meetings/process JSON response structure.
    """
    transcript = ""
    file_type = "text"

    # 1. Obtain transcript
    if file_path:
        logger.info(f"Transcribing audio/video file '{file_path}'...")
        transcription_res = transcribe_file(file_path)
        if transcription_res.get("status") == "error":
            err_msg = " | ".join(transcription_res.get("errors", ["Transcription failed."]))
            raise ValueError(f"Transcription Error: {err_msg}")
        transcript = transcription_res.get("transcript", "")
        file_type = transcription_res.get("file_type", "media")
    elif raw_transcript_input and raw_transcript_input.strip():
        transcript = raw_transcript_input.strip()
    else:
        raise ValueError("Neither a valid media file nor a transcript was provided.")

    if not transcript or not transcript.strip():
        raise ValueError("The meeting transcript is empty or could not be recognized.")

    # 2. LLM Processing & Schema Validation with Long Transcript Chunking
    service = llm_service or LLMService()
    logger.info("Extracting Meeting Intelligence via LLM service...")
    intelligence = process_long_transcript(
        transcript=transcript,
        llm_service=service,
        meeting_context=title
    )

    # 3. Database Persistence (Only validated data persisted)
    logger.info("Persisting validated Meeting Intelligence to database...")
    meeting_id = save_meeting(
        intelligence=intelligence,
        raw_transcript=transcript,
        title=title,
        db_path=db_path
    )

    # 4. Construct Final Response Payload
    return {
        "status": "ok",
        "meeting_id": meeting_id,
        "title": title,
        "summary": intelligence.summary,
        "key_points": intelligence.key_points,
        "decisions": intelligence.decisions,
        "action_items": [item.model_dump() for item in intelligence.action_items],
        "participants": [p.model_dump() for p in intelligence.participants],
        "raw_transcript": transcript,
        "file_type": file_type
    }

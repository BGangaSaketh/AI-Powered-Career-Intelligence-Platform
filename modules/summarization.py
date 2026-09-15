"""
modules/summarization.py
========================
Dedicated Meeting Summarization Module

Input:
  - Meeting transcript text (or pre-extracted MeetingIntelligence)

Output:
  - Executive Summary
  - Key Points (List[str])
  - Decisions (List[str])
"""

from typing import Dict, Any, List, Optional
from modules.schemas import MeetingIntelligence
from modules.llm_service import LLMService
from modules.long_transcript import process_long_transcript


def summarize_meeting(
    transcript: str,
    llm_service: Optional[LLMService] = None,
    meeting_context: str = ""
) -> Dict[str, Any]:
    """
    Summarize a meeting transcript and extract key discussion points and formal decisions.

    Parameters
    ----------
    transcript : str
        The raw meeting transcript text.
    llm_service : LLMService, optional
        LLM Service instance. Creates a default instance if None.
    meeting_context : str, optional
        Optional contextual information.

    Returns
    -------
    Dict[str, Any]
        {
            "summary": str,
            "key_points": list[str],
            "decisions": list[str]
        }
    """
    if not transcript or not transcript.strip():
        return {
            "summary": "Empty transcript provided.",
            "key_points": [],
            "decisions": []
        }

    service = llm_service or LLMService()
    intelligence: MeetingIntelligence = process_long_transcript(
        transcript, service, meeting_context=meeting_context
    )

    return {
        "summary": intelligence.summary,
        "key_points": intelligence.key_points,
        "decisions": intelligence.decisions
    }

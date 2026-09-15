"""
modules/action_extraction.py
=============================
Dedicated Action Item Extraction Engine

Extracts:
  - Task description
  - Assigned participant (or null)
  - Deadline (or null)
  - Priority (High | Medium | Low | Unknown)
  - Status (Pending | In Progress | Completed | Unknown)

Rules:
  1. Never invent an assignee.
  2. Never invent a deadline.
  3. Never invent a priority.
  4. Never invent a status.
  5. Use null or Unknown when information is unavailable.
  6. Support multiple action items.
  7. Associate action items with the meeting.
"""

from typing import List, Dict, Any, Optional
from modules.schemas import ActionItem, MeetingIntelligence
from modules.llm_service import LLMService
from modules.long_transcript import process_long_transcript


def extract_action_items(
    transcript: str,
    llm_service: Optional[LLMService] = None,
    meeting_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract action items from a meeting transcript.

    Parameters
    ----------
    transcript : str
        Meeting transcript text.
    llm_service : LLMService, optional
        LLM service instance.
    meeting_id : str, optional
        Associated meeting identifier.

    Returns
    -------
    List[Dict[str, Any]]
        List of action item dictionaries adhering to schema rules.
    """
    if not transcript or not transcript.strip():
        return []

    service = llm_service or LLMService()
    intelligence: MeetingIntelligence = process_long_transcript(transcript, service)

    action_items_output = []
    for item in intelligence.action_items:
        item_dict = item.model_dump()
        if meeting_id:
            item_dict["meeting_id"] = meeting_id
        action_items_output.append(item_dict)

    return action_items_output

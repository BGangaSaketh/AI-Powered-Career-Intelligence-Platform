"""
modules/participant_mapping.py
===============================
Participant & Responsibility Mapping Module

Rules & Verification:
  1. Participant names are identified correctly.
  2. Names are mapped consistently without premature merging.
  3. Multiple participants are supported.
  4. Unknown participants are handled safely.
  5. Avoid duplicate participant records per meeting.
  6. Responsibilities are explicitly linked to the correct participant.
  7. Participants are associated with the correct meeting ID.

Conservative Matching:
  "Ravi", "Ravi Kumar", and "R. Kumar" are NOT merged automatically
  unless explicit context connects them.
"""

from typing import List, Dict, Any, Optional
from modules.schemas import Participant, MeetingIntelligence
from modules.llm_service import LLMService
from modules.long_transcript import process_long_transcript


def normalize_participant_name(name: str) -> str:
    """Clean participant name string without destructive normalization."""
    if not name or not isinstance(name, str):
        return "Unknown"
    cleaned = name.strip()
    return cleaned if cleaned else "Unknown"


def is_same_participant_conservative(name1: str, name2: str) -> bool:
    """
    Conservative participant identity comparison.
    Returns True ONLY if names match exactly (case-insensitive).
    Distinct variations like "Ravi", "Ravi Kumar", "R. Kumar" remain separate.
    """
    n1 = normalize_participant_name(name1).lower()
    n2 = normalize_participant_name(name2).lower()
    return n1 == n2


def map_participants_and_responsibilities(
    transcript: str,
    llm_service: Optional[LLMService] = None,
    meeting_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract participants and link them with assigned responsibilities.

    Parameters
    ----------
    transcript : str
        Meeting transcript.
    llm_service : LLMService, optional
        LLM service instance.
    meeting_id : str, optional
        Associated meeting ID.

    Returns
    -------
    List[Dict[str, Any]]
        Deduplicated list of participant records with linked responsibilities.
    """
    if not transcript or not transcript.strip():
        return []

    service = llm_service or LLMService()
    intelligence: MeetingIntelligence = process_long_transcript(transcript, service)

    participant_map: Dict[str, List[str]] = {}

    for p in intelligence.participants:
        clean_name = normalize_participant_name(p.name)

        # Conservative matching check against existing keys
        matched_key = None
        for key in participant_map.keys():
            if is_same_participant_conservative(clean_name, key):
                matched_key = key
                break

        if matched_key:
            # Merge responsibilities without duplicating strings
            for resp in p.responsibilities:
                if resp and resp not in participant_map[matched_key]:
                    participant_map[matched_key].append(resp)
        else:
            participant_map[clean_name] = [r for r in p.responsibilities if r]

    # Also link responsibilities from extracted action_items if unmapped
    for item in intelligence.action_items:
        if item.assigned_to and item.task:
            assignee = normalize_participant_name(item.assigned_to)
            matched_key = None
            for key in participant_map.keys():
                if is_same_participant_conservative(assignee, key):
                    matched_key = key
                    break

            if matched_key:
                if item.task not in participant_map[matched_key]:
                    participant_map[matched_key].append(item.task)
            else:
                participant_map[assignee] = [item.task]

    output_list = []
    for name, responsibilities in participant_map.items():
        record = {
            "name": name,
            "responsibilities": responsibilities
        }
        if meeting_id:
            record["meeting_id"] = meeting_id
        output_list.append(record)

    return output_list

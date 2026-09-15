"""
modules/schemas.py
==================
Pydantic Schemas & Response Validation for Meeting Intelligence

Models:
  - ActionItem
  - Participant
  - MeetingIntelligence

Handles strict validation, JSON cleanup, and fallback recovery.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import re


class ActionItem(BaseModel):
    """Represents an extracted task/action item from a meeting transcript."""
    task: str = Field(..., description="Description of the action item task")
    assigned_to: Optional[str] = Field(
        default=None,
        description="Name of participant assigned to task, or null if unassigned"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline or completion timeline, or null if unspecified"
    )
    priority: Literal["High", "Medium", "Low", "Unknown"] = Field(
        default="Unknown",
        description="Priority level: High, Medium, Low, or Unknown"
    )
    status: Literal["Pending", "In Progress", "Completed", "Unknown"] = Field(
        default="Pending",
        description="Current execution status: Pending, In Progress, Completed, or Unknown"
    )

    @field_validator("assigned_to", "deadline", mode="before")
    @classmethod
    def clean_empty_strings(cls, v):
        if v is None or (isinstance(v, str) and (not v.strip() or v.strip().lower() in ("null", "none", "unknown", "n/a"))):
            return None
        return v.strip() if isinstance(v, str) else v

    @field_validator("priority", mode="before")
    @classmethod
    def validate_priority(cls, v):
        if not v or not isinstance(v, str):
            return "Unknown"
        v_title = v.strip().title()
        if v_title in ("High", "Medium", "Low"):
            return v_title
        return "Unknown"

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v):
        if not v or not isinstance(v, str):
            return "Pending"
        v_clean = v.strip().title()
        if v_clean in ("Pending", "Completed"):
            return v_clean
        if v_clean in ("In Progress", "In-Progress", "Inprogress"):
            return "In Progress"
        return "Pending"


class Participant(BaseModel):
    """Represents a meeting participant and their assigned responsibilities."""
    name: str = Field(..., description="Participant full or first name")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="List of specific responsibilities assigned to this participant"
    )

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            return "Unknown"
        return v.strip()

    @field_validator("responsibilities", mode="before")
    @classmethod
    def validate_responsibilities(cls, v):
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            return [str(item).strip() for item in v if item and str(item).strip()]
        return []


class MeetingIntelligence(BaseModel):
    """Complete structured JSON representation of extracted meeting intelligence."""
    meeting_id: Optional[str] = Field(default=None, description="Unique meeting identifier")
    summary: str = Field(default="", description="Executive meeting summary")
    key_points: List[str] = Field(default_factory=list, description="Key discussion points")
    decisions: List[str] = Field(default_factory=list, description="Decisions reached")
    action_items: List[ActionItem] = Field(default_factory=list, description="Action items extracted")
    participants: List[Participant] = Field(default_factory=list, description="Participants & responsibilities")

    @field_validator("summary", mode="before")
    @classmethod
    def validate_summary(cls, v):
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("key_points", "decisions", mode="before")
    @classmethod
    def validate_str_lists(cls, v):
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            return [str(item).strip() for item in v if item and str(item).strip()]
        return []


def parse_and_validate_json(raw_json_str: str) -> MeetingIntelligence:
    """
    Parse a raw string from LLM output (cleaning markdown blocks), 
    and validate it against MeetingIntelligence schema.

    Raises ValueError if JSON is unparseable or validation fails fatally.
    """
    if not raw_json_str or not raw_json_str.strip():
        raise ValueError("LLM returned empty output.")

    cleaned = raw_json_str.strip()

    # Remove markdown code block fences if present (e.g. ```json ... ```)
    if "```" in cleaned:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format from LLM: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"LLM output must be a JSON object, got {type(data).__name__}")

    try:
        return MeetingIntelligence.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc

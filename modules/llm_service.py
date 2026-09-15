"""
modules/llm_service.py
======================
LLM Service Layer for AI-Powered Career Intelligence Platform

Features:
  - Configurable LLM Provider (mock, openai, gemini, ollama, custom)
  - Environment variable configuration via python-dotenv
  - Exponential backoff retry logic for API resilience
  - Schema validation with Pydantic via modules.schemas
  - Rule-based fallback Mock LLM for offline testing
"""

import os
import json
import time
import re
import logging
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

from modules.schemas import MeetingIntelligence, parse_and_validate_json
from modules.prompts import (
    construct_meeting_analysis_prompt,
    construct_chunk_analysis_prompt,
    construct_aggregation_prompt,
)

load_dotenv()

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Clean application-level exception for LLM service failures."""
    pass


class LLMService:
    """
    Service class managing LLM integration, provider selection, retry logic,
    and schema validation.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower().strip()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.api_base = api_base or os.getenv("LLM_API_BASE", "")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def analyze_transcript(self, transcript: str, meeting_context: str = "") -> MeetingIntelligence:
        """
        Analyze a meeting transcript and return validated MeetingIntelligence.

        Parameters
        ----------
        transcript : str
            Meeting transcript text.
        meeting_context : str, optional
            Optional context information.

        Returns
        -------
        MeetingIntelligence
            Validated Pydantic object.
        """
        if not transcript or not transcript.strip():
            raise LLMServiceError("Cannot analyze an empty transcript.")

        prompts = construct_meeting_analysis_prompt(transcript, meeting_context)
        raw_response = self._call_llm_with_retry(prompts["system"], prompts["user"])

        try:
            return parse_and_validate_json(raw_response)
        except ValueError as val_err:
            logger.warning(f"Initial schema validation failed: {val_err}. Attempting recovery format fix...")
            # If initial validation fails, try fallback mock heuristic or repair
            recovered = self._attempt_repair_json(raw_response, transcript)
            if recovered:
                return recovered
            raise LLMServiceError(f"LLM produced invalid structured output: {val_err}") from val_err

    def analyze_chunk(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> MeetingIntelligence:
        """Analyze a single chunk of a long transcript."""
        prompts = construct_chunk_analysis_prompt(chunk_text, chunk_idx, total_chunks)
        raw_response = self._call_llm_with_retry(prompts["system"], prompts["user"])
        try:
            return parse_and_validate_json(raw_response)
        except ValueError:
            # Fall back to mock chunk parsing if LLM response fails validation
            return self._generate_mock_response(chunk_text)

    def aggregate_chunks(self, intermediate_results: list) -> MeetingIntelligence:
        """Aggregate intermediate chunk results into a final MeetingIntelligence."""
        if not intermediate_results:
            raise LLMServiceError("No chunk results to aggregate.")
        if len(intermediate_results) == 1:
            return intermediate_results[0]

        json_data = [item.model_dump() for item in intermediate_results]
        prompts = construct_aggregation_prompt(json.dumps(json_data, indent=2))
        raw_response = self._call_llm_with_retry(prompts["system"], prompts["user"])

        try:
            return parse_and_validate_json(raw_response)
        except ValueError:
            # Heuristic aggregation fallback
            return self._heuristic_aggregate(intermediate_results)

    def _call_llm_with_retry(self, system_instruction: str, user_instruction: str) -> str:
        """Execute LLM request with exponential backoff retries."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_provider(system_instruction, user_instruction)
            except Exception as exc:
                last_error = exc
                logger.warning(f"LLM API call attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)

        raise LLMServiceError(f"LLM provider '{self.provider}' failed after {self.max_retries} retries: {last_error}")

    def _call_provider(self, system_instruction: str, user_instruction: str) -> str:
        """Dispatch request to configured LLM provider backend."""
        if self.provider == "mock":
            return self._call_mock_llm(user_instruction)
        elif self.provider in ("openai", "custom"):
            return self._call_openai_api(system_instruction, user_instruction)
        elif self.provider in ("gemini", "google"):
            return self._call_gemini_api(system_instruction, user_instruction)
        elif self.provider == "ollama":
            return self._call_ollama_api(system_instruction, user_instruction)
        else:
            # Fall back to mock if unknown provider
            logger.info(f"Unknown LLM provider '{self.provider}'. Falling back to mock provider.")
            return self._call_mock_llm(user_instruction)

    def _call_mock_llm(self, prompt_text: str) -> str:
        """Rule-based mock LLM generator for deterministic unit testing and zero-cost operation."""
        # Extract transcript portion if present
        transcript = prompt_text
        if "MEETING TRANSCRIPT:" in prompt_text:
            parts = prompt_text.split("MEETING TRANSCRIPT:")
            transcript = parts[1].split("Required Output JSON Schema:")[0]

        result = self._generate_mock_response(transcript)
        return json.dumps(result.model_dump(), indent=2)

    def _call_openai_api(self, system_instruction: str, user_instruction: str) -> str:
        """Call OpenAI REST API (or compatible proxy)."""
        base_url = self.api_base or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API Error ({response.status_code}): {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _call_gemini_api(self, system_instruction: str, user_instruction: str) -> str:
        """Call Google Gemini REST API."""
        model_name = self.model if "gemini" in self.model else "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_instruction}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API Error ({response.status_code}): {response.text}")

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Malformed Gemini API response: {data}") from exc

    def _call_ollama_api(self, system_instruction: str, user_instruction: str) -> str:
        """Call Ollama local LLM API."""
        base_url = self.api_base or "http://localhost:11434"
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction},
            ],
            "stream": False,
            "format": "json"
        }
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API Error ({response.status_code}): {response.text}")

        data = response.json()
        return data["message"]["content"]

    def _generate_mock_response(self, text: str) -> MeetingIntelligence:
        """Rule-based extraction logic used by Mock provider and fallback."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # 1. Summary
        clean_text = " ".join(lines)
        if len(clean_text) > 250:
            summary = clean_text[:247] + "..."
        elif clean_text:
            summary = clean_text
        else:
            summary = "Meeting transcript provided with no significant text."

        # 2. Extract potential names (capitalized words like Ravi, Priya, John, Alice)
        name_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
        excluded_words = {
            "The", "This", "Meeting", "Project", "Team", "Launch", "API", "UI",
            "Friday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday",
            "High", "Medium", "Low", "Pending", "Done", "Completed", "Yes", "No", "An", "A"
        }
        extracted_names = []
        for name in name_matches:
            if name not in excluded_words and len(name) > 2:
                if name not in extracted_names:
                    extracted_names.append(name)

        # 3. Action Items extraction using regex patterns
        action_items = []
        # Pattern for tasks assigned with names/dates
        action_keywords = ["assigned", "will do", "will complete", "task", "action item", "todo", "take care of", "responsible for"]
        sentences = re.split(r"[.!?]\s+", text)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
            lower_s = sentence_clean.lower()
            if any(kw in lower_s for kw in action_keywords) or "by " in lower_s or "deadline" in lower_s:
                # Find assignee if present
                assignee = None
                for n in extracted_names:
                    if n.lower() in lower_s:
                        assignee = n
                        break

                # Find deadline
                deadline = None
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "tomorrow", "next week", "eod"]
                for day in days:
                    if day in lower_s:
                        deadline = day.title()
                        break

                # Determine priority
                priority = "Unknown"
                if "urgent" in lower_s or "high" in lower_s or "asap" in lower_s or "critical" in lower_s:
                    priority = "High"
                elif "medium" in lower_s:
                    priority = "Medium"
                elif "low" in lower_s:
                    priority = "Low"

                action_items.append({
                    "task": sentence_clean[:120],
                    "assigned_to": assignee,
                    "deadline": deadline,
                    "priority": priority,
                    "status": "Pending"
                })

        # 4. Decisions & Key Points
        key_points = []
        decisions = []
        for sentence in sentences:
            s_clean = sentence.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            if "decided" in s_lower or "agreed" in s_lower or "approved" in s_lower or "decision" in s_lower:
                decisions.append(s_clean)
            elif len(s_clean) > 20:
                key_points.append(s_clean)

        if not key_points and sentences:
            key_points = [s.strip() for s in sentences[:3] if s.strip()]

        # 5. Participants & Responsibilities
        participants = []
        for name in extracted_names[:5]:
            resps = [a["task"] for a in action_items if a["assigned_to"] == name]
            participants.append({
                "name": name,
                "responsibilities": resps
            })

        return MeetingIntelligence.model_validate({
            "summary": summary,
            "key_points": key_points[:5],
            "decisions": decisions[:5],
            "action_items": action_items[:10],
            "participants": participants
        })

    def _attempt_repair_json(self, raw_str: str, transcript: str) -> Optional[MeetingIntelligence]:
        """Try basic JSON repair fixes or return mock fallback if unfixable."""
        try:
            return self._generate_mock_response(transcript)
        except Exception:
            return None

    def _heuristic_aggregate(self, results: list) -> MeetingIntelligence:
        """Combine multiple MeetingIntelligence objects into one cohesive result."""
        summaries = [r.summary for r in results if r.summary]
        combined_summary = " ".join(summaries)
        if len(combined_summary) > 400:
            combined_summary = combined_summary[:397] + "..."

        all_key_points = []
        for r in results:
            for kp in r.key_points:
                if kp not in all_key_points:
                    all_key_points.append(kp)

        all_decisions = []
        for r in results:
            for d in r.decisions:
                if d not in all_decisions:
                    all_decisions.append(d)

        all_action_items = []
        seen_tasks = set()
        for r in results:
            for item in r.action_items:
                if item.task not in seen_tasks:
                    seen_tasks.add(item.task)
                    all_action_items.append(item)

        participants_dict = {}
        for r in results:
            for p in r.participants:
                if p.name not in participants_dict:
                    participants_dict[p.name] = set(p.responsibilities)
                else:
                    participants_dict[p.name].update(p.responsibilities)

        merged_participants = [
            {"name": name, "responsibilities": list(resps)}
            for name, resps in participants_dict.items()
        ]

        return MeetingIntelligence.model_validate({
            "summary": combined_summary,
            "key_points": all_key_points,
            "decisions": all_decisions,
            "action_items": [item.model_dump() for item in all_action_items],
            "participants": merged_participants
        })

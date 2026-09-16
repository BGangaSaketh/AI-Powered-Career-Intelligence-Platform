"""
modules/prompts.py
===================
Prompt Templates and Construction for LLM Meeting Intelligence Pipeline

Provides structured, anti-hallucination prompt templates enforcing exact JSON output.
"""

SYSTEM_INSTRUCTION = """You are an expert Meeting Intelligence AI Assistant for the AI-Powered Career Intelligence Platform.
Your job is to analyze meeting transcripts and extract structured, high-accuracy meeting intelligence.

CRITICAL CONSTRAINTS & ANTI-HALLUCINATION RULES:
1. Extract strictly what is explicitly stated in or directly implied by the transcript.
2. NEVER invent or hallucinate participant names, deadlines, priorities, decisions, responsibilities, or action items.
3. If an attribute is missing or unspecified in the transcript:
   - For assigned_to: use null
   - For deadline: use null
   - For priority: use "Unknown"
   - For status: use "Pending" (or "Unknown" if status cannot be determined)
   - For missing lists: return an empty list []
4. Be conservative with participant name matching. Do NOT merge different name forms (e.g., "Ravi" vs "Ravi Kumar" vs "R. Kumar") unless there is explicit context establishing they are the exact same individual.
5. Produce ONLY raw valid JSON adhering strictly to the required schema. Do NOT include conversational text, preamble, or explanations outside the JSON object.
"""

OUTPUT_FORMAT_INSTRUCTIONS = """
Required Output JSON Schema:
{
  "summary": "Concise executive summary of what was discussed, agreed, and decided in the meeting.",
  "key_points": [
    "Key discussion point 1",
    "Key discussion point 2"
  ],
  "decisions": [
    "Formal decision 1 made by team",
    "Formal decision 2 made by team"
  ],
  "action_items": [
    {
      "task": "Clear description of action required",
      "assigned_to": "Participant name string or null",
      "deadline": "Deadline string or null",
      "priority": "High | Medium | Low | Unknown",
      "status": "Pending | In Progress | Completed | Unknown"
    }
  ],
  "participants": [
    {
      "name": "Participant name",
      "responsibilities": [
        "Assigned responsibility or role 1"
      ]
    }
  ]
}
"""


def construct_meeting_analysis_prompt(transcript: str, meeting_context: str = "") -> dict:
    """
    Construct system and user messages for standard meeting transcript analysis.

    Parameters
    ----------
    transcript : str
        The raw meeting transcript text.
    meeting_context : str, optional
        Optional additional context (e.g. meeting title, date).

    Returns
    -------
    dict
        {"system": str, "user": str}
    """
    user_msg_parts = []
    if meeting_context:
        user_msg_parts.append(f"MEETING CONTEXT: {meeting_context}\n")

    user_msg_parts.append("MEETING TRANSCRIPT:")
    user_msg_parts.append(transcript)
    user_msg_parts.append("\n" + OUTPUT_FORMAT_INSTRUCTIONS)
    user_msg_parts.append("\nAnalyze the transcript and return the JSON object now:")

    return {
        "system": SYSTEM_INSTRUCTION,
        "user": "\n".join(user_msg_parts)
    }


def construct_chunk_analysis_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> dict:
    """
    Construct prompt for analyzing a single chunk of a long transcript.
    """
    user_msg = (
        f"This is Part {chunk_index + 1} of {total_chunks} of a long meeting transcript.\n\n"
        f"TRANSCRIPT CHUNK:\n{chunk_text}\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTIONS}\n\n"
        f"Extract key points, decisions, action items, and participants for THIS CHUNK ONLY in JSON format:"
    )

    return {
        "system": SYSTEM_INSTRUCTION,
        "user": user_msg
    }


def construct_aggregation_prompt(intermediate_results_json: str) -> dict:
    """
    Construct prompt for aggregating intermediate chunk extractions into a final unified result.
    """
    system_instruction = (
        "You are an expert Meeting Intelligence AI Assistant. Your task is to aggregate "
        "multiple intermediate JSON extractions from sequential chunks of a long meeting transcript "
        "into a single, cohesive, deduplicated final Meeting Intelligence JSON output.\n"
        "Follow all anti-hallucination rules strictly."
    )

    user_msg = (
        "INTERMEDIATE CHUNK RESULTS:\n"
        f"{intermediate_results_json}\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTIONS}\n\n"
        "Synthesize all intermediate chunk findings into one final, polished JSON output:"
    )

    return {
        "system": system_instruction,
        "user": user_msg
    }


# ── RAG Prompt Templates (Milestone 3 Task 5) ──────────────────────────────

RAG_SYSTEM_PROMPT = """You are an expert Retrieval-Augmented Generation (RAG) Meeting Assistant.
Your task is to answer the user's question using ONLY the retrieved meeting context provided below.

CRITICAL RAG CONSTRAINTS & GROUNDING RULES:
1. Rely SOLELY on the clear facts contained in the supplied meeting context below. Do NOT use outside knowledge.
2. Do NOT invent, assume, or hallucinate any deadlines, participant names, dates, decisions, action items, or responsibilities.
3. If the supplied meeting context does NOT contain enough information to answer the question, state explicitly:
   "I couldn't find enough information in the available meeting records to answer this question."
4. Answer clearly, accurately, and concisely.
5. Reference the relevant meeting title or participant details when present in the retrieved context.
"""


def build_rag_prompt(question: str, context_items: list) -> dict:
    """
    Construct system and user messages for Grounded RAG Question Answering.

    Parameters
    ----------
    question : str
        User's natural language question.
    context_items : list of dict
        Retrieved context snippets containing meeting_id, title, text/relevant_snippet, content_type, etc.

    Returns
    -------
    dict
        {"system": str, "user": str}
    """
    if not context_items:
        context_str = "No relevant meeting context was found in the database."
    else:
        formatted_blocks = []
        for idx, item in enumerate(context_items, 1):
            m_title = item.get("title") or "Meeting"
            m_id = item.get("meeting_id") or "unknown"
            m_date = item.get("date") or item.get("created_at") or ""
            c_type = item.get("content_type") or "text"
            text_snippet = item.get("relevant_snippet") or item.get("text") or ""

            block = (
                f"[Source {idx}] Meeting: \"{m_title}\" (ID: {m_id}, Date: {m_date})\n"
                f"Content Type: {c_type}\n"
                f"Content Evidence: \"{text_snippet.strip()}\""
            )
            formatted_blocks.append(block)
        context_str = "\n\n".join(formatted_blocks)

    user_msg = (
        f"RETRIEVED MEETING CONTEXT:\n"
        f"{context_str}\n\n"
        f"USER QUESTION:\n"
        f"{question.strip()}\n\n"
        f"Answer the user's question accurately and concisely using ONLY the retrieved context above. "
        f"If the information is not present in the context, respond with:\n"
        f"\"I couldn't find enough information in the available meeting records to answer this question.\""
    )

    return {
        "system": RAG_SYSTEM_PROMPT,
        "user": user_msg
    }

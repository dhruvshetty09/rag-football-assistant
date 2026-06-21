"""
output_parser.py — Force structured output and validate it.

Why this file matters for interviews:
  Real production systems need parseable, consistent responses.
  "Error handling" is one of the first things interviewers probe.
  This file is proof you think about the failure cases, not just the happy path.

The LLM is prompted to return JSON in this shape:
  {
    "answer": "...",
    "citations": ["chunk_id_1", "chunk_id_2"],
    "confidence": "high" | "medium" | "low"
  }

parse_response() handles every way this can go wrong.
"""

import json
import re


def parse_response(raw: str) -> dict:
    """
    Extract structured JSON from the LLM response.
    Tries three strategies before falling back to plain text.
    """
    # Strip markdown code fences (```json ... ```) if the model added them
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Strategy 1: the whole string is valid JSON
    try:
        data = json.loads(cleaned)
        return _validate(data)
    except json.JSONDecodeError:
        pass

    # Strategy 2: JSON object is embedded somewhere in the string
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate(data)
        except json.JSONDecodeError:
            pass

    # Strategy 3: give up on structure, return the raw text as the answer
    # This is a graceful degradation — the app still works, just with less structure.
    print("[parser] Warning: could not parse structured output, falling back to raw text")
    return {
        "answer": raw.strip(),
        "citations": [],
        "confidence": "low",
        "parse_error": True,
    }


def _validate(data: dict) -> dict:
    """Ensure all expected keys are present and have the right types."""
    return {
        "answer": str(data.get("answer", "")),
        "citations": list(data.get("citations", [])),
        "confidence": data.get("confidence", "medium"),
    }

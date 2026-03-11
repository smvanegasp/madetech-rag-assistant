"""
@file highlights_service.py
@description Highlights service for the "Highlight with AI" feature.

Finds exact verbatim phrases in documents that support AI-generated answers.
Uses litellm (Groq primary, OpenAI gpt-4o-mini fallback).

Required environment variables:
- GROQ_API_KEY: For primary highlight generation
- OPENAI_API_KEY: For fallback
"""

import json
import os
from typing import List

from litellm import completion


async def get_relevance_highlights(answer: str, document_content: str) -> List[str]:
    """
    Triggers a secondary semantic analysis pass to find supporting text.

    This function powers the "Highlight with AI" feature. When a user views a
    source document, they can click to highlight which specific phrases support
    the chatbot's answer.

    Process:
    1. Receives the AI's previous answer and full document content
    2. Asks Groq to find 5-8 short exact phrases in the document
    3. Returns only verbatim strings (no paraphrasing)
    4. Frontend injects <mark> tags around these phrases

    Design decisions:
    - Uses litellm (Groq primary, OpenAI fallback)
    - Uses JSON mode for structured outputs
    - Instructs AI to avoid markdown characters for better matching
    - Returns short phrases (3-6 words) for precise highlighting

    Args:
        answer: The AI's previous answer (what we're verifying)
        document_content: Full source document to search

    Returns:
        List of verbatim strings to highlight (e.g., ["25 vacation days", "full-time employees"])

    Raises:
        Does not raise; returns empty list on error
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    system_prompt = (
        "You are a precision text extraction engine. Return ONLY valid JSON objects."
    )
    user_prompt = f"""Find 5-8 short, key phrases (3-6 words each) in the DOCUMENT that specifically support the claims in the ANSWER.

STRICT RULES:
1. Return a JSON object with a "highlights" key containing an array of strings.
2. Each string MUST be a LITERALLY EXACT VERBATIM substring from the DOCUMENT.
3. Choose phrases that do not contain markdown characters like *, #, _ to ensure better matching.
4. Be extremely precise with capitalization and punctuation.

Format: {{"highlights": ["phrase 1", "phrase 2", ...]}}

ANSWER:
"{answer}"

DOCUMENT:
"{document_content}"
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Try Groq first (litellm routes via model prefix)
    try:
        response = completion(
            model="groq/openai/gpt-oss-20b",
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        highlights = _parse_highlights_result(result)
        print("Highlights generated with Groq:", len(highlights), "phrases")
        return highlights
    except Exception as groq_error:
        print("Groq failed, falling back to OpenAI:", groq_error)
        try:
            response = completion(
                model="openai/gpt-4o-mini",
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            highlights = _parse_highlights_result(result)
            print(
                "Highlights generated with OpenAI (fallback):",
                len(highlights),
                "phrases",
            )
            return highlights
        except Exception as openai_error:
            print("OpenAI fallback failed:", openai_error)
            return []


def _parse_highlights_result(result: dict | list | None) -> List[str]:
    """
    Parse LLM JSON response into a flat list of highlight phrases.

    Handles variations: {"highlights": [...]}, {"phrases": [...]}, or raw list.
    """
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "highlights" in result:
        return result["highlights"]
    if isinstance(result, dict) and "phrases" in result:
        return result["phrases"]
    for value in (result or {}).values():
        if isinstance(value, list):
            return value
    return []

"""
Query rewriting — turn follow-up questions into standalone search queries.

E.g. "What about parental leave?" + history → "parental leave policy Made Tech"
"""

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.prompts import REWRITE_QUERY_SYSTEM_PROMPT


@retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(5))
def rewrite_query(
    question: str,
    history: list[dict],
    model: str,
) -> str:
    """
    Use an LLM to rewrite the question for better retrieval.

    Incorporates conversation history so follow-ups become standalone queries.
    Returns a short, specific string suitable for semantic search.
    """
    history_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history
    ) if history else "(No prior messages)"

    system_prompt = REWRITE_QUERY_SYSTEM_PROMPT.format(history=history_text, question=question)
    response = completion(model=model, messages=[{"role": "system", "content": system_prompt}])
    return response.choices[0].message.content

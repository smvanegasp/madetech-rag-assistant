"""
Query rewriting — turn follow-up questions into standalone search queries.

E.g. "What about parental leave?" + history → "parental leave policy Made Tech"
"""

from litellm import completion
from tenacity import retry, wait_exponential


@retry(wait=wait_exponential(multiplier=1, min=10, max=240))
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

    message = f"""
You are in a conversation with a user, answering questions about the company Made Tech.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:

[HISTORY STARTS]

{history_text}

[HISTORY ENDS]

And this is the user's current question:

[QUESTION STARTS]

{question}

[QUESTION ENDS]

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=model, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

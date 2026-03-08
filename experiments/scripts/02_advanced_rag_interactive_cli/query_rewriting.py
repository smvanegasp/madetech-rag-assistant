"""Query rewriting for improved retrieval — expands user questions for KB search."""

from litellm import completion


def rewrite_query(
    question: str,
    history: list[dict],
    model: str,
) -> str:
    """
    Rewrite the user's question to be a more specific query that is more likely
    to surface relevant content in the Knowledge Base.
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

"""
LLM reranking — reorder retrieved chunks by relevance to the question.

ChromaDB returns cosine-similar chunks; this step lets an LLM refine the order.
"""

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.models import Result
from utils.prompts import RERANK_CHUNKS_SYSTEM_PROMPT


class RankOrder(BaseModel):
    """Pydantic model for LLM reranking response."""

    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def merge_chunks(chunks_1: list[Result], chunks_2: list[Result]) -> list[Result]:
    """
    Merge two chunk lists, deduplicating by page_content.

    Used when query rewriting fetches a second set of chunks; we combine
    original + rewritten results and remove duplicates.
    """
    merged = list(chunks_1)
    existing = {chunk.page_content for chunk in chunks_1}
    for chunk in chunks_2:
        if chunk.page_content not in existing:
            merged.append(chunk)
            existing.add(chunk.page_content)
    return merged


def _repair_concatenated_order(concatenated: int, n: int) -> list[int] | None:
    """Recover order from LLM output like 124936781011121314151617 when it forgets commas."""
    s = str(concatenated)
    result = []
    i = 0
    valid = set(range(1, n + 1))
    used = set()
    while i < len(s) and len(result) < n:
        one = int(s[i])
        two_val = int(s[i : i + 2]) if i + 1 < len(s) else None
        if one in valid and one not in used:
            result.append(one)
            used.add(one)
            i += 1
        elif two_val is not None and two_val in valid and two_val not in used:
            result.append(two_val)
            used.add(two_val)
            i += 2
        else:
            return None
    return result if len(result) == n and i == len(s) else None


@retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(5))
def rerank(question: str, chunks: list[Result], model: str) -> list[Result]:
    """
    Ask an LLM to reorder chunks by relevance to the question.

    Chunks are numbered 1..N; the LLM returns a permutation (e.g. [3,1,2]).
    On parse failure or invalid output, returns chunks in original order.
    """
    n = len(chunks)
    valid_ids = set(range(1, n + 1))

    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all {n} chunks by relevance, from most to least relevant. Include every chunk ID from 1 to {n} exactly once.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += f"Reply with valid JSON only: {{\"order\": [a, b, c, ...]}} with {n} comma-separated integers."

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": RERANK_CHUNKS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RankOrder,
    )
    reply = response.choices[0].message.content
    parsed = RankOrder.model_validate_json(reply)
    order = parsed.order

    if len(order) == n and set(order) == valid_ids:
        return [chunks[i - 1] for i in order]
    if len(order) == 1 and isinstance(order[0], int) and order[0] > 9:
        repaired = _repair_concatenated_order(order[0], n)
        if repaired is not None:
            return [chunks[i - 1] for i in repaired]
    return list(chunks)

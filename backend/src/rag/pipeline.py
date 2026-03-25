"""
RAG pipeline: retrieval, optional rewriting, optional reranking, and answer generation.

This module orchestrates the full RAG flow. All LLM calls go through litellm.

The LLM decides whether to call the handbook search tool (RAG) or answer directly:
- Phase 1: LLM receives the conversation with a `search_handbook` tool definition.
  If it calls the tool, retrieval runs and Phase 2 generates the grounded answer.
  If it answers directly (e.g. greetings, follow-ups covered by history), no
  retrieval is performed and no source chunks are returned.
"""

import json

from litellm import completion
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.models import Result
from utils.prompts import RAG_SYSTEM_PROMPT, TOOL_DECISION_SYSTEM_PROMPT

from .query_rewriting import rewrite_query
from .reranking import merge_chunks, rerank
from .retrieval import fetch_context_unranked

SEARCH_HANDBOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": (
            "Search the Made Tech handbook for relevant information. "
            "Use this whenever the user asks about company policies, processes, "
            "benefits, roles, or any subject that requires handbook knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise search query that captures the core of what "
                        "the user wants to know."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


def fetch_context(
    original_question: str,
    history: list[dict],
    collection,
    openai_client: OpenAI,
    config: dict,
) -> list[Result]:
    """
    Retrieve and optionally refine context chunks for the LLM.

    Flow:
    1. Fetch retrieval_k chunks for the original question (OpenAI embeddings + ChromaDB)
    2. If use_query_rewriting: rewrite question, fetch again, merge and dedupe
    3. If use_reranking: LLM reorders chunks by relevance
    4. Return top final_k chunks

    Config keys: retrieval.retrieval_k, retrieval.final_k, use_query_rewriting,
    use_reranking, embedding_model, model.
    """
    retrieval_cfg = config.get("retrieval", {})
    retrieval_k = retrieval_cfg.get("retrieval_k", 10)
    final_k = retrieval_cfg.get("final_k", 7)
    use_rewriting = config.get("use_query_rewriting", True)
    use_reranking = config.get("use_reranking", True)
    embedding_model = config.get("embedding_model", "text-embedding-3-large")
    model = config.get("model", "groq/openai/gpt-oss-20b")

    chunks1 = fetch_context_unranked(
        original_question,
        collection,
        openai_client,
        embedding_model,
        retrieval_k,
    )

    if use_rewriting:
        rewritten_question = rewrite_query(original_question, history, model)
        chunks2 = fetch_context_unranked(
            rewritten_question,
            collection,
            openai_client,
            embedding_model,
            retrieval_k,
        )
        chunks = merge_chunks(chunks1, chunks2)
    else:
        chunks = chunks1

    if use_reranking:
        reranked = rerank(original_question, chunks, model)
        return reranked[:final_k]
    return chunks[:final_k]


def _format_context(chunks: list[Result]) -> str:
    """Render retrieved chunks as a single context string for injection into the LLM."""
    return "\n\n---\n\n".join(
        f"Extract from ({chunk.metadata.get('category', '')}) — {chunk.metadata.get('title', '')}:\n{chunk.page_content}"
        for chunk in chunks
    )


def make_rag_messages(
    question: str,
    history: list[dict],
    chunks: list[Result],
) -> list[dict]:
    """
    Build chat messages for the LLM.

    System prompt: RAG_SYSTEM_PROMPT with injected context (chunks joined by ---).
    User messages: conversation history + current question.
    """
    context = _format_context(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(5))
def _call_completion(model: str, messages: list[dict]) -> str:
    """Call LLM completion with retry on transient failures."""
    response = completion(model=model, messages=messages)
    return response.choices[0].message.content


@retry(wait=wait_exponential(multiplier=1, min=10, max=240), stop=stop_after_attempt(5))
def _call_completion_with_tools(model: str, messages: list[dict]):
    """
    Call LLM completion with the search_handbook tool available.

    Returns the raw response Message so callers can inspect .tool_calls.
    """
    response = completion(
        model=model,
        messages=messages,
        tools=[SEARCH_HANDBOOK_TOOL],
        tool_choice="auto",
    )
    return response.choices[0].message


def answer_question(
    question: str,
    history: list[dict] | None = None,
    collection=None,
    openai_client: OpenAI | None = None,
    config: dict | None = None,
) -> tuple[str, list[Result]]:
    """
    Run the RAG pipeline with LLM-decided retrieval and return (answer, chunks).

    Phase 1 — decision:
        The LLM sees the conversation plus the search_handbook tool definition.
        If it decides retrieval is unnecessary (greetings, history-answerable
        follow-ups, off-topic chat) it replies directly; chunks = [].

    Phase 2 — grounded answer (only when tool is called):
        The tool query drives fetch_context(); the retrieved context is passed
        back as a tool result and the LLM generates a grounded answer.

    Args:
        question: User's question.
        history: Conversation history (role/content dicts).
        collection: ChromaDB collection (from get_chroma_collection).
        openai_client: Used for embeddings (retrieval).
        config: Pipeline config (see fetch_context).

    Returns:
        (answer_text, chunks) — chunks is empty when RAG was not used.
    """
    if history is None:
        history = []
    if config is None:
        config = {}
    if openai_client is None:
        openai_client = OpenAI()
    if collection is None:
        raise ValueError(
            "collection is required; use get_chroma_collection() to obtain it"
        )

    model = config.get("model", "groq/openai/gpt-oss-20b")

    # Phase 1: let the LLM decide whether to search the handbook.
    phase1_messages = (
        [{"role": "system", "content": TOOL_DECISION_SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": question}]
    )
    assistant_msg = _call_completion_with_tools(model, phase1_messages)

    # No tool call → return the direct answer with no sources.
    if not assistant_msg.tool_calls:
        return assistant_msg.content, []

    # Phase 2: tool was called — run retrieval then generate a grounded answer.
    tool_call = assistant_msg.tool_calls[0]
    tool_args = json.loads(tool_call.function.arguments)
    tool_query = tool_args.get("query", question)

    chunks = fetch_context(
        tool_query,
        history,
        collection,
        openai_client,
        config,
    )

    context = _format_context(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    # Reconstruct full message thread: system + history + user + assistant
    # (with tool call) + tool result, then generate the final answer.
    phase2_messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
        + [
            {
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": context,
            },
        ]
    )

    answer = _call_completion(model, phase2_messages)
    return answer, chunks

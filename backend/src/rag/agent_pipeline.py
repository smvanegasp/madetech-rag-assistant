"""
Agent-based RAG pipeline using the OpenAI Agents SDK.

Replaces the manual 2-phase tool-calling approach in pipeline.py with a
declarative agent + @function_tool decorator. The SDK handles the tool-call
loop automatically: if the LLM decides to search the handbook, the tool
executes, the result is fed back, and the LLM generates a grounded answer
— all within a single Runner.run_sync() call.

Run from repo root:
    uvicorn backend.src.app:app --reload --port 9481
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agents import Agent, Runner, function_tool, RunContextWrapper, OpenAIChatCompletionsModel
from agents.stream_events import RunItemStreamEvent
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Async Groq client — lazily initialized on first use (after dotenv loads).
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_groq_client: AsyncOpenAI | None = None


def _get_groq_client() -> AsyncOpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

from utils.models import Result
from utils.prompts import TOOL_DECISION_SYSTEM_PROMPT, RAG_ANSWERING_INSTRUCTIONS

from .pipeline import fetch_context
from ..contact_service import send_contact_email


@dataclass
class RAGContext:
    """Carries dependencies and mutable state through the agent tool loop.

    The SDK injects this as ctx.context in tool functions. It is never
    sent to the LLM — purely a local object for reading/writing state.
    """

    collection: Any
    openai_client: OpenAI
    config: dict
    history: list[dict]
    retrieved_chunks: list[Result] = field(default_factory=list)


def _format_context(chunks: list[Result]) -> str:
    """Render retrieved chunks as a numbered context string."""
    return "\n\n---\n\n".join(
        f"[{i}] Extract from ({chunk.metadata.get('category', '')}) — {chunk.metadata.get('title', '')}:\n{chunk.page_content}"
        for i, chunk in enumerate(chunks, 1)
    )


@function_tool
def search_handbook(ctx: RunContextWrapper[RAGContext], query: str) -> str:
    """Search the handbook for a single topic.
    Use this for simple questions that need one search, or as a follow-up
    search after plan_searches if you need additional information.
    IMPORTANT: Keep the query focused on the topic itself. Do NOT include
    'Made Tech' or 'handbook' in the query — these are noise words that
    hurt search quality since every document contains them.
    Good: 'parental leave policy'
    Bad: 'Made Tech handbook parental leave policy'
    """
    try:
        chunks = fetch_context(
            query,
            ctx.context.history,
            ctx.context.collection,
            ctx.context.openai_client,
            ctx.context.config,
        )
    except Exception as e:
        logger.warning("search_handbook failed (%s: %s)", type(e).__name__, e)
        return "This search failed due to a temporary issue. Try answering with the context you already have, or attempt a different search."

    ctx.context.retrieved_chunks.extend(chunks)
    context_str = _format_context(chunks)
    return f"{RAG_ANSWERING_INSTRUCTIONS}\n\n{context_str}"


@function_tool
def plan_searches(ctx: RunContextWrapper[RAGContext], queries: list[str]) -> str:
    """Plan and execute multiple handbook searches at once.
    Use this when the question requires information from multiple topics
    (e.g., comparing roles, asking about several benefits, multi-part questions).

    Provide a list of 2-4 distinct search queries. Each query should target
    a different topic — do not include duplicate or overlapping queries.
    IMPORTANT: Do NOT include 'Made Tech' or 'handbook' in queries — these
    are noise words. Focus on the topic: 'parental leave', 'Data Engineer role', etc.

    After reviewing the results, if you need more information you can still
    call search_handbook for a follow-up search. If the results already
    contain enough information, skip any remaining planned queries and
    answer directly.
    """
    all_chunks = []
    seen_content = set()
    results_parts = []
    failed_queries = []

    for i, query in enumerate(queries, 1):
        try:
            chunks = fetch_context(
                query,
                ctx.context.history,
                ctx.context.collection,
                ctx.context.openai_client,
                ctx.context.config,
            )
        except Exception as e:
            logger.warning("plan_searches: query %d/%d failed (%s: %s), skipping",
                           i, len(queries), type(e).__name__, e)
            failed_queries.append(query)
            continue

        new_chunks = []
        for chunk in chunks:
            key = chunk.page_content[:200]
            if key not in seen_content:
                seen_content.add(key)
                new_chunks.append(chunk)
        all_chunks.extend(new_chunks)
        if new_chunks:
            results_parts.append(f"--- Search {i}: \"{query}\" ({len(new_chunks)} unique chunks) ---\n\n{_format_context(new_chunks)}")

    ctx.context.retrieved_chunks.extend(all_chunks)
    combined = "\n\n".join(results_parts)

    if failed_queries and not results_parts:
        return "All searches failed due to a temporary issue. Please answer based on conversation context, or try again."

    if failed_queries:
        skipped = ", ".join(f'"{q}"' for q in failed_queries)
        combined += f"\n\n(Note: {len(failed_queries)} search(es) failed and were skipped: {skipped}. Answer using the results above.)"

    return f"{RAG_ANSWERING_INSTRUCTIONS}\n\n{combined}"


def _validate_contact_fields(name: str, email: str, message: str) -> str | None:
    """Return an error message if any field is missing/placeholder, or None if valid."""
    if not name or name.lower() in ("unknown", "n/a", "user", "name"):
        return "You don't have the user's name yet. Ask them for their name before calling this tool."
    if not email or "@" not in email:
        return "You don't have a valid email address yet. Ask the user for their email before calling this tool."
    if not message or len(message.strip()) < 5:
        return "You don't have a message yet. Ask the user what they'd like to say before calling this tool."
    return None


@function_tool
def send_feedback(ctx: RunContextWrapper[RAGContext], name: str, email: str, message: str) -> str:
    """Send user feedback about the Nexus assistant.
    ONLY use this tool when ALL of these conditions are met:
    1. The user has EXPLICITLY asked to send feedback (not just mentioned it in passing)
    2. The user has provided their name, email, and feedback message in the conversation
    3. This is the ONLY action in this turn — never combine with search_handbook or plan_searches
    Do NOT call this tool proactively or as an intermediate step.
    """
    error = _validate_contact_fields(name, email, message)
    if error:
        return error
    try:
        send_contact_email("feedback", name, email, message)
        return "Feedback sent successfully. Thank the user and let them know their feedback has been received."
    except Exception as e:
        return f"Failed to send feedback: {e}. Apologize and suggest they try again later."


@function_tool
def get_in_touch(ctx: RunContextWrapper[RAGContext], name: str, email: str, message: str) -> str:
    """Send a contact request to get in touch with the creator of this app.
    ONLY use this tool when ALL of these conditions are met:
    1. The user has EXPLICITLY asked to contact or get in touch with the creator (not just mentioned it in passing)
    2. The user has provided their name, email, and message in the conversation
    3. This is the ONLY action in this turn — never combine with search_handbook or plan_searches
    Do NOT call this tool proactively or as an intermediate step.
    """
    error = _validate_contact_fields(name, email, message)
    if error:
        return error
    try:
        send_contact_email("contact", name, email, message)
        return "Contact request sent successfully. Thank the user and let them know their message has been delivered."
    except Exception as e:
        return f"Failed to send contact request: {e}. Apologize and suggest they try again later."


def _extract_tool_steps(result) -> list[dict]:
    """Extract tool call metadata from the agent RunResult.

    Skips tool calls that were rejected by validation (e.g., missing contact fields).
    Pairs each ToolCallItem with its following ToolCallOutputItem to check the output.
    """
    from agents.items import ToolCallItem, ToolCallOutputItem

    # Build list of (call, output) pairs
    items = result.new_items
    steps = []
    order = 1
    for i, item in enumerate(items):
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        tool_name = getattr(raw, "name", "unknown")
        arguments_str = getattr(raw, "arguments", "{}")
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else {}
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        # Check if the next item is a ToolCallOutputItem with a validation rejection
        output_str = ""
        if i + 1 < len(items) and isinstance(items[i + 1], ToolCallOutputItem):
            output_str = str(items[i + 1].output) if items[i + 1].output else ""

        # Skip failed validation calls (contact/feedback tools that were rejected)
        if output_str.startswith("You don't have"):
            continue

        steps.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "order": order,
        })
        order += 1
    return steps


def _build_agent(config: dict) -> Agent[RAGContext]:
    """Create the Nexus agent with the search tool."""
    model_name = config.get("model", "groq/openai/gpt-oss-20b")
    # Strip "groq/" prefix — Groq's API expects "openai/gpt-oss-20b" directly
    raw_model_name = model_name.removeprefix("groq/")
    today_str = date.today().strftime("%B %d, %Y")
    instructions = TOOL_DECISION_SYSTEM_PROMPT.format(today=today_str)

    return Agent[RAGContext](
        name="Nexus",
        instructions=instructions,
        tools=[search_handbook, plan_searches, send_feedback, get_in_touch],
        model=OpenAIChatCompletionsModel(
            model=raw_model_name,
            openai_client=_get_groq_client(),
        ),
    )


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(3),
)
async def answer_question_agent(
    question: str,
    history: list[dict] | None = None,
    collection=None,
    openai_client: OpenAI | None = None,
    config: dict | None = None,
) -> tuple[str, list[Result], list[dict]]:
    """
    Run the agent-based RAG pipeline and return (answer, chunks).

    Drop-in replacement for pipeline.answer_question() with the same
    signature and return type. The agent decides whether to search the
    handbook or answer directly — no manual phase orchestration needed.

    Args:
        question: User's question.
        history: Conversation history (role/content dicts).
        collection: ChromaDB collection (from get_chroma_collection).
        openai_client: Used for embeddings (retrieval).
        config: Pipeline config (model, retrieval params, approach flags).

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

    rag_context = RAGContext(
        collection=collection,
        openai_client=openai_client,
        config=config,
        history=history,
    )

    agent = _build_agent(config)

    # Build input messages: history + current question
    input_messages = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            input_messages.append({"role": role, "content": content})
    input_messages.append({"role": "user", "content": question})

    result = await Runner.run(
        agent,
        input=input_messages,
        context=rag_context,
        max_turns=6,
    )

    tool_steps = _extract_tool_steps(result)
    return result.final_output, rag_context.retrieved_chunks, tool_steps


_STREAM_MAX_ATTEMPTS = 3
_STREAM_RETRY_BASE_DELAY = 5


async def answer_question_agent_streamed(
    question: str,
    history: list[dict] | None = None,
    collection=None,
    openai_client: OpenAI | None = None,
    config: dict | None = None,
):
    """
    Async generator that yields SSE events as the agent processes the query.

    Retries up to 3 times on transient failures (matching the non-streaming
    path). Once tool_step events have been yielded to the caller, retrying
    is not possible so the error is surfaced immediately.

    Yields dicts with "event" and "data" keys:
    - {"event": "tool_step", "data": {"tool_name": ..., "arguments": ..., "order": ...}}
    - {"event": "done", "data": {"content": ..., "sources": [...], "tool_steps": [...]}}
    """
    if history is None:
        history = []
    if config is None:
        config = {}
    if openai_client is None:
        openai_client = OpenAI()
    if collection is None:
        raise ValueError("collection is required")

    agent = _build_agent(config)

    input_messages = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            input_messages.append({"role": role, "content": content})
    input_messages.append({"role": "user", "content": question})

    for attempt in range(1, _STREAM_MAX_ATTEMPTS + 1):
        rag_context = RAGContext(
            collection=collection,
            openai_client=openai_client,
            config=config,
            history=history,
        )

        result = Runner.run_streamed(
            agent,
            input=input_messages,
            context=rag_context,
            max_turns=6,
        )

        tool_order = 0
        has_yielded = False
        try:
            async for event in result.stream_events():
                if (
                    isinstance(event, RunItemStreamEvent)
                    and event.name == "tool_called"
                ):
                    try:
                        raw = event.item.raw_item
                        tool_name = getattr(raw, "name", "unknown")
                        arguments_str = getattr(raw, "arguments", "{}")
                        try:
                            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else {}
                        except (json.JSONDecodeError, TypeError):
                            arguments = {}

                        tool_order += 1
                        has_yielded = True
                        yield {
                            "event": "tool_step",
                            "data": {
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "order": tool_order,
                            },
                        }
                    except Exception:
                        pass

            tool_steps = _extract_tool_steps(result)
            yield {
                "event": "done",
                "data": {
                    "content": result.final_output,
                    "chunks": rag_context.retrieved_chunks,
                    "tool_steps": tool_steps,
                },
            }
            return
        except Exception as e:
            is_last_attempt = attempt == _STREAM_MAX_ATTEMPTS

            if has_yielded or is_last_attempt:
                logger.error("Stream failed (attempt %d/%d, %s): %s",
                             attempt, _STREAM_MAX_ATTEMPTS, type(e).__name__, e)
                content = getattr(result, "final_output", None) or \
                    "I encountered an issue processing your request. Please try again."
                yield {
                    "event": "done",
                    "data": {
                        "content": content,
                        "chunks": rag_context.retrieved_chunks,
                        "tool_steps": [],
                        "isError": True,
                    },
                }
                return

            delay = _STREAM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("Stream attempt %d/%d failed (%s: %s), retrying in %ds...",
                           attempt, _STREAM_MAX_ATTEMPTS, type(e).__name__, e, delay)
            await asyncio.sleep(delay)

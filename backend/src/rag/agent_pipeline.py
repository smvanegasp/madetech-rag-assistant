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

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agents import Agent, Runner, function_tool, RunContextWrapper
from agents.extensions.models.litellm_model import LitellmModel
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

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
    """Search the Made Tech handbook for relevant information.
    Use this whenever the user asks about company policies, processes,
    benefits, roles, or any subject that requires handbook knowledge.
    """
    chunks = fetch_context(
        query,
        ctx.context.history,
        ctx.context.collection,
        ctx.context.openai_client,
        ctx.context.config,
    )
    ctx.context.retrieved_chunks.extend(chunks)
    context_str = _format_context(chunks)
    return f"{RAG_ANSWERING_INSTRUCTIONS}\n\n{context_str}"


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
    Use this when the user wants to share feedback, report a problem, or suggest
    an improvement about the app.
    IMPORTANT: You MUST have all three fields before calling this tool:
    - name: the user's full name
    - email: a valid email address
    - message: the feedback content
    Do NOT call this tool until you have explicitly collected all three from the user.
    If any are missing, ask for them one by one before calling.
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
    Use this when the user wants to reach out, connect, ask about the project,
    or contact the person who built Nexus.
    IMPORTANT: You MUST have all three fields before calling this tool:
    - name: the user's full name
    - email: a valid email address
    - message: what they want to discuss or say
    Do NOT call this tool until you have explicitly collected all three from the user.
    If any are missing, ask for them one by one before calling.
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
    """Extract tool call metadata from the agent RunResult."""
    from agents.items import ToolCallItem

    steps = []
    order = 1
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            tool_name = getattr(raw, "name", "unknown")
            arguments_str = getattr(raw, "arguments", "{}")
            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
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
    today_str = date.today().strftime("%B %d, %Y")
    instructions = TOOL_DECISION_SYSTEM_PROMPT.format(today=today_str)

    return Agent[RAGContext](
        name="Nexus",
        instructions=instructions,
        tools=[search_handbook, send_feedback, get_in_touch],
        model=LitellmModel(model=model_name),
    )


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(3),
)
def answer_question_agent(
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

    result = Runner.run_sync(
        agent,
        input=input_messages,
        context=rag_context,
        max_turns=6,
    )

    tool_steps = _extract_tool_steps(result)
    return result.final_output, rag_context.retrieved_chunks, tool_steps

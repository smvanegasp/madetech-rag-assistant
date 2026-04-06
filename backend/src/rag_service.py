"""
RAG service — orchestrates retrieval-augmented generation for handbook queries.

This is the main service consumed by /api/chat. It delegates to the rag pipeline
(retrieval → optional rewriting → optional reranking → LLM generation) and
maps results to API-ready SourceChunk objects.

Configuration (config.yaml):
- approach.use_query_rewriting: Expand follow-up questions for better retrieval
- approach.use_reranking: LLM reorders chunks by relevance before generation

Required env: GROQ_API_KEY, OPENAI_API_KEY
"""

import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from utils.models import Message, SourceChunk

from utils.models import HandbookDoc
from .config_loader import load_config
from .rag import get_chroma_collection
from .rag.agent_pipeline import answer_question_agent, answer_question_agent_streamed
from .rag.keyword_search import HandbookBM25Index

load_dotenv(override=True)


def _history_to_messages(history: List[Message]) -> list[dict]:
    """
    Convert Message objects to the dict format expected by the RAG pipeline.

    The pipeline uses list[dict] with "role" and "content" keys.
    """
    return [{"role": m.role, "content": m.content} for m in history]


class RAGService:
    """
    RAG service for handbook queries using Chroma vector database and Groq.

    Supports configurable approaches via config.yaml:
    - basic_rag: Simple retrieval (use_query_rewriting=false, use_reranking=false)
    - with_reranking: LLM reranking of chunks
    - with_rewriting: Query rewriting + dual retrieval
    - with_rewriting_and_reranking: Both enabled (recommended)
    """

    def __init__(self, config: dict | None = None, handbook_docs: list[HandbookDoc] | None = None):
        config = config if config is not None else load_config()
        self.config = config
        self.collection_name = config["vector_db"]["collection_name"]
        self.chroma_database = config["vector_db"]["database"]

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.openai_client = OpenAI(api_key=openai_api_key)

        chroma_api_key = os.getenv("CHROMA_API_KEY")
        if not chroma_api_key:
            raise ValueError("CHROMA_API_KEY environment variable not set")
        chroma_tenant = os.getenv("TENANT_CHROMA")
        if not chroma_tenant:
            raise ValueError("TENANT_CHROMA environment variable not set")

        self.collection = get_chroma_collection(
            self.collection_name, chroma_api_key, chroma_tenant, self.chroma_database
        )

        approach_desc = (
            "with_rewriting_and_reranking"
            if config.get("use_query_rewriting") and config.get("use_reranking")
            else "with_reranking"
            if config.get("use_reranking")
            else "with_rewriting"
            if config.get("use_query_rewriting")
            else "basic_rag"
        )
        # Build BM25 keyword search index if enabled and docs available
        self.bm25_index = None
        if config.get("use_keyword_search", False) and handbook_docs:
            self.bm25_index = HandbookBM25Index(handbook_docs)

        print(
            f"RAG service initialized ({approach_desc}, keyword_search={'on' if self.bm25_index else 'off'}) "
            f"with Chroma Cloud collection '{self.collection_name}'"
        )

    def _extract_sources(
        self, chunks: List[Any], max_sources: int = 15
    ) -> List[SourceChunk]:
        """
        Convert retrieval Result objects to SourceChunk for the API response.

        Deduplicates by (doc_id, snippet) to avoid returning the same chunk
        multiple times when accumulated across multiple tool calls. Truncates
        long snippets at sentence boundaries when > 800 chars.
        """
        sources = []
        seen = set()
        for chunk in chunks:
            meta = (
                chunk.metadata
                if hasattr(chunk, "metadata")
                else chunk.get("metadata", {})
            )
            page = (
                chunk.page_content
                if hasattr(chunk, "page_content")
                else chunk.get("page_content", "")
            )
            doc_id = (
                meta.get("doc_id")
                or meta.get("id")
                or meta.get("source_file", "unknown")
            )
            snippet = page.strip()
            if len(snippet) > 800:
                truncate_at = snippet.rfind(".", 0, 800)
                if truncate_at > 400:
                    snippet = snippet[: truncate_at + 1]
                else:
                    snippet = snippet[:800] + "..."
            # Deduplicate by content hash
            key = (doc_id, snippet[:200])
            if key in seen:
                continue
            seen.add(key)
            sources.append(SourceChunk(docId=doc_id, snippet=snippet))
            if len(sources) >= max_sources:
                break
        return sources

    async def get_rag_response(
        self,
        query: str,
        history: List[Message],
    ) -> Dict[str, Any]:
        """
        Generate a RAG response: answer text plus source citations.

        Args:
            query: User's question.
            history: Last 30 messages (converted for pipeline).

        Returns:
            {"content": str, "sources": List[SourceChunk]}
            On error, returns a friendly message and empty sources.
        """
        try:
            history_msgs = _history_to_messages(history)
            history_msgs = _history_to_messages(history[-30:])
            pipeline_config = {
                **self.config,
                "use_query_rewriting": self.config.get("use_query_rewriting", False),
                "use_reranking": self.config.get("use_reranking", False),
                "bm25_index": self.bm25_index,
            }
            answer, chunks, tool_steps = await asyncio.to_thread(
                answer_question_agent,
                query,
                history=history_msgs,
                collection=self.collection,
                openai_client=self.openai_client,
                config=pipeline_config,
            )
            sources = self._extract_sources(chunks)
            return {"content": answer, "sources": sources, "tool_steps": tool_steps}
        except Exception as e:
            print(f"RAG service error: {e}")
            return {
                "content": "I'm having trouble processing your request right now. Please try again in a moment.",
                "sources": [],
                "tool_steps": [],
            }

    async def get_rag_response_streamed(
        self,
        query: str,
        history: List[Message],
    ):
        """
        Async generator that yields SSE events as the agent processes the query.

        Yields tool_step events in real-time, then a final done event with the answer.
        """
        history_msgs = _history_to_messages(history[-30:])
        pipeline_config = {
            **self.config,
            "use_query_rewriting": self.config.get("use_query_rewriting", False),
            "use_reranking": self.config.get("use_reranking", False),
            "bm25_index": self.bm25_index,
        }

        try:
            async for event in answer_question_agent_streamed(
                query,
                history=history_msgs,
                collection=self.collection,
                openai_client=self.openai_client,
                config=pipeline_config,
            ):
                if event["event"] == "tool_step":
                    yield event
                elif event["event"] == "done":
                    data = event["data"]
                    sources = self._extract_sources(data["chunks"])
                    yield {
                        "event": "done",
                        "data": {
                            "content": data["content"],
                            "sources": [s.model_dump() for s in sources],
                            "tool_steps": data["tool_steps"],
                        },
                    }
        except Exception as e:
            print(f"RAG service streaming error: {e}")
            yield {
                "event": "done",
                "data": {
                    "content": "I'm having trouble processing your request right now. Please try again in a moment.",
                    "sources": [],
                    "tool_steps": [],
                    "isError": True,
                },
            }

"""
RAG pipeline: retrieval, optional rewriting, optional reranking, and answer generation.

This module orchestrates the full RAG flow. All LLM calls go through litellm.
"""

from litellm import completion
from openai import OpenAI
from tenacity import retry, wait_exponential

from utils.models import Result
from utils.prompts import RAG_SYSTEM_PROMPT
from .query_rewriting import rewrite_query
from .reranking import merge_chunks, rerank
from .retrieval import fetch_context_unranked


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
    context = "\n\n---\n\n".join(
        f"Extract from ({chunk.metadata.get('category', '')}) — {chunk.metadata.get('title', '')}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait_exponential(multiplier=1, min=10, max=240))
def _call_completion(model: str, messages: list[dict]) -> str:
    """Call LLM completion with retry on transient failures."""
    response = completion(model=model, messages=messages)
    return response.choices[0].message.content


def answer_question(
    question: str,
    history: list[dict] | None = None,
    collection=None,
    openai_client: OpenAI | None = None,
    config: dict | None = None,
) -> tuple[str, list[Result]]:
    """
    Run the full RAG pipeline and return the generated answer plus retrieved chunks.

    Args:
        question: User's question.
        history: Conversation history (role/content dicts).
        collection: ChromaDB collection (from get_chroma_collection).
        openai_client: Used for embeddings (retrieval).
        config: Pipeline config (see fetch_context).

    Returns:
        (answer_text, chunks) — chunks are Result objects for source extraction.
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

    chunks = fetch_context(
        question,
        history,
        collection,
        openai_client,
        config,
    )
    messages = make_rag_messages(question, history, chunks)
    model = config.get("model", "groq/openai/gpt-oss-20b")
    return _call_completion(model, messages), chunks

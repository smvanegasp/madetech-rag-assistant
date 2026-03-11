"""RAG pipeline orchestration — retrieval, reranking, and answer generation."""

from litellm import completion
from openai import OpenAI
from query_rewriting import rewrite_query
from reranking import merge_chunks, rerank
from retrieval import fetch_context_unranked
from utils.models import Result
from utils.prompts import RAG_SYSTEM_PROMPT


def fetch_context(
    original_question: str,
    history: list[dict],
    collection,
    openai_client: OpenAI,
    config: dict,
) -> list[Result]:
    """
    Full retrieval pipeline: optionally rewrite query, fetch (original + rewritten
    if rewriting enabled), merge, optionally rerank, and return top final_k chunks.

    Config flags:
      use_query_rewriting: if True, rewrite query and do dual retrieval (default: True)
      use_reranking: if True, LLM-rerank chunks before returning (default: True)
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
    """Build the message list for the LLM: system prompt with context + history + question."""
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


def answer_question(
    question: str,
    history: list[dict] | None = None,
    collection=None,
    openai_client: OpenAI | None = None,
    config: dict | None = None,
) -> tuple[str, list[Result]]:
    """
    Answer a question using the advanced RAG pipeline.

    Returns (answer_text, retrieved_chunks).
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
    response = completion(model=model, messages=messages)
    return response.choices[0].message.content, chunks

"""ChromaDB retrieval and semantic search for the RAG pipeline."""

import chromadb
from openai import OpenAI

from utils.models import Result


def get_chroma_collection(collection_name: str, api_key: str, tenant: str, database: str):
    """Connect to ChromaDB Cloud and return the specified collection."""
    chroma = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )
    return chroma.get_collection(collection_name)


def fetch_context_unranked(
    question: str,
    collection,
    openai_client: OpenAI,
    embedding_model: str,
    n_results: int,
) -> list[Result]:
    """
    Embed the question and retrieve the top-n similar chunks from ChromaDB.

    Returns a list of Result objects (page_content + metadata).
    """
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=[question],
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks

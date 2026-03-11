"""
ChromaDB retrieval — embed query and fetch similar chunks.

Uses OpenAI embeddings (not ChromaDB's built-in) to match the ingestion pipeline.
"""

from chromadb import PersistentClient
from openai import OpenAI

from utils.models import Result


def get_chroma_collection(db_path: str, collection_name: str):
    """
    Connect to ChromaDB and return the named collection.

    db_path: Path to the ChromaDB data directory (e.g. backend/data/vector_db).
    """
    chroma = PersistentClient(path=db_path)
    return chroma.get_collection(collection_name)


def fetch_context_unranked(
    question: str,
    collection,
    openai_client: OpenAI,
    embedding_model: str,
    n_results: int,
) -> list[Result]:
    """
    Semantic search: embed question, query ChromaDB, return top-n chunks.

    Uses OpenAI embeddings (must match ingestion model). ChromaDB returns
    documents and metadatas, which we wrap as Result objects.
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

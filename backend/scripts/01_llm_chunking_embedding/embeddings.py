"""Embedding generation and ChromaDB storage."""

from chromadb import PersistentClient
from openai import OpenAI

from utils.models import Result


def create_embeddings(
    chunks: list[Result],
    db_path: str,
    collection_name: str,
    embedding_model: str,
) -> None:
    """
    Generate embeddings for chunks and store them in ChromaDB.

    Replaces any existing collection with the same name. Uses the
    OpenAI embeddings API (via the configured client).

    Args:
        chunks: List of Result objects with page_content and metadata.
        db_path: Path to the ChromaDB persistent store directory.
        collection_name: Name of the collection to create or replace.
        embedding_model: Embedding model identifier (e.g. text-embedding-3-large).
    """
    chroma = PersistentClient(path=db_path)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    client = OpenAI()
    emb = client.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

"""Embedding generation and ChromaDB storage."""

import chromadb
from openai import OpenAI
from utils.models import Result


def create_embeddings(
    chunks: list[Result],
    collection_name: str,
    embedding_model: str,
    chroma_api_key: str,
    chroma_tenant: str,
    chroma_database: str,
) -> None:
    """
    Generate embeddings for chunks and store them in ChromaDB Cloud.

    Replaces any existing collection with the same name. Uses the
    OpenAI embeddings API (via the configured client).

    Args:
        chunks: List of Result objects with page_content and metadata.
        collection_name: Name of the Chroma collection to create or replace.
        embedding_model: Embedding model identifier (e.g. text-embedding-3-large).
        chroma_api_key: Chroma Cloud API key.
        chroma_tenant: Chroma Cloud tenant identifier.
        chroma_database: Chroma Cloud database name (e.g. madetech_handbook).
    """
    chroma = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    client = OpenAI()
    emb = client.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    batch_size = 250
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=texts[start:end],
            metadatas=metas[start:end],
        )
        print(f"  Added batch {start // batch_size + 1} ({end if end < len(ids) else len(ids)}/{len(ids)} items)")

    print(f"Vectorstore created with {collection.count()} documents")

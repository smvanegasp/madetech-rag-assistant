"""
LLM chunking and embedding pipeline.

Loads handbook documents from the Made Tech handbook, splits them into semantic
chunks via an LLM (with headlines and summaries), generates embeddings, and
stores them in ChromaDB for RAG retrieval.

Configuration is read from config.yaml in this script's directory. Use
document_limit to cap the number of documents (e.g. for testing).

Run from repo root:

    python -m experiments.scripts.01_llm_chunking_embedding.main
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Path setup — must happen before local imports
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent.parent
REPO_ROOT = EXPERIMENTS_PATH.parent
BACKEND_PATH = REPO_ROOT / "backend"

if str(EXPERIMENTS_PATH) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_PATH))

from chunking import parallel_chunk_generation  # noqa: E402
from config import load_config  # noqa: E402
from embeddings import create_embeddings  # noqa: E402
from utils.handbook_loader import load_handbook_documents  # noqa: E402

# Load env
env_path = BACKEND_PATH / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[OK] Loaded environment from {env_path}")
else:
    load_dotenv(override=True)
    print("[WARNING] Backend .env not found, using default environment")


def main() -> None:
    """
    Run the full pipeline: load docs, chunk via LLM, embed, and store in ChromaDB.

    Reads config from config.yaml, loads handbook documents (optionally limited
    by document_limit), generates semantic chunks per document, creates
    embeddings, and persists them to ChromaDB.
    """
    config = load_config(script_dir=SCRIPT_DIR)
    handbook_path = REPO_ROOT / config.get("handbook_path")
    document_limit = config.get("document_limit")
    average_chunk_size = config.get("average_chunk_size")
    chunking_model = config.get("chunking_model")
    max_tokens = config.get("max_tokens")
    chunking_workers = config.get("chunking_workers", 1)
    vector_db = config.get("vector_db")
    embedding_model = config.get("embedding_model")

    documents = load_handbook_documents(handbook_path)
    if document_limit is not None:
        documents = documents[:document_limit]
    print(f"Loaded {len(documents)} documents from {handbook_path}")

    chunks = parallel_chunk_generation(
        documents,
        model=chunking_model,
        max_tokens=max_tokens,
        average_chunk_size=average_chunk_size,
        workers=chunking_workers,
    )
    print(f"Generated {len(chunks)} chunks")

    db_path = SCRIPT_DIR / vector_db.get("path", "preprocessed_db")
    create_embeddings(
        chunks,
        db_path=str(db_path),
        collection_name=vector_db.get("collection_name", "docs"),
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    main()

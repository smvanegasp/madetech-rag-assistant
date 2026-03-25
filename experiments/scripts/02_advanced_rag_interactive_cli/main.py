"""
Advanced RAG pipeline.

Runs retrieval-augmented generation using the ChromaDB from 01_llm_chunking_embedding.
Features: query rewriting, dual retrieval (original + rewritten), LLM reranking, and answer generation.

Configuration is read from config.yaml in this script's directory.

Run from repo root:

    python -m experiments.scripts.02_advanced_rag.main
    python -m experiments.scripts.02_advanced_rag.main "What cycling benefits do I have?"
"""

import os
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import load_config  # noqa: E402
from openai import OpenAI  # noqa: E402
from rag import answer_question  # noqa: E402
from retrieval import get_chroma_collection  # noqa: E402

# Load env
env_path = BACKEND_PATH / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[OK] Loaded environment from {env_path}")
else:
    load_dotenv(override=True)
    print("[WARNING] Backend .env not found, using default environment")


def main() -> None:
    """Run the RAG pipeline: load config, connect to ChromaDB Cloud, answer question(s)."""
    config = load_config(script_dir=SCRIPT_DIR)
    vector_db = config.get("vector_db") or {}
    collection_name = vector_db.get("collection_name", "docs")
    chroma_database = vector_db.get("database")

    chroma_api_key = os.getenv("CHROMA_API_KEY")
    if not chroma_api_key:
        raise ValueError("CHROMA_API_KEY environment variable not set")
    chroma_tenant = os.getenv("TENANT_CHROMA")
    if not chroma_tenant:
        raise ValueError("TENANT_CHROMA environment variable not set")

    collection = get_chroma_collection(collection_name, chroma_api_key, chroma_tenant, chroma_database)
    print(f"Collection '{collection_name}' has {collection.count()} chunks", flush=True)
    print(
        "\nType a question and press Enter. Type 'quit' or 'exit' to stop.\n",
        flush=True,
    )

    openai_client = OpenAI()
    history: list[dict] = []

    while True:
        try:
            print("You: ", end="", flush=True)
            question = input().strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        answer, chunks = answer_question(
            question,
            history=history,
            collection=collection,
            openai_client=openai_client,
            config=config,
        )
        print("\n--- Answer ---\n", flush=True)
        print(answer, flush=True)
        print(flush=True)

        # Keep conversation history for follow-up context
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

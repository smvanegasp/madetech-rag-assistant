"""
Configuration loader for the RAG pipeline.

Loads config.yaml from the backend root (or RAG_CONFIG_PATH env).
Resolves relative paths and merges approach flags for the pipeline.
"""

import os
from pathlib import Path

import yaml

# Backend root: parent of the src/ directory
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load RAG pipeline configuration from a YAML file.

    Args:
        config_path: Path to the config file. If None, uses RAG_CONFIG_PATH
            env var or backend/config.yaml.

    Returns:
        Parsed configuration with resolved paths. Merges approach flags
        (use_query_rewriting, use_reranking) into top-level for pipeline use.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = config_path
    if path is None:
        env_path = os.getenv("RAG_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
        else:
            path = _BACKEND_ROOT / "config.yaml"

    path = Path(path)
    if not path.exists():
        return _default_config()

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return _default_config()

    # Resolve vector_db path relative to backend root
    vector_db = config.get("vector_db") or {}
    db_path = vector_db.get("path", "data/vector_db")
    if not Path(db_path).is_absolute():
        db_path = str((_BACKEND_ROOT / db_path).resolve())
    config.setdefault("vector_db", {})
    config["vector_db"]["path"] = db_path
    config["vector_db"].setdefault("collection_name", "docs")

    # Merge approach flags into top level for pipeline
    approach = config.get("approach") or {}
    config["use_query_rewriting"] = approach.get("use_query_rewriting", False)
    config["use_reranking"] = approach.get("use_reranking", False)

    # Ensure retrieval, embedding_model, model have defaults
    config.setdefault("retrieval", {})
    config["retrieval"].setdefault("retrieval_k", 20)
    config["retrieval"].setdefault("final_k", 10)
    config.setdefault("embedding_model", "text-embedding-3-large")
    config.setdefault("model", "groq/openai/gpt-oss-20b")

    return config


def _default_config() -> dict:
    """
    Return defaults when config.yaml is missing.

    Uses basic_rag (no rewriting, no reranking) and backend/data/vector_db.
    """
    vector_db_path = (_BACKEND_ROOT / "data" / "vector_db").resolve()
    return {
        "vector_db": {"path": str(vector_db_path), "collection_name": "docs"},
        "embedding_model": "text-embedding-3-large",
        "retrieval": {"retrieval_k": 20, "final_k": 10},
        "model": "groq/openai/gpt-oss-20b",
        "use_query_rewriting": False,
        "use_reranking": False,
    }

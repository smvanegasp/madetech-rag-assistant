"""Configuration for multi-source questions generation."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """Configuration loaded from config.yaml."""

    # How many document groups to randomly sample for evaluation.
    n_groups: int

    # Random seed used when sampling groups — keeps runs reproducible.
    seed: int

    # Number of QA pairs generated per sampled document group.
    questions_per_group: int

    # Quality cutoff: a QA record is kept in the filtered output only if
    # every critique dimension (relevance, standalone, groundedness) scores
    # >= this value.
    min_filter_score: int

    # LiteLLM model string used for both QA generation and critiquing.
    # Example: "groq/openai/gpt-oss-20b"
    model: str

    # Path to the handbook documents folder, relative to the project root.
    handbook_path: str

    # Path to the backend .env file that holds API keys, relative to
    # the project root.
    env_path: str

    # OpenAI embedding model used to compute document similarity vectors.
    embedding_model: str

    # Path to the embeddings cache file, relative to the script directory.
    # Loaded on re-runs to avoid re-computing expensive embeddings.
    embeddings_cache_path: str

    # Top-K most similar neighbours to consider for each anchor document.
    similarity_k: int

    # Minimum cosine similarity threshold: neighbours below this value are
    # excluded from the document group.
    min_similarity: float

    # Probability of using 2 documents total (anchor + 1 extra).
    # With probability (1 - two_docs_ratio), 3 documents are used instead.
    two_docs_ratio: float

    # Maximum output tokens for each QA generation LLM call.
    max_tokens: int

    def handbook_dir(self, project_root: Path) -> Path:
        """Resolve handbook directory relative to project root."""
        return project_root / self.handbook_path

    def env_file(self, project_root: Path) -> Path:
        """Resolve .env file path relative to project root."""
        return project_root / self.env_path

    def embeddings_cache_file(self, script_dir: Path) -> Path:
        """Resolve embeddings cache file path relative to the script directory."""
        return script_dir / self.embeddings_cache_path


def load_config(config_path: Path) -> Config:
    """Load configuration from a YAML file, falling back to sensible defaults."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(
        n_groups=data.get("n_groups", 10),
        seed=data.get("seed", 42),
        questions_per_group=data.get("questions_per_group", 5),
        min_filter_score=data.get("min_filter_score", 4),
        model=data.get("model", "groq/openai/gpt-oss-20b"),
        handbook_path=data.get("handbook_path", "backend/data/handbook"),
        env_path=data.get("env_path", "backend/.env"),
        embedding_model=data.get("embedding_model", "text-embedding-3-large"),
        embeddings_cache_path=data.get("embeddings_cache_path", "embeddings_cache.npy"),
        similarity_k=data.get("similarity_k", 7),
        min_similarity=data.get("min_similarity", 0.3),
        two_docs_ratio=data.get("two_docs_ratio", 0.8),
        max_tokens=data.get("max_tokens", 2048),
    )

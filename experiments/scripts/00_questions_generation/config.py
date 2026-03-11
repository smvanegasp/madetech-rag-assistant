"""Configuration for unified questions generation (single-source and multi-source)."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetSplitConfig:
    """Proportions for splitting the quality-filtered questions into validation and test sets."""

    # Fraction of filtered questions assigned to the validation set.
    validation_ratio: float

    # Fraction of filtered questions assigned to the test set.
    # validation_ratio + test_ratio must equal 1.0.
    test_ratio: float


@dataclass
class SingleSourceConfig:
    """Parameters specific to the single-source generation pipeline."""

    # How many handbook documents to randomly sample for evaluation.
    n_documents: int

    # Number of QA pairs generated per sampled document.
    questions_per_document: int


@dataclass
class MultiSourceConfig:
    """Parameters specific to the multi-source generation pipeline."""

    # How many document groups to randomly sample for evaluation.
    n_groups: int

    # Number of QA pairs generated per sampled document group.
    questions_per_group: int

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

    def embeddings_cache_file(self, script_dir: Path) -> Path:
        """Resolve embeddings cache file path relative to the script directory."""
        return script_dir / self.embeddings_cache_path


@dataclass
class Config:
    """Unified configuration loaded from config.yaml."""

    # Random seed used when sampling documents or groups — keeps runs reproducible.
    seed: int

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

    # Maximum number of tokens the model may produce in a single QA generation
    # call. Keeping this bounded avoids runaway token usage on long documents.
    max_tokens: int

    # Pipeline-specific configurations.
    single_source: SingleSourceConfig
    multi_source: MultiSourceConfig
    dataset_split: DatasetSplitConfig

    def handbook_dir(self, project_root: Path) -> Path:
        """Resolve handbook directory relative to project root."""
        return project_root / self.handbook_path

    def env_file(self, project_root: Path) -> Path:
        """Resolve .env file path relative to project root."""
        return project_root / self.env_path


def load_config(config_path: Path) -> Config:
    """Load unified configuration from a YAML file, falling back to sensible defaults."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    single = data.get("single_source", {})
    multi = data.get("multi_source", {})
    split = data.get("dataset_split", {})

    return Config(
        seed=data.get("seed", 42),
        min_filter_score=data.get("min_filter_score", 4),
        model=data.get("model", "groq/openai/gpt-oss-20b"),
        handbook_path=data.get("handbook_path", "backend/data/handbook"),
        env_path=data.get("env_path", "backend/.env"),
        max_tokens=data.get("max_tokens", 2048),
        single_source=SingleSourceConfig(
            n_documents=single.get("n_documents", 10),
            questions_per_document=single.get("questions_per_document", 3),
        ),
        multi_source=MultiSourceConfig(
            n_groups=multi.get("n_groups", 10),
            questions_per_group=multi.get("questions_per_group", 5),
            embedding_model=multi.get("embedding_model", "text-embedding-3-large"),
            embeddings_cache_path=multi.get("embeddings_cache_path", "embeddings_cache.npy"),
            similarity_k=multi.get("similarity_k", 7),
            min_similarity=multi.get("min_similarity", 0.3),
            two_docs_ratio=multi.get("two_docs_ratio", 0.8),
        ),
        dataset_split=DatasetSplitConfig(
            validation_ratio=split.get("validation_ratio", 0.7),
            test_ratio=split.get("test_ratio", 0.3),
        ),
    )

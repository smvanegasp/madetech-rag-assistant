"""Configuration for single-source questions generation."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """Configuration loaded from config.yaml."""

    # How many handbook documents to randomly sample for evaluation.
    n_documents: int

    # Random seed used when sampling documents — keeps runs reproducible.
    seed: int

    # Number of QA pairs generated per sampled document.
    questions_per_document: int

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

    def handbook_dir(self, project_root: Path) -> Path:
        """Resolve handbook directory relative to project root."""
        return project_root / self.handbook_path

    def env_file(self, project_root: Path) -> Path:
        """Resolve .env file path relative to project root."""
        return project_root / self.env_path


def load_config(config_path: Path) -> Config:
    """Load configuration from a YAML file, falling back to sensible defaults."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(
        n_documents=data.get("n_documents", 10),
        seed=data.get("seed", 42),
        questions_per_document=data.get("questions_per_document", 3),
        min_filter_score=data.get("min_filter_score", 4),
        model=data.get("model", "groq/openai/gpt-oss-20b"),
        handbook_path=data.get("handbook_path", "backend/data/handbook"),
        env_path=data.get("env_path", "backend/.env"),
    )

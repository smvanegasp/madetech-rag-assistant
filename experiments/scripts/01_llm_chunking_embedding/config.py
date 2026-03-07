"""Configuration loading for the LLM chunking and embedding pipeline."""

from pathlib import Path

import yaml


def load_config(config_path: Path | None = None, script_dir: Path | None = None) -> dict:
    """
    Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the config file. If None, uses config.yaml
            in the script directory.
        script_dir: Directory containing the script (used when config_path is None).
            Required when config_path is None.

    Returns:
        Parsed configuration as a dictionary. Use .get() for optional keys.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If neither config_path nor script_dir is provided when needed.
    """
    path = config_path
    if path is None:
        if script_dir is None:
            raise ValueError("Either config_path or script_dir must be provided")
        path = script_dir / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

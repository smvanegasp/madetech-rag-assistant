"""Entry point for unified eval question generation (single-source and multi-source)."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports so that sibling modules
# (config, generator, export, plots) and shared utils are findable on sys.path.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up three levels: 00_questions_generation -> scripts -> experiments -> repo root
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent.parent  # experiments/
BACKEND_PATH = PROJECT_ROOT / "backend"

# Add script dir, experiments/ and backend/ so that "utils.*" imports resolve correctly.
for p in [str(SCRIPT_DIR), str(EXPERIMENTS_PATH), str(BACKEND_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Local imports — script is always run directly (`python main.py`), so bare
# imports are used throughout. The path manipulation above makes them findable.
from config import load_config  # noqa: E402
from export import (  # noqa: E402
    build_eval_dataframe,
    export_eval_markdown,
    save_csv,
    save_eval_jsonl,
    save_filtered_eval_jsonl,
)
from generator import (  # noqa: E402
    build_document_groups,
    compute_or_load_embeddings,
    generate_multi_source_eval_dataset,
    generate_single_source_eval_dataset,
)
from plots import (  # noqa: E402
    plot_before_after_bar,
    plot_category_before_after_bar,
    plot_scores_distribution,
)
from utils.handbook_loader import load_handbook_documents  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_file: Path) -> logging.Logger:
    """
    Configure the root logger with two handlers:
      - Console (StreamHandler): WARNING and above, concise format.
      - File (FileHandler):      DEBUG and above, full format including
                                 timestamp, level, and module name.
                                 Written to log_file inside the logs dir.

    Returns the root logger so main() can hold a reference to it.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console_fmt = logging.Formatter("%(levelname)-8s | %(message)s")
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — WARNING+ only; tqdm progress bars provide the visual feedback
    # so routine INFO messages would just interleave with the bars.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_fmt)

    # File — DEBUG+ so every embedding call, group build, LLM call, and
    # critique score is captured for post-run inspection.
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # LiteLLM is very chatty at DEBUG level (HTTP requests, token counts, retries).
    # Restrict it to WARNING so its noise doesn't pollute the log file.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    return root


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_and_plot(eval_records, output_dir: Path, min_filter_score: int) -> None:
    """Save all output artefacts and generate analysis plots for one pipeline phase."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_eval_jsonl(eval_records, output_dir / "eval_questions.jsonl")

    filtered_records = save_filtered_eval_jsonl(
        eval_records,
        output_dir / "eval_questions_filtered.jsonl",
        min_score=min_filter_score,
    )

    df_all = build_eval_dataframe(eval_records)
    save_csv(df_all, output_dir / "eval_questions.csv")
    export_eval_markdown(eval_records, output_dir / "eval_questions.md")

    df_filtered = build_eval_dataframe(filtered_records)

    plot_before_after_bar(
        df_all,
        df_filtered,
        figures_dir / "scores_before_after_bar.png",
        quality_cutoff=min_filter_score,
    )
    plot_category_before_after_bar(
        df_all,
        df_filtered,
        figures_dir / "scores_by_category_before_after.png",
        quality_cutoff=min_filter_score,
    )
    plot_scores_distribution(
        df_all,
        figures_dir / "scores_distribution.png",
        quality_cutoff=min_filter_score,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Config ---
    config_path = SCRIPT_DIR / "config.yaml"
    config = load_config(config_path)

    # --- Output and log directories ---
    # Both pipeline phases share the same timestamp so their outputs are
    # co-located by run under output/single/{timestamp}/ and output/multi/{timestamp}/.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    single_output_dir = SCRIPT_DIR / "output" / "single" / timestamp
    multi_output_dir = SCRIPT_DIR / "output" / "multi" / timestamp

    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{timestamp}.log"

    logger = setup_logging(log_file)
    logger.debug("Loaded config from %s | values: %s", config_path, config)
    logger.debug("Single-source output: %s", single_output_dir)
    logger.debug("Multi-source output: %s", multi_output_dir)
    logger.debug("Log file: %s", log_file)

    # --- Environment variables ---
    # Prefer the backend .env for API keys; fall back to whatever dotenv finds.
    env_file = config.env_file(PROJECT_ROOT)
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.debug("Loaded environment variables from %s", env_file)
    else:
        load_dotenv(override=True)
        logger.warning(
            "Backend .env not found at %s — using default environment", env_file
        )

    # --- Handbook documents ---
    # Both pipelines share the same document corpus; loaded once here.
    handbook_dir = config.handbook_dir(PROJECT_ROOT)
    documents = load_handbook_documents(handbook_dir)
    logger.debug("Loaded %d handbook documents from %s", len(documents), handbook_dir)

    # =========================================================================
    # Phase 1: Single-source questions
    # =========================================================================

    logger.debug(
        "Starting single-source phase: n_documents=%d, questions_per_document=%d",
        config.single_source.n_documents,
        config.single_source.questions_per_document,
    )

    single_records = generate_single_source_eval_dataset(
        documents=documents,
        cfg=config.single_source,
        seed=config.seed,
        model=config.model,
        max_tokens=config.max_tokens,
    )
    logger.debug(
        "Single-source phase produced %d eval records", len(single_records)
    )

    single_output_dir.mkdir(parents=True, exist_ok=True)
    _save_and_plot(single_records, single_output_dir, config.min_filter_score)
    logger.debug("Single-source outputs saved to %s", single_output_dir)

    # =========================================================================
    # Phase 2: Multi-source questions
    # =========================================================================

    # Embeddings are cached in a .npy file next to the script so subsequent
    # runs skip the OpenAI call entirely.
    cache_path = config.multi_source.embeddings_cache_file(SCRIPT_DIR)
    vectors = compute_or_load_embeddings(
        documents,
        cache_path=cache_path,
        embedding_model=config.multi_source.embedding_model,
    )
    logger.debug(
        "Embeddings ready: shape=%s, cache=%s", vectors.shape, cache_path
    )

    # Each group contains an anchor document plus its most similar neighbours.
    doc_groups = build_document_groups(
        documents,
        vectors,
        k=config.multi_source.similarity_k,
        min_sim=config.multi_source.min_similarity,
    )
    logger.debug(
        "Built %d document groups (K=%d, min_sim=%.2f)",
        len(doc_groups),
        config.multi_source.similarity_k,
        config.multi_source.min_similarity,
    )

    logger.debug(
        "Starting multi-source phase: n_groups=%d, questions_per_group=%d",
        config.multi_source.n_groups,
        config.multi_source.questions_per_group,
    )

    multi_records = generate_multi_source_eval_dataset(
        documents=documents,
        doc_groups=doc_groups,
        cfg=config.multi_source,
        seed=config.seed,
        model=config.model,
        max_tokens=config.max_tokens,
    )
    logger.debug(
        "Multi-source phase produced %d eval records", len(multi_records)
    )

    multi_output_dir.mkdir(parents=True, exist_ok=True)
    _save_and_plot(multi_records, multi_output_dir, config.min_filter_score)
    logger.debug("Multi-source outputs saved to %s", multi_output_dir)

    logger.debug(
        "Run complete. Single: %d records | Multi: %d records | timestamp: %s",
        len(single_records),
        len(multi_records),
        timestamp,
    )


if __name__ == "__main__":
    main()

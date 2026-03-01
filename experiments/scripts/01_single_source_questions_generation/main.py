"""Entry point for single-source eval question generation."""

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
# Navigate up three levels: 01_single_source_questions_generation -> evaluation
# -> research -> repo root
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent.parent  # research/
BACKEND_PATH = PROJECT_ROOT / "backend"

# Add research/ and backend/ so that "utils.*" imports resolve correctly.
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
from generator import generate_eval_dataset  # noqa: E402
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
      - Console (StreamHandler): INFO and above, concise format.
      - File (FileHandler):      DEBUG and above, full format including
                                 timestamp, level, and module name.
                                 Written to log_file inside the output dir.

    Returns the root logger so main() can hold a reference to it.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter further

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

    # File — DEBUG+ so every LLM call, doc sample, and record detail is captured
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
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Config ---
    config_path = SCRIPT_DIR / "config.yaml"
    config = load_config(config_path)

    # --- Output and log directories ---
    # Each run gets its own timestamped output folder so artefacts never overwrite
    # each other. Logs go to a shared logs/ folder, one file per run, so they
    # survive independently of the output folder.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = SCRIPT_DIR / "output" / timestamp
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{timestamp}.log"

    # Logging is started before any work so the full run is captured.
    logger = setup_logging(log_file)
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Log file: %s", log_file)
    logger.debug("Loaded config from %s | values: %s", config_path, config)

    # --- Environment variables ---
    # Prefer the backend .env for API keys; fall back to whatever dotenv finds.
    env_file = config.env_file(PROJECT_ROOT)
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.debug("Loaded environment variables from %s", env_file)
    else:
        load_dotenv(override=True)
        # WARNING is shown on console so the user is alerted to a potential
        # missing API key before any LLM calls are made.
        logger.warning(
            "Backend .env not found at %s — using default environment", env_file
        )

    # --- Handbook documents ---
    handbook_dir = config.handbook_dir(PROJECT_ROOT)
    documents = load_handbook_documents(handbook_dir)
    logger.debug("Loaded %d handbook documents from %s", len(documents), handbook_dir)

    # --- QA generation + critique ---
    # Samples n_docs documents, generates questions_per_doc QA pairs each,
    # then critiques every question on three dimensions (relevance, standalone,
    # groundedness). This is the most time-consuming step; tqdm bars show progress.
    eval_records = generate_eval_dataset(
        documents,
        n_docs=config.n_documents,
        seed=config.seed,
        questions_per_doc=config.questions_per_document,
        model=config.model,
    )
    logger.debug(
        "Generated %d eval records from %d sampled documents",
        len(eval_records),
        config.n_documents,
    )

    # --- Save outputs ---
    save_eval_jsonl(eval_records, output_dir / "eval_questions.jsonl")

    filtered_records = save_filtered_eval_jsonl(
        eval_records,
        output_dir / "eval_questions_filtered.jsonl",
        min_score=config.min_filter_score,
    )

    df_all = build_eval_dataframe(eval_records)
    save_csv(df_all, output_dir / "eval_questions.csv")

    export_eval_markdown(eval_records, output_dir / "eval_questions.md")

    # --- Plots ---
    # Build a separate filtered DataFrame for the before/after comparisons.
    df_filtered = build_eval_dataframe(filtered_records)

    plot_before_after_bar(
        df_all,
        df_filtered,
        figures_dir / "scores_before_after_bar.png",
        quality_cutoff=config.min_filter_score,
    )
    plot_category_before_after_bar(
        df_all,
        df_filtered,
        figures_dir / "scores_by_category_before_after.png",
        quality_cutoff=config.min_filter_score,
    )
    plot_scores_distribution(
        df_all,
        figures_dir / "scores_distribution.png",
        quality_cutoff=config.min_filter_score,
    )

    logger.debug("All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()

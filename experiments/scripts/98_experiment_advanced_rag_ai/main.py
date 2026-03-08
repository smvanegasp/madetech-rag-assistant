"""
Advanced RAG experiment runner.

Runs the four experiment variants (basic RAG, +reranking, +rewriting, +both)
against questions from 00_questions_generation output. Uses the latest
timestamp folder by default.

Configuration is read from config.yaml in this script's directory.

Run from repo root:

    python -m experiments.scripts.03_experiment_advanced_rag.main
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Path setup — must happen before local imports
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent.parent
REPO_ROOT = EXPERIMENTS_PATH.parent
BACKEND_PATH = REPO_ROOT / "backend"
ADVANCED_RAG_SCRIPT = SCRIPT_DIR.parent / "02_advanced_rag"

for p in [str(EXPERIMENTS_PATH), str(SCRIPT_DIR), str(ADVANCED_RAG_SCRIPT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from openai import OpenAI  # noqa: E402
from rag import answer_question  # noqa: E402
from retrieval import get_chroma_collection  # noqa: E402

# Load env
env_path = BACKEND_PATH / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)


def _load_experiment_config() -> dict:
    """Load experiment config from config.yaml."""
    config_path = SCRIPT_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_latest_questions_dir(config: dict) -> Path:
    """Resolve the questions directory (latest timestamp or explicit)."""
    questions_cfg = config.get("questions", {})
    source = Path(questions_cfg.get("source", "../00_questions_generation/output"))
    source = (SCRIPT_DIR / source).resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Questions source does not exist: {source}. "
            "Run 00_questions_generation first."
        )

    if questions_cfg.get("use_latest", True) and not questions_cfg.get("timestamp"):
        # Find latest timestamp folder (format: YYYY-MM-DD_HH-MM-SS)
        subdirs = [d for d in source.iterdir() if d.is_dir()]
        timestamp_dirs = [d for d in subdirs if d.name.count("-") >= 2 and "_" in d.name]
        if not timestamp_dirs:
            raise FileNotFoundError(
                f"No timestamp folders found in {source}. "
                "Run 00_questions_generation first."
            )
        latest = max(timestamp_dirs, key=lambda d: d.name)
        return latest

    timestamp = questions_cfg.get("timestamp")
    if timestamp:
        return source / timestamp
    return source


def _load_questions(questions_dir: Path, config: dict) -> list[dict]:
    """Load questions from JSONL files (test.jsonl, validation.jsonl)."""
    files = config.get("questions", {}).get("files", ["test.jsonl"])
    records = []
    seen = set()
    for filename in files:
        path = questions_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                q = rec.get("question", "")
                if q and q not in seen:
                    seen.add(q)
                    records.append(rec)
    return records


def main() -> None:
    """Run all experiment variants and save results."""
    exp_config = _load_experiment_config()
    questions_dir = _get_latest_questions_dir(exp_config)
    questions = _load_questions(questions_dir, exp_config)

    if not questions:
        print("[ERROR] No questions loaded. Check config.questions.files and path.")
        sys.exit(1)

    print(f"[OK] Loaded {len(questions)} questions from {questions_dir}")

    # Base RAG config
    base = exp_config.get("base_config", {})
    vector_db = base.get("vector_db", {})
    db_path_str = vector_db.get("path", "../01_llm_chunking_embedding/output/preprocessed_db")
    db_path = (SCRIPT_DIR / db_path_str).resolve()
    collection_name = vector_db.get("collection_name", "docs")

    if not db_path.exists():
        print(f"[ERROR] ChromaDB path does not exist: {db_path}")
        print("Run 01_llm_chunking_embedding first.")
        sys.exit(1)

    collection = get_chroma_collection(str(db_path), collection_name)
    print(f"[OK] ChromaDB collection '{collection_name}' has {collection.count()} chunks")

    openai_client = OpenAI()
    experiments = exp_config.get("experiments", [])

    # Output directory for this run
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = SCRIPT_DIR / "output" / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {output_dir}")

    for exp in experiments:
        name = exp.get("name", "unnamed")
        desc = exp.get("description", "")
        print(f"\n--- Experiment: {name} ({desc}) ---")

        # Merge base config with experiment overrides
        config = {**base, "use_query_rewriting": exp.get("use_query_rewriting", False),
                  "use_reranking": exp.get("use_reranking", False)}

        results = []
        for i, qa in enumerate(questions):
            question = qa.get("question", "")
            if not question:
                continue
            print(f"  [{i + 1}/{len(questions)}] {question[:60]}...", end=" ", flush=True)
            try:
                answer, _ = answer_question(
                    question,
                    history=[],
                    collection=collection,
                    openai_client=openai_client,
                    config=config,
                )
                results.append({
                    "question": question,
                    "expected_answer": qa.get("answer", ""),
                    "question_type": qa.get("question_type", ""),
                    "model_answer": answer,
                })
                print("OK")
            except Exception as e:
                print(f"FAIL: {e}")
                results.append({
                    "question": question,
                    "expected_answer": qa.get("answer", ""),
                    "question_type": qa.get("question_type", ""),
                    "model_answer": None,
                    "error": str(e),
                })

        # Save results for this experiment
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(results)} results to {out_path}")

    print(f"\n[DONE] All experiments saved to {output_dir}")


if __name__ == "__main__":
    main()

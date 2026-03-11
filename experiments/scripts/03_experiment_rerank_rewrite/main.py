"""
Rerank/Rewrite experiment runner.

Runs the four RAG variants (basic, +reranking, +rewriting, +both) against
evaluation questions and scores each answer with an LLM-as-judge (accuracy,
completeness, relevance). Uses the latest 00_questions_generation output by default.

Configuration is read from config.yaml in this script's directory.

Run from repo root:

    python -m experiments.scripts.03_experiment_rerank_rewrite.main
"""

import json
import random
import sys
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Path setup — must happen before local imports
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent.parent
REPO_ROOT = EXPERIMENTS_PATH.parent
BACKEND_PATH = REPO_ROOT / "backend"
ADVANCED_RAG_SCRIPT = SCRIPT_DIR.parent / "02_advanced_rag_interactive_cli"

for p in [str(EXPERIMENTS_PATH), str(SCRIPT_DIR), str(ADVANCED_RAG_SCRIPT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import load_config  # noqa: E402
from retrieval import get_chroma_collection  # noqa: E402

# Load env
env_path = BACKEND_PATH / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[OK] Loaded environment from {env_path}")
else:
    load_dotenv(override=True)
    print("[WARNING] Backend .env not found, using default environment")


def _get_questions_dir(config: dict) -> Path:
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
        subdirs = [d for d in source.iterdir() if d.is_dir()]
        timestamp_dirs = [
            d for d in subdirs if d.name.count("-") >= 2 and "_" in d.name
        ]
        if not timestamp_dirs:
            raise FileNotFoundError(
                f"No timestamp folders found in {source}. "
                "Run 00_questions_generation first."
            )
        return max(timestamp_dirs, key=lambda d: d.name)

    timestamp = questions_cfg.get("timestamp")
    if timestamp:
        return source / timestamp
    return source


_worker_collection = None
_worker_client = None


def _init_eval_worker(db_path_str: str, collection_name: str, env_path: str) -> None:
    """Initialize worker process with ChromaDB collection and OpenAI client."""
    global _worker_collection, _worker_client
    from openai import OpenAI

    if Path(env_path).exists():
        load_dotenv(env_path, override=True)
    from retrieval import get_chroma_collection

    _worker_collection = get_chroma_collection(db_path_str, collection_name)
    _worker_client = OpenAI()


def _process_qa(qa, exp_config: dict) -> dict:
    """Evaluate a single QA pair (runs in worker process)."""
    global _worker_collection, _worker_client
    from evaluation import evaluate_answer

    try:
        eval_result, generated_answer, _ = evaluate_answer(
            qa,
            config=exp_config,
            collection=_worker_collection,
            openai_client=_worker_client,
            model=exp_config["model"],
        )
        return {
            "question": qa.question,
            "expected_answer": qa.answer,
            "question_type": qa.question_type,
            "accuracy": eval_result.accuracy,
            "completeness": eval_result.completeness,
            "relevance": eval_result.relevance,
            "feedback": eval_result.feedback,
            "generated_answer": generated_answer,
        }
    except Exception as e:
        return {
            "question": qa.question,
            "expected_answer": qa.answer,
            "question_type": qa.question_type,
            "accuracy": None,
            "completeness": None,
            "relevance": None,
            "feedback": str(e),
            "generated_answer": None,
            "error": str(e),
        }


def _load_eval_questions(questions_dir: Path, config: dict):
    """Load QAPairWithTS from JSONL files."""
    from utils.models import QAPairWithTS

    questions_cfg = config.get("questions", {})
    files = questions_cfg.get("files", ["validation.jsonl"])
    questions = []
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
                data = json.loads(line)
                q = data.get("question", "")
                if q and q not in seen:
                    seen.add(q)
                    questions.append(QAPairWithTS(**data))
    return questions


def main() -> None:
    """Run all experiment variants with LLM-as-judge evaluation and save results."""
    config = load_config(script_dir=SCRIPT_DIR)
    load_from = config.get("load_from")
    output_base = config.get("output", "output")
    output_base_path = (SCRIPT_DIR / output_base).resolve()

    if load_from:
        load_path = (SCRIPT_DIR / load_from).resolve()
        if not load_path.exists():
            print(f"[ERROR] load_from path does not exist: {load_path}")
            sys.exit(1)
        with open(load_path / "results.json", encoding="utf-8") as f:
            all_results = json.load(f)
        df_summary = pd.read_csv(load_path / "summary.csv")
        print(f"Loaded results from {load_path}")
        print(f"\n{df_summary.to_string(index=False)}\n")
        if config.get("save_plots", True):
            from plots import save_plots

            save_plots(load_path, all_results, df_summary)
        return

    # Resolve ChromaDB
    vector_db = config.get("vector_db", {})
    db_path_str = vector_db.get("path", "../../../backend/data/vector_db")
    db_path = (SCRIPT_DIR / db_path_str).resolve()
    collection_name = vector_db.get("collection_name", "docs")

    if not db_path.exists():
        print(f"[ERROR] ChromaDB path does not exist: {db_path}")
        print("Run 01_llm_chunking_embedding first.")
        sys.exit(1)

    collection = get_chroma_collection(str(db_path), collection_name)
    print(
        f"[OK] ChromaDB collection '{collection_name}' has {collection.count()} chunks"
    )

    # Load questions
    questions_dir = _get_questions_dir(config)
    questions = _load_eval_questions(questions_dir, config)
    if not questions:
        print("[ERROR] No questions loaded. Check config.questions.files and path.")
        sys.exit(1)

    n_q = config.get("questions", {}).get("n_questions")
    seed = config.get("questions", {}).get("random_seed", 42)
    if n_q is not None:
        rng = random.Random(seed)
        questions = rng.sample(questions, min(n_q, len(questions)))
    print(f"[OK] Loaded {len(questions)} questions from {questions_dir}")

    # Build base RAG config
    retrieval_cfg = config.get("retrieval", {})
    base_config = {
        "retrieval": retrieval_cfg,
        "embedding_model": config.get("embedding_model", "text-embedding-3-large"),
        "model": config.get("model", "groq/openai/gpt-oss-20b"),
    }

    workers = config.get("workers", 1)

    experiments = config.get("experiments", [])
    all_results = {}

    for exp in tqdm(experiments, desc="Experiments"):
        name = exp.get("name", "unnamed")
        exp_config = {
            **base_config,
            "use_query_rewriting": exp.get("use_query_rewriting", False),
            "use_reranking": exp.get("use_reranking", False),
        }
        all_results[name] = []
        process_qa = partial(_process_qa, exp_config=exp_config)
        with Pool(
            processes=workers,
            initializer=_init_eval_worker,
            initargs=(str(db_path), collection_name, str(BACKEND_PATH / ".env")),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(process_qa, questions),
                total=len(questions),
                desc=name,
                leave=False,
            ):
                all_results[name].append(result)

    # Save results
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_base_path / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {results_path}")

    # Build summary
    summary_rows = []
    for name, records in all_results.items():
        acc = [r["accuracy"] for r in records if r.get("accuracy") is not None]
        comp = [r["completeness"] for r in records if r.get("completeness") is not None]
        rel = [r["relevance"] for r in records if r.get("relevance") is not None]
        mean_acc = float(np.mean(acc)) if acc else 0.0
        mean_comp = float(np.mean(comp)) if comp else 0.0
        mean_rel = float(np.mean(rel)) if rel else 0.0
        overall = (mean_acc + mean_comp + mean_rel) / 3.0
        summary_rows.append(
            {
                "experiment": name,
                "mean_accuracy": mean_acc,
                "mean_completeness": mean_comp,
                "mean_relevance": mean_rel,
                "overall_score": overall,
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print summary
    from tabulate import tabulate

    print(f"\nQuestions evaluated per experiment: {len(questions)}\n")
    print(
        tabulate(
            df_summary,
            headers="keys",
            tablefmt="github",
            showindex=False,
            floatfmt=".2f",
        )
    )
    best_row = df_summary.loc[df_summary["overall_score"].idxmax()]
    print(f"\nBest configuration: {best_row['experiment']}")
    print(
        f"  Mean accuracy: {best_row['mean_accuracy']:.2f} | completeness: {best_row['mean_completeness']:.2f} | relevance: {best_row['mean_relevance']:.2f}"
    )

    # Plots
    if config.get("save_plots", True):
        from plots import save_plots

        save_plots(output_dir, all_results, df_summary)


if __name__ == "__main__":
    main()

"""
API evaluation experiment runner.

Tests the live RAG API endpoint against evaluation question datasets and scores
each answer with an LLM-as-judge (accuracy, completeness, relevance). The API
configuration (model, reranking, rewriting) is controlled server-side via
backend/config.yaml — each run here evaluates the API as currently configured.

Configuration is read from config.yaml in this script's directory.

Run from repo root:

    python -m experiments.scripts.04_api_evaluation.main
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Path setup — must happen before local imports
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent.parent
REPO_ROOT = EXPERIMENTS_PATH.parent
BACKEND_PATH = REPO_ROOT / "backend"

for p in [str(EXPERIMENTS_PATH), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import load_config  # noqa: E402

# Load env (needed for judge_model API keys)
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
            print(f"[WARNING] Question file not found: {path}")
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


def _health_check(api_config: dict) -> bool:
    """Verify the API is reachable and healthy."""
    url = f"{api_config['base_url']}/api/health"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"[OK] API healthy — {data.get('documents_loaded', '?')} documents loaded")
        return True
    except Exception as e:
        print(f"[ERROR] API health check failed: {e}")
        return False


def _process_qa(qa, api_config: dict, judge_model: str) -> dict:
    """Evaluate a single QA pair: call API then judge the answer."""
    from evaluation import call_api, judge_answer

    base = {
        "question": qa.question,
        "expected_answer": qa.answer,
        "question_type": qa.question_type,
    }

    # Step 1: Call the live API
    try:
        generated_answer, sources, latency = call_api(qa.question, api_config)
    except Exception as e:
        return {
            **base,
            "accuracy": None, "completeness": None, "relevance": None,
            "feedback": None, "generated_answer": None,
            "latency_seconds": None, "sources": [],
            "api_error": str(e), "judge_error": None,
        }

    # Step 2: LLM-as-judge
    try:
        eval_result = judge_answer(
            qa.question,
            generated_answer,
            qa.answer,
            model=judge_model,
        )
    except Exception as e:
        return {
            **base,
            "accuracy": None, "completeness": None, "relevance": None,
            "feedback": None, "generated_answer": generated_answer,
            "latency_seconds": round(latency, 2), "sources": sources,
            "api_error": None, "judge_error": str(e),
        }

    return {
        **base,
        "accuracy": eval_result.accuracy,
        "completeness": eval_result.completeness,
        "relevance": eval_result.relevance,
        "feedback": eval_result.feedback,
        "generated_answer": generated_answer,
        "latency_seconds": round(latency, 2),
        "sources": sources,
        "api_error": None, "judge_error": None,
    }


def _build_summary(all_results: dict) -> pd.DataFrame:
    """Build a summary DataFrame from results (one row per experiment)."""
    summary_rows = []
    for name, records in all_results.items():
        total = len(records)
        acc = [r["accuracy"] for r in records if r.get("accuracy") is not None]
        comp = [r["completeness"] for r in records if r.get("completeness") is not None]
        rel = [r["relevance"] for r in records if r.get("relevance") is not None]
        lat = [r["latency_seconds"] for r in records if r.get("latency_seconds") is not None]
        api_errors = sum(1 for r in records if r.get("api_error"))
        judge_errors = sum(1 for r in records if r.get("judge_error"))
        mean_acc = float(np.mean(acc)) if acc else 0.0
        mean_comp = float(np.mean(comp)) if comp else 0.0
        mean_rel = float(np.mean(rel)) if rel else 0.0
        overall = (mean_acc + mean_comp + mean_rel) / 3.0
        mean_lat = float(np.mean(lat)) if lat else 0.0
        p50_lat = float(np.median(lat)) if lat else 0.0
        p95_lat = float(np.percentile(lat, 95)) if lat else 0.0
        summary_rows.append({
            "experiment": name,
            "total_questions": total,
            "api_failures": api_errors,
            "judge_failures": judge_errors,
            "success_rate": round((total - api_errors - judge_errors) / total, 2) if total else 0.0,
            "mean_accuracy": mean_acc,
            "mean_completeness": mean_comp,
            "mean_relevance": mean_rel,
            "overall_score": overall,
            "mean_latency_s": mean_lat,
            "p50_latency_s": p50_lat,
            "p95_latency_s": p95_lat,
        })
    return pd.DataFrame(summary_rows)


def _load_comparison_runs(compare_paths: list[str]) -> tuple[dict, pd.DataFrame]:
    """Load results and summaries from previous runs for comparison."""
    merged_results = {}
    merged_summaries = []
    for rel_path in compare_paths:
        load_path = (SCRIPT_DIR / rel_path).resolve()
        if not load_path.exists():
            print(f"[WARNING] compare_with path not found: {load_path}")
            continue
        results_file = load_path / "results.json"
        summary_file = load_path / "summary.csv"
        if not results_file.exists() or not summary_file.exists():
            print(f"[WARNING] Missing results.json or summary.csv in {load_path}")
            continue
        with open(results_file, encoding="utf-8") as f:
            prev_results = json.load(f)
        prev_summary = pd.read_csv(summary_file)
        merged_results.update(prev_results)
        merged_summaries.append(prev_summary)
        print(f"[OK] Loaded comparison run from {load_path}")
    df_merged = pd.concat(merged_summaries, ignore_index=True) if merged_summaries else pd.DataFrame()
    return merged_results, df_merged


def main() -> None:
    """Run API evaluation with LLM-as-judge and save results."""
    config = load_config(script_dir=SCRIPT_DIR)
    load_from = config.get("load_from")
    output_base = config.get("output", "output")
    output_base_path = (SCRIPT_DIR / output_base).resolve()

    # --- Load-from mode: skip evaluation, just load and (re)plot ---
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

        compare_paths = config.get("compare_with", [])
        compare_results, compare_summary = {}, pd.DataFrame()
        if compare_paths:
            compare_results, compare_summary = _load_comparison_runs(compare_paths)

        if config.get("save_plots", True):
            from plots import save_plots
            save_plots(load_path, all_results, df_summary, compare_results, compare_summary)
        return

    # --- Fresh run ---
    api_config = config.get("api", {})
    experiment_name = config.get("experiment_name", "api_run")
    judge_model = config.get("judge_model", "groq/llama-3.3-70b-versatile")
    workers = config.get("workers", 1)

    # Health check
    if not _health_check(api_config):
        print("[ERROR] API is not reachable. Start it with: uvicorn src.app:app --reload --port 9481")
        sys.exit(1)

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

    # Run evaluation
    print(f"\n[{experiment_name}] judge={judge_model}, workers={workers}")
    all_results = {experiment_name: []}

    def process(qa):
        return _process_qa(qa, api_config, judge_model)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, qa): qa for qa in questions}
        for future in tqdm(as_completed(futures), total=len(questions), desc=experiment_name):
            result = future.result()
            all_results[experiment_name].append(result)

    # Save results
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_base_path / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {results_path}")

    # Build and save summary
    df_summary = _build_summary(all_results)
    summary_path = output_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print summary
    from tabulate import tabulate

    print(f"\nQuestions evaluated: {len(questions)}\n")
    print(tabulate(df_summary, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    row = df_summary.iloc[0]
    print(f"\nExperiment: {row['experiment']}")
    print(f"  Mean accuracy: {row['mean_accuracy']:.2f} | completeness: {row['mean_completeness']:.2f} | relevance: {row['mean_relevance']:.2f}")
    print(f"  Latency — mean: {row['mean_latency_s']:.1f}s | p50: {row['p50_latency_s']:.1f}s | p95: {row['p95_latency_s']:.1f}s")

    # Load comparison runs
    compare_paths = config.get("compare_with", [])
    compare_results, compare_summary = {}, pd.DataFrame()
    if compare_paths:
        compare_results, compare_summary = _load_comparison_runs(compare_paths)

    # Plots
    if config.get("save_plots", True):
        from plots import save_plots
        save_plots(output_dir, all_results, df_summary, compare_results, compare_summary)


if __name__ == "__main__":
    main()

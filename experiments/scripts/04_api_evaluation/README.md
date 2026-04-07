# API Evaluation Experiment

Tests the live RAG API endpoint (`POST /api/chat`) against evaluation question datasets and scores each answer with an LLM-as-judge (accuracy, completeness, relevance). Supports cross-run comparison to evaluate different backend configurations.

## How It Works

Unlike `03_experiment_rerank_rewrite` which calls the RAG pipeline directly, this experiment hits the **live API** over HTTP. The API's RAG configuration (model, reranking, rewriting, keyword search) is controlled server-side via `backend/config.yaml`. Each run evaluates the API as currently configured.

**Workflow to compare configurations:**

1. Set the desired approach in `backend/config.yaml`
2. Restart the API server
3. Set a descriptive `experiment_name` in this experiment's `config.yaml`
4. Run the evaluation
5. Repeat steps 1-4 with a different configuration and name
6. Set `compare_with` to the previous output folder(s) for comparison plots

## Requirements

Install experiment dependencies (from repo root):

```bash
pip install -r experiments/requirements.txt
```

Environment variables needed in `backend/.env`:
- `GROQ_API_KEY` — judge model (and API if using Groq)
- `OPENAI_API_KEY` — judge model fallback

## Prerequisites

1. **00_questions_generation** — Run first to generate eval questions (`validation.jsonl` / `test.jsonl`).
2. **Live API** — Must be running at the configured URL.

## Usage

Start the API server (from `backend/`):

```bash
uvicorn src.app:app --reload --port 9481
```

Run the evaluation (from repo root):

```bash
python -m experiments.scripts.04_api_evaluation.main
```

To reload a previous run (view results or regenerate plots):

```yaml
# Edit config.yaml
load_from: "output/2026-04-07_12-00-00"
```

To compare multiple runs:

```yaml
experiment_name: "with_reranking"
compare_with:
  - "output/2026-04-07_12-00-00"   # api_baseline run
```

## Configuration

Edit `config.yaml` in this directory:

| Key | Description | Default |
|-----|-------------|---------|
| `api.base_url` | API server URL | `http://localhost:9481` |
| `api.endpoint` | Chat endpoint path | `/api/chat` |
| `api.timeout_seconds` | Request timeout | `120` |
| `questions.source` | Path to `00_questions_generation/output` | `../00_questions_generation/output` |
| `questions.use_latest` | Use most recent timestamp folder | `true` |
| `questions.files` | JSONL files to load (`validation.jsonl` or `test.jsonl`) | `["validation.jsonl"]` |
| `questions.n_questions` | Limit number of questions (`null` = all) | `null` |
| `questions.random_seed` | Seed for reproducible sampling | `42` |
| `judge_model` | LLM for scoring (separate from API model) | `groq/llama-3.3-70b-versatile` |
| `experiment_name` | Label for this run | `api_baseline` |
| `workers` | Parallel threads | `3` |
| `output` | Output directory | `output` |
| `load_from` | Load previous run (`null` = fresh run) | `null` |
| `save_plots` | Generate plots after run | `true` |
| `compare_with` | Previous output dirs for cross-run comparison | `[]` |

## Output

Results are written to `output/{timestamp}/`:

| File | Contents |
|------|----------|
| `results.json` | Per-question: question, expected/generated answer, accuracy, completeness, relevance, feedback, latency, sources |
| `summary.csv` | Mean accuracy, completeness, relevance, overall score, mean/p50/p95 latency, failure rates |

### Single-Run Plots

| Plot | Description |
|------|-------------|
| `summary_scores.png` | Mean scores by experiment |
| `summary_scores_median.png` | Median scores by experiment |
| `by_question_type.png` | Mean scores by question type (single vs multi-source) |
| `by_question_type_median.png` | Median scores by question type |
| `distribution_histograms.png` | Score distribution histograms |
| `distribution_ecdf.png` | Empirical CDF of scores |
| `latency_summary.png` | Mean/p50/p95 latency |
| `failure_rates.png` | API vs judge error rates |
| `latency_distribution.png` | Latency box plot |

### Cross-Run Comparison Plots (when `compare_with` is set)

| Plot | Description |
|------|-------------|
| `comparison_scores.png` | Mean scores across all runs |
| `comparison_latency.png` | Latency across all runs |
| `comparison_overall.png` | Overall score across all runs |
| `comparison_ecdf.png` | Empirical CDF across all runs |

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Loads YAML configuration |
| `config.yaml` | All configuration (API URL, questions, judge, comparison) |
| `evaluation.py` | API HTTP call + LLM-as-judge scoring |
| `main.py` | Entry point: health check, run eval, save results, generate plots |
| `plots.py` | Single-run and cross-run comparison visualizations |

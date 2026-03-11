> **Note:** This script has been superseded by [`00_questions_generation`](../00_questions_generation/README.md), which runs both the single-source and multi-source pipelines in a single execution with a unified configuration. This folder is kept for reference.

# Multi-Source Questions Generation

Generates evaluation question pairs that require combining information from **two or more** handbook documents to answer. Each question is scored on the same three dimensions as the single-source pipeline: relevance, standalone clarity, and groundedness.

## Structure

- **main.py** — Entry point: loads config, sets up logging, runs the pipeline, writes outputs
- **config.yaml** — Parameters (n_groups, seed, questions_per_group, embedding settings, similarity thresholds, quality cutoff, model, paths)
- **config.py** — Config dataclass and YAML loading
- **generator.py** — Embedding computation, document grouping, QA generation, critique, and eval dataset logic
- **export.py** — JSONL, CSV, and Markdown serialization
- **plots.py** — Bar charts and score distribution plots

## Configuration (config.yaml)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_groups` | Number of document groups to sample | 10 |
| `seed` | Random seed for group selection | 42 |
| `questions_per_group` | QA pairs generated per document group | 5 |
| `min_filter_score` | Quality cutoff: keep records where every dimension >= this | 4 |
| `model` | LLM for QA generation and critique | groq/openai/gpt-oss-20b |
| `handbook_path` | Path to handbook (relative to project root) | backend/data/handbook |
| `env_path` | Path to .env for API keys | backend/.env |
| `embedding_model` | OpenAI model used to embed documents | text-embedding-3-large |
| `embeddings_cache_path` | Cache file for embeddings (relative to script dir) | embeddings_cache.npy |
| `similarity_k` | Top-K neighbours considered per anchor document | 7 |
| `min_similarity` | Minimum cosine similarity to include a neighbour | 0.3 |
| `two_docs_ratio` | Probability of using 2 docs (vs 3) per generation call | 0.8 |
| `max_tokens` | Maximum output tokens per QA generation call | 2048 |

## How to Run

From this directory:

```bash
cd experiments/scripts/02_multi_source_questions_generation
python main.py
```

## Pipeline

```
1. Load config and set up logging + output directories
2. Load environment variables (API keys)
3. Load handbook documents
4. Compute or load document embeddings  ← cached in embeddings_cache.npy
5. Build document groups via cosine similarity
6. Sample n_groups groups and generate QA pairs from each
7. Critique every QA pair (relevance, standalone, groundedness)
8. Save JSONL / CSV / Markdown outputs
9. Generate analysis plots
```

On the first run, step 4 calls the OpenAI embeddings API and saves the result to `embeddings_cache.npy`. All subsequent runs load from the cache, so only the LLM generation and critique calls incur API cost.

## Logging

| Stream | Level | What you see |
|--------|-------|--------------|
| Console (stdout) | WARNING | Warnings and errors only (e.g. missing `.env`). Silent on a clean run. |
| File (`logs/{timestamp}.log`) | DEBUG | Everything: embedding cache hits/misses, group build stats, per-group generation, individual LLM critique calls, scores, file saves. |

Progress during the two long-running phases is shown via `tqdm` bars:

```
Generating QA pairs: 100%|████████| 10/10 [01:30<00:00,  9.0s/group]
Critiquing QA pairs: 100%|████████| 50/50 [02:15<00:00,  2.7s/pair]
```

## Output

Creates `output/{timestamp}/` with:

- `eval_questions.jsonl` — All generated records
- `eval_questions_filtered.jsonl` — Records passing quality cutoff (all dimensions >= min_filter_score)
- `eval_questions.csv` — Tabular format
- `eval_questions.md` — Human-readable Markdown
- `figures/` — `scores_before_after_bar.png`, `scores_by_category_before_after.png`, `scores_distribution.png`

Creates `logs/{timestamp}.log` with the full DEBUG-level log of the entire run.

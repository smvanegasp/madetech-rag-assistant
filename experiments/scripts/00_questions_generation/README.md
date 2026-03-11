# Questions Generation

Generates evaluation QA pairs for RAG system assessment in a single run. Both pipelines execute sequentially:

1. **Single-source** — questions grounded in one handbook document each
2. **Multi-source** — questions that require synthesising information from two or more related documents

Each question is scored on three dimensions: relevance, standalone clarity, and groundedness.

## Structure

- **main.py** — Entry point: loads config, runs both pipelines, writes outputs
- **config.yaml** — All parameters, split into shared and per-pipeline sections
- **config.py** — Typed dataclasses (`Config`, `SingleSourceConfig`, `MultiSourceConfig`) and YAML loader
- **generator.py** — All generation functions: single-source QA, embedding computation, document grouping, multi-source QA, and critique logic
- **export.py** — JSONL, CSV, and Markdown serialisation
- **plots.py** — Bar charts and score distribution plots

## Configuration (config.yaml)

### Shared parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `seed` | Random seed for document/group sampling | 42 |
| `min_filter_score` | Quality cutoff: keep records where every dimension >= this | 4 |
| `model` | LLM for QA generation and critique | groq/openai/gpt-oss-20b |
| `handbook_path` | Path to handbook (relative to project root) | backend/data/handbook |
| `env_path` | Path to .env for API keys | backend/.env |
| `max_tokens` | Maximum output tokens per QA generation call | 2048 |

### `single_source` section

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_documents` | Number of handbook documents to sample | 10 |
| `questions_per_document` | QA pairs generated per document | 3 |

### `multi_source` section

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_groups` | Number of document groups to sample | 10 |
| `questions_per_group` | QA pairs generated per document group | 5 |
| `embedding_model` | OpenAI model used to embed documents | text-embedding-3-large |
| `embeddings_cache_path` | Cache file for embeddings (relative to script dir) | embeddings_cache.npy |
| `similarity_k` | Top-K neighbours considered per anchor document | 7 |
| `min_similarity` | Minimum cosine similarity to include a neighbour | 0.3 |
| `two_docs_ratio` | Probability of using 2 docs (vs 3) per generation call | 0.8 |

## How to Run

From this directory:

```bash
cd experiments/scripts/00_questions_generation
python main.py
```

## Pipeline

```
1. Load config and set up logging + log file
2. Load environment variables (API keys)
3. Load handbook documents (shared between both phases)

── Phase 1: Single-source ──────────────────────────────────────────────────
4. Sample n_documents documents (reproducible via seed)
5. Generate questions_per_document QA pairs per document
6. Critique every QA pair (relevance, standalone, groundedness)
7. Save outputs to output/single/{timestamp}/

── Phase 2: Multi-source ───────────────────────────────────────────────────
8. Compute or load document embeddings  ← cached in embeddings_cache.npy
9. Build document groups via cosine similarity
10. Sample n_groups groups and generate questions_per_group QA pairs from each
11. Critique every QA pair (relevance, standalone, groundedness)
12. Save outputs to output/multi/{timestamp}/
```

On the first run, step 8 calls the OpenAI embeddings API and saves the result to `embeddings_cache.npy`. All subsequent runs load from the cache, so only the LLM generation and critique calls incur API cost.

Both phases share the same `timestamp` string so their outputs are co-located by run.

## Logging

| Stream | Level | What you see |
|--------|-------|--------------|
| Console (stdout) | WARNING | Warnings and errors only (e.g. missing `.env`). Silent on a clean run. |
| File (`logs/{timestamp}.log`) | DEBUG | Everything: config values, embedding cache hits/misses, group build stats, per-doc/group generation, individual LLM critique calls, scores, file saves. |

Progress during the long-running phases is shown via `tqdm` bars — that is the only console output on a clean run:

```
[single] Generating QA pairs: 100%|████████| 7/7 [00:20<00:00,  2.9s/doc]
[single] Critiquing QA pairs: 100%|████████| 21/21 [00:57<00:00,  2.7s/pair]
[multi] Generating QA pairs:  100%|████████| 7/7 [01:03<00:00,  9.0s/group]
[multi] Critiquing QA pairs:  100%|████████| 35/35 [01:35<00:00,  2.7s/pair]
```

## Output

Both pipeline phases produce the same set of artefacts under their respective subdirectory:

```
output/
├── single/
│   └── {timestamp}/
│       ├── eval_questions.jsonl           # All generated records
│       ├── eval_questions_filtered.jsonl  # Records passing quality cutoff
│       ├── eval_questions.csv             # Tabular format
│       ├── eval_questions.md              # Human-readable Markdown
│       └── figures/
│           ├── scores_before_after_bar.png
│           ├── scores_by_category_before_after.png
│           └── scores_distribution.png
└── multi/
    └── {timestamp}/
        ├── eval_questions.jsonl
        ├── eval_questions_filtered.jsonl
        ├── eval_questions.csv
        ├── eval_questions.md
        └── figures/
            ├── scores_before_after_bar.png
            ├── scores_by_category_before_after.png
            └── scores_distribution.png
```

Creates `logs/{timestamp}.log` with the full DEBUG-level log of the entire run.

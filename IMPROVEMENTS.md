# MadeTech RAG Assistant — Improvement Roadmap

---

## What's Done

### Phase 1 — Critical Fixes & Quick Wins

| Item | Status |
|------|--------|
| iPhone text box zoom bug | Fixed: `maximum-scale=1.0` + `text-base` on mobile |
| iPhone text box growing too tall | Fixed: 3-line cap with cached line metrics |
| iPad feedback button overlapping input | Fixed: header button on touch/tablet, floating on desktop only |
| AWS layout scroll | Fixed: `h-dvh` + `overflow: hidden` |
| Tablet experience verified | Manually tested on iPhone and iPad |
| Favicon | Inline SVG book icon on emerald-green square |
| Named the assistant "Nexus" | Updated everywhere: page title, header, welcome screen, modal, system prompts |
| First sample question | Updated to showcase capabilities |
| Try again button on error | Re-sends original message with error cleared from history |
| Dynamic date in system prompt | `{today}` formatted at runtime |
| Self-description ("Tell me about this app") | Answered from system prompt, not RAG |
| Repo link in system prompt | `https://github.com/smvanegasp/madetech-rag-assistant` |
| Handbook summary in system prompt | 150+ documents, 6 categories, full file index with human-readable names |
| Budget constraints verified | Groq $5/month, OpenAI $5 pre-loaded |
| Supabase pause risk | Chat works when DB is down, logging fails gracefully |

### Phase 2 — Frontend UX & RAG Investigation

| Item | Status |
|------|--------|
| Welcome popup timing | 45-second auto-dismiss confirmed |
| Inline source citations | `[n]` stripped from text, clean "Sources:" list below answers |
| FAQ section | Collapsible "What can I help with?" with 5 categories |
| Human-readable names in system prompt | "Delivery Manager" not "delivery_manager" |
| Nexus doesn't assume user identity | System prompt: "you do not know anything about the user" |
| Structural questions skip RAG | Roles/benefits/categories answered from prompt |
| Double query rewriting investigation | Notebook confirmed orchestrator already rewrites — explicit step redundant |
| Latency benchmarking | 3 models tested, basic RAG with gpt-oss-20b won. Latency + failure tracking added |

### Phase 3 — SDK Migration & Agent Architecture

| Item | Status |
|------|--------|
| OpenAI Agents SDK migration | `agent_pipeline.py` with `@function_tool`, `await Runner.run()`, `AsyncOpenAI` + `OpenAIChatCompletionsModel` (fully async) |
| Send feedback tool | `@function_tool` with validation, calls `contact_service.py` |
| Get in touch tool | Same pattern, exclusive action (never combined with searches) |
| Input guardrails | Non-English, off-topic, gibberish handled in system prompt |
| Planning tool (`plan_searches`) | Executes multiple searches at once, deduplicates chunks |
| Multiple RAG tool calls | `max_turns=6`, chunks accumulate, sources capped at 15 |
| Diverse input type handling | Routing rules for all input types in system prompt |
| BM25 hybrid search | `rank-bm25` keyword search merged with semantic search, configurable |
| Real-time SSE tool streaming | `POST /api/chat/stream` sends tool_step events live |
| Live checklist UI | Steps appear with checkmarks, spinner, staggered animation |
| "Surprise me" button | Random question from pool of 50 curated handbook questions |
| Welcome modal redesign | Problem/solution framing, creator attribution, source code link |
| Repo link in app UI | "View source code" in welcome modal |
| Concise response style | Cover key points, offer to elaborate |
| Table formatting | Max 4 columns, prefer 2 |
| Judicious tool calling | Prefer broad searches, never repeat, plan_searches then answer |

---

## What's Next — Actionable Items for Future Sessions

### 4.0 Bug Fixes (High Priority)

- [x] **[CURSOR]** **Mobile: scroll position not reset on new chat** — on iPhone/iPad, creating a new chat from an active session leaves the scroll position mid-page; the welcome screen (sample questions, FAQ) is rendered but not visible until the user scrolls up. Fixed: auto-scroll effect in `ChatArea.tsx` now scrolls to top (with `behavior: 'instant'` + `requestAnimationFrame`) when messages are empty, and to bottom when messages are present.
- [x] **[CURSOR]** **Investigate frequent retry failures** — root cause: `handleRetry` used the non-streaming endpoint (3 backend retries) while `handleSend` used streaming (zero retries), causing ~40% first-query failures. Fixed: (1) 3-attempt retry in `answer_question_agent_streamed`, (2) `handleRetry` switched to streaming, (3) **3-attempt silent frontend retry** in both `handleSend` and `handleRetry` — error message only shown after all attempts fail, (4) error messages cleared when user sends a new message, (5) `config_loader.py` bug where `use_keyword_search` was not merged from `approach` section — BM25 hybrid search was silently disabled.
- [x] **[CURSOR]** **Streaming response bleeds into a newly created chat** — if a query is in-flight (SSE stream active) and the user opens a new chat, streamed tokens/tool-step events from the previous query appear in the new chat. Fixed: guarded `onToolStep` callback with chat ID check and clear `liveToolSteps` on chat switch. In-flight queries now complete in the background and notify via the unread indicator instead of being aborted.

---

### 4.1 Content & Articles

- [ ] **[MANUAL]** Write a blog post / case study about the project. Lead with the user problem (nobody reads HR docs), then the solution. Include the problem statement from the welcome modal as a starting point.
- [ ] **[MANUAL]** Create a one-pager explaining what Nexus does, the problem it solves, and how to use it. Make the link available in the app.
- [ ] **[COLLAB]** Once articles exist, add links to the system prompt and the app UI (sidebar, welcome modal, or footer).

### 4.2 Repository & Documentation

- [ ] **[CLAUDE-CODE]** Improve the README: update with current architecture (Agents SDK, hybrid search, SSE streaming), setup instructions, screenshots, and links to the live app.
- [ ] **[COLLAB]** Set up CI/CD: GitHub Actions → build → deploy to AWS App Runner or HuggingFace. Choose the pipeline, Claude Code writes the workflow files.

### 4.3 Infrastructure

- [ ] **[MANUAL]** Evaluate if ChromaDB Cloud is still the right choice or if an alternative (MongoDB Atlas, Pinecone) would be better for scaling.
- [ ] **[MANUAL]** Check if the handbook markdown files (`backend/data/handbook/`) need updating from the latest Made Tech handbook repo.
- [ ] **[MANUAL]** Decide on a web presence strategy for the project (dedicated page, portfolio integration).

### 4.4 Deferred Items

- [ ] **[COLLAB]** Build popup tutorial / onboarding GIFs: create a 2-4 step walkthrough, then build a carousel component.
- [ ] **[COLLAB]** Add cost monitoring: log token counts per request to Supabase. Not needed while $5/month caps are in place — revisit if budget increases.
- [ ] **[COLLAB]** Add article/resource links in the app UI: blocked until articles are written.

### 4.5 Evaluation & Testing

- [ ] **[CURSOR]** **Batch evaluation script** — write a Python script (e.g. `backend/scripts/evaluate.py`) that reads a validation set (question + expected answer / expected sources), sends each question to the `/api/chat` endpoint (or calls `agent_pipeline.py` directly), and records the response. Output a summary table: question, response, latency, sources retrieved, pass/fail.
- [ ] **[CURSOR]** **Config flag toggles in evaluation** — expose CLI arguments (or a separate `eval_config.yaml`) that mirror the existing `config.yaml` pipeline flags (`use_query_rewriting`, `use_reranking`, `use_keyword_search`). This lets a single evaluation run compare different pipeline configurations side-by-side without changing production config.
- [ ] **[CURSOR]** **Scoring** — at minimum, log whether the expected source document appeared in the retrieved chunks (recall@k). Optionally add an LLM-as-judge step to rate answer quality (1–5) against the expected answer. Keep scoring optional so the script is useful even without ground-truth answers.
- [ ] **[MANUAL]** Review the existing validation set in `backend/` (notebooks / test data) to confirm it is still representative of real user queries before running a full evaluation.
- [ ] **[COLLAB]** Once evaluation results exist, surface key metrics in the README and use them to justify any future pipeline changes.

### 4.7 Potential Improvements (Not Yet Planned)

- [ ] **Streaming answer text**: currently the full answer arrives at once after tool steps stream. Could stream the answer text token-by-token for even better UX.
- [ ] **Experiment with tool configurations**: add/remove tools and test behavior. Investigate if SDK tool descriptions are sufficient without explicit system prompt guidance.
- [ ] **Evaluate disabling query rewriting in production**: confirmed redundant by notebook investigation. Currently `use_query_rewriting: false` in config — could remove the code entirely.
- [ ] **Update experiment framework for new architecture**: experiments still use old `pipeline.py`. Could add an experiment variant that runs through `agent_pipeline.py` for apples-to-apples comparison.
- [ ] **Dark mode testing**: verify the full UX in dark mode across devices.

---

## Architecture Summary (Current State)

```
Frontend (React 19 + Vite)
  ├── SSE streaming via POST /api/chat/stream
  ├── Live tool step checklist during loading
  ├── Sources section below answers
  └── "Surprise me" random question generator

Backend (FastAPI + OpenAI Agents SDK)
  ├── Agent: "Nexus" with AsyncOpenAI → Groq (groq/openai/gpt-oss-20b)
  ├── Tools: search_handbook, plan_searches, send_feedback, get_in_touch
  ├── Hybrid retrieval: semantic (ChromaDB) + BM25 keyword search
  ├── Config-driven: config.yaml controls model, retrieval, approach flags
  └── Chat logging: Supabase PostgreSQL (graceful failure)

Key Config (backend/config.yaml):
  model: groq/openai/gpt-oss-20b
  retrieval_k: 10, final_k: 10
  use_query_rewriting: false
  use_reranking: false
  use_keyword_search: true
```

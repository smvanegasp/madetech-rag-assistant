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
| OpenAI Agents SDK migration | `agent_pipeline.py` with `@function_tool`, `Runner.run_sync()`, `LitellmModel` |
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

### 4.5 Potential Improvements (Not Yet Planned)

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
  ├── Agent: "Nexus" with LitellmModel (groq/openai/gpt-oss-20b)
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
